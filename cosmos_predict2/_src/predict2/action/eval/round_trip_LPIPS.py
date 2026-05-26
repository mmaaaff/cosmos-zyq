"""
Compute round-trip LPIPS between the first and last frame of MP4 videos.

python cosmos_predict2/_src/predict2/action/eval/round_trip_LPIPS.py /inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/outputs_eval/action_conditioned/basic/vjepa2/000001500/20/2_chunks/model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import decord
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the average LPIPS distance between first and last frames for MP4 videos."
    )
    parser.add_argument(
        "video_dir",
        type=Path,
        help="Directory containing MP4 videos.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for MP4 videos recursively.",
    )
    parser.add_argument(
        "--net",
        default="alex",
        choices=("alex", "vgg", "squeeze"),
        help="LPIPS backbone network.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for LPIPS computation.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads used by decord for video decoding.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip unreadable videos instead of failing immediately.",
    )
    return parser.parse_args()


def find_mp4_files(video_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.mp4" if recursive else "*.mp4"
    return sorted(path for path in video_dir.glob(pattern) if path.is_file())


def first_last_frames(video_path: Path, num_threads: int) -> torch.Tensor:
    vr = decord.VideoReader(str(video_path), num_threads=num_threads)
    num_frames = len(vr)
    if num_frames < 2:
        raise ValueError(f"{video_path} has fewer than 2 frames.")

    frames = vr.get_batch([0, num_frames - 1]).asnumpy()
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    return tensor / 127.5 - 1.0


def compute_video_lpips(
    video_path: Path,
    loss_fn: torch.nn.Module,
    device: torch.device,
    num_threads: int,
) -> float:
    frames = first_last_frames(video_path, num_threads=num_threads).to(device)
    with torch.no_grad():
        distance = loss_fn(frames[:1], frames[1:2])
    return float(distance.item())


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir.expanduser().resolve()
    if not video_dir.is_dir():
        raise NotADirectoryError(f"Video directory does not exist: {video_dir}")

    video_paths = find_mp4_files(video_dir, recursive=args.recursive)
    if not video_paths:
        raise FileNotFoundError(f"No MP4 videos found in {video_dir}")

    try:
        import lpips
    except ImportError as exc:
        raise ImportError("Please install lpips first, e.g. `pip install lpips`.") from exc

    device = torch.device(args.device)
    loss_fn = lpips.LPIPS(net=args.net).to(device).eval()

    scores: list[float] = []
    failed: list[tuple[Path, str]] = []
    progress = tqdm(video_paths, desc="Computing round-trip LPIPS", unit="video")
    for video_path in progress:
        try:
            score = compute_video_lpips(
                video_path=video_path,
                loss_fn=loss_fn,
                device=device,
                num_threads=args.num_threads,
            )
        except Exception as exc:
            if not args.skip_errors:
                raise
            failed.append((video_path, str(exc)))
            progress.set_postfix({"avg": f"{sum(scores) / len(scores):.6f}" if scores else "n/a"})
            continue

        scores.append(score)
        progress.set_postfix({"avg": f"{sum(scores) / len(scores):.6f}"})

    if not scores:
        raise RuntimeError("No videos were successfully evaluated.")

    mean_score = sum(scores) / len(scores)
    print(f"Videos evaluated: {len(scores)}")
    print(f"Average round-trip LPIPS: {mean_score:.6f}")

    if failed:
        print(f"Videos skipped: {len(failed)}")
        for path, error in failed:
            print(f"- {path}: {error}")


if __name__ == "__main__":
    main()
