"""Compute delta-LPIPS for same-named MP4 videos in two directories.

Delta-LPIPS is computed as:
    E_t[d_lpips(norm(frame_hat[t+n] - frame_hat[t]), norm(frame[t+n] - frame[t]))]

If two matched videos have different resolutions, target_dir deltas are resized
to the pred_dir resolution before RMS normalization.

python cosmos_predict2/_src/predict2/action/eval/delta_LIPIS.py \
#   outputs_eval/action_conditioned/basic/cosmos_predict_v2p5/000150000/20/model \
  outputs_eval/action_conditioned/basic/OF2/000001500/20/model \
#   outputs_eval/action_conditioned/basic/vjepa2/000001500/20/model \
#   outputs_eval/action_conditioned/basic/cotracker_tau=0/000001500/20/model \
#   outputs_eval/action_conditioned/basic/cotracker_4/000001500/20/model  \
  assets/action_conditioned/basic/bridge1/gt_video_renamed \
  -n 3 \
  --max-frames 36 \
  --net alex \
  --device cuda \
  --batch-size 256 \
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm


@dataclass(frozen=True)
class VideoPair:
    key: str
    pred_path: Path
    target_path: Path
    num_pred_frames: int
    num_target_frames: int
    num_eval_frames: int
    num_deltas: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the mean delta-LPIPS between same-named MP4 videos in two directories. "
            "Frames are sampled every n frames, and LPIPS is computed between adjacent sampled-frame deltas."
        )
    )
    parser.add_argument(
        "pred_dir",
        type=Path,
        help="Directory containing predicted/generated MP4 videos.",
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Directory containing target/reference MP4 videos.",
    )
    parser.add_argument(
        "-n",
        "--interval",
        type=int,
        required=True,
        help="Sample one frame every n frames before computing frame deltas.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Match MP4 videos by relative path recursively instead of only by filename in the top-level directory.",
    )
    parser.add_argument(
        "--net",
        default="alex",
        choices=("alex", "vgg", "squeeze"),
        help="LPIPS backbone network.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for LPIPS computation. Use 'auto' to prefer CUDA when available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of delta pairs evaluated by LPIPS at once.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads used by decord for video decoding.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Only use the first N frames of every matched video pair.",
    )
    parser.add_argument(
        "--rms-eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon used by RMS normalization.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip unreadable or incompatible videos instead of failing immediately.",
    )
    parser.add_argument(
        "--print-per-video",
        action="store_true",
        help="Print the mean delta-LPIPS for each successfully evaluated video.",
    )
    return parser.parse_args()


def find_mp4_files(video_dir: Path, recursive: bool) -> dict[str, Path]:
    candidates = video_dir.rglob("*") if recursive else video_dir.glob("*")
    paths = sorted(path for path in candidates if path.is_file() and path.suffix.lower() == ".mp4")
    if recursive:
        return {path.relative_to(video_dir).as_posix(): path for path in paths}
    return {path.name: path for path in paths}


def import_decord() -> Any:
    try:
        import decord
    except ImportError as exc:
        raise ImportError("Please install decord first, e.g. `pip install decord`.") from exc
    return decord


def import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Please install torch first.") from exc
    return torch


def get_num_frames(video_path: Path, num_threads: int) -> int:
    decord = import_decord()
    return len(decord.VideoReader(str(video_path), num_threads=num_threads))


def build_video_pairs(
    pred_dir: Path,
    target_dir: Path,
    interval: int,
    recursive: bool,
    num_threads: int,
    max_frames: int | None,
    skip_errors: bool,
) -> tuple[list[VideoPair], list[tuple[str, str]]]:
    pred_files = find_mp4_files(pred_dir, recursive=recursive)
    target_files = find_mp4_files(target_dir, recursive=recursive)
    common_keys = sorted(set(pred_files) & set(target_files))
    if not common_keys:
        raise FileNotFoundError(f"No same-named MP4 videos found in {pred_dir} and {target_dir}")

    pairs: list[VideoPair] = []
    failed: list[tuple[str, str]] = []
    for key in common_keys:
        pred_path = pred_files[key]
        target_path = target_files[key]
        try:
            num_pred_frames = get_num_frames(pred_path, num_threads=num_threads)
            num_target_frames = get_num_frames(target_path, num_threads=num_threads)
            num_eval_frames = min(num_pred_frames, num_target_frames)
            if max_frames is not None:
                num_eval_frames = min(num_eval_frames, max_frames)
            num_sampled = len(range(0, num_eval_frames, interval))
            if num_sampled < 2:
                raise ValueError(
                    f"Not enough sampled frames for interval={interval}: "
                    f"pred_frames={num_pred_frames}, target_frames={num_target_frames}, "
                    f"eval_frames={num_eval_frames}"
                )
        except Exception as exc:
            if not skip_errors:
                raise
            failed.append((key, str(exc)))
            continue

        pairs.append(
            VideoPair(
                key=key,
                pred_path=pred_path,
                target_path=target_path,
                num_pred_frames=num_pred_frames,
                num_target_frames=num_target_frames,
                num_eval_frames=num_eval_frames,
                num_deltas=num_sampled - 1,
            )
        )

    if not pairs:
        raise RuntimeError("No video pairs can be evaluated.")
    return pairs, failed


def read_delta_batch(
    video_reader: Any,
    start_indices: list[int],
    end_indices: list[int],
    device: Any,
) -> Any:
    torch = import_torch()
    frame_indices = [idx for pair in zip(start_indices, end_indices) for idx in pair]
    frames = video_reader.get_batch(frame_indices).asnumpy()
    frames_tensor = torch.from_numpy(frames).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
    first = frames_tensor[0::2]
    second = frames_tensor[1::2]
    return second - first


def rms_normalize(delta: Any, rms_eps: float) -> Any:
    torch = import_torch()
    rms = torch.sqrt(torch.mean(delta * delta, dim=(1, 2, 3), keepdim=True) + rms_eps)
    return delta / rms


def resize_target_delta_to_pred(
    pred_delta: Any,
    target_delta: Any,
) -> Any:
    if pred_delta.shape[-2:] == target_delta.shape[-2:]:
        return target_delta
    torch = import_torch()
    return torch.nn.functional.interpolate(
        target_delta,
        size=pred_delta.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def compute_pair_delta_lpips(
    pair: VideoPair,
    loss_fn: Any,
    device: Any,
    interval: int,
    batch_size: int,
    num_threads: int,
    rms_eps: float,
    progress: tqdm,
) -> tuple[float, int]:
    torch = import_torch()
    decord = import_decord()
    pred_reader = decord.VideoReader(str(pair.pred_path), num_threads=num_threads)
    target_reader = decord.VideoReader(str(pair.target_path), num_threads=num_threads)
    sampled_indices = list(range(0, pair.num_eval_frames, interval))

    score_sum = 0.0
    score_count = 0
    processed = 0
    try:
        for batch_start in range(0, pair.num_deltas, batch_size):
            batch_sample_indices = sampled_indices[batch_start : batch_start + batch_size + 1]
            start_indices = batch_sample_indices[:-1]
            end_indices = batch_sample_indices[1:]
            if not start_indices:
                continue

            pred_delta = read_delta_batch(pred_reader, start_indices, end_indices, device=device)
            target_delta = read_delta_batch(target_reader, start_indices, end_indices, device=device)
            target_delta = resize_target_delta_to_pred(pred_delta=pred_delta, target_delta=target_delta)
            pred_delta = rms_normalize(pred_delta, rms_eps=rms_eps)
            target_delta = rms_normalize(target_delta, rms_eps=rms_eps)

            with torch.inference_mode():
                distances = loss_fn(pred_delta, target_delta).flatten()

            batch_count = int(distances.numel())
            score_sum += float(distances.sum().item())
            score_count += batch_count
            processed += batch_count
            progress.update(batch_count)
    except Exception:
        progress.update(pair.num_deltas - processed)
        raise

    return score_sum, score_count


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError(f"--interval must be > 0, got {args.interval}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {args.batch_size}")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError(f"--max-frames must be > 0, got {args.max_frames}")

    pred_dir = args.pred_dir.expanduser().resolve()
    target_dir = args.target_dir.expanduser().resolve()
    if not pred_dir.is_dir():
        raise NotADirectoryError(f"Prediction directory does not exist: {pred_dir}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"Target directory does not exist: {target_dir}")

    pairs, failed = build_video_pairs(
        pred_dir=pred_dir,
        target_dir=target_dir,
        interval=args.interval,
        recursive=args.recursive,
        num_threads=args.num_threads,
        max_frames=args.max_frames,
        skip_errors=args.skip_errors,
    )

    try:
        import lpips
    except ImportError as exc:
        raise ImportError("Please install lpips first, e.g. `pip install lpips`.") from exc

    torch = import_torch()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device(device_name)
    loss_fn = lpips.LPIPS(net=args.net).to(device).eval()

    total_score_sum = 0.0
    total_score_count = 0
    evaluated_pairs = 0
    total_deltas = sum(pair.num_deltas for pair in pairs)
    progress = tqdm(total=total_deltas, desc="Computing delta-LPIPS", unit="delta")
    for pair in pairs:
        try:
            pair_score_sum, pair_score_count = compute_pair_delta_lpips(
                pair=pair,
                loss_fn=loss_fn,
                device=device,
                interval=args.interval,
                batch_size=args.batch_size,
                num_threads=args.num_threads,
                rms_eps=args.rms_eps,
                progress=progress,
            )
        except Exception as exc:
            if not args.skip_errors:
                progress.close()
                raise
            failed.append((pair.key, str(exc)))
            progress.set_postfix({"avg": f"{total_score_sum / total_score_count:.6f}" if total_score_count else "n/a"})
            continue

        total_score_sum += pair_score_sum
        total_score_count += pair_score_count
        evaluated_pairs += 1
        pair_mean = pair_score_sum / pair_score_count
        progress.set_postfix({"avg": f"{total_score_sum / total_score_count:.6f}"})
        if args.print_per_video:
            tqdm.write(f"{pair.key}: {pair_mean:.6f} ({pair_score_count} delta points)")
    progress.close()

    if total_score_count == 0:
        raise RuntimeError("No delta-LPIPS points were successfully evaluated.")

    print(f"Video pairs evaluated: {evaluated_pairs}")
    print(f"Delta points evaluated: {total_score_count}")
    print(f"Average delta-LPIPS: {total_score_sum / total_score_count:.6f}")
    if failed:
        print(f"Videos skipped or failed: {len(failed)}")
        for key, error in failed:
            print(f"- {key}: {error}")


if __name__ == "__main__":
    main()
