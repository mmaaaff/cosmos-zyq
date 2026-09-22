"""
Extract frames from mp4 videos in a directory and concatenate each video's
sampled frames horizontally into one image.

Example:
python assets/action_conditioned/basic/extract_frames.py \
    assets/action_conditioned/basic/bridge1/gt_video_renamed \
    --frame-interval 4 \
    --max-video-frames 32

Output:
    /path/videos_images/video_name.jpg
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXT = ".mp4"


def get_output_dir(input_dir: Path) -> Path:
    return Path(f"{input_dir}_images")


def collect_mp4_videos(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    videos = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() == VIDEO_EXT
    ]

    if len(videos) == 0:
        raise RuntimeError(f"No mp4 videos found in directory: {input_dir}")

    return videos


def extract_sampled_frames(
    video_path: Path,
    frame_interval: int,
    max_video_frames: int | None,
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    frame_idx = 0
    last_frame = None
    last_frame_idx = None
    last_sampled_idx = None
    stopped_by_max_frames = False

    try:
        while True:
            if max_video_frames is not None and frame_idx >= max_video_frames:
                stopped_by_max_frames = True
                break

            ret, frame = cap.read()
            if not ret:
                break

            last_frame = frame
            last_frame_idx = frame_idx

            if frame_idx % frame_interval == 0:
                frames.append(frame)
                last_sampled_idx = frame_idx

            frame_idx += 1
    finally:
        cap.release()

    if (
        max_video_frames is not None
        and not stopped_by_max_frames
        and last_frame is not None
        and last_sampled_idx != last_frame_idx
    ):
        frames.append(last_frame)

    if len(frames) == 0:
        raise RuntimeError(f"No frames were extracted from: {video_path}")

    return frames


def save_horizontal_strip(frames: list[np.ndarray], output_path: Path) -> None:
    image = np.concatenate(frames, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise RuntimeError(f"Failed to save image: {output_path}")


def process_video(
    video_path: Path,
    output_dir: Path,
    frame_interval: int,
    max_video_frames: int | None,
) -> None:
    output_path = output_dir / f"{video_path.stem}.jpg"
    frames = extract_sampled_frames(
        video_path=video_path,
        frame_interval=frame_interval,
        max_video_frames=max_video_frames,
    )
    save_horizontal_strip(frames, output_path)
    print(f"[OK] {video_path} -> {output_path} ({len(frames)} frame(s))")


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be an integer.") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive.")

    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract sampled frames from mp4 videos in a directory, concatenate "
            "each video's frames horizontally, and save images to "
            "<input_dir>_images."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing mp4 videos. Only the current directory is scanned.",
    )
    parser.add_argument(
        "--frame-interval",
        type=positive_int,
        default=1,
        help="Sample one frame every N frames. Default: 1.",
    )
    parser.add_argument(
        "--max-video-frames",
        "--max-frames",
        dest="max_video_frames",
        type=positive_int,
        default=None,
        help=(
            "Maximum number of source video frames to process. Videos with more "
            "frames are truncated before sampling."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = get_output_dir(input_dir)
    videos = collect_mp4_videos(input_dir)

    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path exists but is not a directory: {output_dir}")
        shutil.rmtree(output_dir)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(videos)} mp4 video(s).")
    print(f"Frame interval: {args.frame_interval}")
    if args.max_video_frames is not None:
        print(f"Max source video frames: {args.max_video_frames}")

    failed = 0
    for video_path in videos:
        try:
            process_video(
                video_path=video_path,
                output_dir=output_dir,
                frame_interval=args.frame_interval,
                max_video_frames=args.max_video_frames,
            )
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {video_path}: {exc}")

    if failed > 0:
        raise SystemExit(f"Failed to process {failed} video(s).")


if __name__ == "__main__":
    main()
