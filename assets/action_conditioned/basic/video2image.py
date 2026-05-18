"""
python assets/action_conditioned/basic/video2image.py /inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/outputs/action_conditioned/basic/OF/000001000/20/model/compare \
  --frame-interval 6 \
  --max-frames 6
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def resize_frame(frame, size=None):
    """
    size: None or (width, height)
    """
    if size is None:
        return frame

    width, height = size
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def get_output_path(video_path, output_dir=None, output_path=None):
    """
    默认输出：
        input.mp4 -> input.jpg

    如果指定 output_dir：
        /path/to/input.mp4 -> output_dir/input.jpg

    如果指定 output_path：
        仅在处理单个视频文件时使用。
    """
    video_path = Path(video_path)

    if output_path is not None:
        return Path(output_path)

    if output_dir is not None:
        return Path(output_dir) / f"{video_path.stem}.jpg"

    return video_path.with_suffix(".jpg")


def extract_frames_to_image(
    video_path,
    output_path=None,
    output_dir=None,
    resize=None,
    target_fps=None,
    frame_interval=None,
    max_frames=None,
):
    video_path = Path(video_path)
    save_path = get_output_path(
        video_path=video_path,
        output_dir=output_dir,
        output_path=output_path,
    )

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_fps <= 0:
        cap.release()
        raise RuntimeError(f"Failed to read video FPS: {video_path}")

    print("=" * 80)
    print(f"Input video: {video_path}")
    print(f"Original FPS: {original_fps:.2f}")
    print(f"Total frames: {total_frames}")

    if frame_interval is not None:
        if frame_interval <= 0:
            cap.release()
            raise ValueError("--frame-interval must be positive.")
        step = frame_interval
    elif target_fps is not None:
        if target_fps <= 0:
            cap.release()
            raise ValueError("--target-fps must be positive.")
        step = max(1, round(original_fps / target_fps))
    else:
        step = 1

    print(f"Sampling every {step} frame(s).")

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            frame = resize_frame(frame, resize)
            frames.append(frame)

            if max_frames is not None and len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames were extracted from: {video_path}")

    print(f"Extracted frames: {len(frames)}")

    concat_img = np.concatenate(frames, axis=1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(save_path), concat_img)

    if not success:
        raise RuntimeError(f"Failed to save image: {save_path}")

    print(f"Saved to: {save_path}")
    print(f"Output image shape: {concat_img.shape}")


def collect_videos(input_path):
    """
    input_path 可以是：
    1. 单个视频文件
    2. 包含视频的目录

    若为目录，只读取当前目录下的视频，不递归子目录。
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f"Input file is not a supported video: {input_path}")
        return [input_path]

    if input_path.is_dir():
        videos = [
            p for p in sorted(input_path.iterdir())
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        ]

        if len(videos) == 0:
            raise RuntimeError(f"No video files found in directory: {input_path}")

        return videos

    raise ValueError(f"Unsupported input path: {input_path}")


def parse_resize(value):
    """
    Parse resize string like:
        320x240
    """
    if value is None:
        return None

    try:
        width, height = value.lower().split("x")
        return int(width), int(height)
    except Exception:
        raise argparse.ArgumentTypeError(
            "Resize format should be like 320x240"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames from video(s) and concatenate them horizontally. "
            "Input can be a single video file or a directory containing videos."
        )
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to input video file or directory containing videos."
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=(
            "Path to output image. Only valid when input is a single video file. "
            "Default: same directory and same filename as input video, with .jpg suffix."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to save output images. "
            "Default: same directory as each input video."
        )
    )

    parser.add_argument(
        "--resize",
        type=parse_resize,
        default=None,
        help="Resize each extracted frame, e.g. 320x240."
    )

    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help=(
            "Target sampling FPS. For example, if video is 30 FPS and "
            "target-fps=2, sample about 2 frames per second."
        )
    )

    parser.add_argument(
        "--frame-interval",
        type=int,
        default=None,
        help="Sample every N frames. This has higher priority than --target-fps."
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract from each video."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    videos = collect_videos(input_path)

    if input_path.is_dir() and args.output is not None:
        raise ValueError(
            "--output can only be used when input is a single video file. "
            "Use --output-dir for directory input."
        )

    print(f"Found {len(videos)} video(s).")

    for video_path in videos:
        try:
            extract_frames_to_image(
                video_path=video_path,
                output_path=args.output if len(videos) == 1 else None,
                output_dir=args.output_dir,
                resize=args.resize,
                target_fps=args.target_fps,
                frame_interval=args.frame_interval,
                max_frames=args.max_frames,
            )
        except Exception as e:
            print(f"[ERROR] Failed to process {video_path}: {e}")


if __name__ == "__main__":
    main()