"""
DIR_A="/inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/outputs/action_conditioned/basic/OF/000001000/20/model"
python /inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/assets/action_conditioned/basic/concate_videos.py \
  --dir_a $DIR_A \
  --dir_b /inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/outputs/action_conditioned/basic/original/iter_000150000_model_ema_fp32.pt \
  --dir_c /inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/outputs/action_conditioned/basic/gt \
  --output_dir ${DIR_A}/compare \
  --fps 3 \
  --overwrite
"""
# a: left top; b: right top; c: bottom

import argparse
import subprocess
from pathlib import Path

import cv2


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def list_videos_recursive(video_dir: Path):
    """
    递归列出目录及其所有子目录下的视频文件。

    返回:
        {
            relative_path: absolute_path
        }

    例如:
        A/exp1/a.mp4 -> "exp1/a.mp4"
        A/exp2/a.mp4 -> "exp2/a.mp4"
    """
    videos = {}

    for p in video_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            rel_path = p.relative_to(video_dir)
            videos[rel_path] = p

    return videos


def get_video_info(cap: cv2.VideoCapture, path: Path):
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")

    return frame_count, width, height


def resize_to(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def run_ffmpeg_h264_transcode(
    input_path: Path,
    output_path: Path,
    fps: float | None = None,
    overwrite: bool = True,
):
    """
    将 OpenCV 生成的临时视频转码为 H.264 mp4。

    最终编码:
        video codec: H.264 / libx264
        pixel format: yuv420p
        container: mp4
    """
    cmd = ["ffmpeg"]

    if overwrite:
        cmd.append("-y")
    else:
        cmd.append("-n")

    cmd += [
        "-i",
        str(input_path),
    ]

    if fps is not None:
        cmd += ["-r", str(fps)]

    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg first, "
            "or make sure it is available in PATH."
        )


def concat_three_videos_to_temp(
    path_a: Path,
    path_b: Path,
    path_c: Path,
    temp_output_path: Path,
    output_fps: float,
):
    """
    用 OpenCV 生成临时视频。

    注意：
    这里不直接输出最终文件。
    最终 H.264 文件由 ffmpeg 转码得到。
    """
    cap_a = cv2.VideoCapture(str(path_a))
    cap_b = cv2.VideoCapture(str(path_b))
    cap_c = cv2.VideoCapture(str(path_c))

    writer = None

    try:
        frames_a, width_a, height_a = get_video_info(cap_a, path_a)
        frames_b, width_b, height_b = get_video_info(cap_b, path_b)
        frames_c, width_c, height_c = get_video_info(cap_c, path_c)

        # 只要求 A 和 B 帧数相同
        if frames_a != frames_b:
            raise ValueError(
                f"Frame count mismatch between A and B for {path_a}: "
                f"A={frames_a}, B={frames_b}"
            )

        # C 不能比 A/B 更长
        if frames_c > frames_a:
            raise ValueError(
                f"C has more frames than A/B for {path_a}: "
                f"A/B={frames_a}, C={frames_c}"
            )

        # 若 C 更短，则按 C 的长度输出，相当于截断 A/B
        output_frames = frames_c

        # A 和 B 左右拼接，要求尺寸一致
        if width_a != width_b or height_a != height_b:
            raise ValueError(
                f"Size mismatch between A and B for {path_a}: "
                f"A=({width_a}, {height_a}), B=({width_b}, {height_b})"
            )

        # C 缩放到 A.width + B.width，保持比例
        target_c_width = width_a + width_b
        target_c_height = int(height_c * target_c_width / width_c)

        output_width = width_a + width_b
        output_height = height_a + target_c_height

        temp_output_path.parent.mkdir(parents=True, exist_ok=True)

        # OpenCV 临时写 mp4v，之后再用 ffmpeg 转 H.264
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(temp_output_path),
            fourcc,
            output_fps,
            (output_width, output_height),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Failed to create temp video: {temp_output_path}")

        for idx in range(output_frames):
            ret_a, frame_a = cap_a.read()
            ret_b, frame_b = cap_b.read()
            ret_c, frame_c = cap_c.read()

            if not (ret_a and ret_b and ret_c):
                raise RuntimeError(
                    f"Failed to read frame {idx} from one of the videos: "
                    f"{path_a}"
                )

            if frame_a.shape[1] != width_a or frame_a.shape[0] != height_a:
                frame_a = resize_to(frame_a, width_a, height_a)

            if frame_b.shape[1] != width_b or frame_b.shape[0] != height_b:
                frame_b = resize_to(frame_b, width_b, height_b)

            frame_c_scaled = resize_to(frame_c, target_c_width, target_c_height)

            top = cv2.hconcat([frame_a, frame_b])
            final_frame = cv2.vconcat([top, frame_c_scaled])

            writer.write(final_frame)

    finally:
        cap_a.release()
        cap_b.release()
        cap_c.release()

        if writer is not None:
            writer.release()


def concat_three_videos_h264(
    path_a: Path,
    path_b: Path,
    path_c: Path,
    output_path: Path,
    output_fps: float,
    overwrite: bool,
    keep_temp: bool = False,
):
    """
    对外使用的主函数：
    1. OpenCV 写临时视频
    2. ffmpeg 转码为 H.264
    3. 删除临时视频
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_output_path = output_path.with_name(
        output_path.stem + ".tmp" + output_path.suffix
    )

    if temp_output_path.exists():
        temp_output_path.unlink()

    concat_three_videos_to_temp(
        path_a=path_a,
        path_b=path_b,
        path_c=path_c,
        temp_output_path=temp_output_path,
        output_fps=output_fps,
    )

    run_ffmpeg_h264_transcode(
        input_path=temp_output_path,
        output_path=output_path,
        fps=output_fps,
        overwrite=overwrite,
    )

    if not keep_temp and temp_output_path.exists():
        temp_output_path.unlink()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir_a", type=str, required=True)
    parser.add_argument("--dir_b", type=str, required=True)
    parser.add_argument("--dir_c", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="FPS of output videos.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output videos.",
    )

    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep temporary mp4v videos.",
    )

    args = parser.parse_args()

    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)
    dir_c = Path(args.dir_c)
    output_dir = Path(args.output_dir)

    videos_a = list_videos_recursive(dir_a)
    videos_b = list_videos_recursive(dir_b)
    videos_c = list_videos_recursive(dir_c)

    common_names = sorted(
        set(videos_a.keys()) & set(videos_b.keys()) & set(videos_c.keys())
    )

    if not common_names:
        raise RuntimeError("No common video relative paths found in A, B, C directories.")

    missing_in_a = (set(videos_b.keys()) | set(videos_c.keys())) - set(videos_a.keys())
    missing_in_b = (set(videos_a.keys()) | set(videos_c.keys())) - set(videos_b.keys())
    missing_in_c = (set(videos_a.keys()) | set(videos_b.keys())) - set(videos_c.keys())

    if missing_in_a or missing_in_b or missing_in_c:
        print("[Warning] Some videos are not shared by all three directories.")
        if missing_in_a:
            print(f"Missing in A: {[str(x) for x in sorted(missing_in_a)]}")
        if missing_in_b:
            print(f"Missing in B: {[str(x) for x in sorted(missing_in_b)]}")
        if missing_in_c:
            print(f"Missing in C: {[str(x) for x in sorted(missing_in_c)]}")

    print(f"Found {len(common_names)} common videos.")

    for rel_path in common_names:
        path_a = videos_a[rel_path]
        path_b = videos_b[rel_path]
        path_c = videos_c[rel_path]

        # 保留相对目录结构输出
        output_path = output_dir / rel_path

        # 为了保证最终是 mp4 + H.264，这里强制输出后缀为 .mp4
        output_path = output_path.with_suffix(".mp4")

        if output_path.exists() and not args.overwrite:
            print(f"[Skip] Output exists: {output_path}")
            continue

        print(f"[Processing] {rel_path}")

        concat_three_videos_h264(
            path_a=path_a,
            path_b=path_b,
            path_c=path_c,
            output_path=output_path,
            output_fps=args.fps,
            overwrite=args.overwrite,
            keep_temp=args.keep_temp,
        )

        print(f"[Done] {output_path}")


if __name__ == "__main__":
    main()