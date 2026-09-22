"""
Vertically concatenate same-name jpg images from multiple directories and add
one label column on the left.

Example:

python3 assets/action_conditioned/basic/concat_labeled_images.py \
  --dirs \
    assets/action_conditioned/basic/bridge1/gt_video_renamed_images \
    outputs_eval/action_conditioned/basic/cosmos_predict_v2p5/000150000/20/model_images \
    outputs_eval/action_conditioned/basic/OF2/000001500/20/model_images \
    outputs_eval/action_conditioned/basic/vjepa2/000001500/20/model_images \
    outputs_eval/action_conditioned/basic/cotracker_tau=0/000001500/20/model_images \
    outputs_eval/action_conditioned/basic/cotracker_4/000001500/20/model_images \
    outputs_eval/action_conditioned/basic/mixed_reward_0.7of_0.3vjepa1/000000900/20/model_images \
  --labels \
    "Ground Truth" \
    "Original Model" \
    "W/ Optical Flow" \
    "W/ V-JEPA2" \
    "W/ Cotracker3 tau=0" \
    "W/ CoTracker3 tau=1" \
    "W/ Mixed Reward" \
  --output-dir /inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/assets/eval

Output:
    /path/a_images_vconcat/image_name.jpg
"""

import argparse
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg"}

cv2 = None
np = None


def require_image_deps() -> None:
    global cv2, np

    if cv2 is not None and np is not None:
        return

    try:
        import cv2 as cv2_module
        import numpy as np_module
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This script requires opencv-python and numpy. "
            "Please install them in the Python environment used to run it."
        ) from exc

    cv2 = cv2_module
    np = np_module


def get_default_output_dir(first_input_dir: Path) -> Path:
    return Path(f"{first_input_dir}_vconcat")


def collect_images(image_dir: Path) -> dict[str, Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {image_dir}")

    if not image_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {image_dir}")

    images = {}
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images[path.name] = path

    if len(images) == 0:
        raise RuntimeError(f"No jpg/jpeg images found in directory: {image_dir}")

    return images


def find_common_image_names(image_maps: list[dict[str, Path]]) -> list[str]:
    common_names = set(image_maps[0].keys())
    for image_map in image_maps[1:]:
        common_names &= set(image_map.keys())

    if len(common_names) == 0:
        raise RuntimeError("No same-name jpg/jpeg images found across all directories.")

    return sorted(common_names)


def load_images(image_name: str, image_maps: list[dict[str, Path]]):
    images = []
    for image_map in image_maps:
        image_path = image_map[image_name]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        images.append(image)
    return images


def resize_to_min_size(images):
    min_height = min(image.shape[0] for image in images)
    min_width = min(image.shape[1] for image in images)
    target_size = (min_width, min_height)

    resized = []
    for image in images:
        if image.shape[1] != min_width or image.shape[0] != min_height:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        resized.append(image)

    return resized


def build_label_panel(
    labels: list[str],
    row_height: int,
    label_width: int,
    font_scale: float,
    font_thickness: int,
    label_padding: int,
    background: tuple[int, int, int],
    text_color: tuple[int, int, int],
):
    panel = np.full(
        (row_height * len(labels), label_width, 3),
        background,
        dtype=np.uint8,
    )

    for idx, label in enumerate(labels):
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )
        x = max(label_padding, label_width - label_padding - text_width)
        y = idx * row_height + (row_height + text_height) // 2
        cv2.putText(
            panel,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

    return panel


def build_labeled_concat(
    images,
    labels: list[str],
    font_scale: float,
    font_thickness: int,
    label_padding: int,
    background: tuple[int, int, int],
    text_color: tuple[int, int, int],
):
    resized_images = resize_to_min_size(images)
    row_height = resized_images[0].shape[0]

    text_widths = [
        cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )[0][0]
        for label in labels
    ]
    label_width = max(text_widths) + label_padding * 2

    label_panel = build_label_panel(
        labels=labels,
        row_height=row_height,
        label_width=label_width,
        font_scale=font_scale,
        font_thickness=font_thickness,
        label_padding=label_padding,
        background=background,
        text_color=text_color,
    )
    image_panel = np.concatenate(resized_images, axis=0)

    return np.concatenate([label_panel, image_panel], axis=1)


def process_image_name(
    image_name: str,
    image_maps: list[dict[str, Path]],
    labels: list[str],
    output_dir: Path,
    font_scale: float,
    font_thickness: int,
    label_padding: int,
    background: tuple[int, int, int],
    text_color: tuple[int, int, int],
) -> None:
    images = load_images(image_name, image_maps)
    output = build_labeled_concat(
        images=images,
        labels=labels,
        font_scale=font_scale,
        font_thickness=font_thickness,
        label_padding=label_padding,
        background=background,
        text_color=text_color,
    )

    output_path = output_dir / image_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(output_path), output)
    if not ok:
        raise RuntimeError(f"Failed to save image: {output_path}")

    print(f"[OK] {image_name} -> {output_path}")


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be an integer.") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive.")

    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be a number.") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive.")

    return parsed


def parse_color(value: str) -> tuple[int, int, int]:
    try:
        parts = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Color must be formatted like 255,255,255.") from exc

    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise argparse.ArgumentTypeError("Color must be three integers in [0, 255].")

    # OpenCV uses BGR internally.
    return parts[2], parts[1], parts[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find same-name jpg/jpeg images across N directories, resize each "
            "group to the smallest width and height in that group, concatenate "
            "them vertically, and draw N labels on the left."
        )
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        type=Path,
        help="Input directories containing jpg/jpeg images.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels to draw on the left. The count must match --dirs.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <first_input_dir>_vconcat.",
    )
    parser.add_argument(
        "--font-scale",
        type=positive_float,
        default=1.0,
        help="OpenCV label font scale. Default: 1.0.",
    )
    parser.add_argument(
        "--font-thickness",
        type=positive_int,
        default=2,
        help="OpenCV label font thickness. Default: 2.",
    )
    parser.add_argument(
        "--label-padding",
        type=positive_int,
        default=16,
        help="Horizontal padding for the left label column. Default: 16.",
    )
    parser.add_argument(
        "--background",
        type=parse_color,
        default=(255, 255, 255),
        help="Background color as R,G,B. Default: 255,255,255.",
    )
    parser.add_argument(
        "--text-color",
        type=parse_color,
        default=(0, 0, 0),
        help="Text color as R,G,B. Default: 0,0,0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.dirs) != len(args.labels):
        raise ValueError(f"--dirs count ({len(args.dirs)}) must match --labels count ({len(args.labels)}).")

    require_image_deps()

    input_dirs = [path.expanduser().resolve() for path in args.dirs]
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else get_default_output_dir(input_dirs[0])
    )

    image_maps = [collect_images(image_dir) for image_dir in input_dirs]
    image_names = find_common_image_names(image_maps)

    print(f"Input directories: {len(input_dirs)}")
    for image_dir, label in zip(input_dirs, args.labels):
        print(f"  {label}: {image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Common images: {len(image_names)}")

    failed = 0
    for image_name in image_names:
        try:
            process_image_name(
                image_name=image_name,
                image_maps=image_maps,
                labels=args.labels,
                output_dir=output_dir,
                font_scale=args.font_scale,
                font_thickness=args.font_thickness,
                label_padding=args.label_padding,
                background=args.background,
                text_color=args.text_color,
            )
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {image_name}: {exc}")

    if failed > 0:
        raise SystemExit(f"Failed to process {failed} image(s).")


if __name__ == "__main__":
    main()
