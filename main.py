"""
Person Re-Identification System — Phase 3
==========================================

Entry point for the multi-camera detection, tracking, and
person re-identification pipeline.

Usage::

    # Without ReID (Phase 1/2 mode)
    python main.py --videos cam1.mp4 cam2.mp4

    # With ReID (Phase 3)
    python main.py --videos cam1.mp4 cam2.mp4 --reid

    # Custom similarity threshold
    python main.py --videos cam1.mp4 cam2.mp4 --reid --similarity-threshold 0.70

    # Headless mode
    python main.py --videos cam1.mp4 cam2.mp4 --reid --no-display
"""

import argparse
import sys

from camera.camera_manager import CameraManager


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3 — Multi-Camera Person Detection, Tracking "
            "& Re-Identification (YOLOv8s + DeepSORT + OSNet)"
        )
    )
    parser.add_argument(
        "--videos",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to input video file(s).  One camera pipeline "
        "is created per video.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Base output directory (default: outputs/).  Each camera "
        "writes to <output>/camera_<id>/.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Path to YOLO weights (default: yolov8s.pt, auto-downloads).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Minimum detection confidence (default: 0.6).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Compute device: cpu, cuda, mps, or '' for auto (default: '').",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable real-time display windows (headless mode).",
    )
    parser.add_argument(
        "--min-bbox-width",
        type=int,
        default=40,
        help="Minimum detection bbox width in pixels (default: 40).",
    )
    parser.add_argument(
        "--min-bbox-height",
        type=int,
        default=60,
        help="Minimum detection bbox height in pixels (default: 60).",
    )

    # ── Phase 3: ReID flags ───────────────────────────────────────────
    parser.add_argument(
        "--reid",
        action="store_true",
        help="Enable Person Re-Identification using OSNet.  "
        "Requires torchreid to be installed.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Minimum cosine similarity for a ReID match "
        "(default: 0.75).  Only used when --reid is enabled.",
    )
    parser.add_argument(
        "--reid-interval",
        type=int,
        default=15,
        help="Frames between ReID re-extractions per track "
        "(default: 15).  Lower = more accurate, slower.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed ReID matching decisions "
        "(candidate rejections, match scores).",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    try:
        manager = CameraManager(
            video_paths=args.videos,
            output_dir=args.output,
            model_path=args.model,
            confidence_threshold=args.confidence,
            device=args.device,
            min_bbox_width=args.min_bbox_width,
            min_bbox_height=args.min_bbox_height,
            display=not args.no_display,
            reid_enabled=args.reid,
            similarity_threshold=args.similarity_threshold,
            reid_interval=args.reid_interval,
            debug=args.debug,
        )

        results = manager.run_all()
        manager.print_summary(results)

    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except ImportError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
