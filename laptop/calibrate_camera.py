#!/usr/bin/env python3
"""Calibrate a pinhole camera from saved ChArUco board photographs.

    python3 laptop/calibrate_camera.py --images calibration_images --debug-dir detections

Defaults: 8 x 11 squares, 15 mm squares, 11 mm markers, standard marker IDs
starting at zero. Requires OpenCV 4.8+ with aruco.CharucoDetector.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


DICTIONARIES = tuple(f"DICT_6X6_{count}" for count in (50, 100, 250, 1000))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_paths(inputs: list[str]) -> list[Path]:
    """Accept folders, individual files, and quoted glob patterns; deduplicate."""
    paths = set()
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            matches = [p for p in path.iterdir()
                       if p.suffix.lower() in IMAGE_SUFFIXES]
        elif path.is_file():
            matches = [path]
        else:
            matches = [Path(p) for p in glob.glob(str(path), recursive=True)]
        files = [p.resolve() for p in matches if p.is_file()]
        if not files:
            raise ValueError(f"No images found for {item!r}.")
        paths.update(files)
    return sorted(paths)


def make_board(squares_x=8, squares_y=11, square_mm=15.0, marker_mm=11.0,
               dictionary="DICT_6X6_1000", legacy_pattern=False):
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
        raise ValueError(
            f"OpenCV {cv2.__version__} lacks CharucoDetector. Install dependencies: "
            'python3 -m pip install "opencv-python>=4.8,<5" "numpy>=1.24"'
        )
    if squares_x < 2 or squares_y < 2:
        raise ValueError("Board dimensions must each be at least 2 squares.")
    if not (math.isfinite(square_mm) and math.isfinite(marker_mm)
            and 0 < marker_mm < square_mm):
        raise ValueError("Lengths must be finite, with 0 < marker-mm < square-mm.")
    if dictionary not in DICTIONARIES:
        raise ValueError(f"Unsupported dictionary: {dictionary}")
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    if squares_x * squares_y // 2 > len(aruco_dict.bytesList):
        raise ValueError("The selected dictionary has too few markers for this board.")
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_mm / 1000, marker_mm / 1000, aruco_dict
    )
    board.setLegacyPattern(legacy_pattern)
    return board


def detect_chessboard(gray, board):
    """Detect a complete grid; refine its corners on original-resolution pixels."""
    squares_x, squares_y = board.getChessboardSize()
    pattern = (squares_x - 1, squares_y - 1)
    for scale in (1, 2):
        detection_image = gray if scale == 1 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        found, corners = cv2.findChessboardCorners(
            detection_image, pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            # Account for resize's pixel-center convention. Intrinsics stay at
            # the original resolution; upsampling adds no image information.
            corners = (corners + 0.5) / scale - 0.5
            corners = cv2.cornerSubPix(
                gray, corners, (2, 2), (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001),
            )
            ids = np.arange(len(corners), dtype=np.int32).reshape(-1, 1)
            return corners, ids, [], None
    return None, None, [], None


def calibrate_images(paths, board, min_corners=8, min_views=10, debug_dir=None,
                     detector_type="charuco"):
    """Detect correspondences and return JSON-compatible intrinsics and diagnostics."""
    if min_views < 3:
        raise ValueError("Use at least 3 views; 15–30 varied views are recommended.")
    if not 4 <= min_corners <= len(board.getChessboardCorners()):
        raise ValueError("min-corners must be between 4 and the board's corner count.")
    if detector_type not in ("charuco", "chessboard"):
        raise ValueError("detector must be charuco or chessboard")
    detector = cv2.aruco.CharucoDetector(board) if detector_type == "charuco" else None
    object_points, image_points, views, skipped = [], [], [], []
    image_size = None
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    for index, path in enumerate(paths):
        path = Path(path)
        # Explicitly ignore EXIF rotation: all pixels stay in sensor orientation.
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
        if gray is None:
            reason = "cannot read image"
            print(f"SKIP {path.name}: {reason}")
            skipped.append({"image": str(path), "reason": reason})
            continue
        size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            raise ValueError(
                f"Mixed image resolutions: {path} is {size}, expected {image_size}. "
                "Use images from the same camera at the same resolution."
            )

        if detector_type == "chessboard":
            corners, ids, marker_corners, marker_ids = detect_chessboard(gray, board)
        else:
            corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
        count = 0 if ids is None else len(ids)
        reason = None
        if count < min_corners:
            reason = f"only {count} {detector_type} corners (need {min_corners})"
        elif board.checkCharucoCornersCollinear(ids):
            reason = "detected corners are collinear"

        if debug_dir is not None:
            preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(preview, marker_corners, marker_ids)
            if count:
                if detector_type == "chessboard":
                    sx, sy = board.getChessboardSize()
                    cv2.drawChessboardCorners(preview, (sx - 1, sy - 1), corners, True)
                else:
                    cv2.aruco.drawDetectedCornersCharuco(preview, corners, ids)
            status = "SKIP" if reason else "USE"
            target = debug_dir / f"{index:04d}_{path.stem}_{status}.png"
            if target.resolve() == path.resolve():
                raise ValueError(f"Debug output would overwrite input: {path}")
            if not cv2.imwrite(str(target), preview):
                raise OSError(f"Cannot write debug image: {target}")

        if reason:
            print(f"SKIP {path.name}: {reason}")
            skipped.append({"image": str(path), "reason": reason})
            continue
        objects, pixels = board.matchImagePoints(corners, ids)
        object_points.append(objects)
        image_points.append(pixels)
        views.append({"image": str(path), "corners": count})
        print(f"USE  {path.name}: {count} corners")

    if len(views) < min_views:
        raise ValueError(
            f"Only {len(views)} usable images; need {min_views}. "
            "Check board dimensions, marker layout, focus, and debug images. "
            "For an older printed board, try --legacy-pattern."
        )

    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    if (not math.isfinite(rms) or not np.isfinite(matrix).all()
            or not np.isfinite(distortion).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0):
        raise ValueError("Calibration did not produce finite, valid intrinsics.")
    for view, objects, pixels, rvec, tvec in zip(
            views, object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objects, rvec, tvec, matrix, distortion)
        residual = pixels.reshape(-1, 2) - projected.reshape(-1, 2)
        view["rms_reprojection_error_px"] = float(
            np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))
        )
    return {
        "opencv_version": cv2.__version__,
        "model": "opencv_pinhole",
        "feature_detector": detector_type,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "camera_matrix": matrix.tolist(),
        "dist_coeffs": distortion.reshape(-1).tolist(),
        "dist_coeffs_order": ["k1", "k2", "p1", "p2", "k3"],
        "fx": float(matrix[0, 0]), "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]), "cy": float(matrix[1, 2]),
        "rms_reprojection_error_px": float(rms),
        "num_views": len(views),
        "views": views,
        "skipped": skipped,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", nargs="+", required=True,
                        help="image folders, files, or quoted glob patterns")
    parser.add_argument("--output", type=Path, default=Path("camera_intrinsics.json"))
    parser.add_argument("--dictionary", choices=DICTIONARIES, default="DICT_6X6_1000")
    parser.add_argument("--detector", choices=("charuco", "chessboard"), default="charuco",
                        help="chessboard uses the complete grid without decoding markers")
    parser.add_argument("--squares-x", type=int, default=8, help="squares across (default: 8)")
    parser.add_argument("--squares-y", type=int, default=11, help="squares down (default: 11)")
    parser.add_argument("--square-mm", type=float, default=15.0)
    parser.add_argument("--marker-mm", type=float, default=11.0)
    parser.add_argument("--legacy-pattern", action="store_true",
                        help="match boards generated before OpenCV 4.6")
    parser.add_argument("--min-corners", type=int, default=8)
    parser.add_argument("--min-views", type=int, default=10)
    parser.add_argument("--debug-dir", type=Path, help="save annotated detection images")
    args = parser.parse_args(argv)
    try:
        board = make_board(args.squares_x, args.squares_y, args.square_mm,
                           args.marker_mm, args.dictionary, args.legacy_pattern)
        paths = image_paths(args.images)
        if args.output.resolve() in paths:
            raise ValueError("Output must not overwrite an input image.")
        result = calibrate_images(paths, board, args.min_corners,
                                  args.min_views, args.debug_dir, args.detector)
        result["board"] = {
            "squares_x": args.squares_x, "squares_y": args.squares_y,
            "square_length_mm": args.square_mm, "marker_length_mm": args.marker_mm,
            "dictionary": args.dictionary, "legacy_pattern": args.legacy_pattern,
            "marker_ids": board.getIds().reshape(-1).tolist(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    except (ValueError, OSError, cv2.error) as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 1

    print(f"\nCalibrated using {result['num_views']} images at "
          f"{result['image_width']}x{result['image_height']}")
    print("Camera matrix (pixels):\n", np.array(result["camera_matrix"]))
    print("Distortion [k1, k2, p1, p2, k3]:", result["dist_coeffs"])
    print(f"RMS reprojection error: {result['rms_reprojection_error_px']:.4f} px")
    print("Highest-error images:")
    for view in sorted(result["views"],
                       key=lambda v: v["rms_reprojection_error_px"], reverse=True)[:5]:
        print(f"  {view['rms_reprojection_error_px']:.4f} px  {view['image']}")
    if result["num_views"] < 15:
        print("Consider collecting 15–30 images with varied tilts and positions.")
    if result["rms_reprojection_error_px"] > 1.0:
        print("High reprojection error: inspect detections, blur, and board flatness.")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
