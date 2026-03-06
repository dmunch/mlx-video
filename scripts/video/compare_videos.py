#!/usr/bin/env python3
"""Compare two videos frame-by-frame with quality metrics.

Useful for validating MLX ports against reference PyTorch implementations.
Reports PSNR, SSIM, per-frame differences, temporal coherence, and color
fidelity. Optionally saves a side-by-side diff video.

Usage:
    # Basic comparison
    python scripts/video/compare_videos.py reference.mp4 test.mp4

    # Save side-by-side diff visualization
    python scripts/video/compare_videos.py ref.mp4 test.mp4 --diff-video diff.mp4

    # Compare only first 64 frames
    python scripts/video/compare_videos.py ref.mp4 test.mp4 --max-frames 64

    # Adjust SSIM window size (default: 7)
    python scripts/video/compare_videos.py ref.mp4 test.mp4 --ssim-win 11
"""

import argparse
import sys

import cv2
import numpy as np


def load_video(path, max_frames=None):
    """Load video frames as float32 numpy arrays (0-255 range)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: cannot open {path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.astype(np.float32))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


def compute_psnr(a, b):
    """Peak Signal-to-Noise Ratio between two frames."""
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


def compute_ssim(a, b, win_size=7):
    """Structural Similarity Index (per-channel, averaged).

    Uses the standard SSIM formula with default constants.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(win_size, 1.5)
    window = kernel @ kernel.T

    ssim_channels = []
    for c in range(a.shape[2]):
        ac, bc = a[:, :, c], b[:, :, c]
        mu_a = cv2.filter2D(ac, -1, window)
        mu_b = cv2.filter2D(bc, -1, window)

        mu_a_sq = mu_a**2
        mu_b_sq = mu_b**2
        mu_ab = mu_a * mu_b

        sigma_a_sq = cv2.filter2D(ac**2, -1, window) - mu_a_sq
        sigma_b_sq = cv2.filter2D(bc**2, -1, window) - mu_b_sq
        sigma_ab = cv2.filter2D(ac * bc, -1, window) - mu_ab

        num = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
        den = (mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2)
        ssim_map = num / den
        ssim_channels.append(np.mean(ssim_map))

    return np.mean(ssim_channels)


def temporal_coherence(frames):
    """Mean frame-to-frame difference (lower = smoother)."""
    if len(frames) < 2:
        return 0.0
    diffs = []
    for i in range(1, len(frames)):
        diffs.append(np.mean(np.abs(frames[i] - frames[i - 1])))
    return np.mean(diffs)


def color_histogram_distance(a, b, bins=64):
    """Chi-squared distance between color histograms."""
    dist = 0.0
    for c in range(3):
        ha, _ = np.histogram(a[:, :, c], bins=bins, range=(0, 256))
        hb, _ = np.histogram(b[:, :, c], bins=bins, range=(0, 256))
        ha = ha.astype(np.float64) / (ha.sum() + 1e-10)
        hb = hb.astype(np.float64) / (hb.sum() + 1e-10)
        dist += np.sum((ha - hb) ** 2 / (ha + hb + 1e-10)) / 2
    return dist / 3


def make_diff_frame(a, b, scale=5.0):
    """Create a heatmap visualization of the absolute difference."""
    diff = np.mean(np.abs(a - b), axis=2)
    diff_scaled = np.clip(diff * scale, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_scaled, cv2.COLORMAP_JET)
    return heatmap


def analyze(ref_frames, test_frames, ssim_win=7):
    """Compute per-frame and aggregate metrics."""
    n = min(len(ref_frames), len(test_frames))

    psnrs = []
    ssims = []
    mean_diffs = []
    max_diffs = []
    color_dists = []

    for i in range(n):
        r, t = ref_frames[i], test_frames[i]
        psnrs.append(compute_psnr(r, t))
        ssims.append(compute_ssim(r, t, ssim_win))
        absdiff = np.abs(r - t)
        mean_diffs.append(np.mean(absdiff))
        max_diffs.append(np.max(absdiff))
        color_dists.append(color_histogram_distance(r, t))

    ref_tc = temporal_coherence(ref_frames[:n])
    test_tc = temporal_coherence(test_frames[:n])

    return {
        "num_frames": n,
        "psnr": np.array(psnrs),
        "ssim": np.array(ssims),
        "mean_diff": np.array(mean_diffs),
        "max_diff": np.array(max_diffs),
        "color_dist": np.array(color_dists),
        "ref_temporal_coherence": ref_tc,
        "test_temporal_coherence": test_tc,
    }


def print_report(results, ref_path, test_path):
    """Print a formatted comparison report."""
    n = results["num_frames"]
    psnr = results["psnr"]
    ssim = results["ssim"]
    md = results["mean_diff"]
    mx = results["max_diff"]
    cd = results["color_dist"]

    print("=" * 72)
    print("VIDEO COMPARISON REPORT")
    print("=" * 72)
    print(f"  Reference: {ref_path}")
    print(f"  Test:      {test_path}")
    print(f"  Frames compared: {n}")
    print()

    print("AGGREGATE METRICS")
    print("-" * 40)
    print(f"  PSNR (dB):    mean={np.mean(psnr):6.2f}  min={np.min(psnr):6.2f}  max={np.max(psnr):6.2f}")
    print(f"  SSIM:         mean={np.mean(ssim):.4f}  min={np.min(ssim):.4f}  max={np.max(ssim):.4f}")
    print(f"  Mean diff:    mean={np.mean(md):6.2f}  min={np.min(md):6.2f}  max={np.max(md):6.2f}")
    print(f"  Max diff:     mean={np.mean(mx):6.1f}   min={np.min(mx):6.1f}   max={np.max(mx):6.1f}")
    print(f"  Color dist:   mean={np.mean(cd):.4f}  min={np.min(cd):.4f}  max={np.max(cd):.4f}")
    print()

    print("TEMPORAL COHERENCE (mean frame-to-frame diff, lower = smoother)")
    print("-" * 40)
    print(f"  Reference: {results['ref_temporal_coherence']:.2f}")
    print(f"  Test:      {results['test_temporal_coherence']:.2f}")
    ratio = results["test_temporal_coherence"] / (results["ref_temporal_coherence"] + 1e-10)
    print(f"  Ratio:     {ratio:.2f}x {'(test is smoother)' if ratio < 1 else '(test is jerkier)' if ratio > 1.05 else '(similar)'}")
    print()

    # Identify worst frames
    print("WORST FRAMES (by PSNR)")
    print("-" * 40)
    worst_idx = np.argsort(psnr)[:5]
    for i in worst_idx:
        print(f"  Frame {i:4d}: PSNR={psnr[i]:6.2f} dB  SSIM={ssim[i]:.4f}  mean_diff={md[i]:.2f}")
    print()

    # Quality assessment
    mean_psnr = np.mean(psnr)
    mean_ssim = np.mean(ssim)
    print("QUALITY ASSESSMENT")
    print("-" * 40)
    if mean_psnr > 40:
        grade = "Excellent"
    elif mean_psnr > 35:
        grade = "Good"
    elif mean_psnr > 30:
        grade = "Fair"
    elif mean_psnr > 25:
        grade = "Poor"
    else:
        grade = "Very different"
    print(f"  Overall: {grade} (PSNR={mean_psnr:.1f} dB, SSIM={mean_ssim:.4f})")
    if mean_psnr < 30:
        print("  ⚠  Videos differ significantly — likely a bug or different generation seed")
    print("=" * 72)


def save_diff_video(ref_frames, test_frames, output_path, fps, scale=5.0):
    """Save a side-by-side video: reference | test | diff heatmap."""
    n = min(len(ref_frames), len(test_frames))
    h, w = ref_frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w * 3, h))

    for i in range(n):
        r = ref_frames[i].astype(np.uint8)
        t = test_frames[i].astype(np.uint8)
        d = make_diff_frame(ref_frames[i], test_frames[i], scale)
        combined = np.hstack([r, t, d])
        out.write(combined)

    out.release()
    print(f"Diff video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two videos frame-by-frame with quality metrics"
    )
    parser.add_argument("reference", help="Path to reference video")
    parser.add_argument("test", help="Path to test video")
    parser.add_argument(
        "--diff-video", help="Save side-by-side diff visualization to this path"
    )
    parser.add_argument(
        "--max-frames", type=int, help="Compare only first N frames"
    )
    parser.add_argument(
        "--ssim-win", type=int, default=7, help="SSIM window size (default: 7)"
    )
    parser.add_argument(
        "--diff-scale",
        type=float,
        default=5.0,
        help="Diff heatmap amplification (default: 5.0)",
    )
    parser.add_argument(
        "--csv", help="Export per-frame metrics to CSV file"
    )
    args = parser.parse_args()

    print(f"Loading reference: {args.reference}")
    ref_frames, ref_fps = load_video(args.reference, args.max_frames)
    print(f"  → {len(ref_frames)} frames, {ref_fps:.1f} fps, {ref_frames[0].shape[1]}x{ref_frames[0].shape[0]}")

    print(f"Loading test: {args.test}")
    test_frames, test_fps = load_video(args.test, args.max_frames)
    print(f"  → {len(test_frames)} frames, {test_fps:.1f} fps, {test_frames[0].shape[1]}x{test_frames[0].shape[0]}")

    if ref_frames[0].shape != test_frames[0].shape:
        print(f"Warning: resolution mismatch {ref_frames[0].shape} vs {test_frames[0].shape}")
        print("Resizing test frames to match reference...")
        h, w = ref_frames[0].shape[:2]
        test_frames = [
            cv2.resize(f, (w, h), interpolation=cv2.INTER_LANCZOS4)
            for f in test_frames
        ]

    print("Computing metrics...")
    results = analyze(ref_frames, test_frames, args.ssim_win)
    print()
    print_report(results, args.reference, args.test)

    if args.diff_video:
        save_diff_video(ref_frames, test_frames, args.diff_video, ref_fps, args.diff_scale)

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "psnr", "ssim", "mean_diff", "max_diff", "color_dist"])
            for i in range(results["num_frames"]):
                writer.writerow([
                    i,
                    f"{results['psnr'][i]:.4f}",
                    f"{results['ssim'][i]:.6f}",
                    f"{results['mean_diff'][i]:.4f}",
                    f"{results['max_diff'][i]:.1f}",
                    f"{results['color_dist'][i]:.6f}",
                ])
        print(f"Per-frame metrics saved to {args.csv}")


if __name__ == "__main__":
    main()
