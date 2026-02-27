
import numpy as np
from pathlib import Path
from typing import Optional


def save_video(frames: np.ndarray, output_path: str, fps: int = 16):
    """Save video frames to MP4.

    Args:
        frames: Video frames [T, H, W, 3] uint8
        output_path: Output file path
        fps: Frames per second
    """
    try:
        import imageio
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    except ImportError:
        try:
            import cv2
            h, w = frames.shape[1], frames.shape[2]
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        except (ImportError, Exception):
            # Last resort: save as individual PNGs
            from PIL import Image
            out_dir = Path(output_path).parent / Path(output_path).stem
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(out_dir / f"frame_{i:04d}.png")
            print(f"  (no video encoder available, saved {len(frames)} frames to {out_dir}/)")


def bilateral_filter(image: np.ndarray, d: int = 5, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Apply bilateral filter to reduce grid artifacts while preserving edges.

    Args:
        image: Input image as uint8 numpy array (H, W, C)
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space

    Returns:
        Filtered image
    """
    try:
        import cv2
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    except ImportError:
        # Fallback to simple Gaussian blur if cv2 not available
        return gaussian_blur(image, kernel_size=3)


def gaussian_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply Gaussian blur.

    Args:
        image: Input image as uint8 numpy array (H, W, C)
        kernel_size: Size of the Gaussian kernel (must be odd)

    Returns:
        Blurred image
    """
    try:
        import cv2
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    except ImportError:
        # Simple box blur fallback
        from scipy.ndimage import uniform_filter
        return uniform_filter(image, size=(kernel_size, kernel_size, 1)).astype(np.uint8)


def unsharp_mask(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.0) -> np.ndarray:
    """Apply unsharp masking to enhance edges after blur.

    Args:
        image: Input image as uint8 numpy array
        kernel_size: Size of the Gaussian kernel
        sigma: Gaussian sigma
        amount: Strength of sharpening

    Returns:
        Sharpened image
    """
    try:
        import cv2
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    except ImportError:
        return image


def reduce_grid_artifacts(
    video: np.ndarray,
    method: str = "bilateral",
    strength: float = 1.0,
) -> np.ndarray:
    """Reduce grid artifacts in video frames.

    Args:
        video: Video as numpy array (F, H, W, C) uint8
        method: "bilateral", "gaussian", or "frequency"
        strength: How strong to apply the filter (0-1)

    Returns:
        Processed video
    """
    if method == "bilateral":
        d = max(3, int(5 * strength))
        sigma = 50 + 50 * strength
        processed = np.stack([
            bilateral_filter(frame, d=d, sigma_color=sigma, sigma_space=sigma)
            for frame in video
        ])
    elif method == "gaussian":
        kernel_size = max(3, int(3 + 4 * strength))
        if kernel_size % 2 == 0:
            kernel_size += 1
        processed = np.stack([
            gaussian_blur(frame, kernel_size=kernel_size)
            for frame in video
        ])
    elif method == "frequency":
        processed = np.stack([
            remove_grid_frequency(frame, grid_size=8)
            for frame in video
        ])
    else:
        raise ValueError(f"Unknown method: {method}")

    # Optionally sharpen to recover some detail
    if strength < 1.0:
        # Blend with original based on strength
        alpha = strength
        processed = (alpha * processed + (1 - alpha) * video).astype(np.uint8)

    return processed


def remove_grid_frequency(frame: np.ndarray, grid_size: int = 8) -> np.ndarray:
    """Remove grid-frequency components using FFT.

    Args:
        frame: Input frame (H, W, C) uint8
        grid_size: Expected grid periodicity in pixels

    Returns:
        Filtered frame
    """
    result = np.zeros_like(frame)

    for c in range(frame.shape[2]):
        channel = frame[:, :, c].astype(np.float32)
        h, w = channel.shape

        # FFT
        fft = np.fft.fft2(channel)
        fft_shifted = np.fft.fftshift(fft)

        # Create notch filter at grid frequencies
        cy, cx = h // 2, w // 2
        mask = np.ones((h, w), dtype=np.float32)

        # Attenuate frequencies at grid periodicity
        freq_y = h // grid_size
        freq_x = w // grid_size

        for fy in range(-2, 3):
            for fx in range(-2, 3):
                if fy == 0 and fx == 0:
                    continue
                y_pos = cy + fy * freq_y
                x_pos = cx + fx * freq_x
                if 0 <= y_pos < h and 0 <= x_pos < w:
                    # Gaussian attenuation around the frequency
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            yy, xx = y_pos + dy, x_pos + dx
                            if 0 <= yy < h and 0 <= xx < w:
                                dist = np.sqrt(dy**2 + dx**2)
                                mask[yy, xx] *= min(1.0, dist / 3.0)

        # Apply mask and inverse FFT
        fft_filtered = fft_shifted * mask
        channel_filtered = np.fft.ifft2(np.fft.ifftshift(fft_filtered)).real

        result[:, :, c] = np.clip(channel_filtered, 0, 255).astype(np.uint8)

    return result



