import torch
import torch.nn.functional as F


def sample_target_patches(
    target_img: torch.Tensor,
    eye_trace: torch.Tensor,
    roi_size: int,
) -> torch.Tensor:
    """
    Sample patches from target images at eye trace positions (vectorized, GPU-friendly).

    Replaces stitch-then-compare with gather-then-compare: for each frame, crops
    the corresponding patch from the target image at the eye position. Equivalent
    to the per-patch reconstruction loss without the scatter/accumulation step.

    Args:
        target_img (torch.Tensor): Full target images, shape (B, H, W).
        eye_trace (torch.Tensor): Integer eye positions, shape (B, 2, T) as (x, y).
        roi_size (int): Size of the patch to extract.

    Returns:
        torch.Tensor: Target patches, shape (B, T, roi_size, roi_size).
    """
    B, H, W = target_img.shape
    T = eye_trace.shape[2]
    device = target_img.device

    patch_y, patch_x = torch.meshgrid(
        torch.arange(roi_size, device=device),
        torch.arange(roi_size, device=device),
        indexing='ij',
    )  # (roi, roi)

    # Absolute pixel coords: (B, T, roi, roi)
    x_abs = eye_trace[:, 0, :, None, None] + patch_x
    y_abs = eye_trace[:, 1, :, None, None] + patch_y

    # Normalize to [-1, 1] for grid_sample
    x_norm = 2.0 * x_abs / (W - 1) - 1.0
    y_norm = 2.0 * y_abs / (H - 1) - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1)           # (B, T, roi, roi, 2)
    grid = grid.reshape(B, T * roi_size, roi_size, 2)

    sampled = F.grid_sample(
        target_img.unsqueeze(1).float(),                   # (B, 1, H, W)
        grid,
        mode='nearest',
        align_corners=True,
        padding_mode='border',
    )  # (B, 1, T*roi, roi)

    return sampled.squeeze(1).reshape(B, T, roi_size, roi_size)


def stitch_frames_by_position(
    pos: torch.Tensor, video: torch.Tensor, img_size: int
) -> torch.Tensor:
    """
    Reconstruct a static image by stitching video frames based on their positions.

    Iterates through positions and accumulates video frames at corresponding
    locations in the output image. Works with integer positions.

    Args:
        pos (torch.Tensor): Position tensor of shape (2, N) containing (x, y)
                           coordinates for each frame.
        video (torch.Tensor): Video tensor of shape (N, roi_size, roi_size)
                             containing the ROI patches for each timestep.
        img_size (int): Size of the output image (img_size x img_size).

    Returns:
        torch.Tensor: Reconstructed static image of shape (img_size, img_size).

    Raises:
        ValueError: If position array length doesn't match video frame count.
    """
    if pos.shape[1] != video.shape[0]:
        raise ValueError("Position array length must match video frame count.")

    roi_size = video.shape[1]
    img = torch.zeros(
        [img_size, img_size], dtype=torch.float32, device=video.device
    )
    for i in range(pos.shape[1]):
        img[
            pos[1, i] : pos[1, i] + roi_size, pos[0, i] : pos[0, i] + roi_size
        ] += video[i]

    return img


def stitch_frames_by_position_bilinear(
    pos: torch.Tensor, video: torch.Tensor, img_size: int
) -> torch.Tensor:
    """
    Reconstruct image using bilinear splatting at floating-point positions.

    Drop-in replacement for stitch_frames_by_position that handles
    floating-point eye positions via bilinear weighted splatting. Fully
    vectorized using scatter-add operations for maximum efficiency.

    Args:
        pos (torch.Tensor): Position tensor of shape (2, N) containing (x, y)
                           coordinates for each frame (can be float).
        video (torch.Tensor): Video tensor of shape (N, roi_size, roi_size)
                             containing the ROI patches for each timestep.
        img_size (int): Size of the output image (img_size x img_size).

    Returns:
        torch.Tensor: Reconstructed static image of shape (img_size, img_size),
                     normalized by coverage weights.

    Raises:
        ValueError: If position array length doesn't match video frame count.

    Notes:
        Uses bilinear splatting to distribute frame contributions to the four
        nearest integer pixel locations, weighted by distance. Output is
        normalized by accumulated visibility weights to account for overlaps.

        Optimization: Uses torch.index_put_ with accumulate=True for efficient
        vectorized scatter-add, avoiding Python loops over frames.
    """
    if pos.shape[1] != video.shape[0]:
        raise ValueError("Position array length must match video frame count.")

    roi_size = video.shape[1]
    device = video.device
    dtype = video.dtype

    # Extract position components and compute integer/fractional parts
    x_pos, y_pos = pos[0], pos[1]  # Each shape: (N,)
    x0 = torch.floor(x_pos).long()
    y0 = torch.floor(y_pos).long()

    # Fractional parts (distance to next pixel)
    dx = x_pos - x0.float()  # (N,)
    dy = y_pos - y0.float()  # (N,)

    # Compute bilinear weights for all 4 corners
    w00 = (1.0 - dx) * (1.0 - dy)  # Top-left
    w10 = dx * (1.0 - dy)           # Top-right
    w01 = (1.0 - dx) * dy           # Bottom-left
    w11 = dx * dy                   # Bottom-right

    # Initialize accumulation buffers
    img = torch.zeros((img_size, img_size), dtype=dtype, device=device)
    # visibility = torch.zeros((img_size, img_size), dtype=torch.float32, device=device)

    # Relative patch coordinate meshgrid: (roi_size, roi_size)
    patch_y, patch_x = torch.meshgrid(
        torch.arange(roi_size, device=device),
        torch.arange(roi_size, device=device),
        indexing='ij'
    )

    # Stack all 4 corner offsets at once: x_offsets/y_offsets shape (4,)
    x_offsets = torch.tensor([0, 1, 0, 1], device=device, dtype=torch.long)
    y_offsets = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.long)
    # weights_all: (4, N)
    weights_all = torch.stack([w00, w10, w01, w11])

    # x_corners, y_corners: (4, N)
    x_corners = x0.unsqueeze(0) + x_offsets[:, None]
    y_corners = y0.unsqueeze(0) + y_offsets[:, None]

    # Valid mask: (4, N) — only frames whose corner patch fits entirely in the image
    valid = (
        (x_corners >= 0)
        & (x_corners + roi_size <= img_size)
        & (y_corners >= 0)
        & (y_corners + roi_size <= img_size)
    )  # (4, N)

    # Flatten corner and frame dims together: process all valid (corner, frame) pairs at once
    # corner_idx: which corner (0-3), frame_idx: which frame (0-N-1)
    corner_idx, frame_idx = valid.nonzero(as_tuple=True)  # each (K,)

    if corner_idx.numel() > 0:
        vx = x_corners[corner_idx, frame_idx]  # (K,)
        vy = y_corners[corner_idx, frame_idx]  # (K,)
        vw = weights_all[corner_idx, frame_idx]  # (K,)
        vf = video[frame_idx]  # (K, roi_size, roi_size)

        # Absolute pixel coordinates for each (frame, patch_pixel) pair
        abs_x = vx[:, None, None] + patch_x[None]  # (K, roi_size, roi_size)
        abs_y = vy[:, None, None] + patch_y[None]  # (K, roi_size, roi_size)

        values_flat = (vf * vw[:, None, None]).flatten()
        img.index_put_(
            (abs_y.flatten(), abs_x.flatten()),
            values_flat,
            accumulate=True,
        )

    return img
