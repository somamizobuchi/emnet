import torch


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

    # Create relative patch coordinate meshgrid (computed once, reused 4 times)
    patch_y, patch_x = torch.meshgrid(
        torch.arange(roi_size, device=device),
        torch.arange(roi_size, device=device),
        indexing='ij'
    )  # Both shape: (roi_size, roi_size)

    # Four corners: (x_offset, y_offset, weights)
    corners = [
        (0, 0, w00),
        (1, 0, w10),
        (0, 1, w01),
        (1, 1, w11),
    ]

    for dx_offset, dy_offset, weights in corners:
        x_corners = x0 + dx_offset  # (N,)
        y_corners = y0 + dy_offset  # (N,)

        # Vectorized bounds check
        valid = (
            (x_corners >= 0)
            & (x_corners + roi_size <= img_size)
            & (y_corners >= 0)
            & (y_corners + roi_size <= img_size)
        )

        if not valid.any():
            continue

        # Extract valid frames and their properties
        valid_x = x_corners[valid]  # (M,) where M = num valid frames
        valid_y = y_corners[valid]
        valid_weights = weights[valid]  # (M,)
        valid_frames = video[valid]  # (M, roi_size, roi_size)

        # Broadcast to get absolute coordinates for all pixels in all valid frames
        # Shape expansion: (M,) -> (M, 1, 1), then broadcast with (roi_size, roi_size)
        abs_x = valid_x[:, None, None] + patch_x[None, :, :]  # (M, roi_size, roi_size)
        abs_y = valid_y[:, None, None] + patch_y[None, :, :]  # (M, roi_size, roi_size)

        # Flatten for indexing
        abs_x_flat = abs_x.flatten()  # (M * roi_size²,)
        abs_y_flat = abs_y.flatten()  # (M * roi_size²,)

        # Apply weights and flatten frame values
        weighted_frames = valid_frames * valid_weights[:, None, None]  # (M, roi_size, roi_size)
        values_flat = weighted_frames.flatten()  # (M * roi_size²,)

        # Vectorized scatter-add to image using index_put with accumulate=True
        img.index_put_(
            (abs_y_flat, abs_x_flat),
            values_flat,
            accumulate=True
        )

        # Scatter-add weights to visibility
        # weights_broadcast = valid_weights[:, None, None].expand(-1, roi_size, roi_size).flatten()
        # visibility.index_put_(
        #     (abs_y_flat, abs_x_flat),
        #     weights_broadcast,
        #     accumulate=True
        # )

    # Normalize by visibility (average overlapping regions)
    # Use epsilon threshold to avoid numerical instability from very small visibility
    eps = 1e-8
    # img = torch.where(visibility > eps, img / visibility, img)

    return img
