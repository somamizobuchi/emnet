import torch


def stitch_frames_by_position(
    pos: torch.Tensor, video: torch.Tensor, img_size: int
) -> torch.Tensor:
    """
    Reconstruct a static image by stitching video frames based on their positions.

    Iterates through positions and accumulates video frames at corresponding
    locations in the output image.

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
