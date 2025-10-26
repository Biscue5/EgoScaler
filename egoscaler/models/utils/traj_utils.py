import numpy as np

def preprocess_traj(traj: np.ndarray, num_steps: int, return_padding_mask: bool = False):
    """
    Downsample the trajectory to the specified number of steps.
    If T < num_steps, pad the trajectory with the last point.

    Parameters:
    - traj (np.ndarray): Original trajectory with shape (T, D).
    - num_steps (int): Number of timesteps after downsampling.
    - return_padding_mask (bool): Whether to return the padding mask.

    Returns:
    - np.ndarray: Downsampled and padded trajectory with shape (num_steps, D).
    - (optional) np.ndarray: Padding mask with shape (num_steps,).
    """
    T, D = traj.shape

    if T >= num_steps:
        # Calculate indices for evenly spaced sampling
        indices = np.linspace(0, T - 1, num_steps).astype(int)
        # Sample the trajectory using the calculated indices
        sampled_traj = traj[indices]
        padding_mask = np.ones(num_steps, dtype=int)  # No padding
    else:
        # No downsampling needed, padding is required
        sampled_traj = traj.copy()
        # Calculate the number of padding steps
        pad_length = num_steps - T
        # Pad the trajectory with the last point
        pad = np.tile(traj[-1], (pad_length, 1))
        # Add padding to the trajectory
        sampled_traj = np.vstack([sampled_traj, pad])
        # Create the padding mask
        padding_mask = np.concatenate([np.ones(T, dtype=int), np.zeros(pad_length, dtype=int)])

    if return_padding_mask:
        return sampled_traj, padding_mask
    return sampled_traj