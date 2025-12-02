import re
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.spatial.transform import Rotation as R

def final_displacement_error(gen_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    """
    Computes the Final Displacement Error (FDE) between generated and ground truth trajectories.

    FDE is defined as the L2 distance between the predicted final position and the ground truth final position
    of the trajectory, averaged across the entire batch.

    Args:
        gen_traj (np.ndarray): Generated trajectories of shape (batch_size, seq_length, feature_dim).
        gt_traj (np.ndarray): Ground truth trajectories of shape (batch_size, seq_length, feature_dim).

    Returns:
        float: A scalar representing the FDE.
    """
    # Ensure the input arrays have the same shape
    if gen_traj.shape != gt_traj.shape:
        raise ValueError(f"Shape mismatch: gen_traj shape {gen_traj.shape} and gt_traj shape {gt_traj.shape} must be the same.")
    
    # Extract the positional components (assuming the first 3 features are positions)
    gen_pos = gen_traj[:, :, :3]  # Shape: (batch_size, seq_length, 3)
    gt_pos = gt_traj[:, :, :3]    # Shape: (batch_size, seq_length, 3)    
    
    # Extract the final positions from the trajectories
    gen_final = gen_pos[:, -1, :]  # Shape: (batch_size, 3)
    gt_final = gt_pos[:, -1, :]    # Shape: (batch_size, 3)

    # Compute the difference between the final positions
    diff = gt_final - gen_final  # Shape: (batch_size, 3)

    # Compute the L2 norm (Euclidean distance) for each sample in the batch
    l2_norm = np.linalg.norm(diff, axis=-1)  # Shape: (batch_size,)

    # Compute the mean over the batch
    fde = np.mean(l2_norm)

    return fde

def initial_displacement_error(gen_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    """
    Computes the Final Displacement Error (FDE) between generated and ground truth trajectories.

    FDE is defined as the L2 distance between the predicted final position and the ground truth final position
    of the trajectory, averaged across the entire batch.

    Args:
        gen_traj (np.ndarray): Generated trajectories of shape (batch_size, seq_length, feature_dim).
        gt_traj (np.ndarray): Ground truth trajectories of shape (batch_size, seq_length, feature_dim).

    Returns:
        float: A scalar representing the FDE.
    """
    # Ensure the input arrays have the same shape
    if gen_traj.shape != gt_traj.shape:
        raise ValueError(f"Shape mismatch: gen_traj shape {gen_traj.shape} and gt_traj shape {gt_traj.shape} must be the same.")
    
    # Extract the positional components (assuming the first 3 features are positions)
    gen_pos = gen_traj[:, :, :3]  # Shape: (batch_size, seq_length, 3)
    gt_pos = gt_traj[:, :, :3]    # Shape: (batch_size, seq_length, 3)    
    
    # Extract the final positions from the trajectories
    gen_inital = gen_pos[:, 0, :]  # Shape: (batch_size, 3)
    gt_inital = gt_pos[:, 0, :]    # Shape: (batch_size, 3)

    # Compute the difference between the final positions
    diff = gt_inital - gen_inital  # Shape: (batch_size, 3)

    # Compute the L2 norm (Euclidean distance) for each sample in the batch
    l2_norm = np.linalg.norm(diff, axis=-1)  # Shape: (batch_size,)

    # Compute the mean over the batch
    ide = np.mean(l2_norm)

    return ide

def average_displacement_error(gen_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    """
    Computes the Average Displacement Error (ADE) between generated and ground truth trajectories.

    ADE is defined as the average L2 distance between the predicted trajectory and the ground truth
    trajectory over all time steps and across the entire batch.

    Args:
        gen_traj (np.ndarray): Generated trajectories of shape (batch_size, seq_length, feature_dim).
        gt_traj (np.ndarray): Ground truth trajectories of shape (batch_size, seq_length, feature_dim).

    Returns:
        float: A scalar representing the ADE.
    """
    # Ensure the input arrays have the same shape
    if gen_traj.shape != gt_traj.shape:
        raise ValueError(f"Shape mismatch: gen_traj shape {gen_traj.shape} and gt_traj shape {gt_traj.shape} must be the same.")
    
    # Extract the positional components (assuming the first 3 features are positions)
    gen_pos = gen_traj[:, :, :3]  # Shape: (batch_size, seq_length, 3)
    gt_pos = gt_traj[:, :, :3]    # Shape: (batch_size, seq_length, 3)

    # Compute the difference between generated and ground truth trajectories
    diff = gt_pos - gen_pos  # Shape: (batch_size, seq_length, 3)

    # Compute the L2 norm (Euclidean distance) along the feature dimension
    # This results in an array of shape (batch_size, seq_length)
    l2_norm = np.linalg.norm(diff, axis=-1)  # Shape: (batch_size, seq_length)

    # Compute the mean over all time steps and batches
    ade = np.mean(l2_norm)

    return ade

def anglar_distance(gen_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    """
    Computes the Geodesic Distance (GD) between generated and ground truth trajectories.

    GD is defined as the angular distance (in radians) between the predicted rotation matrix
    and the ground truth rotation matrix, averaged across all time steps and the entire batch.

    Args:
        gen_traj (np.ndarray): Generated trajectories of shape (batch_size, seq_length, feature_dim).
                               The last 3 dimensions are assumed to represent rotation as rotvec.
        gt_traj (np.ndarray): Ground truth trajectories of shape (batch_size, seq_length, feature_dim).
                              The last 3 dimensions are assumed to represent rotation as rotvec.

    Returns:
        float: A scalar representing the average geodesic distance across all trajectories.
    """
    # Ensure the input arrays have the same shape
    if gen_traj.shape != gt_traj.shape:
        raise ValueError(f"Shape mismatch: gen_traj shape {gen_traj.shape} and gt_traj shape {gt_traj.shape} must be the same.")

    # Extract the rotational components (assuming the last 3 features are rotations)
    gen_rot = gen_traj[:, :, 3:]  # Shape: (batch_size, seq_length, 3)
    gt_rot = gt_traj[:, :, 3:]    # Shape: (batch_size, seq_length, 3)
    
    gd = []  # Store geodesic distances for all time steps and trajectories

    for gen_r, gt_r in zip(gen_rot, gt_rot):
        # Convert rotation vectors to rotation matrices
        gen_rot_mat = R.from_rotvec(gen_r).as_matrix()  # Shape: (seq_length, 3, 3)
        gt_rot_mat = R.from_rotvec(gt_r).as_matrix()    # Shape: (seq_length, 3, 3)

        # Compute geodesic distance for each time step
        for g_mat, t_mat in zip(gen_rot_mat, gt_rot_mat):
            # Compute the relative rotation matrix
            relative_rot = np.dot(t_mat.T, g_mat)  # Shape: (3, 3)

            # Compute the trace of the relative rotation matrix
            trace_value = np.trace(relative_rot)

            # Compute the angular distance (geodesic distance)
            angle_dist = np.arccos(np.clip((trace_value - 1) / 2, -1.0, 1.0))  # Radians
            
            gd.append(angle_dist)
    
    # Compute the mean geodesic distance across all time steps and trajectories
    return np.mean(gd)