import numpy as np
import torch


def decode_pca(coeff, mean_shape,  blend_shapes,  n_components=64, num_points=100):
    x = mean_shape + blend_pca(coeff[:, :n_components], blend_shapes[:n_components])

    x = torch.fft.irfft(torch.complex(x[..., :3], x[..., 3:]), n=num_points - 1, dim=-2, norm='ortho')

    return x
    

def project_pca(data: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """ Project data to the subspace spanned by bases.

    Args:
        data (torch.Tensor): Hair data of shape (batch_size, ...).
        basis (torch.Tensor): Blend shapes of shape (num_blend_shapes, ...).

    Returns:
        (torch.Tensor): Projected parameters of shape (batch_size, num_blend_shapes).
    """
    return torch.einsum('bn,cn->bc', data.flatten(start_dim=1), basis.flatten(start_dim=1))


def blend_pca(coeff: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """ Blend parameters and the corresponding blend shapes.

    Args:
        coeff (torch.Tensor): Parameters (blend shape coefficients) of shape (batch_size, num_blend_shapes).
        basis (torch.Tensor): Blend shapes of shape (num_blend_shapes, ...).

    Returns:
        (torch.Tensor): Blended results of shape (batch_size, ...).
    """
    return torch.einsum('bn,n...->b...', coeff, basis)

def umeyama_similarity(X, Y, with_scaling=True):
    """
    Solve: Y ≈ s * (X @ R^T) + t
    Returns s, R, t
    """
    assert X.shape == Y.shape and X.shape[1] == 3
    n = X.shape[0]

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    X0 = X - mu_x
    Y0 = Y - mu_y

    # covariance
    Sigma = (Y0.T @ X0) / n  # 3x3

    U, D, Vt = np.linalg.svd(Sigma)
    V = Vt.T

    # Handle reflection
    S = np.eye(3)
    if np.linalg.det(U @ V.T) < 0:
        S[2, 2] = -1.0

    R = U @ S @ V.T

    if with_scaling:
        var_x = (X0 ** 2).sum() / n
        s = (D * np.diag(S)).sum() / var_x
    else:
        s = 1.0

    t = mu_y - s * (R @ mu_x)
    return float(s), R, t


def compute_similarity_transform0(A, B):
    """
    Computes similarity transform (sR, t) such that:
    B ≈ s * R @ A + t

    A: source (N x 3)
    B: target (N x 3)

    Returns:(5118, 3)
        s: scale (float)
        R: rotation (3 x 3)
        t: translation (3,)
    """
    assert A.shape == B.shape
    N = A.shape[0]

    # Compute centroids
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    # Center the point clouds
    AA = A - centroid_A
    BB = B - centroid_B

    # Compute covariance matrix
    H = AA.T @ BB / N

    # SVD of covariance
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute scale
    var_A = np.sum(AA ** 2) / N
    scale = np.sum(S) / var_A

    # Compute translation
    t = centroid_B - scale * R @ centroid_A

    return scale, R, t

def compute_similarity_transform(source, target):
    """
    Returns similarity transform: scale, rotation matrix, and translation vector
    to align source to target using the Kabsch algorithm.
    """
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    # Compute rotation using SVD
    U, S, Vt = np.linalg.svd(target_centered.T @ source_centered)
    R = U @ Vt

    # Fix reflection case (det(R) < 0)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # Compute scale
    scale = np.sum(S) / np.sum(source_centered ** 2)

    # Compute translation
    t = target_mean - scale * R @ source_mean

    return scale, R, t

def can2world_transform(can_mesh, s, R, t):
       
    #world_mesh = s * (R @ can_mesh.T).T + t 
    world_mesh = s * (can_mesh @ R.T) + t
    return world_mesh
