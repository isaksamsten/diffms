"""Discrete Flow Matching for categorical graph data.


References:
    - Campbell et al., "A Continuous Time Framework for Discrete Denoising Models", NeurIPS 2022
    - Gat et al., "Discrete Flow Matching", arXiv:2407.15595, 2024
"""

import torch
import torch.nn.functional as F
from src.utils import PlaceHolder


def interpolate_simplex(x0_onehot, prior_probs, t, node_mask=None):
    """Construct the time-t probability vector via linear interpolation on the simplex.

    Args:
        x0_onehot: One-hot encoded clean data.
                   For nodes: (bs, n, dx)  For edges: (bs, n, n, de)
        prior_probs: Prior distribution over classes.
                     For nodes: (dx,)  For edges: (de,)
        t: Time in [0, 1], shape (bs, 1).
        node_mask: Optional mask, (bs, n).

    Returns:
        p_t: Probability vector at time t, same shape as x0_onehot.
    """
    if x0_onehot.dim() == 3:
        # Node features: (bs, n, dx)
        t_broad = t.unsqueeze(-1)  # (bs, 1, 1)
        prior = prior_probs.unsqueeze(0).unsqueeze(0).expand_as(x0_onehot)  # (bs, n, dx)
    elif x0_onehot.dim() == 4:
        # Edge features: (bs, n, n, de)
        t_broad = t.unsqueeze(-1).unsqueeze(-1)  # (bs, 1, 1, 1)
        prior = prior_probs.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(x0_onehot)
    else:
        raise ValueError(f"Unexpected tensor dim: {x0_onehot.dim()}")

    p_t = t_broad * x0_onehot + (1.0 - t_broad) * prior

    return p_t


def sample_categorical(probs, node_mask=None):
    """Sample from categorical distributions defined by probability vectors.

    Args:
        probs: Probability vectors. Nodes: (bs, n, dx), Edges: (bs, n, n, de).
        node_mask: (bs, n), optional.

    Returns:
        Sampled one-hot tensors with the same shape as probs.
    """
    shape = probs.shape
    num_classes = shape[-1]

    flat_probs = probs.reshape(-1, num_classes)
    flat_probs = flat_probs.clamp(min=1e-8)
    flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)

    samples = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)  # (N,)
    onehot = F.one_hot(samples, num_classes=num_classes).float()  # (N, num_classes)
    onehot = onehot.reshape(shape)

    return onehot


def sample_zt(x0_onehot, prior_probs, t, node_mask=None):
    """Sample z_t from the interpolation distribution.

    Args:
        x0_onehot: One-hot clean data. Nodes: (bs, n, dx), Edges: (bs, n, n, de).
        prior_probs: Prior distribution. (dx,) or (de,).
        t: Time values, (bs, 1).
        node_mask: Optional, (bs, n).

    Returns:
        z_t: One-hot sampled state at time t, same shape as x0_onehot.
    """
    p_t = interpolate_simplex(x0_onehot, prior_probs, t, node_mask)
    z_t = sample_categorical(p_t, node_mask)
    return z_t


def compute_flow_matching_loss(pred, X, E, node_mask, lambda_train=None):
    """Compute discrete flow matching cross-entropy loss for nodes and edges.

    Args:
        pred: PlaceHolder with predicted logits.
            X: (bs, n, dx), E: (bs, n, n, de), y: (bs, dy).
        X: Ground-truth one-hot node features (bs, n, dx).
        E: Ground-truth one-hot edge features (bs, n, n, de).
        node_mask: (bs, n).
        lambda_train: [lambda_X, lambda_E, lambda_y].

    Returns:
        Scalar loss.
    """
    if lambda_train is None:
        lambda_train = [0.0, 1.0, 0.0]

    losses = []

    if lambda_train[0] > 0:
        true_X_flat = X.reshape(-1, X.size(-1))  # (bs*n, dx)
        pred_X_flat = pred.X.reshape(-1, pred.X.size(-1))
        mask_X = (true_X_flat != 0.0).any(dim=-1)
        if mask_X.any():
            loss_X = F.cross_entropy(pred_X_flat[mask_X], true_X_flat[mask_X].argmax(dim=-1))
            losses.append(lambda_train[0] * loss_X)

    if lambda_train[1] > 0:
        true_E_flat = E.reshape(-1, E.size(-1))  # (bs*n*n, de)
        pred_E_flat = pred.E.reshape(-1, pred.E.size(-1))
        mask_E = (true_E_flat != 0.0).any(dim=-1)
        if mask_E.any():
            loss_E = F.cross_entropy(pred_E_flat[mask_E], true_E_flat[mask_E].argmax(dim=-1))
            losses.append(lambda_train[1] * loss_E)

    if len(losses) == 0:
        return torch.zeros(1, device=pred.X.device, requires_grad=True).squeeze()

    return sum(losses)


def euler_step_simplex(p_t, pred_x0_prob, t, dt):
    """Perform one Euler step on the probability simplex during sampling.

    Args:
        p_t: Current probability distribution (bs, ..., num_classes).
        pred_x0_prob: Predicted clean data distribution (bs, ..., num_classes).
        t: Current time (bs, 1).
        dt: Step size (scalar).

    Returns:
        p_next: Updated probability distribution.
    """
    if p_t.dim() == 3:
        t_broad = t.unsqueeze(-1)           # (bs, 1, 1)
    elif p_t.dim() == 4:
        t_broad = t.unsqueeze(-1).unsqueeze(-1)  # (bs, 1, 1, 1)
    else:
        raise ValueError(f"Unexpected dim: {p_t.dim()}")

    one_minus_t = (1.0 - t_broad).clamp(min=1e-5)

    velocity = (pred_x0_prob - p_t) / one_minus_t

    p_next = p_t + dt * velocity
    p_next = p_next.clamp(min=0.0)
    p_next = p_next / (p_next.sum(dim=-1, keepdim=True) + 1e-8)

    return p_next


def sample_discrete_flow_matching(
    model_forward_fn,
    X, E, y,
    node_mask,
    prior_X,
    prior_E,
    num_steps=50,
    denoise_nodes=False,
):
    """Run the discrete flow matching sampling loop.

    Starts from the prior distribution at t=0 and integrates the ODE forward
    to t=1 using Euler steps with the learned velocity field.

    Args:
        model_forward_fn: Callable(z_t_X, z_t_E, y, t, node_mask) -> (pred_X_logits, pred_E_logits).
            This should handle computing extra features internally.
        X: Ground-truth node features (used as-is if denoise_nodes=False).
            Shape: (bs, n, dx).
        E: Not used directly (we start from prior). Shape reference only.
        y: Conditioning features (bs, dy).
        node_mask: (bs, n).
        prior_X: Prior distribution over node types (dx,).
        prior_E: Prior distribution over edge types (de,).
        num_steps: Number of Euler integration steps.
        denoise_nodes: If False, keep X fixed (ground truth atoms).

    Returns:
        List of RDKit molecules.
    """
    bs, n, dx = X.shape
    de = E.shape[-1]
    device = X.device
    dt = 1.0 / num_steps

    if denoise_nodes:
        p_X = prior_X.unsqueeze(0).unsqueeze(0).expand(bs, n, -1).to(device)
    else:
        p_X = X.clone()

    p_E = prior_E.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(bs, n, n, -1).to(device)

    for step in range(1, num_steps + 1):
        t = step * dt
        t_tensor = torch.full((bs, 1), t, device=device)

        z_t_E = sample_categorical(p_E)
        E_idx = z_t_E.argmax(dim=-1)
        upper = torch.triu(E_idx, diagonal=1)
        E_idx = upper + upper.transpose(1, 2)
        z_t_E = F.one_hot(E_idx, num_classes=de).float()

        if denoise_nodes:
            z_t_X = sample_categorical(p_X)
        else:
            z_t_X = X.clone()

        pred_X_logits, pred_E_logits = model_forward_fn(z_t_X, z_t_E, y, t_tensor, node_mask)
        pred_X_prob = F.softmax(pred_X_logits, dim=-1)
        pred_E_prob = F.softmax(pred_E_logits, dim=-1)

        if denoise_nodes:
            p_X = euler_step_simplex(p_X, pred_X_prob, t_tensor, dt)
        p_E = euler_step_simplex(p_E, pred_E_prob, t_tensor, dt)

    if denoise_nodes:
        final_X = sample_categorical(p_X)
    else:
        final_X = X.clone()

    final_E = sample_categorical(p_E)

    E_idx = final_E.argmax(dim=-1)
    upper = torch.triu(E_idx, diagonal=1)
    E_idx = upper + upper.transpose(1, 2)
    final_E = F.one_hot(E_idx, num_classes=de).float()

    return final_X, final_E

