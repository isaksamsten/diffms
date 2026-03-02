"""Discrete Flow Matching for categorical graph data.

Implements the interpolation-based flow matching framework for discrete
(categorical) node and edge features, as an alternative to D3PM-style
discrete denoising diffusion.

Key idea: instead of a discrete Markov chain with transition matrices,
we define a continuous-time interpolation on the probability simplex
between data x_0 and a prior distribution p_prior. The neural network
learns to predict the clean data distribution p(x_0 | z_t, t), and
sampling follows an ODE on the simplex with an Euler integrator.

References:
    - Campbell et al., "A Continuous Time Framework for Discrete Denoising Models", NeurIPS 2022
    - Gat et al., "Discrete Flow Matching", arXiv:2407.15595, 2024
"""

import torch
import torch.nn.functional as F
from src.utils import PlaceHolder


def interpolate_simplex(x0_onehot, prior_probs, t, node_mask=None):
    """Construct the time-t probability vector via linear interpolation on the simplex.

    p_t(k) = t * x0(k) + (1 - t) * prior(k)

    At t=1 we recover the clean data; at t=0 we have the prior.

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
    # Reshape t for broadcasting
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

    # Flatten to 2D for multinomial sampling
    flat_probs = probs.reshape(-1, num_classes)
    flat_probs = flat_probs.clamp(min=1e-8)
    flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)

    samples = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)  # (N,)
    onehot = F.one_hot(samples, num_classes=num_classes).float()  # (N, num_classes)
    onehot = onehot.reshape(shape)

    return onehot


def sample_zt(x0_onehot, prior_probs, t, node_mask=None):
    """Sample z_t from the interpolation distribution.

    For each element, with probability t keep x_0, with probability (1-t)
    sample from the prior. This is equivalent to sampling from the categorical
    distribution defined by p_t = t * x_0 + (1-t) * prior.

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

    # --- Node loss ---
    if lambda_train[0] > 0:
        true_X_flat = X.reshape(-1, X.size(-1))  # (bs*n, dx)
        pred_X_flat = pred.X.reshape(-1, pred.X.size(-1))
        mask_X = (true_X_flat != 0.0).any(dim=-1)
        if mask_X.any():
            loss_X = F.cross_entropy(pred_X_flat[mask_X], true_X_flat[mask_X].argmax(dim=-1))
            losses.append(lambda_train[0] * loss_X)

    # --- Edge loss ---
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


def euler_step_simplex(p_t, pred_x0_prob, t, dt, prior_probs):
    """Perform one Euler step on the probability simplex during sampling.

    The velocity field for discrete flow matching at time t is:
        v_t = (x_0_hat - prior) / (1 - eps)

    where x_0_hat is the network's predicted clean distribution.

    The update on the simplex is:
        p_{t+dt} = p_t + dt * v_t

    Then we project back to a valid probability distribution.

    Args:
        p_t: Current probability distribution (bs, ..., num_classes).
        pred_x0_prob: Predicted clean data distribution (bs, ..., num_classes).
        t: Current time (bs, 1).
        dt: Step size (scalar).
        prior_probs: Prior distribution (num_classes,).

    Returns:
        p_next: Updated probability distribution.
    """
    # Velocity: direction from prior toward predicted x_0
    # v_t = (pred_x0 - prior) for the linear interpolation flow
    if p_t.dim() == 3:
        prior = prior_probs.unsqueeze(0).unsqueeze(0).expand_as(p_t)
    elif p_t.dim() == 4:
        prior = prior_probs.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(p_t)
    else:
        raise ValueError(f"Unexpected dim: {p_t.dim()}")

    velocity = pred_x0_prob - prior

    # Euler step
    p_next = p_t + dt * velocity

    # Project back to simplex: clamp and renormalize
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

    # Initialize from prior at t=0
    if denoise_nodes:
        p_X = prior_X.unsqueeze(0).unsqueeze(0).expand(bs, n, -1).to(device)
    else:
        # Keep ground-truth node types
        p_X = X.clone()

    p_E = prior_E.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(bs, n, n, -1).to(device)

    # Euler integration from t=0 to t=1
    for step in range(num_steps):
        t = step * dt
        t_tensor = torch.full((bs, 1), t, device=device)

        # Sample current state from probability distribution
        z_t_E = sample_categorical(p_E)
        # Symmetrize edges
        z_t_E = (z_t_E + z_t_E.transpose(1, 2)) / 2.0
        # Re-sample to make it proper one-hot after symmetrization
        z_t_E_idx = z_t_E.argmax(dim=-1)
        z_t_E = F.one_hot(z_t_E_idx, num_classes=de).float()
        # Force symmetry
        upper = torch.triu(z_t_E_idx, diagonal=1)
        lower = upper.transpose(1, 2)
        sym_idx = upper + lower
        z_t_E = F.one_hot(sym_idx, num_classes=de).float()

        if denoise_nodes:
            z_t_X = sample_categorical(p_X)
        else:
            z_t_X = X.clone()

        # Get model prediction: logits -> probabilities
        pred_X_logits, pred_E_logits = model_forward_fn(z_t_X, z_t_E, y, t_tensor, node_mask)
        pred_X_prob = F.softmax(pred_X_logits, dim=-1)
        pred_E_prob = F.softmax(pred_E_logits, dim=-1)

        # Euler step on simplex
        if denoise_nodes:
            p_X = euler_step_simplex(p_X, pred_X_prob, t_tensor, dt, prior_X.to(device))
        p_E = euler_step_simplex(p_E, pred_E_prob, t_tensor, dt, prior_E.to(device))

    # Final sample from the converged distribution
    if denoise_nodes:
        final_X = sample_categorical(p_X)
    else:
        final_X = X.clone()

    final_E = sample_categorical(p_E)

    # Symmetrize final edges
    E_idx = final_E.argmax(dim=-1)
    upper = torch.triu(E_idx, diagonal=1)
    E_idx = upper + upper.transpose(1, 2)
    final_E = F.one_hot(E_idx, num_classes=de).float()

    return final_X, final_E

