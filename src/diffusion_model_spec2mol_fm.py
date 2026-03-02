"""Spec2Mol model using Discrete Flow Matching instead of D3PM diffusion.

This module replaces the discrete denoising diffusion (D3PM) training objective
and sampling loop with a Discrete Flow Matching approach. The key changes are:

1. Training: Linear interpolation on the probability simplex instead of
   Markov chain forward process. Loss is simply cross-entropy on predicting x_0.

2. Sampling: Euler integration of the learned velocity field on the simplex
   (typically 50 steps instead of 500).

3. No transition matrices, posterior distributions, or KL divergences needed.

The encoder, decoder (GraphTransformer), merge layer, and all metrics are
reused unchanged from the original model.
"""

import time
import logging
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Batch
from rdkit import Chem

from models.transformer_model import GraphTransformer
from metrics.abstract_metrics import CrossEntropyMetric
from src.metrics.diffms_metrics import K_ACC_Collection, K_SimilarityCollection, Validity
from src import utils
from src.mist.models.spectra_encoder import SpectraEncoderGrowing
from src.diffusion.discrete_flow_matching import (
    sample_zt,
    compute_flow_matching_loss,
    euler_step_simplex,
    sample_categorical,
)


class Spec2MolFlowMatching(pl.LightningModule):
    """Spectrum-to-Molecule generation via Discrete Flow Matching.

    Replaces the D3PM-style discrete diffusion with flow matching on the
    probability simplex. The neural network architecture (encoder + merge +
    GraphTransformer decoder) is identical; only the training objective and
    sampling loop differ.
    """

    def __init__(self, cfg, dataset_infos, train_metrics, visualization_tools,
                 extra_features, domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.T = cfg.model.diffusion_steps  # reused as num_sampling_steps
        self.num_sampling_steps = getattr(cfg.model, 'flow_matching_steps', 50)
        self.val_num_samples = cfg.general.val_samples_to_generate
        self.test_num_samples = cfg.general.test_samples_to_generate

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        # --- Training loss ---
        # Flow matching loss is computed directly via compute_flow_matching_loss.
        self.lambda_train = cfg.model.lambda_train

        # --- Validation / test metrics ---
        # Flow matching has no KL / logp decomposition. We track:
        #   - X_CE / E_CE: per-component cross-entropy (fast, every epoch)
        #   - FM loss as a scalar (logged as val/loss)
        #   - Sampling-based metrics: acc_at_k, tanimoto, cosine, validity
        self.val_X_CE = CrossEntropyMetric()
        self.val_E_CE = CrossEntropyMetric()
        self.val_k_acc = K_ACC_Collection(list(range(1, self.val_num_samples + 1)))
        self.val_sim_metrics = K_SimilarityCollection(list(range(1, self.val_num_samples + 1)))
        self.val_validity = Validity()

        self.test_X_CE = CrossEntropyMetric()
        self.test_E_CE = CrossEntropyMetric()
        self.test_k_acc = K_ACC_Collection(list(range(1, self.test_num_samples + 1)))
        self.test_sim_metrics = K_SimilarityCollection(list(range(1, self.test_num_samples + 1)))
        self.test_validity = Validity()

        self.train_metrics = train_metrics
        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        # --- Cross-attention feature flag ---
        self.use_cross_attention = getattr(cfg.model, 'cross_attention', False)

        # --- Decoder (GraphTransformer) ---
        hidden_size = getattr(cfg.model, 'encoder_hidden_dim', 256)
        self.decoder = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
            cross_attention=self.use_cross_attention,
            d_peak=hidden_size,
        )

        try:
            if cfg.general.decoder is not None:
                state_dict = torch.load(cfg.general.decoder, map_location='cpu')
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                    cleaned_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith('model.'):
                            k = k[6:]
                            cleaned_state_dict[k] = v
                    state_dict = cleaned_state_dict
                self.decoder.load_state_dict(state_dict, strict=not self.use_cross_attention)
        except Exception as e:
            logging.info(f"Could not load decoder: {e}")

        # --- Encoder ---
        magma_modulo = getattr(cfg.model, 'encoder_magma_modulo', 512)

        self.encoder = SpectraEncoderGrowing(
            inten_transform='float',
            inten_prob=0.1,
            remove_prob=0.5,
            peak_attn_layers=2,
            num_heads=8,
            pairwise_featurization=True,
            embed_instrument=False,
            cls_type='ms1',
            set_pooling='cls',
            spec_features='peakformula',
            mol_features='fingerprint',
            form_embedder='pos-cos',
            output_size=4096,
            hidden_size=hidden_size,
            spectra_dropout=0.1,
            top_layers=1,
            refine_layers=4,
            magma_modulo=magma_modulo,
        )

        try:
            if cfg.general.encoder is not None:
                self.encoder.load_state_dict(torch.load(cfg.general.encoder), strict=True)
        except Exception as e:
            logging.info(f"Could not load encoder: {e}")

        # --- Merge layer ---
        self.denoise_nodes = getattr(cfg.dataset, 'denoise_nodes', False)
        self.merge = getattr(cfg.dataset, 'merge', 'none')

        if self.merge == 'merge-encoder_output-linear':
            self.merge_function = nn.Linear(hidden_size, cfg.dataset.morgan_nbits)
        elif self.merge == 'merge-encoder_output-mlp':
            self.merge_function = nn.Sequential(
                nn.Linear(hidden_size, 1024),
                nn.SiLU(),
                nn.Linear(1024, cfg.dataset.morgan_nbits),
            )
        elif self.merge == 'downproject_4096':
            self.merge_function = nn.Linear(4096, cfg.dataset.morgan_nbits)

        # --- Prior distributions ---
        # Flow matching uses a prior distribution instead of transition matrices.
        # We reuse the marginal/uniform distribution from the dataset.
        if cfg.model.transition == 'marginal':
            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)
            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
        else:
            x_marginals = torch.ones(self.Xdim_output) / self.Xdim_output
            e_marginals = torch.ones(self.Edim_output) / self.Edim_output

        self.register_buffer('prior_X', x_marginals)
        self.register_buffer('prior_E', e_marginals)
        logging.info(f"[Flow Matching] Prior X: {self.prior_X}, Prior E: {self.prior_E}")

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.val_counter = 1

    # ================================================================
    # Merge helper (extracted to avoid duplication)
    # ================================================================
    def _apply_merge(self, data, output, aux):
        """Apply the merge strategy to set data.y from encoder outputs.

        Also stores peak tokens and mask for cross-attention when enabled.
        """
        if self.merge == 'mist_fp':
            data.y = aux["int_preds"][-1]
        elif self.merge == 'merge-encoder_output-linear':
            data.y = self.merge_function(aux['h0'])
        elif self.merge == 'merge-encoder_output-mlp':
            data.y = self.merge_function(aux['h0'])
        elif self.merge == 'downproject_4096':
            data.y = self.merge_function(output)

        # Store peak tokens for cross-attention (always extracted; only used
        # downstream when self.use_cross_attention is True).
        self._peak_tokens = aux.get("peak_tensor", None)   # (B, Np, hidden_size)
        self._peak_mask = aux.get("peak_mask", None)        # (B, Np), True=valid

        return data

    def _get_peak_context(self):
        """Return (peak_tokens, peak_mask) for cross-attention, or (None, None)."""
        if self.use_cross_attention:
            return self._peak_tokens, self._peak_mask
        return None, None

    # ================================================================
    # Forward pass (feeds into GraphTransformer with optional cross-attention)
    # ================================================================
    def forward(self, noisy_data, extra_data, node_mask,
                peak_tokens=None, peak_mask=None):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.decoder(X, E, y, node_mask,
                            peak_tokens=peak_tokens, peak_mask=peak_mask)

    # ================================================================
    # Extra features (same as original)
    # ================================================================
    def compute_extra_data(self, noisy_data):
        """Compute extra features (cycles, eigenvalues, molecular features, timestep)."""
        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    # ================================================================
    # TRAINING: Flow Matching interpolation + cross-entropy loss
    # ================================================================
    def apply_noise_flow_matching(self, X, E, y, node_mask):
        """Sample time t and construct z_t via simplex interpolation.

        Instead of the D3PM forward process (x_0 @ Q_t), we use:
            z_t ~ Cat(t * x_0 + (1-t) * prior)

        Returns:
            noisy_data dict compatible with compute_extra_data / forward.
        """
        bs = X.size(0)

        # Sample t ~ Uniform(0, 1) — continuous time
        t = torch.rand(bs, 1, device=X.device)

        # Construct z_t for edges via interpolation + sampling
        E_t = sample_zt(E, self.prior_E, t, node_mask)

        # Symmetrize edge samples
        E_t_idx = E_t.argmax(dim=-1)
        upper = torch.triu(E_t_idx, diagonal=1)
        E_t_idx = upper + upper.transpose(1, 2)
        E_t = F.one_hot(E_t_idx, num_classes=self.Edim_output).float()

        # Nodes: either interpolate or keep ground truth
        if self.denoise_nodes:
            X_t = sample_zt(X, self.prior_X, t, node_mask)
        else:
            X_t = X

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {
            't_int': (t * self.T).long(),  # For compatibility with extra_features
            't': t,
            'X_t': z_t.X,
            'E_t': z_t.E,
            'y_t': z_t.y,
            'node_mask': node_mask,
            # These are not needed for flow matching but kept for compatibility
            'beta_t': torch.zeros_like(t),
            'alpha_s_bar': torch.zeros_like(t),
            'alpha_t_bar': torch.zeros_like(t),
        }
        return noisy_data

    def training_step(self, batch, i):
        output, aux = self.encoder(batch)
        data = batch["graph"]
        data = self._apply_merge(data, output, aux)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # Flow matching: interpolate on simplex and sample z_t
        noisy_data = self.apply_noise_flow_matching(X, E, data.y, node_mask)

        extra_data = self.compute_extra_data(noisy_data)
        peak_tokens, peak_mask = self._get_peak_context()
        pred = self.forward(noisy_data, extra_data, node_mask,
                            peak_tokens=peak_tokens, peak_mask=peak_mask)

        # Flow matching loss: cross-entropy between predicted logits and true x_0
        loss = compute_flow_matching_loss(pred, X, E, node_mask, self.lambda_train)

        # Also log the standard train metrics for monitoring
        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E,
                           true_X=X, true_E=E, log=False)

        return {'loss': loss}

    # ================================================================
    # VALIDATION
    # ================================================================
    def validation_step(self, batch, i):
        output, aux = self.encoder(batch)
        data = batch["graph"]
        data = self._apply_merge(data, output, aux)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # Compute flow matching loss for validation
        noisy_data = self.apply_noise_flow_matching(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        peak_tokens, peak_mask = self._get_peak_context()
        pred = self.forward(noisy_data, extra_data, node_mask,
                            peak_tokens=peak_tokens, peak_mask=peak_mask)

        val_loss = compute_flow_matching_loss(pred, X, E, node_mask, self.lambda_train)

        # Per-component cross-entropy for monitoring
        true_X_flat = X.reshape(-1, X.size(-1))
        pred_X_flat = pred.X.reshape(-1, pred.X.size(-1))
        mask_X = (true_X_flat != 0.).any(dim=-1)
        if mask_X.any():
            self.val_X_CE(pred_X_flat[mask_X], true_X_flat[mask_X])

        true_E_flat = E.reshape(-1, E.size(-1))
        pred_E_flat = pred.E.reshape(-1, pred.E.size(-1))
        mask_E = (true_E_flat != 0.).any(dim=-1)
        if mask_E.any():
            self.val_E_CE(pred_E_flat[mask_E], true_E_flat[mask_E])

        # Sampling-based evaluation (periodic)
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            true_mols = [Chem.inchi.MolFromInchi(data.get_example(idx).inchi) for idx in range(len(data))]
            predicted_mols = [list() for _ in range(len(data))]
            for _ in range(self.val_num_samples):
                for idx, mol in enumerate(self.sample_batch(data)):
                    predicted_mols[idx].append(mol)

            for idx in range(len(data)):
                self.val_k_acc.update(predicted_mols[idx], true_mols[idx])
                self.val_sim_metrics.update(predicted_mols[idx], true_mols[idx])
                self.val_validity.update(predicted_mols[idx])

        return {'loss': val_loss}

    # ================================================================
    # SAMPLING: Euler integration on the probability simplex
    # ================================================================
    @torch.no_grad()
    def sample_batch(self, data: Batch):
        """Generate molecules by integrating the flow matching ODE.

        Instead of 500 reverse diffusion steps, we run ~50 Euler steps
        on the probability simplex from prior (t=0) to data (t=1).
        """
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        X = dense_data.X  # Ground truth nodes (kept fixed if denoise_nodes=False)
        y = data.y
        bs, n, dx = X.shape
        de = self.Edim_output
        device = self.device

        dt = 1.0 / self.num_sampling_steps

        # Initialize edge probabilities from prior
        p_E = self.prior_E.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(bs, n, n, -1).clone()

        if self.denoise_nodes:
            p_X = self.prior_X.unsqueeze(0).unsqueeze(0).expand(bs, n, -1).clone()
        else:
            p_X = X.clone()

        # Euler integration from t=0 (prior) to t=1 (data)
        for step in range(self.num_sampling_steps):
            t_val = step * dt
            t_tensor = torch.full((bs, 1), t_val, device=device)

            # Sample current state from probability distribution
            z_t_E = sample_categorical(p_E)

            # Symmetrize edges
            E_idx = z_t_E.argmax(dim=-1)
            upper = torch.triu(E_idx, diagonal=1)
            E_idx = upper + upper.transpose(1, 2)
            z_t_E = F.one_hot(E_idx, num_classes=de).float()

            if self.denoise_nodes:
                z_t_X = sample_categorical(p_X)
            else:
                z_t_X = X.clone()

            # Forward pass through the model
            noisy_data = {
                'X_t': z_t_X, 'E_t': z_t_E, 'y_t': y,
                't': t_tensor, 'node_mask': node_mask,
            }
            extra_data = self.compute_extra_data(noisy_data)
            peak_tokens, peak_mask = self._get_peak_context()
            pred = self.forward(noisy_data, extra_data, node_mask,
                                peak_tokens=peak_tokens, peak_mask=peak_mask)

            # Convert logits to probabilities
            pred_E_prob = F.softmax(pred.E, dim=-1)

            # Euler step on the simplex for edges
            p_E = euler_step_simplex(p_E, pred_E_prob, t_tensor, dt, self.prior_E)

            if self.denoise_nodes:
                pred_X_prob = F.softmax(pred.X, dim=-1)
                p_X = euler_step_simplex(p_X, pred_X_prob, t_tensor, dt, self.prior_X)

        # Final sample from the converged distribution
        final_E = sample_categorical(p_E)

        # Symmetrize final edges
        E_idx = final_E.argmax(dim=-1)
        upper = torch.triu(E_idx, diagonal=1)
        E_idx = upper + upper.transpose(1, 2)
        final_E = F.one_hot(E_idx, num_classes=de).float()

        if self.denoise_nodes:
            final_X = sample_categorical(p_X)
        else:
            final_X = X.clone()

        # Collapse to discrete and generate molecules
        result = utils.PlaceHolder(X=final_X, E=final_E, y=torch.zeros(bs, 0).to(device))
        result.X = X  # Always use ground-truth nodes for final output
        result = result.mask(node_mask, collapse=True)

        mols = []
        for nodes, adj_mat in zip(result.X, result.E):
            mols.append(self.visualization_tools.mol_from_graphs(nodes, adj_mat))

        return mols

    # ================================================================
    # TEST
    # ================================================================
    def test_step(self, batch, i):
        output, aux = self.encoder(batch)
        data = batch["graph"]
        data = self._apply_merge(data, output, aux)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        noisy_data = self.apply_noise_flow_matching(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        peak_tokens, peak_mask = self._get_peak_context()
        pred = self.forward(noisy_data, extra_data, node_mask,
                            peak_tokens=peak_tokens, peak_mask=peak_mask)

        val_loss = compute_flow_matching_loss(pred, X, E, node_mask, self.lambda_train)

        # Per-component cross-entropy for monitoring
        true_X_flat = X.reshape(-1, X.size(-1))
        pred_X_flat = pred.X.reshape(-1, pred.X.size(-1))
        mask_X = (true_X_flat != 0.).any(dim=-1)
        if mask_X.any():
            self.test_X_CE(pred_X_flat[mask_X], true_X_flat[mask_X])

        true_E_flat = E.reshape(-1, E.size(-1))
        pred_E_flat = pred.E.reshape(-1, pred.E.size(-1))
        mask_E = (true_E_flat != 0.).any(dim=-1)
        if mask_E.any():
            self.test_E_CE(pred_E_flat[mask_E], true_E_flat[mask_E])

        true_mols = [Chem.inchi.MolFromInchi(data.get_example(idx).inchi) for idx in range(len(data))]
        predicted_mols = [list() for _ in range(len(data))]

        for _ in range(self.test_num_samples):
            for idx, mol in enumerate(self.sample_batch(data)):
                predicted_mols[idx].append(mol)

        with open(f"preds/{self.name}_rank_{self.global_rank}_pred_{i}.pkl", "wb") as f:
            pickle.dump(predicted_mols, f)
        with open(f"preds/{self.name}_rank_{self.global_rank}_true_{i}.pkl", "wb") as f:
            pickle.dump(true_mols, f)

        for idx in range(len(data)):
            self.test_k_acc.update(predicted_mols[idx], true_mols[idx])
            self.test_sim_metrics.update(predicted_mols[idx], true_mols[idx])
            self.test_validity.update(predicted_mols[idx])

        return {'loss': val_loss}

    # ================================================================
    # Optimizer / scheduler / lifecycle (reused from original)
    # ================================================================
    def configure_optimizers(self):
        if self.cfg.train.scheduler == 'const':
            return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr,
                                     amsgrad=True, weight_decay=self.cfg.train.weight_decay)
        elif self.cfg.train.scheduler == 'one_cycle':
            opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr,
                                    amsgrad=True, weight_decay=self.cfg.train.weight_decay)
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=self.cfg.train.lr,
                total_steps=stepping_batches,
                pct_start=self.cfg.train.pct_start,
            )
            return [opt], [{'scheduler': scheduler, 'name': 'learning_rate',
                            'interval': 'step', 'frequency': 1}]
        else:
            raise ValueError(f'Unknown scheduler: {self.cfg.train.scheduler}')

    def on_fit_start(self) -> None:
        if self.global_rank == 0:
            logging.info(f"[Flow Matching] Input features: X-{self.Xdim}, E-{self.Edim}, y-{self.ydim}")
            logging.info(f"[Flow Matching] Sampling steps: {self.num_sampling_steps}")
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())

    def on_train_epoch_start(self) -> None:
        self.start_epoch_time = time.time()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = {"train_epoch/epoch": float(self.current_epoch),
                  "train_epoch/time": time.time() - self.start_epoch_time}

        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        for key, value in epoch_at_metrics.items():
            to_log[f"train_epoch/{key}"] = value
        for key, value in epoch_bond_metrics.items():
            to_log[f"train_epoch/{key}"] = value

        self.log_dict(to_log, sync_dist=True)
        if self.global_rank == 0:
            logging.info(f"Epoch {self.current_epoch}: X_CE: {to_log.get('train_epoch/x_CE', -1):.2f}"
                         f" -- E_CE: {to_log.get('train_epoch/E_CE', -1):.2f}"
                         f" -- time: {to_log['train_epoch/time']:.2f}")

    def on_validation_epoch_start(self) -> None:
        self.val_X_CE.reset()
        self.val_E_CE.reset()
        self.val_k_acc.reset()
        self.val_sim_metrics.reset()
        self.val_validity.reset()

    def on_validation_epoch_end(self) -> None:
        metrics = {
            "val/X_CE": self.val_X_CE.compute(),
            "val/E_CE": self.val_E_CE.compute(),
        }

        if self.val_counter % self.cfg.general.sample_every_val == 0:
            for key, value in self.val_k_acc.compute().items():
                metrics[f"val/{key}"] = value
            for key, value in self.val_sim_metrics.compute().items():
                metrics[f"val/{key}"] = value
            metrics["val/validity"] = self.val_validity.compute()

        self.log_dict(metrics, sync_dist=True)

        if self.global_rank == 0:
            logging.info(f"Epoch {self.current_epoch}: Val X_CE {metrics.get('val/X_CE', -1):.4f}"
                         f" -- Val E_CE {metrics.get('val/E_CE', -1):.4f}")

        self.val_counter += 1

    def on_test_epoch_start(self) -> None:
        if self.global_rank == 0:
            logging.info("Starting test...")
        self.test_X_CE.reset()
        self.test_E_CE.reset()
        self.test_k_acc.reset()
        self.test_sim_metrics.reset()
        self.test_validity.reset()

    def on_test_epoch_end(self) -> None:
        metrics = {
            "test/X_CE": self.test_X_CE.compute(),
            "test/E_CE": self.test_E_CE.compute(),
        }

        self.log_dict(metrics, sync_dist=True)
        if self.global_rank == 0:
            logging.info(f"Test X_CE: {metrics['test/X_CE']:.4f} -- Test E_CE: {metrics['test/E_CE']:.4f}")

        log_dict = {}
        for key, value in self.test_k_acc.compute().items():
            log_dict[f"test/{key}"] = value
        for key, value in self.test_sim_metrics.compute().items():
            log_dict[f"test/{key}"] = value
        log_dict["test/validity"] = self.test_validity.compute()

        self.log_dict(log_dict, sync_dist=True)

