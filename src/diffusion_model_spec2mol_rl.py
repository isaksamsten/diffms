import time
import logging
import pickle

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from rdkit import Chem

from src.metrics.abstract_metrics import CrossEntropyMetric
from src.metrics.train_metrics import TrainLossDiscrete
from src.metrics.diffms_metrics import K_ACC_Collection, K_SimilarityCollection, Validity
from src import utils
from src.rewards import RewardFunction


logger = logging.getLogger(__name__)


def _compute_supervised_loss(model, X, E, y, node_mask, model_type):
    """Compute the supervised (KL-regularisation) loss for either model type.

    For flow matching: sample noisy data via interpolation, predict x_0, CE loss.
    For D3PM: sample noisy data via forward diffusion, predict x_0, CE loss.
    """
    if model_type == 'flow_matching':
        from src.diffusion.discrete_flow_matching import compute_flow_matching_loss
        noisy_data = model.apply_noise_flow_matching(X, E, y, node_mask)
        extra_data = model.compute_extra_data(noisy_data)
        peak_tokens, peak_mask = model._get_peak_context()
        pred = model.forward(
            noisy_data, extra_data, node_mask,
            peak_tokens=peak_tokens, peak_mask=peak_mask,
        )
        return compute_flow_matching_loss(pred, X, E, node_mask, model.lambda_train)
    else:
        noisy_data = model.apply_noise(X, E, y, node_mask)
        extra_data = model.compute_extra_data(noisy_data)
        peak_tokens, peak_mask = model._get_peak_context()
        pred = model.forward(
            noisy_data, extra_data, node_mask,
            peak_tokens=peak_tokens, peak_mask=peak_mask,
        )
        return model.train_loss(
            masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
            true_X=X, true_E=E, true_y=y, log=False,
        )


class Spec2MolRLFinetuning(pl.LightningModule):

    def __init__(
        self,
        cfg,
        pretrained_model,
        reward_fn: RewardFunction,
        dataset_infos,
        train_metrics,
        visualization_tools,
        extra_features,
        domain_features,
    ):
        super().__init__()
        self.cfg = cfg
        self.name = cfg.general.name

        self.model_type = getattr(cfg.model, 'model_type', 'diffusion')

        self.model = pretrained_model

        self.reward_fn = reward_fn

        self.kl_coeff = float(getattr(cfg.train, 'rl_kl_coeff', 0.1))
        self.num_samples = int(getattr(cfg.train, 'rl_num_samples', 4))
        self.baseline_momentum = float(getattr(cfg.train, 'rl_baseline_momentum', 0.99))
        self.pg_coeff = float(getattr(cfg.train, 'rl_pg_coeff', 1.0))

        self.rl_algorithm = str(getattr(cfg.train, 'rl_algorithm', 'reinforce'))
        assert self.rl_algorithm in ('reinforce', 'grpo'), \
            f"Unknown rl_algorithm: {self.rl_algorithm}. Must be 'reinforce' or 'grpo'."

        self.grpo_clip_eps = float(getattr(cfg.train, 'grpo_clip_eps', 0.2))
        self.grpo_kl_coeff = float(getattr(cfg.train, 'grpo_kl_coeff', 0.01))

        self.rl_sampling_steps = int(getattr(cfg.train, 'rl_sampling_steps', 0))
        if self.rl_sampling_steps > 0 and self.model_type == 'flow_matching':
            self.model.num_sampling_steps = self.rl_sampling_steps

        for p in self.model.encoder.parameters():
            p.requires_grad = False

        self.register_buffer('reward_baseline', torch.tensor(0.0))

        self.val_num_samples = cfg.general.val_samples_to_generate
        self.test_num_samples = cfg.general.test_samples_to_generate

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

        self.visualization_tools = visualization_tools

        self.save_hyperparameters(ignore=[
            'pretrained_model', 'reward_fn', 'train_metrics',
            'visualization_tools', 'extra_features', 'domain_features',
            'dataset_infos',
        ])
        self.start_epoch_time = None
        self.val_counter = 1

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            output, aux = self.model.encoder(batch)
        data = batch["graph"]
        data = self.model._apply_merge(data, output, aux)

        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        all_rewards = []
        all_log_probs = []

        for _ in range(self.num_samples):
            mols, log_probs = self.model.sample_batch_with_log_probs(data)
            rewards = torch.tensor(
                [self.reward_fn(mol) for mol in mols],
                device=self.device, dtype=torch.float32,
            )
            all_rewards.append(rewards)
            all_log_probs.append(log_probs)

        rewards = torch.stack(all_rewards, dim=1)      # (bs, G)
        log_probs = torch.stack(all_log_probs, dim=1)   # (bs, G)
        mean_reward = rewards.mean()

        if self.rl_algorithm == 'grpo':
            pg_loss, algo_metrics = self._grpo_loss(rewards, log_probs)
        else:
            pg_loss, algo_metrics = self._reinforce_loss(rewards, log_probs, mean_reward)

        sup_loss = _compute_supervised_loss(
            self.model, X, E, data.y, node_mask, self.model_type,
        )

        loss = self.pg_coeff * pg_loss + self.kl_coeff * sup_loss

        bs = X.size(0)
        validity = (rewards > 0).float().mean()  # rough proxy

        log_dict = {
            'rl_train/loss': loss,
            'rl_train/pg_loss': pg_loss,
            'rl_train/sup_loss': sup_loss,
            'rl_train/mean_reward': mean_reward,
            'rl_train/reward_std': rewards.std(),
            'rl_train/approx_validity': validity,
        }
        log_dict.update(algo_metrics)
        self.log_dict(log_dict, prog_bar=True, sync_dist=True, batch_size=bs)

        return loss

    def _reinforce_loss(self, rewards, log_probs, mean_reward):
        """Standard REINFORCE with EMA running baseline.

        Args:
            rewards:     (bs, G)  scalar rewards per sample.
            log_probs:   (bs, G)  differentiable log-probs.
            mean_reward: scalar   mean of rewards (for baseline update).

        Returns:
            pg_loss:     scalar.
            metrics:     dict of extra things to log.
        """
        advantages = rewards - self.reward_baseline     # (bs, G)
        pg_loss = -(advantages.detach() * log_probs).mean()

        with torch.no_grad():
            self.reward_baseline = (
                self.baseline_momentum * self.reward_baseline
                + (1.0 - self.baseline_momentum) * mean_reward
            )

        return pg_loss, {'rl_train/baseline': self.reward_baseline}

    def _grpo_loss(self, rewards, log_probs):
        """Group Relative Policy Optimisation.

        An explicit KL penalty between the current and sampling-time
        policy is added (approximated from the ratio).

        Args:
            rewards:   (bs, G)  scalar rewards per sample.
            log_probs: (bs, G)  differentiable log-probs from replayed step.

        Returns:
            pg_loss:  scalar.
            metrics:  dict of extra things to log.
        """
        bs, G = rewards.shape
        eps = self.grpo_clip_eps

        group_mean = rewards.mean(dim=1, keepdim=True)   # (bs, 1)
        group_std = rewards.std(dim=1, keepdim=True)     # (bs, 1)
        advantages = (rewards - group_mean) / (group_std + 1e-8)  # (bs, G)

        log_probs_old = log_probs.detach()
        log_ratio = log_probs - log_probs_old             # (bs, G)
        ratio = torch.exp(log_ratio)                       # (bs, G)

        # surrogate
        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages.detach()
        pg_loss = -torch.min(surr1, surr2).mean()

        approx_kl = (ratio - 1.0 - log_ratio).mean()
        pg_loss = pg_loss + self.grpo_kl_coeff * approx_kl

        metrics = {
            'rl_train/grpo_adv_mean': advantages.mean(),
            'rl_train/grpo_adv_std': advantages.std(),
            'rl_train/grpo_ratio_mean': ratio.mean(),
            'rl_train/grpo_approx_kl': approx_kl,
            'rl_train/grpo_clip_frac': ((ratio - 1.0).abs() > eps).float().mean(),
        }

        return pg_loss, metrics

    def _eval_forward_pass(self, data, X, E, node_mask):
        """Run a single noisy forward pass and return (pred, loss) for monitoring."""
        if self.model_type == 'flow_matching':
            from src.diffusion.discrete_flow_matching import compute_flow_matching_loss
            noisy_data = self.model.apply_noise_flow_matching(X, E, data.y, node_mask)
            extra_data = self.model.compute_extra_data(noisy_data)
            peak_tokens, peak_mask = self.model._get_peak_context()
            pred = self.model.forward(
                noisy_data, extra_data, node_mask,
                peak_tokens=peak_tokens, peak_mask=peak_mask,
            )
            loss = compute_flow_matching_loss(pred, X, E, node_mask, self.model.lambda_train)
        else:
            noisy_data = self.model.apply_noise(X, E, data.y, node_mask)
            extra_data = self.model.compute_extra_data(noisy_data)
            peak_tokens, peak_mask = self.model._get_peak_context()
            pred = self.model.forward(
                noisy_data, extra_data, node_mask,
                peak_tokens=peak_tokens, peak_mask=peak_mask,
            )
            loss = self.model.train_loss(
                masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                true_X=X, true_E=E, true_y=data.y, log=False,
            )
        return pred, loss

    def validation_step(self, batch, i):
        with torch.no_grad():
            output, aux = self.model.encoder(batch)
        data = batch["graph"]
        data = self.model._apply_merge(data, output, aux)

        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        pred, val_loss = self._eval_forward_pass(data, X, E, node_mask)

        true_E_flat = E.reshape(-1, E.size(-1))
        pred_E_flat = pred.E.reshape(-1, pred.E.size(-1))
        mask_E = (true_E_flat != 0.).any(dim=-1)
        if mask_E.any():
            self.val_E_CE(pred_E_flat[mask_E], true_E_flat[mask_E])

        true_X_flat = X.reshape(-1, X.size(-1))
        pred_X_flat = pred.X.reshape(-1, pred.X.size(-1))
        mask_X = (true_X_flat != 0.).any(dim=-1)
        if mask_X.any():
            self.val_X_CE(pred_X_flat[mask_X], true_X_flat[mask_X])

        if self.val_counter % self.cfg.general.sample_every_val == 0:
            true_mols = [
                Chem.inchi.MolFromInchi(data.get_example(idx).inchi)
                for idx in range(len(data))
            ]
            predicted_mols = [list() for _ in range(len(data))]
            batch_rewards = []

            for _ in range(self.val_num_samples):
                sample_mols = self.model.sample_batch(data)
                for idx, mol in enumerate(sample_mols):
                    predicted_mols[idx].append(mol)
                    batch_rewards.append(self.reward_fn(mol))

            for idx in range(len(data)):
                self.val_k_acc.update(predicted_mols[idx], true_mols[idx])
                self.val_sim_metrics.update(predicted_mols[idx], true_mols[idx])
                self.val_validity.update(predicted_mols[idx])

            avg_reward = sum(batch_rewards) / max(len(batch_rewards), 1)
            self.log('val/mean_reward', avg_reward, sync_dist=True)

        return {'loss': val_loss}

    def test_step(self, batch, i):
        with torch.no_grad():
            output, aux = self.model.encoder(batch)
        data = batch["graph"]
        data = self.model._apply_merge(data, output, aux)

        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        pred, val_loss = self._eval_forward_pass(data, X, E, node_mask)

        true_E_flat = E.reshape(-1, E.size(-1))
        pred_E_flat = pred.E.reshape(-1, pred.E.size(-1))
        mask_E = (true_E_flat != 0.).any(dim=-1)
        if mask_E.any():
            self.test_E_CE(pred_E_flat[mask_E], true_E_flat[mask_E])

        true_X_flat = X.reshape(-1, X.size(-1))
        pred_X_flat = pred.X.reshape(-1, pred.X.size(-1))
        mask_X = (true_X_flat != 0.).any(dim=-1)
        if mask_X.any():
            self.test_X_CE(pred_X_flat[mask_X], true_X_flat[mask_X])

        true_mols = [
            Chem.inchi.MolFromInchi(data.get_example(idx).inchi)
            for idx in range(len(data))
        ]
        predicted_mols = [list() for _ in range(len(data))]
        batch_rewards = []

        for _ in range(self.test_num_samples):
            sample_mols = self.model.sample_batch(data)
            for idx, mol in enumerate(sample_mols):
                predicted_mols[idx].append(mol)
                batch_rewards.append(self.reward_fn(mol))

        with open(f"preds/{self.name}_rank_{self.global_rank}_pred_{i}.pkl", "wb") as f:
            pickle.dump(predicted_mols, f)
        with open(f"preds/{self.name}_rank_{self.global_rank}_true_{i}.pkl", "wb") as f:
            pickle.dump(true_mols, f)

        for idx in range(len(data)):
            self.test_k_acc.update(predicted_mols[idx], true_mols[idx])
            self.test_sim_metrics.update(predicted_mols[idx], true_mols[idx])
            self.test_validity.update(predicted_mols[idx])

        return {'loss': val_loss}

    def configure_optimizers(self):
        rl_lr = float(getattr(self.cfg.train, 'rl_lr', self.cfg.train.lr * 0.1))
        params = [p for p in self.model.decoder.parameters() if p.requires_grad]
        if hasattr(self.model, 'merge_function'):
            merge_params = [p for p in self.model.merge_function.parameters() if p.requires_grad]
            params = params + merge_params

        logger.info(f"RL optimizer: {len(params)} parameter groups, lr={rl_lr}")

        if getattr(self.cfg.train, 'rl_scheduler', 'const') == 'const':
            return torch.optim.AdamW(
                params, lr=rl_lr, amsgrad=True,
                weight_decay=self.cfg.train.weight_decay,
            )
        elif self.cfg.train.rl_scheduler == 'one_cycle':
            opt = torch.optim.AdamW(
                params, lr=rl_lr, amsgrad=True,
                weight_decay=self.cfg.train.weight_decay,
            )
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=rl_lr, total_steps=stepping_batches,
                pct_start=self.cfg.train.pct_start,
            )
            return [opt], [{
                'scheduler': scheduler, 'name': 'learning_rate',
                'interval': 'step', 'frequency': 1,
            }]
        else:
            raise ValueError(f'Unknown scheduler: {self.cfg.train.rl_scheduler}')

    def on_fit_start(self) -> None:
        if self.global_rank == 0:
            logger.info(f"[RL Finetuning] algorithm={self.rl_algorithm}")
            logger.info(f"[RL Finetuning] model_type={self.model_type}")
            logger.info(f"[RL Finetuning] reward={self.reward_fn}")
            logger.info(f"[RL Finetuning] kl_coeff={self.kl_coeff}, "
                        f"num_samples={self.num_samples}, "
                        f"pg_coeff={self.pg_coeff}")
            if self.rl_algorithm == 'grpo':
                logger.info(f"[RL Finetuning] GRPO clip_eps={self.grpo_clip_eps}, "
                            f"kl_coeff={self.grpo_kl_coeff}")
            n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.parameters())
            logger.info(f"[RL Finetuning] Trainable params: {n_trainable:,} / {n_total:,}")
            if self.model_type == 'diffusion':
                logger.info(f"[RL Finetuning] D3PM sampling: {self.model.T} reverse steps")
            else:
                logger.info(f"[RL Finetuning] FM sampling: {self.model.num_sampling_steps} Euler steps")

    def on_train_epoch_start(self) -> None:
        self.start_epoch_time = time.time()

    def on_train_epoch_end(self) -> None:
        elapsed = time.time() - self.start_epoch_time
        self.log('rl_train/epoch_time', elapsed, sync_dist=True)
        if self.global_rank == 0:
            logger.info(
                f"Epoch {self.current_epoch}: "
                f"baseline={self.reward_baseline.item():.4f}, "
                f"time={elapsed:.1f}s"
            )

    def on_validation_epoch_start(self) -> None:
        self.val_X_CE.reset()
        self.val_E_CE.reset()
        self.val_k_acc.reset()
        self.val_sim_metrics.reset()
        self.val_validity.reset()

    def on_validation_epoch_end(self) -> None:
        metrics = {
            'val/X_CE': self.val_X_CE.compute(),
            'val/E_CE': self.val_E_CE.compute(),
        }

        if self.val_counter % self.cfg.general.sample_every_val == 0:
            for key, value in self.val_k_acc.compute().items():
                metrics[f'val/{key}'] = value
            for key, value in self.val_sim_metrics.compute().items():
                metrics[f'val/{key}'] = value
            metrics['val/validity'] = self.val_validity.compute()
        else:
            metrics['val/mean_reward'] = float('-inf')

        self.log_dict(metrics, sync_dist=True)
        if self.global_rank == 0:
            logger.info(
                f"Epoch {self.current_epoch}: "
                f"Val X_CE={metrics.get('val/X_CE', -1):.4f} "
                f"Val E_CE={metrics.get('val/E_CE', -1):.4f}"
            )
        self.val_counter += 1

    def on_test_epoch_start(self) -> None:
        if self.global_rank == 0:
            logger.info("Starting RL test...")
        self.test_X_CE.reset()
        self.test_E_CE.reset()
        self.test_k_acc.reset()
        self.test_sim_metrics.reset()
        self.test_validity.reset()

    def on_test_epoch_end(self) -> None:
        metrics = {
            'test/X_CE': self.test_X_CE.compute(),
            'test/E_CE': self.test_E_CE.compute(),
        }
        self.log_dict(metrics, sync_dist=True)

        log_dict = {}
        for key, value in self.test_k_acc.compute().items():
            log_dict[f'test/{key}'] = value
        for key, value in self.test_sim_metrics.compute().items():
            log_dict[f'test/{key}'] = value
        log_dict['test/validity'] = self.test_validity.compute()
        self.log_dict(log_dict, sync_dist=True)

        if self.global_rank == 0:
            logger.info(
                f"Test X_CE={metrics['test/X_CE']:.4f} "
                f"E_CE={metrics['test/E_CE']:.4f} "
                f"validity={log_dict.get('test/validity', -1):.4f}"
            )

