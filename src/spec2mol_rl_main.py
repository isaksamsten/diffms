
import os
import sys
import pathlib
import warnings
import logging

import torch
torch.cuda.empty_cache()

_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

try:
    torch.set_float32_matmul_precision('medium')
except Exception:
    pass

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from src.diffusion_model_spec2mol import Spec2MolDenoisingDiffusion
from src.diffusion_model_spec2mol_fm import Spec2MolFlowMatching
from src.diffusion_model_spec2mol_rl import Spec2MolRLFinetuning
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.analysis.visualization import MolecularVisualization
from src.datasets import spec2mol_dataset
from src.rewards import build_reward_from_cfg


warnings.filterwarnings("ignore", category=PossibleUserWarning)


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    logger = logging.getLogger("spec2mol_rl")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler("spec2mol_rl.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logging.info(cfg)

    dataset_config = cfg["dataset"]
    if dataset_config["name"] not in ("canopus", "msg"):
        raise NotImplementedError(f"Unknown dataset {cfg['dataset']}")

    datamodule = spec2mol_dataset.Spec2MolDataModule(cfg)
    dataset_infos = spec2mol_dataset.Spec2MolDatasetInfos(datamodule, cfg)

    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )

    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    visualization_tools = MolecularVisualization(
        cfg.dataset.remove_h, dataset_infos=dataset_infos,
    )

    model_kwargs = {
        'dataset_infos': dataset_infos,
        'train_metrics': train_metrics,
        'visualization_tools': visualization_tools,
        'extra_features': extra_features,
        'domain_features': domain_features,
    }

    model_type = getattr(cfg.model, 'model_type', 'diffusion')
    model_cls = Spec2MolFlowMatching if model_type == 'flow_matching' else Spec2MolDenoisingDiffusion

    ckpt_path = getattr(cfg.train, 'rl_pretrained_checkpoint', None)
    if ckpt_path is not None and ckpt_path != 'null':
        logging.info(f"Loading pretrained {model_type} model from {ckpt_path}")
        pretrained = model_cls.load_from_checkpoint(ckpt_path, **model_kwargs)
    else:
        logging.info(f"Initialising {model_type} model from config (no pretrained checkpoint)")
        pretrained = model_cls(cfg=cfg, **model_kwargs)

    reward_fn = build_reward_from_cfg(cfg)
    logging.info(f"Reward function: {reward_fn}")

    os.makedirs('preds/', exist_ok=True)
    os.makedirs('logs/', exist_ok=True)
    os.makedirs(f'logs/{cfg.general.name}', exist_ok=True)

    rl_model = Spec2MolRLFinetuning(
        cfg=cfg,
        pretrained_model=pretrained,
        reward_fn=reward_fn,
        dataset_infos=dataset_infos,
        train_metrics=train_metrics,
        visualization_tools=visualization_tools,
        extra_features=extra_features,
        domain_features=domain_features,
    )

    callbacks = [LearningRateMonitor(logging_interval='step')]

    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename='{epoch}',
            monitor='val/E_CE',
            save_top_k=3,
            mode='min',
            every_n_epochs=1,
        )
        last_ckpt = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename='last',
            every_n_epochs=1,
        )

        reward_ckpt = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename='best_reward_{epoch}',
            monitor='val/mean_reward',
            save_top_k=1,
            mode='max',
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        )
        callbacks.extend([checkpoint_callback, last_ckpt, reward_ckpt])

    name = cfg.general.name

    loggers = [
        CSVLogger(save_dir=f"logs/{name}", name=name),
        WandbLogger(
            name=name, save_dir=f"logs/{name}",
            project=cfg.general.wandb_name,
            log_model=False,
            config=utils.cfg_to_dict(cfg),
        ),
    ]

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy="ddp_find_unused_parameters_true",
        accelerator='gpu' if use_gpu else 'cpu',
        devices=cfg.general.gpus if use_gpu else 1,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=cfg.general.name == 'debug',
        callbacks=callbacks,
        log_every_n_steps=50 if name != 'debug' else 1,
        logger=loggers,
    )

    if not cfg.general.test_only:
        trainer.fit(rl_model, datamodule=datamodule)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(rl_model, datamodule=datamodule)
    else:
        trainer.test(rl_model, datamodule=datamodule, ckpt_path=cfg.general.test_only)


if __name__ == '__main__':
    main()

