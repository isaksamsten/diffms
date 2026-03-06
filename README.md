# DiffMS: Diffusion Generation of Molecules Conditioned on Mass Spectra

![teaser](./figs/diffms-animation.gif)

This is the codebase for our preprint [DiffMS: Diffusion Generation of Molecules Conditioned on Mass Spectra](https://arxiv.org/abs/2502.09571).

The DiffMS codebase is adapted from [DiGress](https://github.com/cvignac/DiGress).

## Environment installation
This code was tested with PyTorch 2.3.1, cuda 11.8 and torch_geometrics 2.3.1

  - Download anaconda/miniconda if needed
  - Create a conda environment with rdkit:

    ```
    conda create -y -c conda-forge -n diffms rdkit=2024.09.4 python=3.9
    conda activate diffms
    ```

  - OR for a faster installation, you can use mamba:

    ```
    mamba create -y -n diffms rdkit=2024.09.4 python=3.9
    mamba activate diffms
    ```

  - Install a corresponding version of pytorch, for example:

    ```pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118```

  - Run:

    ```pip install -e .```


## Dataset Download/Processing

We provide a series of scripts to download/process the pretraining and finetuning datasets. To download/setup the datasets, run the scripts in the data_processing/ folder in order:

```
bash data_processing/00_download_fp2mol_data.sh
bash data_processing/01_download_canopus_data.sh
bash data_processing/02_download_msg_data.sh
bash data_processing/03_preprocess_fp2mol.sh
```

## Run the code

For fingerprint-molecule pretraining run [fp2mol_main.py](src/fp2mol_main.py). You will need to set the dataset in [config.yaml](configs/config.yaml) to 'fp2mol'. The primary pretraining dataset in our paper is referred to as 'combined' in the [fp2mol.yaml](configs/dataset/fp2mol.yaml) config.

To finetune the end-to-end model on spectra-molecule generation, run [spec2mol_main.py](src/spec2mol_main.py). You will also need to set the dataset in [config.yaml](configs/config.yaml) to 'msg' for MassSpecGym or 'canopus' for NPLIB1.

### Discrete Flow Matching (experimental)

To use Discrete Flow Matching instead of the default D3PM diffusion, set in [model_default.yaml](configs/model/model_default.yaml):

```yaml
model_type: 'flow_matching'
flow_matching_steps: 50
```


## Pretrained Checkpoints

We provide checkpoints for the end-to-end finetuned DiffMS model as well as the pretrained encoder/decoder weights [here](https://zenodo.org/records/15122968).

### Download

```bash
bash data_processing/04_download_checkpoints.sh
```

This downloads the Zenodo archive ([link](https://zenodo.org/records/15122968)) containing:
- **`diffms_checkpoints.tar.gz`** — encoder, decoder, and finetuned model weights → extracted to `checkpoints/`
- **`msg_preprocessed.tar.gz`** — preprocessed MassSpecGym data → extracted to `data/msg/`

### Loading Pretrained Weights

Set paths in [general_default.yaml](configs/general/general_default.yaml). Check `ls checkpoints/` for exact filenames after extraction.

**Option A: Load individual encoder/decoder for finetuning**
```yaml
# configs/general/general_default.yaml
encoder: 'checkpoints/<encoder_filename>'     # pretrained spectra encoder
decoder: 'checkpoints/<decoder_filename>'     # pretrained GraphTransformer
```
The encoder weights are loaded directly. The decoder checkpoint uses the FP→Mol format (keys prefixed with `model.`), which is automatically stripped during loading.

**Option B: Load the full end-to-end model**
```yaml
# configs/general/general_default.yaml
load_weights: 'checkpoints/<model_filename>'
```
This loads all weights (encoder + merge + decoder) from a finetuned checkpoint using `strict=False`, so any architecture mismatches are skipped gracefully.

**Option C: Resume training or run test-only**
```yaml
# configs/general/general_default.yaml
resume: 'checkpoints/<model_filename>'       # resume training from checkpoint
# OR
test_only: 'checkpoints/<model_filename>'    # evaluate only
```

### Using Checkpoints with Discrete Flow Matching

The pretrained encoder and decoder weights are **fully compatible** with the Discrete Flow Matching model. The architecture is identical, only the training objective differs. To use them:

```yaml
# configs/model/model_default.yaml
model_type: 'flow_matching'       # use DFM instead of D3PM
flow_matching_steps: 50           # Euler sampling steps (default: 50)

# configs/general/general_default.yaml
encoder: 'checkpoints/<encoder_filename>'
decoder: 'checkpoints/<decoder_filename>'
```


### Finetuning Strategies

Control which parts of the model are frozen during finetuning:

```yaml
# configs/general/general_default.yaml
encoder_finetune_strategy: 'freeze'        # freeze | ft-unfold | freeze-unfold | freeze-transformer | ft-transformer
decoder_finetune_strategy: 'freeze-input'  # freeze | ft-input | freeze-input | ft-transformer | freeze-transformer | ft-output
```

## RL Finetuning (Stage 3)

After supervised training, the model can be further finetuned with reinforcement learning to optimise non-differentiable molecular properties.

The encoder is frozen and only the decoder is updated. A supervised loss on the same training batch acts as KL regularisation to prevent mode collapse.

### Quick start

```bash
python -m src.spec2mol_rl_main \
    general.name=rl_validity \
    train=train_rl \
    train.rl_pretrained_checkpoint=<path_to_pretrained_checkpoint> \
    train.rl_max_train_samples=10000 \
    train.rl_max_val_samples=50
```

### Configuration


```yaml
rl_algorithm: 'reinforce'     # 'reinforce' or 'grpo'
rl_reward: 'validity'         # 'validity' | 'qed' | 'sa' | 'composite'
rl_num_samples: 4             # molecules sampled per input for the policy gradient
rl_kl_coeff: 0.1              # weight of supervised loss (KL regularisation)
rl_pg_coeff: 1.0              # weight of policy gradient loss
rl_lr: 2e-5                   # learning rate (decoder only)
```

Works with both D3PM (`model.model_type=diffusion`) and Flow Matching
(`model.model_type=flow_matching`).

## License

DiffMS is released under the [MIT](LICENSE.txt) license.

## Contact

If you have any questions, please reach out to mbohde@tamu.edu

## Reference
If you find this codebase useful in your research, please kindly cite the following manuscript
```
@article{bohde2025diffms,
  title={DiffMS: Diffusion Generation of Molecules Conditioned on Mass Spectra},
  author={Bohde, Montgomery and Manjrekar, Mrunali and Wang, Runzhong and Ji, Shuiwang and Coley, Connor W},
  journal={arXiv preprint arXiv:2502.09571},
  year={2025}
}
```
