# magnetogram-denoising

Code for *Self-supervised Deep-learning Denoising of Solar Line-of-sight Magnetograms and Its
Applications* (Park, Lee & Moon). A small U-Net is trained with Noise2Noise on pairs of SDO/HMI
45 s magnetograms taken 45 s apart, with no clean target. Gaussian noise of a random level is added
during training, so one model works over a threefold range of noise levels, and the same model is
applied without retraining to SOHO/MDI and Solar Orbiter/PHI-HRT magnetograms.

This repository holds everything needed to retrain the models of the paper and to recompute every
number, table and figure of it from the released data. The data, the model outputs and the trained
checkpoints are in a separate data repository (URL to be added).

## Results of the paper

| test | result |
|---|---|
| HMI, 2013 December (672 hours), 21-frame stack reference | PSNR 31.47 dB, residual sigma 3.04 G (input 25.57 dB, 7.86 G); adaptive TV 31.03 dB, BM3D 30.98 dB |
| HMI, 2019-2020 (1,947 hours), six years after training | 29.00 dB (input 22.36 dB) |
| HMI-MDI weak-field disagreement, 2,468 pairs | 14.89 to 2.71 G (82%) |
| HMI-PHI weak-field disagreement, 14 pairs | 12.10 to 3.42 G (72%), the same as two HMI frames 270 s apart (ratio 1.015) |

`analysis.py` prints the rest (Tables 1 and 2, Sections 3-4, Appendices); `notebooks/04_results.ipynb`
shows them with Figs. 2-5.

## Install

```
pip install -r requirements.txt               # torch, numpy, scipy, scikit-image, matplotlib
pip install -r requirements-baselines.txt     # optional: BM3D (non-commercial license, see THIRD_PARTY.md)
```

## Data

Download the data repository and pass its folder as `--data-root` (notebooks: `DATA_ROOT`).
Every sample is one `npz` file with a `meta` entry (JSON) naming its source record (JSOC `T_REC`,
MDI and PHI file names) and how it was made. File names give the UTC time of the observation.

| folder | content |
|---|---|
| `frames/train`, `frames/val` | 145,586 HMI `hmi.M_45s` frames, central 512 x 512 px, float16 (the values the training reads) |
| `frames/test` | the 14,160 frames of the 672 test hours, float32 |
| `stacks/<split>` | 9,501 hourly groups: the input frame and its 21-frame stack (train, val, test, test_xcycle) |
| `pairs/mdi` | 2,468 HMI-MDI pairs on the MDI grid (HMI reprojected), with the 0.85 R_sun disk |
| `pairs/phi` | 14 HMI-PHI pairs on the HMI and HRT grids, with the 45 s / 270 s HMI references |
| `manifests` | the files of every split and group (`frames.csv`, `stacks.csv`) and the source records |
| `checkpoints` | the 19 trained models, `<name>/model.pt`, with `SHA256SUMS` |
| `outputs`, `results` | the model outputs and the paper's tables, numbers and figure data, made by `analysis.py` |

The preprocessing (reprojection, co-alignment, stacking of the frames) was done once with the
authors' research code; its products are what is released. The data are released under
CC BY 4.0.

## Quick start

| notebook | |
|---|---|
| `notebooks/01_data.ipynb` | look at the frames, stacks and pairs |
| `notebooks/02_train.ipynb` | train a model; compare a checkpoint with the released one |
| `notebooks/03_test.ipynb` | score a checkpoint (Tables 1 and 2) |
| `notebooks/04_results.ipynb` | the tables (printed, saved as `.csv` / `.txt`) and Figs. 2-5 |

From the command line:

```
python train.py --model n2n --data-root <data> --out-dir runs/n2n               # the model of the paper
python test.py --ckpt runs/n2n/model.pt --data-root <data> --table 1a 2a 2b
python analysis.py --data-root <data> --ckpt-dir <data>/checkpoints --out-dir results
```

## Reproducing the paper

```
bash reproduce.sh <data> <output folder>      # all 19 models (about 15 h on one GPU), then analysis.py
```

`reproduce.sh` lists the options of every model. Training is deterministic: with the same data,
seed, package versions (`requirements.txt`: Python 3.12, PyTorch 2.13.0 with CUDA 13.0 and cuDNN
9.2) and GPU type (we used an NVIDIA GeForce RTX 3090), `model.pt` is byte-identical to the
released checkpoint (compare with `checkpoints/SHA256SUMS`; keep the file name, which is stored
inside the file). On other GPUs or package versions the trained weights can differ in the last
bits; we have not measured how far the numbers move then. Evaluation is less sensitive: scoring
the released checkpoints on another platform (an Apple GPU) gave the paper's numbers at their
printed precision. `test.py` and `analysis.py` reproduce the paper's numbers from the released
checkpoints. One exception to bitwise reproduction: the `bm3d` package gives slightly different
outputs from run to run on the same machine (relative differences of about 1e-7), so the BM3D rows
are reproduced at their printed precision only.

Checked on our machine: retraining all 19 models with `reproduce.sh` from the released data gave
checkpoints whose weights are bitwise equal to the released ones, and `analysis.py` then
reproduced every table, number and figure-data file (the BM3D values to about 1e-7).

## Files

| file | content |
|---|---|
| `options.py` | the options; the defaults are the settings of the paper |
| `network.py` | the U-Net (and its depth/width variants) and the Park et al. (2020) pix2pix networks |
| `pipeline.py` | data loading, stacking, normalization, metrics |
| `train.py`, `validation.py` | training (Noise2Noise, Noise2Void, supervised U-Net, Park et al. 2020) and its validation |
| `test.py` | the model rows of Tables 1 and 2 |
| `analysis.py` | every other number of the paper, the tables, the figure data |
| `figures.py` | Figs. 2-5 (Fig. 1 is `assets/fig1-pipeline.png`) |
| `reproduce.sh` | the training commands of all models, then `analysis.py` |

## License

MIT (`LICENSE`). Reimplemented methods and third-party packages: `THIRD_PARTY.md`.
