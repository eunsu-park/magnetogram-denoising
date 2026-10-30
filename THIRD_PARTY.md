# Third-party methods and packages

The code in this repository is released under the MIT license (`LICENSE`). The parts below
reimplement published methods or depend on packages with their own licenses. No third-party
source code is redistributed here.

## Reimplemented methods

- **Park et al. (2020)**: `network.py` (`Generator`, `Discriminator`), `train.py`
  (`train_park2020`). Park, E., Moon, Y.-J., Lim, D., & Lee, H. 2020, ApJL, 891, L4,
  doi:10.3847/2041-8213/ab74d2. A PyTorch port of the pix2pix architecture (U-Net generator,
  70 x 70 PatchGAN discriminator, cGAN + L1 loss) of the authors' reference implementation
  (https://github.com/eunsu-park/solar_magnetogram_denoising). The batch size (4 instead of 1) and
  the strong-field restoration after denoising (`pipeline.restore_strong`) are ours.
- **Noise gating**: `analysis.noise_gate`. DeForest, C. E. 2017, ApJ, 838, 155. An implementation
  of the method as described in the paper (Hann-apodized 3-D Fourier chunks, hard gate at gamma
  times the white-noise level); it uses no code from the published noise-gating software.
- **Noise2Void**: `pipeline.FrameMasked`, `train.py --model n2v`. Krull, A., Buchholz, T.-O., &
  Jug, F. 2019, CVPR. Our implementation of the blind-spot training with the U-Net of this
  repository.
- **Figures 2-5**: `figures.py` is generated from a co-author's plotting notebooks, with their
  plot settings as keyword defaults.

## Packages

- **`bm3d`** (Tampere University; the BM3D baseline of Tables 1 and 2 and the injection test).
  Licensed for **non-commercial use only**. It is not installed by `requirements.txt`; users who
  accept that license install it separately (`requirements-baselines.txt`). The paper used
  bm3d 4.0.3.
- `torch`, `numpy`, `scipy`, `scikit-image`, `matplotlib`: under their own open-source licenses
  (BSD, PSF-compatible); used as installed packages only.
