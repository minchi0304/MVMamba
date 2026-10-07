# MVMamba

This repository contains the implementation used for the experiments in our anonymous submission. It supports multi-camera detection and tracking on Wildtrack, MultiviewX, CityTrack, MVCrowdTrack, and GMVD.

## Environment

The provided `Dockerfile` uses Python 3.10, PyTorch 2.0.1, torchvision 0.15.2, CUDA 11.8, and Mamba-SSM 1.2.0.post1.

To install the remaining dependencies:

```bash
pip install -r requirements.txt --no-build-isolation
```

## Dataset preparation

Place the datasets under a common root directory:

```text
<DATA_ROOT>/
├── Wildtrack/
├── MultiviewX/
├── CityTrack/
├── MVCrowdTrack/
└── GMVD/
```

For GMVD, update `train_datapath.csv` and `test_datapath.csv` so that each entry points to the corresponding sequence. These CSV files must be placed in the project root.

## Training

The following command trains the model on Wildtrack:

```bash
python main.py \
  --dataset wildtrack \
  --data_root <DATA_ROOT> \
  --motion_model mamba
```

Replace `wildtrack` with `multiviewx`, `citytrack`, `mvcrowdtrack`, or `gmvd_train` as needed.

## Test

Pass a trained checkpoint through `--resume` to run testing:

```bash
python main.py \
  --dataset wildtrack \
  --data_root <DATA_ROOT> \
  --resume <CHECKPOINT_PATH>
```

Test outputs are saved under `logs/`.

## Tracking options

The main tracking controls are:

- `--high_cls_thres`: confidence threshold for high-confidence detections
- `--motion_model`: `mamba`, `kalman`, or `diffusion` (`mamba` is the default)
- `--motion_history_len`: number of `[x, y, dx, dy]` trajectory tokens (default: `5`)
- `--motion_normalization`: normalize coordinates and deltas by BEV width/height (`bev`) or keep raw grid units (`none`)
- `--use_offset`: `1` to use BEV sub-grid offsets, `0` to disable them

The Mamba predictor estimates the normalized forward displacement. During tracking it is converted back to world-grid units and applied as
`predicted_xy = previous_xy + predicted_delta`. A matched detection then replaces the predicted state and becomes the newest trajectory observation.

Use `python main.py --help` for the complete list of options.

## Notes

- Pretrained checkpoints and datasets are not included.
- Training logs, generated visualizations, and test results are written to `logs/`.
- Author and affiliation information is intentionally omitted for anonymous review.
