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
  --motion_model mamba \
  --motion_loss niw \
  --evidential_head separate
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
- `--motion_model`: `mamba` or `kalman` (`mamba` is the default)
- `--motion_history_len`: number of `[x, y, dx, dy]` trajectory tokens (default: `5`)
- `--mamba_history_mode`: `window` uses the fixed trajectory window; `stateful` keeps each track's Mamba convolution and SSM state across frames
- `--motion_history_source`: `detection` keeps detection-corrected history, `motion_immediate` starts motion rollout after the first detection, and `motion_warmup` uses two detections before motion rollout
- `--motion_normalization`: normalize coordinates and deltas by BEV width/height (`bev`) or keep raw grid units (`none`)
- `--motion_loss`: `niw` predicts a correlated 2D offset distribution; `smooth_l1` keeps the deterministic baseline
- `--evidential_lambda`: NIW evidence regularization coefficient (default: `1e-3`)
- `--evidential_head`: `separate` uses independent mean/evidence MLPs; `joint` uses one MLP with seven outputs
- `--use_offset`: `1` to use BEV sub-grid offsets, `0` to disable them

The Mamba predictor estimates the normalized forward displacement. During tracking it is converted back to world-grid units and applied as
`predicted_xy = previous_xy + predicted_delta`. A matched detection then replaces the predicted state and becomes the newest trajectory observation.

With `--motion_loss niw`, the predictive covariance is converted to world-grid
units and recorded without changing association. Per-epoch values are written to
`motion_uncertainty_epoch_<epoch>.csv`; the file for the best tracking checkpoint
is copied to `motion_uncertainty_best_tracking.csv`. The CSV contains the predicted
position, mean displacement, full symmetric 2D covariance, and whether the motion
prediction matched a detection.

For persistent recurrent inference, add `--mamba_history_mode stateful`. In this mode, `--motion_history_len` still controls the training sample length, while tracking inference carries the Mamba state forward without truncating it to that window.

Use `python main.py --help` for the complete list of options.

## Notes

- Pretrained checkpoints and datasets are not included.
- Training logs, generated visualizations, and test results are written to `logs/`.
- Author and affiliation information is intentionally omitted for anonymous review.
