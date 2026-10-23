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
  --evidential_head separate \
  --mamba_uncertainty_mode unified
```

Replace `wildtrack` with `multiviewx`, `citytrack`, `mvcrowdtrack`, or `gmvd_train` as needed.

With `--multi_task_loss uncertainty`, each active task uses
`exp(-s_task) * loss_task + s_task`, including NIW motion loss. The active tasks
are occupancy detection and motion.
`--multi_task_loss plain` sums their losses with coefficient `1` each.

### Training Mamba with detection histories

`--motion_train_history gt` keeps the existing GT-history training.
`--motion_train_history detection` jointly trains the detector and Mamba using
actual detector observations, with autoregressive motion predictions at missed
observations (`--motion_history_missing motion`, the default). Both models train
from epoch 1, using the initial detector's observations even while its performance
is poor. No trained detector checkpoint or detector-only warm-up stage is required.
`--train_from` optionally initializes weights from a checkpoint with a new
optimizer/scheduler (`--resume` is test-only). When using it, select the same Mamba
architecture/head/projection options to continue its motion weights. Missing and
unused checkpoint keys are printed. `-pr` initializes the ImageNet backbone only.

For example, to train both models on MultiviewX from the beginning:

```bash
python main.py -d multiviewx -pr --data_root /dataset \
  --motion_train_history detection --motion_history_missing motion \
  --multi_task_loss uncertainty --batch_size 1 --epochs 100 \
  --motion_model mamba --motion_normalization bev --motion_history_len 10 \
  --motion_loss niw --evidential_head separate --mamba_history_mode window \
  --mamba_uncertainty_mode separate_full --mamba_uncertainty_gate none \
  --track_match_thresh 50 --track_low_match_thresh 80 \
  --bev_gaussian_sigma 1.5 --aug
```

At the start of **each epoch**, the current detector runs on the training split
to refresh detections and Beta uncertainty. The latest snapshot is stored in
`motion_detections_train.pt` under the new log directory. This adds one detector
inference pass per epoch. Only this snapshot pass uses unaugmented frames to keep
history coordinates consistent. Detector training still uses `--aug` when supplied;
its augmented targets and the independent motion histories keep their respective
coordinate systems. Detector updates within an epoch appear in the next epoch's
snapshot. `--batch_size` still counts image frames. Histories are attached by
sequence/target frame, so image batches can remain shuffled.

`--motion_detection_cache <PATH>` optionally creates or reuses an initial cache
for epoch 1. Reuse checks the current detector parameters/buffers, optional checkpoint
hash, training GT/frame range, camera projections, and decoder settings. Later
epochs refresh the run-local cache; they do not overwrite the external cache.

Detection candidates use the tracker's confidence floor
`max(0, high_cls_thres - 0.1)`. They are matched one-to-one to GT identities using
a metric distance gate: 1 m by default, or 2 m for CityTrack/MVCrowdTrack. Override
it with `--motion_detection_match_thresh` (meters, not the grid-unit tracking
association thresholds). Unmatched detections are not training trajectories.

Detection tokens are `[x, y, dx, dy, u_det, 0, 0, 0, 1, 0, 0]`. When detection is
missing after initialization, the current Mamba predicts the next position and
NIW distribution. The token becomes
`[x, y, dx, dy, 0, cov_xx, cov_yy, cov_xy, 0, 1, kappa]`. Consecutive misses
feed these predictions back; a matched detection replaces the predicted state
at its time and restores the detection token. Covariance is the one-step total
predictive covariance, in grid units squared, as in tracking history. Mamba
applies its existing BEV normalization and optional covariance decomposition.

GT provides identity assignment and the next-position target, never replacement
input coordinates. The displacement target is
`next_GT_xy - last_history_xy`, including when the last history state is a motion
prediction. Detection positions use the BEV cell coordinates directly, without
sub-grid correction. The target frame needs GT but does not need detection.

Training uses bounded trajectory windows. Each window starts from an actual
detection, with the existing repeated-first-state padding before that observation.
Windows with no detection to initialize from, or gaps in the dataset timestamps,
are excluded. A batch without a motion history has no motion loss or motion task
weight update; an entirely empty snapshot does not abort training. This is based
on available samples, not a scheduled warm-up or a detector performance target.
Time steps with missing detections after initialization are filled
by motion, not compressed. `--motion_history_missing skip` retains the
detection-only experiment, excluding incomplete windows instead. Stateful tracking
still uses persistent state at inference; training samples remain bounded windows.

Cached detector observations and autoregressive generation are detached. The
final trajectory loss trains Mamba, including its detection/motion uncertainty
projections and gated blocks; occupancy loss trains the detector. Gradients do
not pass through detection decoding/GT assignment or through the earlier rollout
predictions. Identity assignment uses GT, so this does not simulate tracker ID
switches. No artificial detection dropout is added: a dataset with few misses
will provide few motion tokens.

`motion_training_samples_epoch_<N>.json` records cache hashes, matching settings,
sample counts, and exclusions. The training log records task weights, losses,
detection/motion token counts, mean Beta uncertainty, and mean motion covariance
trace. Per-epoch tracking evaluation and best-checkpoint outputs remain available.
To test explicit gating, use `--mamba_uncertainty_mode none` with
`--mamba_uncertainty_gate b` or `delta`.

## Test

Pass a trained checkpoint through `--resume` to run testing:

```bash
python main.py \
  --dataset wildtrack \
  --data_root <DATA_ROOT> \
  --resume <CHECKPOINT_PATH>
```

Test outputs are saved under `logs/`.

The BEV occupancy head uses a two-parameter Beta evidential prediction at every
cell. Detection confidence is `alpha / (alpha + beta)`, while raw occupancy
uncertainty is `2 / (alpha + beta)` and is retained for each decoded detection.

## Tracking options

The main tracking controls are:

- `--high_cls_thres`: confidence threshold for high-confidence detections
- `--motion_model`: `mamba` or `kalman` (`mamba` is the default)
- `--motion_history_len`: number of trajectory tokens (default: `10`)
- `--mamba_history_mode`: `window` uses the fixed trajectory window; `stateful` keeps each track's Mamba convolution and SSM state across frames
- `--mamba_uncertainty_mode`: `none` uses `[x, y, dx, dy]`; `unified` projects kinematics, source masks, detection uncertainty, and motion covariance together; `separate` adds source-specific uncertainty projections to a shared kinematic projection; `separate_full` independently projects the full detection and motion tokens
- `--mamba_uncertainty_gate`: explicit SSM gating, `none` (default), `b`, or `delta`
- `--mamba_uncertainty_weight`: detection reliability mapping, `linear`, `rational`, or `cosine` (default); also used for motion in `shared` mode
- `--motion_uncertainty_weight`: `shared` (default) preserves the common mapping; `tanh` uses `1 - tanh(trace)` directly for motion
- `--motion_uncertainty_type`: `aleatoric`, `epistemic`, or `total` (default) covariance trace for the explicit motion gate
- `--motion_uncertainty_ref`: positive trace reference for mapping motion uncertainty to `[0, 1)` (initial default: `1e-4`)
- `--motion_cost`: `euclidean` keeps center-distance association; `mahalanobis` uses the NIW covariance for the primary association stage
- `--mahalanobis_gate`: squared Mahalanobis gate (default: `5.991`, the 95% threshold for two dimensions)
- `--motion_cov_scale`: calibration multiplier applied to NIW covariance before association
- `--motion_cov_floor`: minimum covariance eigenvalue used when computing the inverse
- `--motion_normalization`: normalize coordinates and deltas by BEV width/height (`bev`) or keep raw grid units (`none`)
- `--motion_loss`: `niw` predicts a correlated 2D offset distribution; `smooth_l1` keeps the deterministic baseline
- `--evidential_lambda`: NIW evidence regularization coefficient (default: `1e-3`)
- `--evidential_head`: `separate` uses independent mean/evidence MLPs; `joint` uses one MLP with seven outputs

The Mamba predictor estimates the normalized forward displacement. During tracking it is converted back to world-grid units and applied as
`predicted_xy = previous_xy + predicted_delta`. A matched detection then replaces the predicted state and becomes the newest trajectory observation.

With `--motion_loss niw`, the predictive covariance is converted to world-grid
units and recorded without changing association. Per-epoch values are written to
`motion_uncertainty_epoch_<epoch>.csv`; the file for the best tracking checkpoint
is copied to `motion_uncertainty_best_tracking.csv`. The CSV contains the predicted
position, mean displacement, full symmetric 2D covariance, and whether the motion
prediction matched a detection.

The implicit uncertainty token is
`[x, y, dx, dy, u_det, cov_xx, cov_yy, cov_xy, is_det, is_motion]`.
A matched detection stores its Beta occupancy uncertainty and zero motion
covariance. A frame without a matched detection stores the predicted NIW
covariance and zero detection uncertainty. Covariance is normalized by the
squared BEV axis scales before entering Mamba. Short histories retain the
existing padding rule: the first state and source fields are repeated at the
front, while only the repeated `dx, dy` values are set to zero.

With `--motion_train_history gt`, motion samples contain only kinematics. They are converted to
detection-source tokens with zero uncertainty during training. The uncertainty
projections therefore start from zero so that a checkpoint is not perturbed by
untrained uncertainty weights. Training the uncertainty response itself
requires detector/motion-derived historical tokens. Detection-history training
above supplies real Beta uncertainty to the detection inputs and explicit gates.

In `separate_full`, detection tokens use a `5 -> d_model` projection over
`[x, y, dx, dy, u_det]`, while motion tokens use a separate `7 -> d_model`
projection over `[x, y, dx, dy, cov_xx, cov_yy, cov_xy]`. The source masks
select exactly one projection for each time step.

For an association uncertainty ablation, disable implicit uncertainty inputs and
select covariance-aware association:

```bash
python main.py \
  --motion_model mamba \
  --motion_loss niw \
  --mamba_uncertainty_mode none \
  --motion_cost mahalanobis \
  --mahalanobis_gate 5.991
```

Mahalanobis association is applied to the primary tracked/lost track pool. The
unconfirmed-track association remains Euclidean. Consecutive missed frames add
the predicted displacement covariances, and a matched detection resets the
accumulated position covariance. If a valid NIW covariance is unavailable, the
primary association falls back to the Euclidean cost.

For persistent recurrent inference, add `--mamba_history_mode stateful`. In this mode, `--motion_history_len` still controls the training sample length, while tracking inference carries the Mamba state forward without truncating it to that window.

### Explicit uncertainty gates

Both window and stateful inference support three fixed reliability mappings:

```text
linear:   r = 1 - u
rational: r = (1 - u) / (1 + u)
cosine:   r = (1 + cos(pi * u)) / 2

detection: u = Beta occupancy uncertainty
motion:    u = s / (s + motion_uncertainty_ref), s = trace(selected covariance)
```

All three mappings decrease from `r = 1` at `u = 0` to `r = 0` at `u = 1`.
With `--motion_uncertainty_weight shared` (default), the same selected mapping
applies to detection and motion, independently of the `b`/`delta` gate location
and covariance component. All existing mappings remain available.

For NIW parameters `Psi = L @ L.T`, `kappa`, and `nu` in two dimensions:

```text
aleatoric covariance = Psi / (nu - 3)
epistemic covariance = Psi / (kappa * (nu - 3))
total covariance     = aleatoric + epistemic
```

The selected covariance is expressed in the same coordinate system as the
motion model before taking its trace. With BEV normalization, the trace is
`cov_xx / W^2 + cov_yy / H^2`. Without normalization, it is in raw grid units
squared. The initial reference `1e-4` is a tunable starting value, not a
calibrated uncertainty threshold; `s = reference` gives `u = 0.5`, hence
`r = 0.5` for linear/cosine and `r = 1/3` for rational.
Detection uncertainty uses its existing `[0, 1]` scale directly.

For the detection-linear / motion-tanh experiment, use:

```bash
--motion_model mamba --motion_loss niw \
--mamba_uncertainty_mode none --mamba_uncertainty_gate b \
--mamba_uncertainty_weight linear --motion_uncertainty_weight tanh \
--motion_uncertainty_type total
```

This applies `r_det = 1 - u_det` and `r_motion = 1 - tanh(s)` directly.
The tanh mode uses the selected covariance trace in model coordinates (after
BEV normalization when enabled), with no additional scale parameter,
`trace/(trace+ref)` transform, or second weighting function. Consequently,
`--motion_uncertainty_ref` is unused in this mode. If the normalized trace is
small, motion weights remain close to one (e.g. `s = 0.001` gives about `0.999`).
Use `--motion_uncertainty_weight shared` to return to the previous motion
mapping. Detection can still independently select linear, rational, or cosine.

The two gate locations are separate experiments. Here `delta` is the positive
SSM time step after softplus, **not** the predicted spatial displacement:

```text
b:     h_t = exp(delta * A)     * h_(t-1) + r * delta * B * x_t
delta: h_t = exp(r * delta * A) * h_(t-1) + r * delta * B * x_t
```

At `r = 0`, `b` suppresses the new write while keeping the original decay;
`delta` freezes the SSM state. C, the D skip, the output gate, convolution
cache, and residual connection retain their usual behavior. Thus zero
reliability does not freeze the entire block or eliminate the current token
from its output.

A matched detection supplies its own occupancy uncertainty. An unmatched
frame supplies the one-step covariance and kappa of the prediction used for
that frame, not those of the next prediction. This conditional displacement
covariance does not include propagated history uncertainty. The accumulated
position covariance used by Mahalanobis association remains separate. Full
covariance is retained for association and existing CSV output; the trace
reduction is used only for gating. Gate histories append kappa to the existing
10-field token; kappa is not an additional projected input feature.

For the explicit-only experiment, add these options to the dataset's usual
training or test command:

```bash
--motion_model mamba --motion_loss niw \
--mamba_uncertainty_mode none --motion_cost euclidean \
--mamba_uncertainty_gate b \
--mamba_uncertainty_weight cosine \
--motion_uncertainty_type total --motion_uncertainty_ref 1e-4
```

Replace `cosine` with `linear` or `rational` to compare reliability mappings.
Replace `b` with `delta` to change the gate location, and select `aleatoric`,
`epistemic`, or `total` for the covariance ablation. Use
`--mamba_uncertainty_gate none` for the ungated baseline. The implicit modes
can also be combined with the explicit gate, but the CLI default implicit
mode is `unified`, so explicitly set `none` for an isolated gate comparison.

GT-only training histories still have zero uncertainty, giving `r = 1`.
The explicit mapping has no learned parameters and acts directly on real
uncertainties during tracking, but this training setup does not teach Mamba
to adapt to noisy histories or non-unit gates. Evaluate this as a fixed
inference-time gating experiment. Gating adds no checkpoint parameters, so a
checkpoint with otherwise identical model/head settings can be evaluated
with all gate choices using `--resume <checkpoint>`.

Use `python main.py --help` for the complete list of options.

## Notes

- Pretrained checkpoints and datasets are not included.
- Training logs, generated visualizations, and test results are written to `logs/`.
- Author and affiliation information is intentionally omitted for anonymous review.
