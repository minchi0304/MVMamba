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
  --mamba_uncertainty_mode unified \
  --track_match_thresh 50 --track_low_match_thresh 80
```

Replace `wildtrack` with `multiviewx`, `citytrack`, `mvcrowdtrack`, or `gmvd_train` as needed.

Camera-view BEV features are always fused with max pooling.

With `--multi_task_loss uncertainty`, each active task uses
`exp(-s_task) * loss_task + s_task`, including NIW motion loss. The active tasks
are occupancy detection and motion.
`--multi_task_loss plain` sums their losses with coefficient `1` each.

### Detector-history joint training

The default `--motion_train_history detection` trains the detector and Mamba
together. `--batch_size 1` still means one **target multiview frame** per optimizer
step, and target frames are shuffled. For each target `t`, the **current** detector
decodes up to `--motion_history_len` preceding frames in chronological order.
These forwards run one frame at a time in evaluation mode without gradients;
they do not update BatchNorm statistics or use a previous-epoch cache. No target
or future detection enters the motion history. Histories stay inside each dataset
root/train split and restart at gaps in the sampled frame sequence.

Past detections are matched one-to-one to GT identities within
`--motion_train_match_thresh` (default `20`, in original world-grid units, separate
from tracking association thresholds). GT positions are never inserted as inputs.
A history starts at a matched detection with confidence at least
`--high_cls_thres`; later observations may use confidence above
`max(0, high_cls_thres - 0.1)`, as in tracking. Missed observations use detached
Mamba predictions autoregressively, including predicted covariance and NIW kappa
when explicit gating is enabled. Histories expire after more than
`--track_buffer` missed frames (default `5`, also used during tracking).
This is supervised identity association, not training on tracker ID switches or
unmatched false positives.

The final motion forward has gradients and predicts
`GT_xy(t) - history_xy(t-1)`. A short history uses repeated-first-state padding;
an object with no detected history contributes no motion loss. The current target
frame supplies the detection loss, and both losses share **one optimizer step**.
There is no gradient through decoding, identity assignment, or autoregressive
history generation; the motion model learns from the resulting input tokens.
Without any valid motion samples, only detection is updated. Logs report sample
counts, observed/predicted token counts before padding, mean input uncertainty,
and mean predictive covariance trace in original world-grid units squared.

This costs up to `motion_history_len` extra detector inferences per target.
Early training can have few motion samples until the detector finds people;
covariance inputs occur only when a detected history contains a subsequent miss.
Use `--motion_train_history gt` for the previous GT-only baseline, whose motion
target remains `next_GT_xy - last_GT_xy` and which requires a complete GT window.

`-pr` initializes the ImageNet backbone. `--resume` loads a trained checkpoint
for evaluation. Image augmentation remains available with `--aug`.

Training uses independent trajectory windows. During tracking, matched detections
update the history and missed detections use autoregressive motion predictions.
This also applies when stateful Mamba tracking is selected.

### Bidirectional Mamba

Select `--motion_model bimamba --mamba_history_mode window` to encode the available
past trajectory in both directions. Each residual block has independent forward
and backward Mamba branches. The backward output is returned to chronological
order, and their average is added to the residual. The final history features
are mean-pooled before the existing Smooth L1 or NIW head. The velocity skip
still uses the most recent chronological token.

The forward/reverse alignment follows the bidirectional scan pattern in
[Vision Mamba](https://github.com/hustvl/Vim/blob/main/mamba-1p1p1/mamba_ssm/modules/mamba_simple.py).
This implementation gives each direction its own full Mamba parameters and uses
the installed kernels; it is not the shared-projection Vim implementation.
`--mamba_layers` counts bidirectional blocks, each containing two Mamba branches.

GT training, BEV normalization, repeated-first-state padding, implicit uncertainty
inputs, explicit B/delta gates, and Euclidean association remain available.
Reliability values are reversed with their tokens on the backward branch. Only
the supplied history is used; the next-frame target is never part of the input.

BiMamba recomputes the window for each prediction and requires `window` mode.
It cannot use the one-token `stateful` cache. Select the same motion model when
training and evaluating its checkpoint; Mamba and BiMamba have different block
parameters. No Docker dependency change is required.

```bash
python main.py -d multiviewx -pr --data_root /dataset --aug \
  --motion_model bimamba --mamba_history_mode window \
  --motion_loss niw --evidential_head separate --multi_task_loss uncertainty \
  --mamba_uncertainty_mode separate_full --mamba_uncertainty_gate none \
  --track_match_thresh 50 --track_low_match_thresh 80 --bev_gaussian_sigma 1.5
```

## Test

Pass a trained checkpoint through `--resume` to run testing:

```bash
python main.py \
  --dataset wildtrack \
  --data_root <DATA_ROOT> \
  --resume <CHECKPOINT_PATH> \
  --track_match_thresh 50 --track_low_match_thresh 80
```

Test outputs are saved under `logs/`.

Training saves two independent best tracking checkpoints in the run directory:

- `Best_Multiview_Tracking_IDF1_<dataset>.pth`: highest IDF1, with MOTA as the tie-breaker.
- `Best_Multiview_Tracking_MOTA_<dataset>.pth`: highest MOTA, with IDF1 as the tie-breaker.

The existing MODA-based detection checkpoint is also saved. The final training
summary reports the selected epoch and metrics for each tracking criterion.

The BEV occupancy head uses a two-parameter Beta evidential prediction at every
cell. Detection confidence is `alpha / (alpha + beta)`, while raw occupancy
uncertainty is `2 / (alpha + beta)` and is retained for each decoded detection.

## Tracking options

Both primary and unconfirmed-track association use Euclidean center distance. The
Wildtrack commands above set the primary and unconfirmed-track thresholds to
`50` and `80`, respectively; the generic CLI defaults remain `75` and `100`.
NIW motion training, uncertainty history inputs, explicit uncertainty gates,
and covariance logging remain available with Euclidean association.

The main tracking controls are:

- `--high_cls_thres`: confidence threshold for high-confidence detections
- `--track_match_thresh`: Euclidean distance threshold for primary association (`50` for the Wildtrack configuration)
- `--track_low_match_thresh`: Euclidean distance threshold for unconfirmed-track association (`80` for the Wildtrack configuration)
- `--motion_model`: `mamba`, `bimamba`, or `kalman` (`mamba` is the default)
- `--motion_history_len`: number of trajectory tokens (default: `10`)
- `--mamba_history_mode`: `window` uses the fixed trajectory window; `stateful` keeps each track's unidirectional Mamba convolution and SSM state across frames. BiMamba supports `window` only.
- `--mamba_uncertainty_mode`: `none` uses `[x, y, dx, dy]`; `unified` projects kinematics, source masks, detection uncertainty, and motion covariance together; `separate` adds source-specific uncertainty projections to a shared kinematic projection; `separate_full` independently projects the full detection and motion tokens
- `--mamba_uncertainty_gate`: explicit SSM gating, `none` (default), `b`, or `delta`
- `--mamba_uncertainty_weight`: detection reliability mapping, `linear`, `rational`, or `cosine` (default); motion always uses `1 - tanh(trace)`
- `--motion_uncertainty_type`: `aleatoric`, `epistemic`, or `total` (default) covariance trace for the explicit motion gate
- `--motion_loss`: `niw` predicts a correlated 2D offset distribution; `smooth_l1` keeps the deterministic baseline
- `--niw_regularizer`: `mahalanobis` (default) uses Psi-whitened error for the NIW evidence penalty; `euclidean` uses BEV-normalized L2 error
- `--niw_reg_weight`: NIW evidence regularization coefficient (default: `1e-3`; finite and non-negative)
- `--evidential_head`: `separate` uses independent mean/evidence MLPs; `joint` uses one MLP with seven outputs

Motion coordinates, displacements, and covariance are always normalized by the
BEV width/height. The Mamba predictor estimates the normalized forward displacement. During tracking it is converted back to world-grid units and applied as
`predicted_xy = previous_xy + predicted_delta`. A matched detection then replaces the predicted state and becomes the newest trajectory observation.

Both Mamba and BiMamba support the same NIW regularizer options. With error
`e = target_delta_normalized - predicted_delta_normalized` and `Psi = L @ L.T`:

```text
mahalanobis: R = sqrt(e.T @ inverse(Psi) @ e + eps) * (kappa + nu)
euclidean:   R = norm(e, 2) * (kappa + nu)
loss = Student-t NLL + niw_reg_weight * R
```

To train with the Euclidean evidence penalty, add:

```bash
--motion_loss niw --niw_regularizer euclidean --niw_reg_weight 1e-3
```

The Student-t NLL and predictive covariance formula are the same for both
regularizers. These options affect NIW training only; tracking association
always uses Euclidean distance. The L2 penalty uses BEV-normalized displacement
coordinates, not world-grid distances. Its scale differs from the Psi-whitened
penalty, so compare regularization weights on validation data. The default
`mahalanobis` and weight `1e-3` preserve the previous training objective.

With `--motion_loss niw`, the predictive covariance is converted to world-grid
units and recorded without changing association. Per-epoch values are written to
`motion_uncertainty_epoch_<epoch>.csv`; the files for the two best tracking checkpoints
are copied to `motion_uncertainty_best_tracking_idf1.csv` and
`motion_uncertainty_best_tracking_mota.csv`. Corresponding tracking predictions
are saved as `mota_pred_best_tracking_idf1.txt` and `mota_pred_best_tracking_mota.txt`.
The CSV contains the predicted
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

GT motion samples contain only kinematics. They are converted to
detection-source tokens with zero uncertainty during training. The uncertainty
projections start from zero so that a checkpoint is not perturbed by untrained
uncertainty weights. GT-only training does not supply nonzero detection or motion
uncertainty; these values enter the history during tracking.

In `separate_full`, detection tokens use a `5 -> d_model` projection over
`[x, y, dx, dy, u_det]`, while motion tokens use a separate `7 -> d_model`
projection over `[x, y, dx, dy, cov_xx, cov_yy, cov_xy]`. The source masks
select exactly one projection for each time step.

For persistent recurrent inference, add `--mamba_history_mode stateful`. In this mode, `--motion_history_len` still controls the training sample length, while tracking inference carries the Mamba state forward without truncating it to that window.

### Explicit uncertainty gates

Both window and stateful inference support three detection reliability mappings:

```text
linear:   r = 1 - u
rational: r = (1 - u) / (1 + u)
cosine:   r = (1 + cos(pi * u)) / 2

detection: u = Beta occupancy uncertainty
motion:    r = 1 - tanh(s), s = trace(selected covariance)
```

All three mappings decrease from `r = 1` at `u = 0` to `r = 0` at `u = 1`.
The selected mapping applies only to detection. Motion always uses `1 - tanh(s)`,
independently of the `b`/`delta` gate location and covariance component.

For NIW parameters `Psi = L @ L.T`, `kappa`, and `nu` in two dimensions:

```text
aleatoric covariance = Psi / (nu - 3)
epistemic covariance = Psi / (kappa * (nu - 3))
total covariance     = aleatoric + epistemic
```

The selected covariance is expressed in the same coordinate system as the
motion model before taking its trace. The BEV-normalized trace is
`cov_xx / W^2 + cov_yy / H^2`.
Detection uncertainty uses its existing `[0, 1]` scale directly.

For the detection-linear / motion-tanh experiment, use:

```bash
--motion_model mamba --motion_loss niw \
--mamba_uncertainty_mode none --mamba_uncertainty_gate b \
--mamba_uncertainty_weight linear \
--motion_uncertainty_type total
```

This applies `r_det = 1 - u_det` and `r_motion = 1 - tanh(s)` directly.
Motion uses the selected covariance trace in model coordinates (after
BEV normalization), with no additional scale parameter or second weighting function.
If the normalized trace is
small, motion weights remain close to one (e.g. `s = 0.001` gives about `0.999`).
Detection can independently select linear, rational, or cosine.

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
covariance does not include propagated history uncertainty. Full covariance is
retained for uncertainty history inputs and existing CSV output; the trace
reduction is used only for gating. Gate histories append kappa to the existing
10-field token; kappa is not an additional projected input feature.

For the explicit-only experiment, add these options to the dataset's usual
training or test command:

```bash
--motion_model mamba --motion_loss niw \
--mamba_uncertainty_mode none \
--mamba_uncertainty_gate b \
--mamba_uncertainty_weight cosine \
--motion_uncertainty_type total
```

Replace `cosine` with `linear` or `rational` to compare detection reliability mappings.
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
