# MVMamba

This repository contains the implementation used for the experiments in our anonymous submission. It supports multi-camera detection and tracking on Wildtrack, MultiviewX, CityTrack, MVCrowdTrack, and GMVD.

## Environment

The provided `Dockerfile` uses Python 3.10, PyTorch 2.0.1, torchvision 0.15.2, CUDA 11.8, and Mamba-SSM 1.2.0.post1.

To install the remaining dependencies:

```bash
pip install -r requirements.txt --no-build-isolation
```

### RTX 5090 Docker environment

Use `Dockerfile.rtx5090` for an RTX 5090. It provides Python 3.10, CUDA 12.8.1,
PyTorch 2.7.1, torchvision 0.22.1, Mamba-SSM 2.3.0, and causal-conv1d 1.6.0.
The original `Dockerfile` and `requirements.txt` describe the older environment;
the new image uses `requirements.rtx5090.txt` instead.

The RTX 5090 has compute capability `12.0` (`sm_120`), and PyTorch's Blackwell
support starts with the CUDA 12.8 builds in PyTorch 2.7.
See [NVIDIA's GPU list](https://developer.nvidia.com/cuda/gpus) and the
[PyTorch 2.7 release notes](https://pytorch.org/blog/pytorch-2-7/).
Both CUDA extensions are built from source against the installed PyTorch.
The selected [Mamba build](https://github.com/state-spaces/mamba/blob/v2.3.0/setup.py)
and [causal-conv1d build](https://github.com/Dao-AILab/causal-conv1d/blob/v1.6.0/setup.py)
explicitly include `sm_120`; setting `TORCH_CUDA_ARCH_LIST` alone is insufficient
for older releases that hard-code their compiler targets. The application still
uses the Mamba-1 `Mamba` class, including its existing uncertainty gates.

On the host, install an up-to-date NVIDIA driver that supports the RTX 5090 and
CUDA 12.8.1. NVIDIA lists Linux 570.124.06 / Windows 572.61 as the corresponding
toolkit driver versions; use those or newer drivers
([CUDA release notes](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html)).
Linux hosts also need the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
On Windows, use Docker Desktop's WSL 2 backend with Linux containers
([Docker GPU setup](https://docs.docker.com/desktop/features/gpu/)).
The CUDA toolkit itself is included in the image.

Build from the repository root (the same command works in Bash and PowerShell):

```bash
docker build -f Dockerfile.rtx5090 -t mvmamba:rtx5090 .
```

The first build compiles both CUDA extensions and can take a while. It does not
need GPU access. Compilation defaults to two concurrent jobs; if the compiler
is killed for lack of host RAM, build with `--build-arg MAX_JOBS=1`.
The build checks package dependencies, direct native extension imports, and
Python source syntax without running the application.

With this Mamba/Triton combination, importing `mamba_ssm` initializes Triton's
autotuner and requires an active GPU driver. Ordinary Docker builds have no GPU
access, so the build reads Mamba's version from package metadata. Both
`python main.py --help` and Mamba execution are checked after building, using
`docker run --gpus all`. See the
[Triton autotuner initialization](https://github.com/triton-lang/triton/blob/v3.3.1/python/triton/runtime/autotuner.py).

Ninja is pinned to `1.11.1.4`: `1.11.1.1` has malformed wheel metadata that
causes `pip check` to report "is not supported on this platform" with pip 24.2+.
See the [upstream Ninja issue](https://github.com/scikit-build/ninja-python-distributions/issues/256).
Build tools are checked before compiling the CUDA extensions.

On the RTX 5090 host, verify GPU access and run an actual Mamba forward/backward
pass before training (these commands also work in Bash and PowerShell):

```bash
docker run --rm --gpus all mvmamba:rtx5090 nvidia-smi
docker run --rm --gpus all mvmamba:rtx5090 python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; from mamba_ssm import Mamba; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.__version__, torch.version.cuda); x = torch.randn(2, 16, 32, device='cuda', requires_grad=True); m = Mamba(d_model=32, d_state=16, d_conv=4, expand=2).cuda(); y = m(x); y.square().mean().backward(); torch.cuda.synchronize(); assert torch.isfinite(y).all() and torch.isfinite(x.grad).all(); print('Mamba CUDA forward/backward OK')"
docker run --rm --gpus all mvmamba:rtx5090 python main.py --help
docker run --rm --gpus all mvmamba:rtx5090 python -m unittest discover -s tests -p '*mamba*.py' -v
```

The existing Mamba tests also exercise BiMamba, explicit uncertainty gates, and
stateful inference. Run the forward/backward check first: even importing these
tests requires GPU access with this environment. A successful image build alone
does not validate GPU execution.

For training on Linux / WSL, replace `/absolute/path/to/datasets` with the host
directory containing `Wildtrack/`, `MultiviewX/`, etc.:

```bash
docker run --rm -it --gpus all --shm-size=8g \
  -v "$PWD:/workspace" \
  -v "/absolute/path/to/datasets:/dataset" \
  mvmamba:rtx5090 \
  python main.py --dataset wildtrack --data_root /dataset --pretrained \
  --motion_model mamba --motion_loss niw --evidential_head separate \
  --mamba_uncertainty_mode unified \
  --track_match_thresh 50 --track_low_match_thresh 80
```

PowerShell equivalent, with an example dataset root of `E:\datasets`:

```powershell
docker run --rm -it --gpus all --shm-size=8g `
  -v "${PWD}:/workspace" `
  -v "E:\datasets:/dataset" `
  mvmamba:rtx5090 `
  python main.py --dataset wildtrack --data_root /dataset --pretrained `
  --motion_model mamba --motion_loss niw --evidential_head separate `
  --mamba_uncertainty_mode unified `
  --track_match_thresh 50 --track_low_match_thresh 80
```

The repository mount keeps checkpoints and logs in the host's `logs/` directory
and exposes local GMVD CSV files. Dataset mounts are writable because loaders
can generate GT files. The detector downloads ImageNet weights on first use,
so allow network access for the first training run. To open a shell, omit
`python main.py` and its arguments from the run command.

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
step, and target frames are shuffled. For each target `t`, up to
`--motion_history_len` preceding frames form the history in chronological order.
With `--motion_train_history detection`, every frame in the training dataset is
always decoded using the **epoch-start detector weights**, before the first
training batch. This is a sequential inference pass, one frame at a time in
evaluation mode without gradients, BatchNorm updates, or optimizer steps.
All histories in that epoch therefore use detections from the same detector
snapshot. During training, cache hits skip image loading and detector inference
for past frames; the current target still has its normal training forward.
The cache is recreated at the start of every training epoch and contains only
CPU arrays of world-grid xy, score, and uncertainty, keyed by dataset root/frame;
it retains no images, GPU features, computation graphs, or motion predictions.

There is no separate cache-mode option; remove `--motion_train_cache` from older
commands. Detector improvements during an epoch reach motion histories at the
next epoch's refresh. The current-frame detector and motion model continue to
train each batch. No target or future detection enters the motion history, even
when a later frame has already been cached. Histories stay inside each dataset
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
GT identity matching and all Mamba predictions are recomputed from each window
with the current motion model, including on detector-cache hits.
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
`history_frames` counts frame visits, `history_inferences` counts history detector
forwards during training, and `cache_hits` reports reused visits and their fraction.
Precomputation has a progress bar and logs its frame count and elapsed time.
The epoch summary reports precompute forwards, in-training history forwards,
and their total separately; detection-history training has zero in-training
history forwards. Total epoch time includes precomputation.

Precomputation adds one detector inference per training frame per epoch, in
addition to the current-frame detector forwards used for joint training.
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
- `--implicit_motion_uncertainty_type`: `aleatoric`, `epistemic`, or `total` (default) covariance for the implicit motion projection; applies to `unified`, `separate`, and `separate_full`
- `--mamba_uncertainty_gate`: explicit SSM gating, `none` (default), `b`, or `delta`
- `--mamba_uncertainty_weight`: detection reliability mapping, `linear`, `rational`, or `cosine` (default); motion always uses `1 - tanh(trace)`
- `--motion_uncertainty_type`: `aleatoric`, `epistemic`, or `total` (default) covariance trace for the explicit motion gate
- `--motion_loss`: `niw` predicts a correlated 2D offset distribution; `smooth_l1` keeps the deterministic baseline
- `--niw_reg_weight`: NIW evidence regularization coefficient (default: `1e-3`; finite and non-negative)
- `--evidential_head`: `separate` uses independent mean/evidence MLPs; `joint` uses one MLP with seven outputs

Motion coordinates, displacements, and covariance are always normalized by the
BEV width/height. The Mamba predictor estimates the normalized forward displacement. During tracking it is converted back to world-grid units and applied as
`predicted_xy = previous_xy + predicted_delta`. A matched detection then replaces the predicted state and becomes the newest trajectory observation.

Both Mamba and BiMamba use a fixed Euclidean NIW evidence penalty. With error
`e = target_delta_normalized - predicted_delta_normalized`:

```text
R = norm(e, 2) * (kappa + nu)
loss = Student-t NLL + niw_reg_weight * R
```

To set the evidence penalty weight, add:

```bash
--motion_loss niw --niw_reg_weight 1e-3
```

The `--niw_regularizer` selector has been removed; omit it from older commands.
The Student-t NLL uses the full predictive scale matrix, and predictive
covariance is still learned. The L2 penalty uses BEV-normalized displacement
coordinates. Its scale differs from the former Psi-whitened penalty, so compare
regularization weights on validation data. The default weight is `1e-3`.

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

Choose the implicit covariance component with
`--implicit_motion_uncertainty_type aleatoric|epistemic|total`. For example,
add these options to the training command for an aleatoric-only implicit ablation:

```bash
--motion_loss niw --motion_train_history detection \
--mamba_uncertainty_mode separate_full --mamba_uncertainty_gate none \
--implicit_motion_uncertainty_type aleatoric
```

Histories retain total covariance. For `aleatoric` or `epistemic`, an eleventh
field stores NIW `kappa`, including when the explicit gate is disabled. Immediately
before projection, the covariance entries `(xx, yy, xy)` are multiplied by
`kappa / (kappa + 1)` for aleatoric or `1 / (kappa + 1)` for epistemic.
Kappa itself is not projected. Detection uncertainty is unchanged.
The option applies to training replay, window inference, and stateful inference;
BiMamba uses the same selection in window mode. NIW loss and returned/logged
covariances still use the full predictive distribution. The explicit gate
independently follows `--motion_uncertainty_type`.

For a training ablation, train each component separately and pass the same
component option when evaluating its checkpoint (`--resume`). Checkpoints store
weights, not this selection; the training log records it in `Settings`.
Changing the option only at inference instead measures sensitivity of that
checkpoint to different inputs. Omitting the option preserves total-covariance
behavior and projection dimensions.

GT motion samples contain only kinematics. They are converted to
detection-source tokens with zero uncertainty during training. The uncertainty
projections start from zero so that a checkpoint is not perturbed by untrained
uncertainty weights. GT-only training does not supply nonzero detection or motion
uncertainty; these values enter the history during tracking.
Consequently, switching covariance components does not change GT-only training
inputs. Use detection histories containing motion rollouts to train with the
selected nonzero covariance component.

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

## Resume evaluation and separate visualization

`python main.py ... --resume CHECKPOINT` does **not** automatically generate
motion covariance/trajectory figures. Run `visualize_motion_uncertainty.py`
separately as shown below. Existing resume options for covariance plotting are
opt-in only: `--plot_motion_uncertainty` explicitly enables that older workflow.
Keep the model settings in the main.py resume command consistent with its checkpoint.

Resume creates only the evaluation datasets and loader. Training images and
training splits are not required. For GMVD, `-d gmvd_train --resume ...` reads
only the evaluation rows of `train_datapath.csv`, skipping training rows before
opening their dataset directories. `-d gmvd_test --resume ...` uses evaluation
rows from `test_datapath.csv`. The selected CSV is still required, and sequence
IDs follow its evaluation-row order.

## Standalone motion covariance visualization from a checkpoint

Run the trained detector and motion model directly from the standalone plotting
script. No pre-existing prediction/GT/detection result files are needed:

```bash
python visualize_motion_uncertainty.py \
  --checkpoint logs/RUN/Best_Multiview_Tracking_IDF1_gmvd_train.pth \
  --data_root /dataset --sequence-id 0 --gmvd-eval-all-frames \
  --consecutive-frames 5 \
  --max-objects 3 --component total --formats png pdf \
  --output-dir outputs/motion_from_checkpoint
```

This mode uses the project's CUDA/Mamba environment, including the RTX 5090
Docker environment. It automatically reads the training `Settings` dictionary
from `log.txt` beside the checkpoint. For relocated weights, use
`--model-config /path/to/training/log.txt` or a JSON object containing the same
settings. Explicit options such as `--mamba_d_model 1024`,
`--mamba_uncertainty_mode separate_full`, `--evidential_head joint`, and
`--motion_history_len 13` override saved settings. If no settings file is found,
specify `--dataset` and the model/tracker options used for that checkpoint.
Weights must match the complete detector and motion model: missing, unexpected,
or differently sized parameters cause an error. Backbone pretraining is not
downloaded when loading a checkpoint.

For GMVD, `--sequence-id N` refers to the zero-based evaluation-row order in
`train_datapath.csv` (`gmvd_train`) or `test_datapath.csv` (`gmvd_test`). Use
`--gmvd-csv PATH` to select another split file. Relative sequence paths are
resolved under `--data_root`. Alternatively, select a sequence with
`--sequence scene5/config1/seq1`; its numeric ID is determined automatically.
Add `--list-sequences` to list the actual ID/path mapping without loading images
or model weights. Only the selected sequence is loaded for inference.

`--frames` selects figures, not independent inference samples. The script runs
every sampled evaluation frame from the beginning of that sequence through the
last requested frame, in order, so earlier detections and motion rollouts build
the history. Requested frame numbers must exist in the evaluation split and
its sampling. Instead of specifying frame numbers, use `--consecutive-frames 5`
to automatically select five existing original IDs spaced by exactly one.
`--start-frame N` sets the exact start ID: `--start-frame 380 --consecutive-frames 5`
plots frames `380, 381, 382, 383, 384`. If any requested frame is outside the
evaluation split or absent after sampling, the command reports the missing IDs
instead of moving to a later window. Without an explicit start, one earlier
frame is left for tracking history. Missing annotation frames break an automatic
run. With neither frame-selection option, the entire selected sequence is
processed and plotted.

For GMVD checkpoint inference, `--gmvd-eval-all-frames` (also accepted as
`--gmvd_eval_all_frames`) overrides the evaluation CSV's `sample_require` with
zero for this run. This changes the inference interval to every available
frame without modifying the CSV or running main.py. Source annotations must
still contain consecutive IDs. Saved CSVs cannot recover skipped frames, so
this option requires `--checkpoint`; use existing sampled IDs for CSV-only plots.

The fresh `motion_uncertainty.csv`, `mota_gt.txt`, `all_res.txt`, and resolved
`inference_settings.json` are saved in a new `OUTPUT_DIR/inference/RUN_ID/`
directory. The script then plots those exact results under
`OUTPUT_DIR/seq_N_NAME/`. Motion CSVs include total covariance and NIW `kappa`,
so `--component total`, `aleatoric`, and `epistemic` are all available. The
component controls only the plotted ellipse; the model's implicit covariance
input remains controlled by `--implicit_motion_uncertainty_type`.

By default, all selected frames are combined in one trajectory figure per
sequence and format (`--layout trajectory`). Use `--layout frames` for separate
frame figures. Files are named `trajectory_FIRSTFRAME_LASTFRAME_COMPONENT.png`
(or PDF/SVG). Each object retains one color for motion and GT throughout the
window: motion uses circles/solid lines and GT crosses/dashed lines. Detections
use black hollow squares/dotted lines. Covariance ellipses share the object's
color. Lines break across missing selected observations and GT identity changes.
The points CSV includes
`object_label`/`object_color`, and settings JSON records the color mapping.
`object_color` records the group palette; detections are rendered in black
in both layouts regardless of their group color.
Figures show GT, high-confidence detections, the pure motion prediction before
detection replacement, and a covariance ellipse with semiaxes
`sqrt(eigenvalues)`. They use equal axis units, no background grid, and no
extra covariance multiplier. `--max-objects N` keeps N stable motion IDs and
at most one nearby GT/detection per ID; this proximity selection is not an
identity annotation or the tracker's actual assignment. Object colors are shared
between motion and GT; source markers and line styles remain distinct. When
no object limit is set, unpaired GT retains a separate GT-ID color and unpaired
detections remain black hollow squares, without an invented detection trajectory. Use `--xlim MIN MAX`
and `--ylim MIN MAX` for a closer view without changing covariance values.

Choose which sources to display with `--sources` (default: `gt detection motion`):

```bash
--sources motion                    # motion predictions and covariance ellipses
--sources motion gt                 # motion and GT
--sources motion detection          # motion and detections
--sources motion detection --hide-covariance  # motion/detection points and trajectories only
```

`--hide-covariance` removes the ellipses and covariance legend and fits automatic
axis limits to the displayed points. The exported CSV retains motion covariance
values. Omit the flag to display ellipses as usual.

This works with both layouts, random object selection, and explicit frame
windows. Only the selected sources appear in the figure, legend, automatic
axis limits, and exported points CSV. Object selection and nearby context
matching still use the supplied inputs before the display filter. In CSV mode,
`--gt` and `--detections` can be omitted if those sources are hidden; for example,
`--motion-csv PATH --sources motion` needs only the motion CSV. Checkpoint mode
still runs the detector and tracker normally to obtain motion predictions.

For a random single object on each invocation, append:

```bash
--object-selection random --max-objects 1
```

This draws uniformly from motion IDs present in the selected frames, after any
`--track-ids` filtering. It uses a fresh random seed independent of the model's
fixed `--seed`, so repeating the command draws again. A draw can happen to pick
the same object; a single eligible candidate always gives that object. Within
one run, the chosen ID stays fixed across the plotted frames, with its nearby
GT/detection context. Larger `--max-objects N` values draw N distinct IDs.
To reproduce a draw, pass `--object-selection-seed SEED`; the actual seed and
selected IDs are printed and saved in `plot_settings_COMPONENT.json`. Omitting
that option enables fresh draws again. Saved inference CSVs can be replotted
with this option to browse random objects without rerunning the model.

CSV-only plotting is still supported with just NumPy and Matplotlib. For
example, replot the inference output with another covariance component without
running the checkpoint again:

```bash
python visualize_motion_uncertainty.py \
  --motion-csv outputs/motion_from_checkpoint/inference/RUN_ID/motion_uncertainty.csv \
  --gt outputs/motion_from_checkpoint/inference/RUN_ID/mota_gt.txt \
  --detections outputs/motion_from_checkpoint/inference/RUN_ID/all_res.txt \
  --frames 100 110 --max-objects 3 --component aleatoric
```

## Notes

- Pretrained checkpoints and datasets are not included.
- Training logs, generated visualizations, and test results are written to `logs/`.
- Author and affiliation information is intentionally omitted for anonymous review.
