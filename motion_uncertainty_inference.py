"""Checkpoint inference backend for visualize_motion_uncertainty.py.

Heavy model imports are deliberately lazy so CSV plotting and sequence listing
also work on machines without PyTorch, Mamba, or a GPU.
"""

import argparse
import ast
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np


# Same inference defaults as main.py; saved training settings take precedence.
OPTIONS = {
    "dataset": (None, str, ("wildtrack", "multiviewx", "citytrack", "mvcrowdtrack", "gmvd_train", "gmvd_test")),
    "data_root": ("/dataset", str, None),
    "high_cls_thres": (.4, float, None),
    "track_match_thresh": (75., float, None),
    "track_low_match_thresh": (100., float, None),
    "track_buffer": (5, int, None),
    "motion_model": ("mamba", str, ("mamba", "bimamba")),
    "motion_history_len": (10, int, None),
    "mamba_history_mode": ("window", str, ("window", "stateful")),
    "mamba_uncertainty_mode": ("unified", str, ("none", "unified", "separate", "separate_full")),
    "implicit_motion_uncertainty_type": ("total", str, ("total", "aleatoric", "epistemic")),
    "mamba_uncertainty_gate": ("none", str, ("none", "b", "delta")),
    "mamba_uncertainty_weight": ("cosine", str, ("linear", "rational", "cosine")),
    "motion_uncertainty_type": ("total", str, ("total", "aleatoric", "epistemic")),
    "mamba_d_model": (128, int, None),
    "mamba_d_state": (16, int, None),
    "mamba_d_conv": (4, int, None),
    "mamba_expand": (2, int, None),
    "mamba_layers": (2, int, None),
    "mamba_dropout": (0., float, None),
    "motion_loss": ("niw", str, ("niw",)),
    "niw_reg_weight": (.001, float, None),
    "evidential_head": ("separate", str, ("separate", "joint")),
    "bev_gaussian_sigma": (2., float, None),
    "multi_task_loss": ("uncertainty", str, ("plain", "uncertainty")),
    "num_workers": (4, int, None),
    "seed": (1, int, None),
}


def add_inference_arguments(parser):
    group = parser.add_argument_group("Checkpoint inference (settings from log.txt, then explicit overrides)")
    group.add_argument("--model-config", type=Path, help="Training log.txt or JSON settings; default: log.txt beside checkpoint")
    group.add_argument("--gmvd-csv", type=Path, help="GMVD split CSV; default: train_datapath.csv or test_datapath.csv in repository")
    group.add_argument("--gmvd-eval-all-frames", "--gmvd_eval_all_frames", action="store_true",
                       help="Disable GMVD evaluation subsampling for this standalone checkpoint run")
    group.add_argument("--device", default="cuda", help="Inference device, e.g. cuda or cuda:1")
    for name, (_, value_type, choices) in OPTIONS.items():
        flags = ["--" + name]
        if name == "dataset":
            flags.insert(0, "-d")
        group.add_argument(*flags, type=value_type, choices=choices, default=None)


def read_settings(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    if Path(path).suffix.lower() == ".json":
        settings = json.loads(text)
    else:
        lines = text.splitlines()
        settings = None
        for index, line in enumerate(lines):
            if line.strip() == "Settings:":
                value = next((s.strip() for s in lines[index + 1:] if s.strip()), "")
                try:
                    settings = ast.literal_eval(value)
                except (SyntaxError, ValueError) as exc:
                    raise ValueError(f"Cannot read Settings dictionary in {path}") from exc
                break
    if not isinstance(settings, dict):
        raise ValueError(f"{path}: expected a JSON object or a log with a Settings dictionary")
    return settings


def resolve_settings(args):
    checkpoint = args.checkpoint.expanduser()
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")
    config = args.model_config
    if config is None and (checkpoint.parent / "log.txt").is_file():
        config = checkpoint.parent / "log.txt"
    saved = read_settings(config) if config is not None else {}
    values = {}
    for name, (default, value_type, choices) in OPTIONS.items():
        override = getattr(args, name)
        value = override if override is not None else saved.get(name, default)
        if value is None:
            raise ValueError(f"Specify --{name}; no value was found in the training settings")
        try:
            values[name] = value_type(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid {name} in model settings: {value!r}") from exc
        if choices and values[name] not in choices:
            raise ValueError(f"{name} must be one of {choices}; got {value!r}")
        if value_type is float and not np.isfinite(values[name]):
            raise ValueError(f"{name} must be finite")
    positive = ("motion_history_len", "mamba_d_model", "mamba_d_state", "mamba_d_conv",
                "mamba_expand", "mamba_layers", "bev_gaussian_sigma", "track_match_thresh", "track_low_match_thresh")
    if any(values[key] <= 0 for key in positive):
        raise ValueError(f"These settings must be positive: {', '.join(positive)}")
    if values["track_buffer"] < 0 or values["num_workers"] < 0 or values["niw_reg_weight"] < 0:
        raise ValueError("track_buffer, num_workers and niw_reg_weight must be nonnegative")
    if not 0 <= values["high_cls_thres"] <= 1 or not 0 <= values["mamba_dropout"] < 1:
        raise ValueError("high_cls_thres must be in [0, 1] and mamba_dropout in [0, 1)")
    if values["motion_model"] == "bimamba" and values["mamba_history_mode"] != "window":
        raise ValueError("BiMamba requires mamba_history_mode=window")
    print(f"Model settings: {config if config is not None else 'CLI/defaults (no training log found)'}", flush=True)
    return argparse.Namespace(**values), config


def sequence_specs(settings, gmvd_csv=None, all_frames=False):
    root = Path(settings.data_root).expanduser()
    if all_frames and not settings.dataset.startswith("gmvd"):
        raise ValueError("--gmvd-eval-all-frames requires gmvd_train or gmvd_test")
    if settings.dataset.startswith("gmvd"):
        split = gmvd_csv or Path(__file__).parent / ("train_datapath.csv" if settings.dataset == "gmvd_train" else "test_datapath.csv")
        specs = []
        with Path(split).open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.reader(stream):
                if not row or row[0].strip().startswith("#") or row[0].strip() == "train":
                    continue
                if len(row) != 4:
                    raise ValueError(f"{split}: expected split,path,train_ratio,sample_require")
                relative = Path(row[1].strip())
                path = relative if relative.is_absolute() else root / relative
                ratio, samples = float(row[2]), int(row[3])
                if not np.isfinite(ratio) or not 0 <= ratio < 1 or samples < 0:
                    raise ValueError(f"{split}: invalid evaluation split settings for {path}")
                specs.append(dict(sequence_id=len(specs), root=str(path), train_ratio=ratio,
                                  sample_require=0 if all_frames else samples,
                                  kind="wildtrack" if path.name == "Wildtrack" else "multiviewx"))
        if not specs:
            raise ValueError(f"{split}: no evaluation sequences")
        return specs
    folder = dict(wildtrack="Wildtrack", multiviewx="MultiviewX", citytrack="CityTrack", mvcrowdtrack="MVCrowdTrack")
    return [dict(sequence_id=0, root=str(root / folder[settings.dataset]), train_ratio=.9,
                 sample_require=0, kind=settings.dataset)]


def choose_sequence(specs, name=None, sequence_id=None):
    candidates = specs
    if name is not None:
        normalized = name.replace("\\", "/").rstrip("/")
        candidates = [s for s in specs if s["root"].replace("\\", "/").rstrip("/") == normalized]
        if not candidates:
            candidates = [s for s in specs if s["root"].replace("\\", "/").endswith("/" + normalized.removeprefix("./"))]
        if len(candidates) != 1:
            raise ValueError("--sequence must identify one sequence; use its full path or scene/config/seq path")
    if sequence_id is not None:
        candidates = [s for s in candidates if s["sequence_id"] == sequence_id]
    if len(candidates) != 1:
        raise ValueError("Choose one matching --sequence or --sequence-id; inspect --list-sequences")
    return candidates[0]


def frame_inventory(frames):
    if not frames:
        return "Available evaluation frames: none"
    steps = sorted({right - left for left, right in zip(frames, frames[1:])})
    preview = str(frames) if len(frames) <= 12 else f"{frames[:6]} ... {frames[-6:]}"
    return (f"Available evaluation frames: count={len(frames)}, range={frames[0]}..{frames[-1]}, "
            f"steps={steps[:8]}, IDs={preview}")


def select_plot_frames(available, requested=None, consecutive=None, start_frame=None):
    available = sorted({int(frame) for frame in available})
    if consecutive is not None:
        if consecutive <= 0:
            raise ValueError("--consecutive-frames must be positive")
        if requested is not None:
            raise ValueError("Use --frames or --consecutive-frames, not both")
    elif start_frame is not None:
        raise ValueError("--start-frame requires --consecutive-frames")
    if not available:
        raise ValueError("The selected evaluation sequence has no frames")
    if consecutive is not None:
        # Retain one earlier observation by default; explicit lower bounds can
        # include the first frame, which has no motion prediction yet.
        lower = start_frame if start_frame is not None else (available[1] if len(available) > 1 else None)
        run = []
        if lower is not None:
            for frame in available:
                if frame < lower:
                    continue
                if run and frame != run[-1] + 1:
                    run = []
                run.append(frame)
                if len(run) == consecutive:
                    return run
        raise ValueError(f"No run of {consecutive} consecutive frame IDs with the requested start/history condition. "
                         f"{frame_inventory(available)}. For GMVD checkpoint inference, "
                         "--gmvd-eval-all-frames disables subsampling; consecutive source annotations are still required.")
    selected = available if requested is None else sorted(set(requested))
    missing = set(selected) - set(available)
    if not selected or missing:
        raise ValueError(f"Requested frames are outside this evaluation split/sampling: {sorted(missing)}. "
                         f"{frame_inventory(available)}. Use --consecutive-frames 5 to automatically select existing frames.")
    return selected


def inference_frames(available, requested=None, consecutive=None, start_frame=None):
    available = sorted({int(frame) for frame in available})
    selected = select_plot_frames(available, requested, consecutive, start_frame)
    # Include every previous evaluation frame, even for a sparse plot request.
    return [frame for frame in available if frame <= selected[-1]], selected


def build_dataset(settings, spec):
    import torchvision.transforms as T
    from dataloader.dataloader import GetDataset
    from dataloader.concat_dataset import ConcatDataset
    from dataloader.wildtrack import Wildtrack
    from dataloader.multiviewx import MultiviewX
    from dataloader.citytrack import CityTrack
    from dataloader.mvcrowdtrack import MVCrowdTrack

    classes = dict(wildtrack=Wildtrack, multiviewx=MultiviewX, citytrack=CityTrack, mvcrowdtrack=MVCrowdTrack)
    base = classes[spec["kind"]](spec["root"])
    if spec["sample_require"] > base.num_frames:
        raise ValueError("GMVD sample_require exceeds the sequence frame count (would produce a zero frame step)")
    transform = T.Compose([T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])
    dataset = GetDataset(base, train=False, transform=transform, data_aug=False,
                         grid_reduce=int(getattr(base, "default_grid_reduce", 4)),
                         img_reduce=int(getattr(base, "default_img_reduce", 8)),
                         train_ratio=spec["train_ratio"], sample_require=spec["sample_require"],
                         bev_gaussian_sigma=settings.bev_gaussian_sigma)
    return ConcatDataset(dataset)


def load_checkpoint(model, checkpoint):
    import torch

    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(weights, dict):
        for key in ("state_dict", "model_state_dict"):
            if isinstance(weights.get(key), dict):
                weights = weights[key]
                break
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Checkpoint must contain a model state_dict")
    if all(key.startswith("module.") for key in weights):
        weights = {key[len("module."):]: value for key, value in weights.items()}
    try:
        model.load_state_dict(weights, strict=True)
    except RuntimeError as exc:
        raise ValueError("Checkpoint does not match the model settings. Use its training log with "
                         "--model-config, or supply the matching architecture options.\n" + str(exc)) from exc


def build_model(settings, checkpoint, device, output):
    from detector.encoder import resnet18
    from detector.multiview_model import MultiView_Detection
    from tracker.mamba_motion_predictor import MambaMotionPredictor
    from tracker.bimamba_motion_predictor import BiMambaMotionPredictor

    backbone = resnet18(pretrained=False, replace_stride_with_dilation=[False, True, True])
    model = MultiView_Detection(backbone, str(output), device=device,
                               multi_task_loss=settings.multi_task_loss, use_motion_loss=True)
    predictor = BiMambaMotionPredictor if settings.motion_model == "bimamba" else MambaMotionPredictor
    model.motion_predictor = predictor(
        history_len=settings.motion_history_len, d_model=settings.mamba_d_model,
        d_state=settings.mamba_d_state, d_conv=settings.mamba_d_conv,
        expand=settings.mamba_expand, num_layers=settings.mamba_layers, dropout=settings.mamba_dropout,
        history_mode=settings.mamba_history_mode, loss_type=settings.motion_loss,
        niw_reg_weight=settings.niw_reg_weight, evidential_head=settings.evidential_head,
        uncertainty_mode=settings.mamba_uncertainty_mode,
        implicit_motion_uncertainty_type=settings.implicit_motion_uncertainty_type,
        uncertainty_gate=settings.mamba_uncertainty_gate, uncertainty_weight=settings.mamba_uncertainty_weight,
        motion_uncertainty_type=settings.motion_uncertainty_type,
    )
    load_checkpoint(model, checkpoint)
    return model.to(device).eval()


def build_logging_tracker(settings, dataset_config, device, predictor):
    from tracker.multitracker import Tracker

    class LoggingTracker(Tracker):
        def _make_detections(self, *args, **kwargs):
            detections = super()._make_detections(*args, **kwargs)
            for track in detections:
                # Request the already-computed NIW kappa for export only. Token
                # width and the configured implicit/explicit inputs stay intact.
                track.requires_motion_kappa = True
            return detections

        def update(self, *args, **kwargs):
            previous = list(self.tracked_stracks) + list(self.lost_stracks)
            result = super().update(*args, **kwargs)
            by_id = {track.track_id: track for track in previous}
            for row in self.last_motion_predictions:
                kappa = by_id[row["track_id"]].motion_kappa
                if kappa is None or not np.isfinite(kappa) or kappa <= 0:
                    raise ValueError("NIW predictor did not provide positive kappa for covariance export")
                row["kappa"] = float(kappa)
            return result

    return LoggingTracker(
        high_cls_thres=settings.high_cls_thres, match_thresh=settings.track_match_thresh,
        unconfirmed_match_thresh=settings.track_low_match_thresh, track_buffer=settings.track_buffer,
        motion_model=settings.motion_model, bev_grid_shape=tuple(dataset_config["base"].base.world_grid_shape),
        device=device, motion_predictor=predictor, motion_history_len=settings.motion_history_len,
        mamba_history_mode=settings.mamba_history_mode, mamba_uncertainty_mode=settings.mamba_uncertainty_mode,
        mamba_uncertainty_gate=settings.mamba_uncertainty_gate,
        implicit_motion_uncertainty_type=settings.implicit_motion_uncertainty_type,
    )


def infer_sequence(model, dataset, settings, spec, device, requested_frames=None):
    import torch
    from utils import utils

    sub_dataset = dataset.datasets[0]
    processed, selected = inference_frames(sub_dataset.frames, requested_frames)
    root = dataset.dataset_list[0]
    config = dataset.dicts[root]
    tracker = build_logging_tracker(settings, config, device, model.motion_predictor)
    # A sampler restricts decoding to the last requested frame, keeping the real
    # ConcatDataset metadata needed by the shared evaluation path.
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                        sampler=range(len(processed)), num_workers=settings.num_workers,
                                        pin_memory=device.type == "cuda")
    tracking, gt, uncertainty = defaultdict(list), defaultdict(list), defaultdict(list)
    detection_rows, scales = [], {}
    print(f"Inferring {root}: {len(processed)} frames from {processed[0]} to {processed[-1]}; "
          f"plotting {len(selected)} frame(s)", flush=True)
    with torch.no_grad():
        for index, (item, _) in enumerate(loader):
            frame = int(torch.as_tensor(item["frame"]).reshape(-1)[0].item())
            if frame != processed[index]:
                raise ValueError("Inference loader must preserve chronological evaluation frame order")
            decoded = utils.decode_frame(model, dataset, item, index, device, settings)
            rows, _ = utils.update_tracking_for_frame(
                loader, item, decoded, root, frame, spec["sequence_id"], {root: tracker},
                tracking, gt, uncertainty, scales, settings, motion_predictor=model.motion_predictor,
            )
            detection_rows.append(rows.detach().cpu().numpy())
            if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(processed):
                print(f"Inference {index + 1}/{len(processed)} (frame {frame})", flush=True)
    detections = np.concatenate(detection_rows, axis=0).reshape(-1, 5)
    gt_rows = [[spec["sequence_id"], frame, pid, -1, -1, -1, -1, 1, x, y, -1]
               for frame, pid, x, y in gt[root]]
    return dict(motion=[dict(sequence=root, **row) for row in uncertainty[root]],
                gt=np.asarray(gt_rows, dtype=float).reshape(-1, 11), detections=detections,
                processed_frames=processed, selected_frames=selected, sequence=root,
                detection_order="xy" if sub_dataset.base.indexing == "xy" else "yx")


def write_inference(output, result, metadata):
    output.mkdir(parents=True, exist_ok=True)
    paths = dict(motion_csv=output / "motion_uncertainty.csv", gt=output / "mota_gt.txt",
                 detections=output / "all_res.txt")
    fields = ["sequence", "frame", "track_id", "matched", "predicted_x", "predicted_y",
              "mean_dx", "mean_dy", "cov_xx", "cov_xy", "cov_yy", "kappa"]
    with paths["motion_csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["motion"])
    np.savetxt(paths["gt"], result["gt"], delimiter=",", fmt="%.10g")
    np.savetxt(paths["detections"], result["detections"], fmt="%.10g")
    metadata.update({key: result[key] for key in ("processed_frames", "selected_frames", "sequence", "detection_order")})
    (output / "inference_settings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths


def run_checkpoint(args):
    settings, config = resolve_settings(args)
    specs = sequence_specs(settings, args.gmvd_csv, all_frames=args.gmvd_eval_all_frames)
    if args.list_sequences:
        for spec in specs:
            print(f"{spec['sequence_id']}: {spec['root']}")
        return False
    spec = choose_sequence(specs, args.sequence, args.sequence_id)
    import torch

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Checkpoint inference with Mamba requires CUDA. Run in the GPU environment used for evaluation.")
    torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    np.random.seed(settings.seed)
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset = build_dataset(settings, spec)
    _, selected = inference_frames(dataset.datasets[0].frames, args.frames, args.consecutive_frames, args.start_frame)
    print(f"Selected plot frames: {selected if len(selected) <= 12 else str(len(selected)) + ' frames'}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output_dir / "inference" / stamp
    model = build_model(settings, args.checkpoint.expanduser(), device, output)
    print(f"Loaded checkpoint: {args.checkpoint} (strict state_dict match)", flush=True)
    result = infer_sequence(model, dataset, settings, spec, device, selected)
    paths = write_inference(output, result, dict(
        checkpoint=str(args.checkpoint.expanduser().resolve()), model_config=str(config) if config else None,
        model_settings=vars(settings), sequence_id=spec["sequence_id"], split=spec,
        gmvd_csv=str(args.gmvd_csv) if args.gmvd_csv else None,
        gmvd_eval_all_frames=args.gmvd_eval_all_frames,
    ))
    for key, value in paths.items():
        setattr(args, key, value)
    args.sequence, args.sequence_id = result["sequence"], spec["sequence_id"]
    args.frames = result["selected_frames"]
    if args.detection_order == "auto":
        args.detection_order = result["detection_order"]
    print(f"Saved inference inputs to {output}", flush=True)
    return True
