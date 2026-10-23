"""Command-line options for joint, detector-only, and motion-only training."""

import argparse
import math


_STAGE_DESCRIPTIONS = {
    'joint': 'GT-history joint training and evaluation',
    'detector': 'Train the Beta EDL detector',
    'motion': 'Train Mamba using a frozen detector',
}


def _add_data_arguments(parser):
    group = parser.add_argument_group('Dataset and loading')
    group.add_argument('-d', '--dataset', type=str, default='wildtrack',
                       choices=['wildtrack', 'multiviewx', 'citytrack', 'mvcrowdtrack', 'gmvd_test', 'gmvd_train'],
                       help='Dataset to use')
    group.add_argument('--data_root', type=str, default='/dataset/Multiview_datasets',
                       help='Root containing dataset folders')
    group.add_argument('-j', '--num_workers', type=int, default=4)


def _add_training_arguments(parser, batch_size):
    group = parser.add_argument_group('Training and logging')
    group.add_argument('-b', '--batch_size', type=int, default=batch_size, metavar='N',
                       help='Frames per batch (joint/detector) or trajectories per batch (motion)')
    group.add_argument('--epochs', type=int, default=120, metavar='N', help='Number of training epochs')
    group.add_argument('--optim', type=str, default='Adam', choices=['SGD', 'Adam'], help='Optimizer')
    group.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='Initial learning rate')
    group.add_argument('--min_lr', type=float, default=1e-6, metavar='min LR',
                       help='Minimum learning rate for the cosine scheduler')
    group.add_argument('--weight_decay', type=float, default=5e-4)
    group.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum')
    group.add_argument('--seed', type=int, default=1, help='Random seed')
    group.add_argument('--log_interval', type=int, default=200, metavar='N',
                       help='Number of batches between training status messages')
    group.add_argument('--output_dir', default=None,
                       help='Run output directory; creates a timestamped logs directory when omitted')


def _add_detector_arguments(parser):
    group = parser.add_argument_group('Detector')
    group.add_argument('--high_cls_thres', type=float, default=0.4,
                       help='Confidence threshold for high-confidence detections')
    group.add_argument('--bev_gaussian_sigma', type=float, default=2.0,
                       help='Gaussian sigma used to generate BEV occupancy targets')


def _add_image_training_arguments(parser):
    group = parser.add_argument_group('Image training and initialization')
    group.add_argument('-pr', '--pretrained', default=False, action='store_true',
                       help='Initialize the image backbone with ImageNet weights')
    group.add_argument('--aug', action='store_true', help='Use train-time random resize/crop augmentation')


def _add_motion_arguments(parser):
    group = parser.add_argument_group('Motion history and loss')
    group.add_argument('--motion_history_len', type=int, default=10,
                       help='Number of trajectory tokens used by learned motion models')
    group.add_argument('--mamba_history_mode', type=str, default='window', choices=['window', 'stateful'],
                       help='Use a fixed trajectory window or persistent per-track Mamba state')
    group.add_argument('--motion_loss', type=str, default='niw', choices=['smooth_l1', 'niw'],
                       help='Motion prediction loss and output head')
    group.add_argument('--evidential_lambda', type=float, default=1e-3,
                       help='Evidence regularization coefficient for NIW motion loss')
    group.add_argument('--evidential_head', type=str, default='separate', choices=['separate', 'joint'],
                       help='Use separate mean/evidence MLPs or one joint NIW MLP')


def _add_mamba_arguments(parser):
    group = parser.add_argument_group('Mamba architecture')
    group.add_argument('--mamba_d_model', type=int, default=128)
    group.add_argument('--mamba_d_state', type=int, default=16)
    group.add_argument('--mamba_d_conv', type=int, default=4)
    group.add_argument('--mamba_expand', type=int, default=2)
    group.add_argument('--mamba_layers', type=int, default=2)
    group.add_argument('--mamba_dropout', type=float, default=0.0)


def _add_uncertainty_arguments(parser):
    group = parser.add_argument_group('Implicit uncertainty input')
    group.add_argument('--mamba_uncertainty_mode', type=str, default='unified',
                       choices=['none', 'unified', 'separate', 'separate_full'],
                       help='Disabled, unified, uncertainty-only branches, or fully source-specific projections')

    group = parser.add_argument_group('Explicit uncertainty gating')
    group.add_argument('--mamba_uncertainty_gate', type=str, default='none', choices=['none', 'b', 'delta'],
                       help='Apply a reliability gate to SSM B or positive delta')
    group.add_argument('--mamba_uncertainty_weight', type=str, default='cosine',
                       choices=['linear', 'rational', 'cosine'],
                       help='Detection reliability mapping; also used for motion with motion_uncertainty_weight=shared')
    group.add_argument('--motion_uncertainty_weight', type=str, default='shared', choices=['shared', 'tanh'],
                       help='Shared mapping after trace/(trace+ref), or directly 1-tanh(trace)')
    group.add_argument('--motion_uncertainty_type', type=str, default='total',
                       choices=['aleatoric', 'epistemic', 'total'],
                       help='NIW covariance component whose trace drives the explicit motion gate')
    group.add_argument('--motion_uncertainty_ref', type=float, default=1e-4,
                       help='Positive trace reference for trace/(trace+ref), in model coordinate units squared')


def _add_tracking_arguments(parser):
    group = parser.add_argument_group('Tracking association')
    group.add_argument('--track_match_thresh', type=float, default=75,
                       help='MVTrackTrans-style center-distance association threshold')
    group.add_argument('--track_low_match_thresh', type=float, default=100,
                       help='MVTrackTrans-style center-distance gate for unconfirmed tracks')
    group.add_argument('--motion_cost', type=str, default='euclidean', choices=['euclidean', 'mahalanobis'],
                       help='Primary track-to-detection motion cost')
    group.add_argument('--mahalanobis_gate', type=float, default=5.991,
                       help='Squared Mahalanobis gate for primary association')
    group.add_argument('--motion_cov_scale', type=float, default=1.0,
                       help='Scale applied to NIW motion covariance during association')
    group.add_argument('--motion_cov_floor', type=float, default=1e-4,
                       help='Minimum covariance eigenvalue during Mahalanobis association')


def _add_joint_arguments(parser):
    group = parser.add_argument_group('Joint training')
    group.add_argument('--motion_model', type=str, default='mamba', choices=['kalman', 'mamba'],
                       help='Motion prediction model used by the tracker')
    group.add_argument('--multi_task_loss', type=str, default='uncertainty', choices=['plain', 'uncertainty'],
                       help='Plain summed loss or learned uncertainty weights for each active task')

    group = parser.add_argument_group('Checkpoint evaluation')
    group.add_argument('--resume', type=str, default=None, help='Evaluate a checkpoint without training')
    group.add_argument('--visualize_all_frames', action='store_true',
                       help='Save per-frame map images under maps_{epoch} during test')
    group.add_argument('--plot_tracking_videos', action='store_true',
                       help='Also save tracking trajectory and camera videos during test')


def _add_motion_training_arguments(parser):
    group = parser.add_argument_group('Frozen detector and motion training')
    group.add_argument('--detector_checkpoint', required=True,
                       help='Trained detector or combined checkpoint; detector remains frozen')
    group.add_argument('--motion_detection_cache', type=str, default=None,
                       help='Create or reuse detections from the frozen detector once per run')
    group.add_argument('--motion_history_missing', choices=['motion', 'skip'], default='motion',
                       help='Fill history misses with current Mamba predictions or exclude incomplete windows')


def create_parser(stage='joint'):
    parser = argparse.ArgumentParser(
        description=_STAGE_DESCRIPTIONS[stage],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_data_arguments(parser)
    _add_training_arguments(parser, batch_size=256 if stage == 'motion' else 1)
    _add_detector_arguments(parser)

    if stage != 'motion':
        _add_image_training_arguments(parser)
    if stage != 'detector':
        _add_motion_arguments(parser)
        _add_mamba_arguments(parser)
        _add_uncertainty_arguments(parser)
        _add_tracking_arguments(parser)
    if stage == 'joint':
        _add_joint_arguments(parser)
    elif stage == 'motion':
        _add_motion_training_arguments(parser)

    # The shared runner also reads these fields in stages that do not expose them.
    parser.set_defaults(
        motion_model='kalman' if stage == 'detector' else 'mamba',
        multi_task_loss='uncertainty' if stage == 'joint' else 'plain',
        motion_history_len=10,
        pretrained=False,
        aug=False,
        resume=None,
        mamba_uncertainty_gate='none',
        visualize_all_frames=False,
        plot_tracking_videos=False,
    )
    return parser


def parse_args(stage='joint', argv=None):
    parser = create_parser(stage)
    args = parser.parse_args(argv)
    if stage != 'joint' and args.dataset == 'gmvd_test':
        parser.error('Use gmvd_train for training; gmvd_test contains evaluation splits')
    if args.mamba_uncertainty_gate != 'none':
        if args.motion_model != 'mamba' or args.motion_loss != 'niw':
            parser.error('explicit Mamba uncertainty gating requires --motion_model mamba --motion_loss niw')
        if args.motion_uncertainty_weight == 'shared' and (
            not math.isfinite(args.motion_uncertainty_ref) or args.motion_uncertainty_ref <= 0
        ):
            parser.error('--motion_uncertainty_ref must be finite and positive')
    return args
