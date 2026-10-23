"""Stage 2: train Mamba on fixed detector observations and current-model rollouts."""

import time

from main import run
from utils.motion_training_data import motion_inputs


def train_motion(model, epoch, data_loader, optimizer, log_interval, *, dataset, device, criterion, args):
    # Keep detector BatchNorm/dropout and all detector parameters frozen.
    model.eval()
    model.motion_predictor.train()
    start = time.perf_counter()
    loss_sum, samples = 0.0, 0
    detection_tokens = motion_tokens = 0
    uncertainty_sum = trace_sum = 0.0
    for batch_index, item in enumerate(data_loader, 1):
        optimizer.zero_grad(set_to_none=True)
        history, target, shape = motion_inputs(model, item, device)
        loss = model.motion_predictor(history, target, bev_grid_shape=shape)
        loss.backward()
        optimizer.step()
        count = history.size(0)
        loss_sum += float(loss.detach()) * count
        samples += count
        is_det, is_motion = history[..., 8].bool(), history[..., 9].bool()
        detection_tokens += int(is_det.sum())
        motion_tokens += int(is_motion.sum())
        uncertainty_sum += float(history[..., 4][is_det].sum())
        trace_sum += float((history[..., 5] + history[..., 6])[is_motion].sum())
        if batch_index % log_interval == 0 or batch_index == len(data_loader):
            print('Motion Epoch: {}, Batch:{}/{}, motion_loss: {:.6f}, stats: {}, Time: {:.3f} (min)'.format(
                epoch, batch_index, len(data_loader), float(loss.detach()),
                model.motion_predictor.last_loss_stats, (time.perf_counter() - start) / 60))
    print('Epoch {} training history: {}'.format(epoch, {
        'samples': samples, 'detection_tokens': detection_tokens, 'motion_tokens': motion_tokens,
        'mean_detection_uncertainty': uncertainty_sum / max(detection_tokens, 1),
        'mean_motion_covariance_trace_grid2': trace_sum / max(motion_tokens, 1),
    }))
    return loss_sum / samples


if __name__ == '__main__':
    run(stage='motion', train_epoch=train_motion)
