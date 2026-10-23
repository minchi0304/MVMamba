"""Stage 1: train the Beta EDL detector without a motion model."""

import time

from main import run


def train_detector(model, epoch, data_loader, optimizer, log_interval, *, dataset, device, criterion, args):
    model.train()
    start = time.perf_counter()
    loss_sum, frames = 0.0, 0
    for batch_index, (item, target) in enumerate(data_loader, 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(dataset, item['root'], item['img'].to(device), item.get('proj_mats'))
        loss = criterion(prediction, target['valid_bev'].to(device).float(), target['center_bev'].to(device))
        loss.backward()
        optimizer.step()
        count = item['img'].size(0)
        loss_sum += float(loss.detach()) * count
        frames += count
        if batch_index % log_interval == 0 or batch_index == len(data_loader):
            print('Detector Epoch: {}, Batch:{}/{}, occ_loss: {:.6f}, Time: {:.3f} (min)'.format(
                epoch, batch_index, len(data_loader), float(loss.detach()), (time.perf_counter() - start) / 60))
    return loss_sum / frames


if __name__ == '__main__':
    run(stage='detector', train_epoch=train_detector)
