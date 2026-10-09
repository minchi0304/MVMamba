import os
import time
import csv
import numpy as np
import random
import argparse
import sys
import datetime
import shutil
import tqdm
import torch
import torch.optim as optim
import torchvision.transforms as T
from collections import defaultdict
from dataloader.wildtrack import Wildtrack
from dataloader.multiviewx import MultiviewX
from dataloader.citytrack import CityTrack
from dataloader.mvcrowdtrack import MVCrowdTrack
from dataloader.dataloader import GetDataset, motion_joint_collate
from dataloader.concat_dataset import ConcatDataset
from utils.logger import Logger
from utils import utils
from utils.loss import CornerNetFocalLoss
from detector.multiview_model import MultiView_Detection
from detector.encoder import resnet18
from tracker.mamba_motion_predictor import MambaMotionPredictor
    
###################################################################################################################################
score = []
best_detection_key, best_detection_idx = None, 0
best_tracking_key, best_tracking_idx = None, 0


def init_fn(worker_id):
    np.random.seed(int(args.seed))    


def print_bev_gaussian_settings(name, dataset):
    print("{} BEV Gaussian settings:".format(name))
    for idx, sub_dataset in enumerate(dataset.datasets):
        sigma = float(getattr(sub_dataset, 'bev_gaussian_sigma', 1.5))
        radius = int(3 * sigma)
        kernel_size = 2 * radius + 1
        dataset_name = getattr(sub_dataset, 'dataset_name', os.path.basename(sub_dataset.root))
        print(
            "  [{}] {}: sigma {:.3f}, radius {}, kernel {}x{}".format(
                idx,
                dataset_name,
                sigma,
                radius,
                kernel_size,
                kernel_size,
            )
        )


def masked_offset_l1_loss(pred, target, valid_mask):
    if target.dim() == 3:
        target = target.unsqueeze(0)
    if valid_mask.dim() == 3:
        valid_mask = valid_mask.unsqueeze(1)

    pred = pred[:, :2]
    target = target[:, :2].to(device=pred.device, dtype=pred.dtype)
    mask = valid_mask.to(device=pred.device, dtype=pred.dtype)
    if mask.size(1) == 2:
        mask = mask.min(dim=1, keepdim=True).values
    elif mask.size(1) != 1:
        raise ValueError("valid_mask channel dimension must be 1 or 2")

    loss_map = torch.abs(pred - target).sum(dim=1, keepdim=True)
    return (loss_map * mask).sum() / mask.sum().clamp_min(1.0)


def train(
    model,
    epoch,
    data_loader,
    optimizer,
    log_interval,
):
    model.train()
    tic = time.time()

    losses = 0

    for batch_idx, (item, target) in enumerate(data_loader):
        optimizer.zero_grad()
        data = item['img'].to(device)
                
        bev_occupancy, bev_offset = model(train_dataset, item['root'], data, item.get('proj_mats'))
        bev_center = utils.get_occupancy_prob(bev_occupancy)

        # occupancy gt
        gt_occ = target['center_bev'].to(bev_center.device)
        gt_valid_bev = target['valid_bev'].to(bev_center.device)

        occ_loss = criterion(bev_occupancy, gt_occ)

        if args.use_offset:
            gt_offset = target['offset_bev'].to(bev_center.device)
            offset_loss = masked_offset_l1_loss(bev_offset, gt_offset, gt_valid_bev)
        else:
            offset_loss = occ_loss.new_zeros(())

        has_motion_sample = args.motion_model == 'mamba' and item['motion_condition'].size(0) > 0
        if has_motion_sample:
            motion_condition = item['motion_condition'].to(bev_center.device)
            motion_target = item['motion_target'].to(bev_center.device)
            motion_bev_grid_shape = item['motion_bev_grid_shape'].to(bev_center.device)
            motion_loss = model.motion_predictor(
                motion_condition,
                motion_target,
                bev_grid_shape=motion_bev_grid_shape,
            )
            motion_stats = model.motion_predictor.last_loss_stats
        else:
            motion_loss = occ_loss.new_zeros(())
            motion_stats = {}

        if args.multi_task_loss == "uncertainty":
            total_loss = torch.exp(-model.instance_weight) * occ_loss + model.instance_weight
            task_weight_parts = ["occ {:.3f}".format(model.instance_weight.item())]

            if args.use_offset:
                total_loss = total_loss + torch.exp(-model.offset_weight) * offset_loss + model.offset_weight
                task_weight_parts.append("offset {:.3f}".format(model.offset_weight.item()))

            if has_motion_sample and args.motion_loss == 'niw':
                total_loss = total_loss + motion_loss
                task_weight_parts.append("motion fixed")
            elif has_motion_sample:
                total_loss = total_loss + torch.exp(-model.motion_weight) * motion_loss + model.motion_weight
            if args.motion_model == 'mamba' and args.motion_loss == 'smooth_l1':
                task_weight_parts.append("motion {:.3f}".format(model.motion_weight.item()))

            task_weight_text = ', weights: ' + ', '.join(task_weight_parts)
        else:
            total_loss = occ_loss + offset_loss + motion_loss
            task_weight_text = ''

        motion_stats_text = ''
        if motion_stats:
            motion_stats_text = (
                ', niw_nll {nll:.4f}, niw_reg {reg:.4f}, '
                'kappa {kappa_mean:.3f}, nu {nu_mean:.3f}'
            ).format(**motion_stats)
        
        total_loss.backward()
        optimizer.step()
        losses += float(total_loss.detach().item())

        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {}, Batch:{}/{},\t occ_loss: {:.3f}, offset_loss: {:.3f}, motion_loss: {:.6f}{}{}, \tTime: {:.3f} (min), maxima: {:.3f}'.format(
                    epoch, (batch_idx + 1), len(data_loader), occ_loss.item(), offset_loss.item(), motion_loss.item(), task_weight_text, motion_stats_text, (time.time()-tic)/60, bev_center.max()))

    print('Train Epoch: {}, Batch:{}/{},\t occ_loss: {:.3f}, offset_loss: {:.3f}, motion_loss: {:.6f}{}{}, \tTime: {:.3f} (min), maxima: {:.3f}'.format(
                    epoch, (batch_idx + 1), len(data_loader), occ_loss.item(), offset_loss.item(), motion_loss.item(), task_weight_text, motion_stats_text, (time.time()-tic)/60, bev_center.max()))

    return losses / len(data_loader)


def test(
    model,
    epoch,
    data_loader,
    res_fpath=None,
    visualize=False,
    plot_tracking=False,
    measure_fps=False,
    visualize_all_frames=False,
    plot_tracking_videos=False,
    plot_tracking_seq_ids=None,
):
    model.eval()

    all_res_rows = []
    trackers_by_root = {}
    tracking_results = defaultdict(list)
    tracking_gt_results = defaultdict(list)
    tracking_uncertainty_results = defaultdict(list)
    tracking_scales = {}
    realtime_update_time = 0.0
    realtime_update_count = 0
    realtime_fps_by_frame = {}
    last_visualization = None
    for batch_idx, (item, target) in enumerate(data_loader):
        realtime_tic = None
        if measure_fps:
            utils.synchronize_cuda()
            realtime_tic = time.perf_counter()

        decoded = utils.decode_frame(model, data_loader.dataset, item, batch_idx, device, args)
        root_name, frame_id, dataset_idx = utils.get_frame_metadata(data_loader, item)
        last_visualization = (target, decoded)
        if visualize and visualize_all_frames:
            utils.save_frame_visualization(
                logdir,
                epoch,
                target,
                decoded,
                frame_id,
                dataset_idx,
            )

        if res_fpath is not None:
            result_rows, realtime_elapsed = utils.update_tracking_for_frame(
                data_loader,
                item,
                decoded,
                root_name,
                frame_id,
                dataset_idx,
                trackers_by_root,
                tracking_results,
                tracking_gt_results,
                tracking_uncertainty_results,
                tracking_scales,
                args,
                measure_fps=measure_fps,
                realtime_tic=realtime_tic,
                motion_predictor=getattr(model, 'motion_predictor', None),
            )
            all_res_rows.append(result_rows)
            if realtime_elapsed is not None:
                realtime_update_time += realtime_elapsed
                realtime_update_count += 1
                if realtime_elapsed > 0.0:
                    realtime_fps_by_frame[(dataset_idx, frame_id)] = 1.0 / realtime_elapsed

    if res_fpath is not None:
        moda = utils.finalize_results(
            epoch,
            data_loader,
            res_fpath,
            plot_tracking,
            measure_fps,
            all_res_rows,
            tracking_results,
            tracking_gt_results,
            tracking_uncertainty_results,
            tracking_scales,
            realtime_update_time,
            realtime_update_count,
            realtime_fps_by_frame,
            score,
            device,
            plot_tracking_videos=plot_tracking_videos,
            plot_tracking_seq_ids=plot_tracking_seq_ids,
        )
    else:
        moda = 0

    if visualize and epoch != 0 and not visualize_all_frames and last_visualization is not None:
        target, decoded = last_visualization
        utils.save_detection_visualization(
            os.path.join(logdir, 'occupancy_map', 'map_' + str(epoch) + '.jpg'),
            decoded['bev_center'],
            target,
            decoded['vis_detection_xy'],
        )

    return np.average(moda)


###################################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--high_cls_thres', type=float, default=0.4, help='Confidence threshold for high-confidence detections')
    parser.add_argument('--track_match_thresh', type=float, default=75, help='MVTrackTrans-style center-distance association threshold')
    parser.add_argument('--track_low_match_thresh', type=float, default=100, help='MVTrackTrans-style center-distance gate for unconfirmed tracks')
    parser.add_argument('--motion_model', type=str, default='mamba', choices=['kalman', 'mamba'], help='Motion prediction model used by the tracker')
    parser.add_argument('--motion_normalization',type=str, default='bev', choices=['bev', 'none'], help='Normalize learned-motion coordinates by BEV grid size or use raw grid coordinates.')
    parser.add_argument('--motion_history_len', type=int, default=5, help='Number of trajectory tokens used by learned motion models')
    parser.add_argument('--mamba_history_mode', type=str, default='window', choices=['window', 'stateful'], help='Use a fixed trajectory window or persistent per-track Mamba state')
    parser.add_argument('--motion_history_source', type=str, default='detection', choices=['detection', 'motion_immediate', 'motion_warmup'], help='Use detection history, motion rollout from frame two, or a two-detection warmup before motion rollout')
    parser.add_argument('--mamba_d_model', type=int, default=64)
    parser.add_argument('--mamba_d_state', type=int, default=16)
    parser.add_argument('--mamba_d_conv', type=int, default=4)
    parser.add_argument('--mamba_expand', type=int, default=2)
    parser.add_argument('--mamba_layers', type=int, default=2)
    parser.add_argument('--mamba_dropout', type=float, default=0.0)
    parser.add_argument('--motion_loss', type=str, default='niw', choices=['smooth_l1', 'niw'], help='Motion prediction loss and output head')
    parser.add_argument('--evidential_lambda', type=float, default=1e-3, help='Evidence regularization coefficient for NIW motion loss')
    parser.add_argument('--evidential_head', type=str, default='separate', choices=['separate', 'joint'], help='Use separate mean/evidence MLPs or one joint NIW MLP')
    parser.add_argument('--use_offset', type=int, default=1, choices=[0, 1], help='Use the BEV sub-grid offset branch (1: use, 0: disable)')
    parser.add_argument('--bev_gaussian_sigma', type=float, default=2.0, help='Gaussian sigma used to generate BEV occupancy targets')
    parser.add_argument('--multi_task_loss', type=str, default='uncertainty', choices=['plain', 'uncertainty'], help='Choose plain summed loss or learned uncertainty multi-task loss')
    parser.add_argument('--data_root', type=str, default='/dataset/Multiview_datasets', help='Root containing dataset folders')
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx', 'citytrack', 'mvcrowdtrack', 'gmvd_test','gmvd_train'], help='Choose dataset wildtrack/multiviewx/citytrack/mvcrowdtrack/gmvd (default: wildtrack)')
    parser.add_argument('-pr', '--pretrained', default=False, action='store_true', help='Use pretrained weights (default: True)')
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N', help='input batch size for training (default: 2)')
    parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 1e-3)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='min LR', help='minimum learning rate for cosine scheduler')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize_all_frames', action='store_true', help='Save per-frame map images under maps_{epoch} during test')
    parser.add_argument('--plot_tracking_videos', action='store_true', help='Also save tracking trajectory and camera videos during test')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--aug', action='store_true', help='Use train-time random resize/crop augmentation')
    parser.add_argument('--view_pool', type=str, default='max', choices=['avg', 'max'], help='Camera-view pooling mode')
    parser.add_argument('--optim', type=str, default='Adam', choices=['SGD', 'Adam'], help='Choose optimizer.(default : SGD)')

    args = parser.parse_args()
    print('Occupancy loss: focal')
    print('Motion model: {}'.format(args.motion_model))
    print('Motion loss: {}'.format(args.motion_loss))
    print('Multi-task loss: {}'.format(args.multi_task_loss))
    
    torch.cuda.empty_cache()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
        
    #Logging
    logdir = f'logs/{args.dataset}_'+datetime.datetime.today().strftime('%d_%m_%Y_%H_%M_%S')
    os.makedirs(logdir, exist_ok=True)
    sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print('Settings:')
    print(vars(args))
    
     # Dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = T.Compose([T.ToTensor(), normalize])
    
    if 'wildtrack' in args.dataset:
        data_path = os.path.expanduser(os.path.join(args.data_root, 'Wildtrack'))
        base0 = Wildtrack(data_path)
    elif 'multiviewx' in args.dataset:
        data_path = os.path.expanduser(os.path.join(args.data_root, 'MultiviewX'))
        base0 = MultiviewX(data_path)
    elif args.dataset == 'citytrack':
        data_path = os.path.expanduser(os.path.join(args.data_root, 'CityTrack'))
        base0 = CityTrack(data_path)
    elif args.dataset == 'mvcrowdtrack':
        data_path = os.path.expanduser(os.path.join(args.data_root, 'MVCrowdTrack'))
        base0 = MVCrowdTrack(data_path)
        
    elif 'gmvd_train' in args.dataset or 'gmvd_test' in args.dataset:
        train_dataset_list = []
        test_dataset_list = []
        
        if 'gmvd_train' == args.dataset: 
            f = open('train_datapath.csv')
        elif 'gmvd_test' == args.dataset:
            f = open('test_datapath.csv')

        data_path = csv.reader(f)

        for i,data_row in enumerate(data_path):

            train_ratio = float(data_row[2])
            sample_require = int(data_row[3])
            sub_path = str(data_row[1]).lstrip('.')
            path = os.path.expanduser("/dataset/Multiview_datasets"+sub_path)
            if data_row[1].split('/')[-1]!='Wildtrack':
                base = MultiviewX(path)
            else:
                base = Wildtrack(path)
                
            if data_row[0]=='train':
                # Train data
                # dataset_obj = GetDataset(base, train=True, transform=train_transform, grid_reduce=4, img_reduce=4, train_ratio=train_ratio, sample_require=sample_require)
                dataset_obj = GetDataset(base, train=True, transform=train_transform, grid_reduce=4, img_reduce=8, train_ratio=train_ratio, sample_require=sample_require, data_aug=args.aug, build_motion_samples=args.motion_model == 'mamba', motion_history_len=args.motion_history_len, bev_gaussian_sigma=args.bev_gaussian_sigma)
                train_dataset_list.append(dataset_obj)
                
            else:
                # test data
                print("inside test")
                # dataset_obj = GetDataset(base, train=False, transform=train_transform, grid_reduce=4, img_reduce=4, train_ratio=train_ratio, sample_require=sample_require)
                dataset_obj = GetDataset(base, train=False, transform=train_transform, grid_reduce=4, img_reduce=8, train_ratio=train_ratio, sample_require=sample_require, data_aug=False, bev_gaussian_sigma=args.bev_gaussian_sigma)
                test_dataset_list.append(dataset_obj)
        
        if 'gmvd_test' == args.dataset:
            train_dataset_list = test_dataset_list

    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
        

    # Train and Test set
    if args.dataset == 'gmvd_train' or args.dataset == 'gmvd_test':
        print("Training datasets")
        train_dataset = ConcatDataset(*train_dataset_list)
        print("Testing datasets")
        test_dataset = ConcatDataset(*test_dataset_list)

    elif args.dataset in ['wildtrack', 'multiviewx', 'citytrack', 'mvcrowdtrack']:
        grid_reduce = int(getattr(base0, 'default_grid_reduce', 4))
        img_reduce = int(getattr(base0, 'default_img_reduce', 8))
        train_dataset_ = GetDataset(base0, train=True, transform=train_transform, grid_reduce=grid_reduce, img_reduce=img_reduce, data_aug=args.aug, build_motion_samples=args.motion_model == 'mamba', motion_history_len=args.motion_history_len, bev_gaussian_sigma=args.bev_gaussian_sigma)
        test_dataset_ = GetDataset(base0, train=False, transform=train_transform, grid_reduce=grid_reduce, img_reduce=img_reduce, data_aug=False, bev_gaussian_sigma=args.bev_gaussian_sigma)

        print("Training datasets")
        train_dataset = ConcatDataset(train_dataset_)
        print("Testing datasets")
        test_dataset = ConcatDataset(test_dataset_)
    
    # Train and Test Data Loader 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.num_workers, pin_memory=True, worker_init_fn=init_fn,
                                            collate_fn=motion_joint_collate if args.motion_model == 'mamba' else None)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=True, worker_init_fn=init_fn)

    print_bev_gaussian_settings("Train", train_dataset)
    print_bev_gaussian_settings("Test", test_dataset)
    print(f'Train Data : {len(train_loader)*args.batch_size}')
    print(f'Test Data : {len(test_loader)}')
    
    # Model
    resnet_model = resnet18(pretrained=args.pretrained, replace_stride_with_dilation=[False, True, True])
    model = MultiView_Detection(
        resnet_model,
        logdir,
        multi_task_loss=args.multi_task_loss,
        view_pool=args.view_pool,
        use_offset=args.use_offset,
        use_motion_loss=args.motion_model == 'mamba' and args.motion_loss == 'smooth_l1',
    )
    if args.motion_model == 'mamba':
        model.motion_predictor = MambaMotionPredictor(
            history_len=args.motion_history_len,
            d_model=args.mamba_d_model,
            d_state=args.mamba_d_state,
            d_conv=args.mamba_d_conv,
            expand=args.mamba_expand,
            num_layers=args.mamba_layers,
            dropout=args.mamba_dropout,
            motion_normalization=args.motion_normalization,
            history_mode=args.mamba_history_mode,
            loss_type=args.motion_loss,
            evidential_lambda=args.evidential_lambda,
            evidential_head=args.evidential_head,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(model)

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location=device),strict=False)
        print("Loaded checkpoint for test: {}".format(args.resume))
        model.eval()
        print('Test loaded model...')
        test(
            model,
            1,
            test_loader,
            os.path.join(logdir, 'test_'+str(args.dataset)+'.txt'),
            visualize=True,
            plot_tracking=True,
            plot_tracking_videos=args.plot_tracking_videos and args.dataset != 'gmvd_test',
            plot_tracking_seq_ids=[-1] if args.dataset == 'gmvd_test' else None,
            measure_fps=True,
            visualize_all_frames=args.visualize_all_frames,
        )
        sys.exit(0)
     
    # Optimizer
    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # LR Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    
    # Loss
    criterion = CornerNetFocalLoss().to(device)
    best_tracking_pred_path = os.path.join(logdir, 'mota_pred_best_tracking.txt')
    for epoch in tqdm.tqdm(range(1, args.epochs+1)):
        print('Training...')
        train(
            model,
            epoch,
            train_loader,
            optimizer,
            args.log_interval,
        )
        
        print('Testing...')
        test(
            model,
            epoch,
            test_loader,
            os.path.join(logdir, 'test_'+str(args.dataset)+'_'+str(epoch)+'.txt'),
            visualize=True,
        )
            
        scheduler.step()
        
        # Save Dictionary
        current_detection_score_key = utils.detection_score_key(score[-1])
        if utils.isbest(current_detection_score_key, best_detection_key):
            best_detection_key = current_detection_score_key
            best_detection_idx = len(score) - 1
            torch.save(model.state_dict(), os.path.join(logdir, 'Best_Multiview_Detection_'+str(args.dataset)+'.pth'))
            print(
                "Saved best detection checkpoint by {}".format(
                    utils.detection_score_text(current_detection_score_key)
                )
            )

        current_tracking_score_key = utils.tracking_score_key(score[-1])
        if utils.isbest(current_tracking_score_key, best_tracking_key):
            best_tracking_key = current_tracking_score_key
            best_tracking_idx = len(score) - 1
            torch.save(model.state_dict(), os.path.join(logdir, 'Best_Multiview_Tracking_'+str(args.dataset)+'.pth'))
            current_mota_pred_path = os.path.join(logdir, 'mota_pred.txt')
            if os.path.exists(current_mota_pred_path):
                shutil.copyfile(current_mota_pred_path, best_tracking_pred_path)
            current_uncertainty_path = os.path.join(
                logdir,
                'motion_uncertainty_epoch_{}.csv'.format(epoch),
            )
            if os.path.exists(current_uncertainty_path):
                shutil.copyfile(
                    current_uncertainty_path,
                    os.path.join(logdir, 'motion_uncertainty_best_tracking.csv'),
                )
            print(
                "Saved best tracking checkpoint by {}".format(
                    utils.tracking_score_text(current_tracking_score_key)
                )
            )
        
    moda, modp, prec, rec, idf1, mota, motp_raw, motp, mt, ml = score[best_detection_idx]
    print(f'####### best detection epoch: {best_detection_idx + 1} ,moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {prec:.1f}%, recall: {rec:.1f}% #######')
    moda, modp, prec, rec, idf1, mota, motp_raw, motp, mt, ml = score[best_tracking_idx]
    print(f'####### best tracking epoch: {best_tracking_idx + 1} ,idf1: {idf1:.1f}%, mota: {mota:.1f}%, motp_raw: {motp_raw:.4f}, motp: {motp:.1f}%, mt: {mt:.1f}%, ml: {ml:.1f}% #######')
    
    print("Training Completed..")

    retain_epochs = [best_detection_idx + 1, best_tracking_idx + 1]
    utils.cleanup(logdir, retain_epochs)
