import os
import time
import csv
import numpy as np
import random
import sys
import datetime
import shutil
import json
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
from utils.loss import EvidentialOccupancyLoss
from detector.multiview_model import MultiView_Detection
from detector.encoder import resnet18
from tracker.mamba_motion_predictor import MambaMotionPredictor
from utils.training_stages import load_detector_weights, freeze_detector
from utils.arguments import create_parser, parse_args
    
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


def train(
    model,
    epoch,
    data_loader,
    optimizer,
    log_interval,
    *, dataset, device, criterion, args,
):
    model.train()
    tic = time.time()

    losses = 0

    for batch_idx, (item, target) in enumerate(data_loader):
        optimizer.zero_grad()
        data = item['img'].to(device)
                
        bev_occupancy = model(dataset, item['root'], data, item.get('proj_mats'))
        bev_center = utils.get_occupancy_prob(bev_occupancy)

        # occupancy gt
        gt_occ = target['center_bev'].to(bev_center.device)
        gt_valid_bev = target['valid_bev'].to(bev_center.device)

        occ_loss = criterion(bev_occupancy, gt_valid_bev.float(), gt_occ)

        has_motion_sample = args.motion_model == 'mamba' and item['motion_bev_grid_shape'].size(0) > 0
        if has_motion_sample:
            motion_condition = item['motion_condition'].to(device)
            motion_target = item['motion_target'].to(device)
            motion_bev_grid_shape = item['motion_bev_grid_shape'].to(device)
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

            if has_motion_sample:
                total_loss = total_loss + torch.exp(-model.motion_weight) * motion_loss + model.motion_weight
                task_weight_parts.append("motion {:.3f}".format(model.motion_weight.item()))

            task_weight_text = ', weights: ' + ', '.join(task_weight_parts)
        else:
            total_loss = occ_loss + motion_loss
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
            print('Train Epoch: {}, Batch:{}/{},\t occ_loss: {:.3f}, motion_loss: {:.6f}{}{}, \tTime: {:.3f} (min), maxima: {:.3f}'.format(
                    epoch, (batch_idx + 1), len(data_loader), occ_loss.item(), motion_loss.item(), task_weight_text, motion_stats_text, (time.time()-tic)/60, bev_center.max()))

    print('Train Epoch: {}, Batch:{}/{},\t occ_loss: {:.3f}, motion_loss: {:.6f}{}{}, \tTime: {:.3f} (min), maxima: {:.3f}'.format(
                    epoch, (batch_idx + 1), len(data_loader), occ_loss.item(), motion_loss.item(), task_weight_text, motion_stats_text, (time.time()-tic)/60, bev_center.max()))

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
    detection_only=False,
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
            if detection_only:
                config = data_loader.dataset.dicts[root_name]
                xy = decoded['eval_xy']
                if config['base'].base.indexing != 'xy':
                    xy = xy[:, [1, 0]]
                all_res_rows.append(torch.cat((
                    xy.new_full((len(xy), 1), frame_id),
                    xy * float(config['base'].grid_reduce),
                    decoded['eval_scores'],
                    xy.new_full((len(xy), 1), dataset_idx),
                ), dim=1))
                continue
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

    if detection_only and res_fpath is not None:
        all_res = utils.stack_detection_results(all_res_rows, device)
        np.savetxt(res_fpath, all_res.cpu().numpy(), '%.8f')
        metrics = utils.evaluate_detection_results(logdir, epoch, data_loader, all_res)
        score.append(list(metrics))
        moda, modp, precision, recall = metrics
        print(f'####### moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {precision:.1f}%, recall: {recall:.1f}% #######')
    elif res_fpath is not None:
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


def run(stage='joint', train_epoch=None, argv=None):
    global args, device, logdir, score
    args = parse_args(stage, argv)
    score = []
    best_detection_key = best_tracking_key = None
    best_detection_idx = best_tracking_idx = 0
    train_epoch = train if train_epoch is None else train_epoch
    print('Occupancy loss: edl')
    print('Training stage: {}'.format(stage))
    if stage != 'detector':
        print('Motion model: {}'.format(args.motion_model))
        print('Motion loss: {}'.format(args.motion_loss))
        print('Motion association cost: {}'.format(args.motion_cost))
        print('Mamba uncertainty input: {}'.format(args.mamba_uncertainty_mode))
        print('Mamba explicit gate: {}, weight: {}, motion weight: {}, motion covariance: {}, trace reference: {}'.format(
            args.mamba_uncertainty_gate, args.mamba_uncertainty_weight,
            args.motion_uncertainty_weight, args.motion_uncertainty_type, args.motion_uncertainty_ref,
        ))
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
    logdir = args.output_dir or f'logs/{args.dataset}_{stage}_' + datetime.datetime.today().strftime('%d_%m_%Y_%H_%M_%S_%f')
    os.makedirs(logdir, exist_ok=True)
    sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print('Settings:')
    print(vars(args))
    with open(os.path.join(logdir, 'settings.json'), 'w') as settings_file:
        json.dump({'stage': stage, **vars(args)}, settings_file, indent=2)
    print('Motion training history: {}'.format('GT' if stage == 'joint' else 'frozen detector' if stage == 'motion' else 'not used'))

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
            path = os.path.expanduser(os.path.join(args.data_root, sub_path.lstrip('/\\')))
            if data_row[1].split('/')[-1]!='Wildtrack':
                base = MultiviewX(path)
            else:
                base = Wildtrack(path)
                
            if data_row[0]=='train':
                # Train data
                # dataset_obj = GetDataset(base, train=True, transform=train_transform, grid_reduce=4, img_reduce=4, train_ratio=train_ratio, sample_require=sample_require)
                dataset_obj = GetDataset(base, train=True, transform=train_transform, grid_reduce=4, img_reduce=8, train_ratio=train_ratio, sample_require=sample_require, data_aug=args.aug, build_motion_samples=stage == 'joint' and args.motion_model == 'mamba', motion_history_len=args.motion_history_len, bev_gaussian_sigma=args.bev_gaussian_sigma)
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
        train_dataset_ = GetDataset(base0, train=True, transform=train_transform, grid_reduce=grid_reduce, img_reduce=img_reduce, data_aug=args.aug, build_motion_samples=stage == 'joint' and args.motion_model == 'mamba', motion_history_len=args.motion_history_len, bev_gaussian_sigma=args.bev_gaussian_sigma)
        test_dataset_ = GetDataset(base0, train=False, transform=train_transform, grid_reduce=grid_reduce, img_reduce=img_reduce, data_aug=False, bev_gaussian_sigma=args.bev_gaussian_sigma)

        print("Training datasets")
        train_dataset = ConcatDataset(train_dataset_)
        print("Testing datasets")
        test_dataset = ConcatDataset(test_dataset_)
    
    # Train and Test Data Loader 
    if stage != 'motion':
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True, worker_init_fn=init_fn,
                                                collate_fn=motion_joint_collate if args.motion_model == 'mamba' else None)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=True, worker_init_fn=init_fn)

    print_bev_gaussian_settings("Train", train_dataset)
    print_bev_gaussian_settings("Test", test_dataset)
    print(f'Train Data : {len(train_dataset)} frames')
    print(f'Test Data : {len(test_loader)}')
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet_model = resnet18(pretrained=args.pretrained, replace_stride_with_dilation=[False, True, True])
    model = MultiView_Detection(
        resnet_model,
        logdir,
        device=device,
        multi_task_loss=args.multi_task_loss,
        use_motion_loss=args.motion_model == 'mamba',
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
            history_mode=args.mamba_history_mode,
            loss_type=args.motion_loss,
            evidential_lambda=args.evidential_lambda,
            evidential_head=args.evidential_head,
            uncertainty_mode=args.mamba_uncertainty_mode,
            uncertainty_gate=args.mamba_uncertainty_gate,
            motion_uncertainty_type=args.motion_uncertainty_type,
            motion_uncertainty_ref=args.motion_uncertainty_ref,
            uncertainty_weight=args.mamba_uncertainty_weight,
            motion_uncertainty_weight=args.motion_uncertainty_weight,
        )

    model.to(device)

    print(model)

    if stage == 'motion':
        load_detector_weights(model, args.detector_checkpoint)
        freeze_detector(model)
        from utils.motion_training_data import prepare_motion_dataset
        cache_start = time.perf_counter()
        motion_dataset = prepare_motion_dataset(model, train_dataset, args, device, logdir)
        cache_seconds = time.perf_counter() - cache_start
        with open(os.path.join(logdir, 'history_preparation.json'), 'w') as output:
            json.dump({'seconds': cache_seconds, 'samples': len(motion_dataset)}, output, indent=2)
        train_loader = torch.utils.data.DataLoader(motion_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=0, pin_memory=True)
        print('Frozen detector; motion samples: {}; history preparation: {:.1f}s'.format(len(motion_dataset), cache_seconds))

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
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.optim == 'Adam':
        optimizer = optim.Adam(trainable_parameters, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        
    if args.optim == 'SGD':
        optimizer = optim.SGD(trainable_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # LR Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    
    # Separate stages each optimize one unweighted task loss.
    criterion = None if stage == 'motion' else EvidentialOccupancyLoss().to(device)
    if criterion is not None:
        print('Occupancy EDL lambda_reg: {}, gamma: {}, eta: {}'.format(criterion.lambda_reg, criterion.gamma, criterion.eta))
    metric_names = ['moda', 'modp', 'precision', 'recall']
    if stage != 'detector':
        metric_names += ['idf1', 'mota', 'motp_raw', 'motp', 'mt', 'ml']
    metrics_path = os.path.join(logdir, 'metrics.csv')
    with open(metrics_path, 'w', newline='') as output:
        csv.writer(output).writerow(['epoch', 'train_loss', 'lr', 'train_seconds', 'eval_seconds'] + metric_names)
    best_tracking_pred_path = os.path.join(logdir, 'mota_pred_best_tracking.txt')
    for epoch in tqdm.tqdm(range(1, args.epochs+1)):
        print('Training...')
        train_start = time.perf_counter()
        train_loss = train_epoch(model, epoch, train_loader, optimizer, args.log_interval,
                                 dataset=train_dataset, device=device, criterion=criterion, args=args)
        train_seconds = time.perf_counter() - train_start
        eval_start = time.perf_counter()
        print('Testing...')
        test(
            model,
            epoch,
            test_loader,
            os.path.join(logdir, 'test_'+str(args.dataset)+'_'+str(epoch)+'.txt'),
            visualize=True,
            detection_only=stage == 'detector',
        )
            
        eval_seconds = time.perf_counter() - eval_start
        with open(metrics_path, 'a', newline='') as output:
            csv.writer(output).writerow([epoch, train_loss, optimizer.param_groups[0]['lr'], train_seconds, eval_seconds] + list(score[-1]))
        torch.save(model.state_dict(), os.path.join(logdir, 'Last_Multiview_{}.pth'.format(args.dataset)))
        if stage == 'motion':
            torch.save(model.motion_predictor.state_dict(), os.path.join(logdir, 'Last_Motion_{}.pth'.format(args.dataset)))
        scheduler.step()
        
        # Save Dictionary
        current_detection_score_key = utils.detection_score_key(score[-1])
        if stage != 'motion' and utils.isbest(current_detection_score_key, best_detection_key):
            best_detection_key = current_detection_score_key
            best_detection_idx = len(score) - 1
            torch.save(model.state_dict(), os.path.join(logdir, 'Best_Multiview_Detection_'+str(args.dataset)+'.pth'))
            print(
                "Saved best detection checkpoint by {}".format(
                    utils.detection_score_text(current_detection_score_key)
                )
            )

        if stage != 'detector':
            current_tracking_score_key = utils.tracking_score_key(score[-1])
            if utils.isbest(current_tracking_score_key, best_tracking_key):
                best_tracking_key = current_tracking_score_key
                best_tracking_idx = len(score) - 1
                torch.save(model.state_dict(), os.path.join(logdir, 'Best_Multiview_Tracking_'+str(args.dataset)+'.pth'))
                if stage == 'motion':
                    torch.save(model.motion_predictor.state_dict(), os.path.join(logdir, 'Best_Motion_{}.pth'.format(args.dataset)))
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

    if stage != 'motion':
        print('Best detection epoch: {}, {}'.format(best_detection_idx + 1, utils.detection_score_text(best_detection_key)))
    if stage != 'detector':
        print('Best tracking epoch: {}, {}'.format(best_tracking_idx + 1, utils.tracking_score_text(best_tracking_key)))
    print('Training Completed..')


if __name__ == '__main__':
    run()
