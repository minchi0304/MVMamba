import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform.imgwarp import warp_perspective
import matplotlib.pyplot as plt
from detector.decoder import Decoder

class MultiView_Detection(nn.Module):
    def __init__(
        self,
        backbone_model,
        logdir,
        latent_dim=512,
        device=torch.device('cuda'),
        multi_task_loss="plain",
        view_pool="avg",
        use_offset=True,
        use_motion_loss=False,
    ):
        super().__init__()
        if multi_task_loss not in ["plain", "uncertainty"]:
            raise ValueError("multi_task_loss must be 'plain' or 'uncertainty'")
        if view_pool not in ["avg", "max"]:
            raise ValueError("view_pool must be 'avg' or 'max'")
        self.logdir = logdir
        self.device = device
        self.feat2d = 128
        self.base_arch = nn.Sequential(*list(backbone_model.children())[:-2]).to(self.device)
        self.multi_task_loss = multi_task_loss
        self.view_pool = view_pool
        self.use_offset = bool(use_offset)
        self.use_motion_loss = bool(use_motion_loss)
        if self.multi_task_loss == "uncertainty":
            self.instance_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            if self.use_offset:
                self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            if self.use_motion_loss:
                self.motion_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.out_channel = 512
   

        shared_out_channels = self.out_channel
        offset_out_channels = 2

        self.decoder = Decoder(in_channels=latent_dim,).to(self.device)
 
        # Ground Plane Convolution
        if self.use_offset:
            self.instance_offset_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, offset_out_channels, kernel_size=1, padding=0),
            ).to(self.device)
    
        center_out_channels = 2
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, center_out_channels, kernel_size=1, padding=0),
        ).to(self.device)
        
        with torch.no_grad():
            self.instance_center_head[-1].bias[0].fill_(-4.0)
            self.instance_center_head[-1].bias[1].fill_(4.0)
    

    def forward(self, dataset, dataset_name, imgs, proj_mats=None):

        B, N, C, H, W = imgs.shape
        if B > 1:
            outputs = []
            for batch_idx in range(B):
                if isinstance(dataset_name, (list, tuple)):
                    sample_dataset_name = dataset_name[batch_idx]
                else:
                    sample_dataset_name = dataset_name
                sample_proj_mats = None if proj_mats is None else proj_mats[batch_idx:batch_idx + 1]
                outputs.append(self.forward(dataset, [sample_dataset_name], imgs[batch_idx:batch_idx + 1], sample_proj_mats))

            bev_occupancy, bev_offset = zip(*outputs)
            return torch.cat(bev_occupancy, dim=0), torch.cat(bev_offset, dim=0)

        sample_dataset_name = dataset_name[0] if isinstance(dataset_name, (list, tuple)) else dataset_name
        config_dict = dataset.dicts[sample_dataset_name]
        upsample_shape = config_dict['upsample_shape']
        reducedgrid_shape = config_dict['reducedgrid_shape']
        if proj_mats is None:
            proj_mats = config_dict['proj_mats']
        elif proj_mats.dim() == 4:
            proj_mats = proj_mats[0]
        coord_map = config_dict['coord_map']

        #vector code
        imgs_vec = imgs[0]  # (N, 3, H, W)
        img_features = self.base_arch(imgs_vec.to(self.device))  # (N, latent_dim, h, w)
        # img_features = self.encoder(imgs_vec.to(self.device))  # (N, latent_dim, h, w)

        img_features = F.interpolate(img_features, upsample_shape, mode='bilinear')  # (N, 512, H/8, W/8)


        proj_mats = proj_mats.float().to(self.device)
        world_features = warp_perspective(img_features, proj_mats, reducedgrid_shape) #(N, 512, Hg/4, Wg/4)

        if self.view_pool == "avg":
            world_features = torch.mean(world_features, dim=0, keepdim=True)  # (1, 512, Hg/4, Wg/4)
        else:
            world_features = torch.max(world_features, dim=0, keepdim=True).values  # (1, 512, Hg/4, Wg/4)
        world_features = torch.cat([world_features, coord_map.repeat([B, 1, 1, 1]).to(self.device)], dim=1)

        # fig = plt.figure()
        # subplt0 = fig.add_subplot(211, title="output")
        # subplt0.imshow(torch.norm(world_features[0].detach(), dim=0).cpu().numpy())
        # plt.savefig(os.path.join(self.logdir,'world_features.jpg'))
        # plt.close(fig)
        
        
        bev_features = self.decoder(world_features)


        bev_occupancy = self.instance_center_head(bev_features)
        if self.use_offset:
            bev_offset = self.instance_offset_head(bev_features)
        else:
            bev_offset = bev_features.new_zeros(
                (bev_features.size(0), 2, bev_features.size(2), bev_features.size(3))
            )

        return bev_occupancy, bev_offset
