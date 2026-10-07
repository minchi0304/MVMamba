import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torch.nn import Linear, Module


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, class_token=False):
        x = x.permute(1, 2, 0)

        num_feats = x.shape[1]
        num_pos_feats = num_feats
        mask = torch.zeros(x.shape[0], x.shape[2], device=x.device).to(torch.bool)
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_y = y_embed[:, :, None] / dim_t
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        return pos_y


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        pos=None,
    ):
        src2 = self.self_attn(
            src,
            src,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        pos=None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q,
            k,
            value=src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class HistoryMotionEmbedding(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        dim_feedforward=512,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        pos_type="sin",
    ):
        super().__init__()
        self.cascade_num = 6
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.trca = nn.ModuleList()
        for _ in range(self.cascade_num):
            self.trca.append(
                TransformerEncoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout,
                    activation,
                    normalize_before,
                )
            )

        self.proj = nn.Linear(4, d_model)
        if pos_type == "sin":
            self.pose_encoding = PositionEmbeddingSine(normalize=True)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0).to(self.cls_token)
        else:
            x = x.to(self.cls_token)

        q_patch = self.proj(x).permute(1, 0, 2)
        pos = self.pose_encoding(q_patch).transpose(0, 1)
        _, batch_size, feature_dim = q_patch.shape

        cls_tokens = repeat(
            self.cls_token,
            "() n d -> b n d",
            b=batch_size,
        ).permute(1, 0, 2).contiguous()
        encoder_patch = torch.cat((q_patch, cls_tokens), dim=0)

        for i in range(self.cascade_num):
            en_out = self.trca[i](src=encoder_patch, pos=pos)
            encoder_patch = en_out

        return en_out[0].view(batch_size, 1, feature_dim).contiguous()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class MFL(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super().__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        return self._layer(x) * gate + bias


class VarianceSchedule(Module):
    def __init__(
        self,
        num_steps,
        mode="linear",
        beta_1=1e-4,
        beta_T=5e-2,
        cosine_s=8e-3,
    ):
        super().__init__()
        assert mode in ("linear", "cosine")
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        else:
            timesteps = torch.arange(num_steps + 1) / num_steps + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()
        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = (
                (1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])
            ) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sigmas_flex", sigmas_flex)
        self.register_buffer("sigmas_inflex", sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility <= 1
        return (
            self.sigmas_flex[t] * flexibility
            + self.sigmas_inflex[t] * (1 - flexibility)
        )


class HMINet(Module):
    def __init__(self, point_dim=2, context_dim=256, tf_layer=3, residual=False):
        super().__init__()
        self.residual = residual
        self.pos_emb = PositionalEncoding(
            d_model=2 * context_dim,
            dropout=0.1,
            max_len=24,
        )
        self.pos_emb2 = PositionalEncoding(
            d_model=context_dim,
            dropout=0.1,
            max_len=24,
        )
        self.concat1 = MFL(point_dim, context_dim // 2, context_dim + 3)
        self.concat1_2 = MFL(
            context_dim // 2,
            context_dim,
            context_dim + 3,
        )
        self.concat1_3 = MFL(
            context_dim,
            2 * context_dim,
            context_dim + 3,
        )
        self.layer = nn.TransformerEncoderLayer(
            d_model=2 * context_dim,
            nhead=4,
            dim_feedforward=4 * context_dim,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.layer,
            num_layers=tf_layer,
        )
        self.layer2 = nn.TransformerEncoderLayer(
            d_model=context_dim,
            nhead=4,
            dim_feedforward=2 * context_dim,
        )
        self.transformer_encoder2 = nn.TransformerEncoder(
            self.layer2,
            num_layers=tf_layer,
        )
        self.concat3 = MFL(
            2 * context_dim,
            context_dim,
            context_dim + 3,
        )
        self.concat4 = MFL(
            context_dim,
            context_dim // 2,
            context_dim + 3,
        )
        self.linear = MFL(
            context_dim // 2,
            point_dim,
            context_dim + 3,
        )

    def forward(self, x, beta, context):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1)
        context = context.view(batch_size, -1)
        time_emb = torch.cat(
            [beta, torch.sin(beta), torch.cos(beta)],
            dim=-1,
        )
        ctx_emb = torch.cat([time_emb, context], dim=-1)
        x = self.concat1_3(
            ctx_emb,
            self.concat1_2(ctx_emb, self.concat1(ctx_emb, x)),
        )
        final_emb = self.pos_emb(x.unsqueeze(0))
        trans = self.transformer_encoder(final_emb).permute(1, 0, 2).squeeze(1)
        trans = self.concat3(ctx_emb, trans)
        final_emb = self.pos_emb2(trans.unsqueeze(0))
        trans = self.transformer_encoder2(final_emb).permute(1, 0, 2).squeeze(1)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)


class D2MP(Module):
    def __init__(self, net, var_sched, eps=0.001, clamp_x0=True):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.eps = eps
        self.weight = True
        self.clamp_x0 = clamp_x0

    def q_sample(self, x_start, noise, t, c):
        time = t.reshape(c.shape[0], *((1,) * (len(c.shape) - 1)))
        return x_start + c * time + torch.sqrt(time) * noise

    def pred_x0_from_xt(self, xt, noise, c, t):
        time = t.reshape(c.shape[0], *((1,) * (len(c.shape) - 1)))
        return xt - c * time - torch.sqrt(time) * noise

    def pred_xtms_from_xt(self, xt, noise, c, t, s):
        time = t.reshape(c.shape[0], *((1,) * (len(c.shape) - 1)))
        s = s.reshape(c.shape[0], *((1,) * (len(c.shape) - 1)))
        mean = xt + c * (time - s) - c * time - s / torch.sqrt(time) * noise
        epsilon = torch.randn_like(mean, device=xt.device)
        sigma = torch.sqrt(s * (time - s) / time)
        return mean + sigma * epsilon

    def forward(self, x_0, context, t=None):
        batch_size, point_dim = x_0.size()
        if t is None:
            t = torch.rand(x_0.shape[0], device=x_0.device) * (1.0 - self.eps) + self.eps

        beta = t.log() / 4
        e_rand = torch.randn_like(x_0).cuda()
        c = -1 * x_0
        x_noisy = self.q_sample(x_start=x_0, noise=e_rand, t=t, c=c)
        t = t.reshape(-1, 1)

        c_pred = self.net(x_noisy, beta=beta, context=context)
        noise_pred = (x_noisy - (t - 1) * c_pred) / t.sqrt()
        if not self.weight:
            loss_c = F.smooth_l1_loss(
                c_pred.view(-1, point_dim),
                c.view(-1, point_dim),
                reduction="mean",
            )
            loss_noise = F.smooth_l1_loss(
                noise_pred.view(-1, point_dim),
                e_rand.view(-1, point_dim),
                reduction="mean",
            )
            loss = 0.5 * loss_c + 0.5 * loss_noise
        else:
            simple_weight1 = (t ** 2 - t + 1) / t
            simple_weight2 = (t ** 2 - t + 1) / (1 - t + self.eps)
            loss_c = F.smooth_l1_loss(
                c_pred.view(-1, point_dim),
                c.view(-1, point_dim),
                reduction="none",
            )
            loss_noise = F.smooth_l1_loss(
                noise_pred.view(-1, point_dim),
                e_rand.view(-1, point_dim),
                reduction="none",
            )
            loss = simple_weight1 * loss_c + simple_weight2 * loss_noise
            loss = loss.mean()
        return loss

    def sample(
        self,
        context,
        sample,
        bestof,
        point_dim=2,
        flexibility=0.0,
        ret_traj=False,
    ):
        traj_list = []
        for _ in range(sample):
            batch_size = context.size(0)
            if bestof:
                x_t = torch.randn([batch_size, point_dim]).to(context.device)
            else:
                x_t = torch.zeros([batch_size, point_dim]).to(context.device)

            self.var_sched.num_steps = 1
            traj = {self.var_sched.num_steps: x_t}

            cur_time = torch.ones((batch_size,), device=x_t.device)
            step = 1.0 / self.var_sched.num_steps
            for t in range(self.var_sched.num_steps, 0, -1):
                s = torch.full((batch_size,), step, device=x_t.device)
                if t == 1:
                    s = cur_time

                x_t = traj[t]
                beta = cur_time.log() / 4
                t_tmp = cur_time.reshape(-1, 1)
                c_pred = self.net(x_t, beta=beta, context=context)
                noise_pred = (x_t - (t_tmp - 1) * c_pred) / t_tmp.sqrt()

                x0 = self.pred_x0_from_xt(
                    x_t,
                    noise_pred,
                    c_pred,
                    cur_time,
                )
                if self.clamp_x0:
                    x0.clamp_(-1.0, 1.0)
                c_pred = -1 * x0
                x_next = self.pred_xtms_from_xt(
                    x_t,
                    noise_pred,
                    c_pred,
                    cur_time,
                    s,
                )
                cur_time = cur_time - s
                traj[t - 1] = x_next.detach()
                if not ret_traj:
                    del traj[t]

            if ret_traj:
                traj_list.append(traj)
            else:
                traj_list.append(traj[0])

        return torch.stack(traj_list)


class DiffusionMotionPredictor(Module):
    def __init__(
        self,
        encoder_dim=256,
        tf_layer=3,
        eps=0.001,
        motion_normalization='bev',
    ):
        super().__init__()
        if motion_normalization not in ('bev', 'none'):
            raise ValueError("motion_normalization must be 'bev' or 'none'")
        self.motion_normalization = motion_normalization
        self.encoder = HistoryMotionEmbedding(d_model=encoder_dim)
        self.diffusion = D2MP(
            net=HMINet(
                point_dim=2,
                context_dim=encoder_dim,
                tf_layer=tf_layer,
                residual=False,
            ),
            var_sched=VarianceSchedule(
                num_steps=100,
                beta_T=5e-2,
                mode="linear",
            ),
            eps=eps,
            clamp_x0=motion_normalization == 'bev',
        )

    @staticmethod
    def _coordinate_scale(bev_grid_shape, reference):
        if torch.is_tensor(bev_grid_shape):
            grid_shape = bev_grid_shape.to(device=reference.device, dtype=reference.dtype)
        else:
            grid_shape = reference.new_tensor(bev_grid_shape)
        if grid_shape.dim() == 1:
            return grid_shape[[1, 0]]
        if grid_shape.dim() == 2:
            return grid_shape[:, [1, 0]]
        raise ValueError("bev_grid_shape must be [2] or [B,2]")

    def _normalize_conditions(self, conditions, bev_grid_shape):
        if self.motion_normalization == 'none':
            return conditions
        scale = self._coordinate_scale(bev_grid_shape, conditions)
        if scale.dim() == 2:
            scale = scale.unsqueeze(1)
        conditions = conditions.clone()
        conditions[..., :2] = conditions[..., :2] / scale
        conditions[..., 2:] = conditions[..., 2:] / scale
        return conditions

    def _normalize_delta(self, delta_position, bev_grid_shape):
        if self.motion_normalization == 'none':
            return delta_position
        scale = self._coordinate_scale(bev_grid_shape, delta_position)
        return delta_position / scale

    def _denormalize_delta(self, delta_position, bev_grid_shape):
        if self.motion_normalization == 'none':
            return delta_position
        scale = self._coordinate_scale(bev_grid_shape, delta_position)
        return delta_position * scale

    def generate(
        self,
        conditions,
        sample=1,
        bestof=True,
        flexibility=0.0,
        ret_traj=False,
        bev_grid_shape=None,
    ):
        if self.motion_normalization == 'bev' and bev_grid_shape is None:
            raise ValueError("bev_grid_shape is required for BEV normalization")
        encoded_conditions = []
        device = next(self.parameters()).device
        for condition in conditions:
            condition = torch.tensor(np.array(condition), dtype=torch.float, device=device)
            condition = self._normalize_conditions(condition, bev_grid_shape)
            if len(condition) != 5:
                padded = condition[-1].repeat((5, 1))
                condition = torch.cat((condition, padded), dim=0)[:5]
            encoded_conditions.append(condition.unsqueeze(0))
        encoded_conditions = torch.cat(encoded_conditions)
        encoded_conditions = self.encoder(encoded_conditions)
        prediction = self.diffusion.sample(
            encoded_conditions,
            sample,
            bestof,
            flexibility=flexibility,
            ret_traj=ret_traj,
        )
        prediction = self._denormalize_delta(prediction, bev_grid_shape)
        return prediction.cpu().detach().numpy()

    def forward(self, condition, delta_position, bev_grid_shape=None):
        if self.motion_normalization == 'bev' and bev_grid_shape is None:
            raise ValueError("bev_grid_shape is required for BEV normalization")
        condition = self._normalize_conditions(condition, bev_grid_shape)
        delta_position = self._normalize_delta(delta_position, bev_grid_shape)
        encoded_condition = self.encoder(condition)
        return self.diffusion(delta_position, encoded_condition)
