import math
import sys
import torch
import torch.nn as nn
import utils
from typing import Iterable
from einops import rearrange, repeat
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def custom_mae_loss(outputs, labels, bool_masked_pos, decode_masked_pos, B):

    #----------------------#
    # 计算基本的均方误差损失
    #----------------------#
    loss = (outputs - labels) ** 2
    loss = loss.mean(dim=-1)
    
    #----------------------#
    # 计算掩码位置的损失
    #----------------------#
    cal_loss_mask = bool_masked_pos[~decode_masked_pos].reshape(B, -1)
    loss          = (loss * cal_loss_mask).sum() / cal_loss_mask.sum()
    
    return loss


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None,
                    num_samples=1, # by default
                    normlize_target_audio: bool = True, patch_size_audio: int = 16,
                    loss_weight=0.1,
                    use_frame_diff_as_target=False, frame_diff_group_size=2,
                    target_diff_weight=None
                    ):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header     = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func        = nn.MSELoss() # video
    loss_func_audio  = nn.MSELoss() # audio

    # for weighted org target + diff target (assert frame_diff_group_size==2!!!)
    if use_frame_diff_as_target and target_diff_weight is not None:
        loss_func_diff = nn.MSELoss()

    loss_func_inter_contrastive_v = nn.CrossEntropyLoss() # video
    loss_func_inter_contrastive_a = nn.CrossEntropyLoss() # audio

    # little modify to avoid index error # NOTE: [2024.7.3 15.41]
    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration

        # NOTE: new added
        if it >= len(lr_schedule_values):
            print(f"=> Warning: 'it' value {it} exceeds 'lr_schedule_values' length {len(lr_schedule_values)}")
            it = len(lr_schedule_values) - 1 # 调整it为最后一个有效索引

        if lr_schedule_values is not None or wd_schedule_values is not None:

            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"] # may occur index error

                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # videos, bool_masked_pos, audios, bool_masked_pos_audio = batch [2024.7.3 15.45]
                    
        videos, bool_masked_pos, decode_masked_pos, audios, bool_masked_pos_audio, decode_masked_pos_audio   = batch # NOTE: 添加decoder masks, 按照顺序进行

        if num_samples > 1:
            videos          = rearrange(videos, 'b c (nt t) h w -> (b nt) c t h w', nt=num_samples)
            bool_masked_pos = repeat(bool_masked_pos, 'b c -> (b nt) c', nt=num_samples)

        # videos
        videos              = videos.to(device, non_blocking=True)
        bool_masked_pos     = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        decode_masked_pos   = decode_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool) # new added

        # audios
        audios                  = audios.to(device, non_blocking=True) # [20, 1, 256, 128]
        bool_masked_pos_audio   = bool_masked_pos_audio.to(device, non_blocking=True).flatten(1).to(torch.bool)
        decode_masked_pos_audio = decode_masked_pos_audio.to(device, non_blocking=True).flatten(1).to(torch.bool) # new added


        # contrastive learning
        contrastive_labels = torch.arange(videos.shape[0], dtype=torch.long).to(device, non_blocking=True)

        with torch.no_grad():
            # calculate the predict label
            mean          = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std           = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            unnorm_videos = videos * std + mean  # in [0, 1]

            # me: for VIDEO
            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size) # keep the same
                videos_norm    = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)') # keep the same
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

            if use_frame_diff_as_target:
                _, _, t_in, h_in, w_in                = unnorm_videos.shape
                t_tokenized, h_tokenized, w_tokenized = t_in // 2, h_in // patch_size, w_in // patch_size

                # calculate frame diff
                videos_patch      = rearrange(videos_patch, 'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)', t=t_tokenized, h=h_tokenized, w=w_tokenized, p0=2, p1=patch_size, p2=patch_size)
                videos_patch_kept = videos_patch[:,:,::frame_diff_group_size]
                videos_patch_diff = videos_patch - videos_patch_kept.repeat_interleave(frame_diff_group_size, dim=2)

                videos_patch_diff[:,:,::frame_diff_group_size] = videos_patch[:,:,::frame_diff_group_size]
                videos_patch                                   = rearrange(videos_patch_diff, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', t=t_tokenized, h=h_tokenized, w=w_tokenized, p0=2, p1=patch_size, p2=patch_size)

            B, _, C   = videos_patch.shape
            # labels  = videos_patch[bool_masked_pos].reshape(B, -1, C)
            labels    = videos_patch[~decode_masked_pos].reshape(B, -1, C) # NOTE: 这里使用了[~decode_masked_pos]

            # me: for AUDIO
            if normlize_target_audio:
                audios_squeeze = rearrange(audios, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size_audio, p2=patch_size_audio)
                audios_squeeze = (audios_squeeze - audios_squeeze.mean(dim=-2, keepdim=True)
                    ) / (audios_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                
                audios_patch   = rearrange(audios_squeeze, 'b n p c -> b n (p c)')
            else:
                audios_patch   = rearrange(audios, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size_audio, p2=patch_size_audio)

            B_audio, _, C_audio   = audios_patch.shape
            # labels_audio        = audios_patch[bool_masked_pos_audio].reshape(B_audio, -1, C_audio)
            labels_audio          = audios_patch[~decode_masked_pos_audio].reshape(B_audio, -1, C_audio) # NOTE: 这里使用了[~decode_masked_pos_audio]

        with torch.cuda.amp.autocast():
            # outputs: x, x_audio, logits_per_video, logits_per_audio
            outputs = model(videos, bool_masked_pos, decode_masked_pos, audios, bool_masked_pos_audio, decode_masked_pos_audio) # NOTE: add decoder masks

            # masked audio-visual reconstruction
            # for weighted org target + diff target (assert frame_diff_group_size==2!!!)

            if use_frame_diff_as_target and target_diff_weight is not None: # NOTE: NO USE
                print("The frame_diff_as_target and target_diff_weight are used!!!")

                labels_new  = rearrange(labels, 'b n (p0 c) -> b n p0 c', p0=2)
                outputs_new = rearrange(outputs[0], 'b n (p0 c) -> b n p0 c', p0=2)

                loss_org    = loss_func(input=outputs_new[:,:,0], target=labels_new[:,:,0]) # keep the same # 如若修改, 直接更换函数定义即可, 但是输入的数目不一样, 需要再看看
                loss_diff   = loss_func_diff(input=outputs_new[:,:,1], target=labels_new[:,:,1]) # keep the same
                loss_video  = loss_org * (1 - target_diff_weight) + loss_diff * target_diff_weight

            else: # NOTE: our choice

                # loss_video  = loss_func(input=outputs[0], target=labels) # old [2024.7.3 16.17]
                loss_video    = custom_mae_loss(outputs=outputs[0], labels=labels, bool_masked_pos=bool_masked_pos, decode_masked_pos=decode_masked_pos, B=B) # NOTE: new added for VIDEO


            # loss_audio = loss_func_audio(input=outputs[1], target=labels_audio) # only for audio
            loss_audio   = custom_mae_loss(outputs=outputs[1], labels=labels_audio, bool_masked_pos=bool_masked_pos_audio, decode_masked_pos=decode_masked_pos_audio, B=B_audio) # NOTE: new added for AUDIO

            # MAE Loss
            loss = loss_video + loss_audio

            loss_hcmcl = 0
            for logits_per_video_inter, logits_per_audio_inter in zip(outputs[2], outputs[3]):

                loss_hcmcl += 0.5 * (
                    loss_func_inter_contrastive_v(logits_per_video_inter, contrastive_labels) +
                    loss_func_inter_contrastive_a(logits_per_audio_inter, contrastive_labels)
                )

            loss = loss + loss_weight * loss_hcmcl

        #----------------------------#
        # 当loss不可用时, 训练停止
        # 也就是报nan时终止训练
        #----------------------------#
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        #----------------------#
        # 优化器清零
        # 并对参数进行更新
        #----------------------#
        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order  = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm        = loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize() # 进程同步

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]

        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        #--------------------------#
        # 对log_writer进行更新
        #--------------------------#
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    #----------------------------#
    # 从所有进程中汇聚全部的stats
    #----------------------------#
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
