import copy
import functools
import os
from os.path import basename
import numpy as np

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from scripts.autoencoder import get_model
import wandb
from torchvision.utils import make_grid, save_image

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        data_cls,
        val_data,
        batch_size,
        cls_batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        max_steps=1000000,
        eval_interval = 5000,
        ce_weight = 0,
        label_smooth=0.0,
        use_hdfs = False,
        grad_clip = 1.0,
        local_rank = 0,
        autoencoder_path=None,
        betas=(0.9, 0.999),
        cls_cond_training = False,
        train_classifier=False,
        scale_factor=0.18215,
        autoencoder_stride=8,
        autoencoder_type='KL',
        warm_up_iters=-1,
        encode_online=False,
        encode_cls=False,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.data_cls = data_cls
        self.val_data = val_data
        self.cls_batch_size = cls_batch_size
        self.ce_weight = ce_weight
        self.eval_interval = eval_interval
        self.warm_up_iters = warm_up_iters

        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.max_steps = max_steps
        self.label_smooth = label_smooth
        self.use_hdfs = use_hdfs
        self.grad_clip = grad_clip
        self.local_rank = local_rank
        self.autoencoder = None
        self.betas = betas
        self.cls_cond_training = cls_cond_training
        self.train_classifier = train_classifier
        self.encode_online = encode_online
        self.encode_cls = encode_cls
        if autoencoder_path is not None:
            self.autoencoder = get_model(autoencoder_path, autoencoder_type, autoencoder_stride, scale_factor).cuda()

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        dist.barrier()
        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            grad_clip=self.grad_clip,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay, betas=self.betas,
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            (not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps) 
            and (self.step + self.resume_step < self.max_steps)
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)

            if self.step > 0 and \
                (self.step + self.resume_step) % self.eval_interval == 0 and \
                self.ce_weight > 0.0 and \
                self.val_data is not None:

                self.eval_cls()

            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                self.model.train()

                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def eval_cls(self):
        self.ddp_model.eval()

        def gather(obj_to_gather):
            obj_sum = obj_to_gather.sum()
            obj_num = obj_to_gather.numel()
            buffer = th.zeros(2).cuda()
            buffer[0] = obj_sum
            buffer[1] = obj_num

            dist.all_reduce(buffer)
            mean = buffer[0] / buffer[1]
            return mean

        with th.no_grad():
            test_corrects, test_losses = [], []
            for img, labeldict in self.val_data:
                bs = img.shape[0]
                sub_labels = labeldict['y'].to(dist_util.dev())
                img = img.to(dist_util.dev())
                if self.encode_online:
                    with th.no_grad():
                        img = self.autoencoder.encode(img)

                time = th.zeros(bs, device=dist_util.dev()).float()
                logits = self.ddp_model(img, time, cls_mode=True)

                loss_test = th.nn.functional.cross_entropy(logits, sub_labels, reduction="none")
                test_losses.append(loss_test)

                correct = (logits.max(1)[1] == sub_labels).float()
                test_corrects.append(correct)

            test_losses = th.cat(test_losses)
            test_corrects = th.cat(test_corrects)

            loss_test = gather(test_losses)
            correct = gather(test_corrects)
                        
            logger.logkv_mean(f"loss test", loss_test)
            logger.logkv_mean(f"test acc", correct)

            if dist.get_rank() == 0:
                wandb.log({'loss_test': loss_test, 'test_acc': correct}, step=self.step + self.resume_step)

        self.ddp_model.train()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        
        if self.warm_up_iters > 0:
            frac = (self.step + self.resume_step) / self.warm_up_iters
            lr = self.lr * (frac if frac < 1 else 1)
            for param_group in self.opt.param_groups:
                param_group["lr"] = lr
                
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()

        if self.cls_batch_size > 0:
            batch_cls_t, cond_cls_t = next(self.data)
            batch_cls, cond_cls = next(self.data_cls)
            
            micro_cls = batch_cls.to(dist_util.dev())
            if self.encode_online or self.encode_cls:
                with th.no_grad():
                    micro_cls = self.autoencoder.encode(micro_cls)
                
            micro_cond_cls = {
                k: v.to(dist_util.dev())
                for k, v in cond_cls.items()
            }

            micro_cls_t = batch_cls_t.to(dist_util.dev())
            micro_cond_cls_t = {
                k: v.to(dist_util.dev())
                for k, v in cond_cls_t.items()
            }
            

            # sample noised sample
            t, weights = self.schedule_sampler.sample(micro_cls_t.shape[0], dist_util.dev())
            noise = th.randn_like(micro_cls_t)
            micro_cls_noised = self.diffusion.q_sample(micro_cls_t, t, noise=noise)

            t_cls = np.random.choice(1, size=(micro_cls.shape[0],))
            t_cls = th.from_numpy(t_cls).long().to(dist_util.dev())
            micro_cls_clean = micro_cls

            micro_cls = th.cat([micro_cls_noised, micro_cls_clean]).detach()
            t_cls = th.cat([t, t_cls])

            sqrt_alphas_cumprod = th.from_numpy(self.diffusion.sqrt_alphas_cumprod).to(device=t_cls.device)[t_cls].float()
            
            with self.ddp_model.no_sync():
                logits_cls = self.ddp_model(micro_cls, t_cls, cls_mode=True)
                cls_gt = th.cat([micro_cond_cls_t['y'], micro_cond_cls['y']])
                loss_cls = th.nn.CrossEntropyLoss(label_smoothing=self.label_smooth, reduction='none')(logits_cls, cls_gt)
                loss_cls = (loss_cls * sqrt_alphas_cumprod).mean()

                logger.logkv_mean("loss_ce_x0", loss_cls.item())
                loss_cls = self.ce_weight * loss_cls
                self.mp_trainer.backward(loss_cls)

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            if self.encode_online:
                with th.no_grad():
                    micro = self.autoencoder.encode(micro)

            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            micro_cond['label_smooth'] = self.label_smooth
            micro_cond['train_classifier'] = False 
            
            rand_idx = th.rand(micro_cond['y'].shape, device=dist_util.dev()) < 0.1
            micro_cond['y'][rand_idx] = self.model.num_classes

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if last_batch or not self.use_ddp:
                self.mp_trainer.backward(loss)
            else:
                with self.ddp_model.no_sync():
                    self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

            if self.use_hdfs:
                hdfs_step = self.step + self.resume_step
                with open('hdfs_step.txt', 'w') as f:
                    f.write(str(hdfs_step))

                os.system('hdfs dfs -mkdir hdfs://haruna/home/byte_arnold_hl_vc/user/qsguo/{}'.format(get_blob_logdir()))
                os.system('hdfs dfs -put -f hdfs_step.txt hdfs://haruna/home/byte_arnold_hl_vc/user/qsguo/{}'.format(get_blob_logdir()))

                filename = f"model{(self.step+self.resume_step):06d}.pt"
                hdfs_cmd = 'hdfs dfs -put {} hdfs://haruna/home/byte_arnold_hl_vc/user/qsguo/{} & \n'.format(bf.join(get_blob_logdir(), filename), get_blob_logdir())
                for ema_rate in self.ema_rate:
                    filename = f"ema_{ema_rate}_{(self.step+self.resume_step):06d}.pt"
                    hdfs_cmd += 'hdfs dfs -put {} hdfs://haruna/home/byte_arnold_hl_vc/user/qsguo/{} & \n'.format(bf.join(get_blob_logdir(), filename), get_blob_logdir())
                
                filename = f"opt{(self.step+self.resume_step):06d}.pt"
                hdfs_cmd += 'hdfs dfs -put {} hdfs://haruna/home/byte_arnold_hl_vc/user/qsguo/{} & \n'.format(bf.join(get_blob_logdir(), filename), get_blob_logdir())
                os.system(hdfs_cmd)



def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    ckpt_steps = -1
    for name in bf.listdir(get_blob_logdir()):
        if 'model' in name:
            basename = name.split('.')[0]
            step = int(basename[5:])
            ckpt_steps = max(step, ckpt_steps)

    if ckpt_steps > 0:
        ckpt_steps = str(ckpt_steps).zfill(6)
        ckpt_path = bf.join(get_blob_logdir(), f'model{ckpt_steps}.pt')
    else:
        ckpt_path = None
        
    return ckpt_path


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for quartile in range(4):
            logger.get_current().name2val[f"{key}_q{quartile}"] = 0
            logger.get_current().name2cnt[f"{key}_q{quartile}"] = 0
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
