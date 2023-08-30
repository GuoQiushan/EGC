"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

from tqdm import tqdm
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    egc_model_and_diffusion_defaults,
    create_egc_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.image_datasets import get_val_ldm_data

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_egc_model_and_diffusion(
        **args_to_dict(args, egc_model_and_diffusion_defaults().keys())
    )
    
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    val_data = get_val_ldm_data(
        data_dir=args.val_data_dir,
        img_num=50000,
        batch_size=args.val_batch_size,
        class_cond=True,
        double_z=args.double_z,
        sample_z=args.sample_z,
        scale_factor=args.scale_factor,
    )

    test_corrects = []
    for image, label in tqdm(val_data):
        image = image.to(dist_util.dev())
        label = label['y'].to(dist_util.dev())
        with th.no_grad():
            time = th.zeros([image.shape[0]], dtype=th.long, device=dist_util.dev())
            pred = model(image, time, cls_mode=True)
            correct = (pred.max(1)[1] == label).float()
            test_corrects.append(correct)
    
    test_corrects = th.cat(test_corrects)
    test_acc = float(test_corrects.mean() * 100)
    
    logger.log(f"Test Acc: {test_acc}")


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        batch_size=16,
        use_ddim=False,
        model_path="",
        autoencoder_path="",
        local_rank=0,
        autoencoder_type = 'KL',
        autoencoder_stride='8',
        sample_z=False,
        double_z=True,
        scale_factor=0.18215,
        val_data_dir='',
        val_batch_size=16,
    )
    defaults.update(egc_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
