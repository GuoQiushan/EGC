# EGC: Image Generation and Classification via a Single Energy-Based Model (ICCV 2023)

#### <p align="center"><a href="https://guoqiushan.github.io/egc.github.io/">Project Page</a> | <a href="https://arxiv.org/abs/2304.02012">Paper</a> | <a href="https://arxiv.org/abs/2304.02012">ArXiv</a></p>

![avatar](./doc/overview.png)

# Download pre-trained models

We have some released checkpoints for the main models in the paper. 

Here are the download links for each model checkpoint:
 * 256x256 ImageNet (78.97% Top-1 Acc + FID 6.05): [256x256_EGC.pt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007956_connect_hku_hk/EWdM8xRsLcRHjm8Sp0zDGeYBAB-3mu_PEDlX4ex4bFcdkQ)
 * 256x256 ImageNet (72.39% Top-1 Acc + FID 6.77): [256x256_EGC.pt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007956_connect_hku_hk/EQ_6FEQ8VdZPrDeZse3okmEB0mSupZz4gmJMqjKD86MOHQ)

More checkpoints and training scripts will be released soon.

# Sampling from pre-trained models

To sample from 256x256 ImageNet EGC model, you can use the `run_imagenet_egc_latent_sample_cond.sh`.
Here, we provide flags for sampling from all of these models.

For this example, we will generate 50000 samples with batch size 8 and 100 ddim steps. Feel free to change these values.

```
OPT="--batch_size 8 --num_samples 50000 --use_ddim True --timestep_respacing ddim100"
```

The classifier guidiance scale `--classifier_scale` is be recommended to set as `6.0` to reproduce the FID score. 

# Image Classification with pre-trained models

To test the image classification performance of 256x256 ImageNet EGC model, you can use the `run_imagenet_egc_eval_cls.sh`.

For this example, you can run the following cmd:

```
./run_imagenet_ldm_eval_cls.sh $LOGDIR 1 1 0 127.0.0.1 $CKPT_PATH --val_data_dir=./data/imagenet256_features_val
```

# Training models
To reproduce the ImageNet result, you can run:

```
./run_imagenet_egc_train.sh $LOGDIR $GPU_NUM $NODE_NUN $RANK $MASTER_ADDR
```

Make sure that the `GPU_NUM * NODE_NUN * batch_size = 512` and `batch_size = cls_batch_size` in the shell script. You may change `microbatch` to reduce the memory cost.

## Prepare ImageNet data

The ImageNet-1k dataset should be organized as following:

```text
EGC
├── data
│   ├── imagenet
│   │   ├── train/
│   │   ├── val/
│   ├── imagenet256_features/
│   ├── imagenet256_features_val/
```

Besides, download the [autoencoder_kl.pth](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007956_connect_hku_hk/EV3hjamqcHZDofc3Usjwy-QBpeZ2mppTAzASMppjhsf16g?e=Qpf6xd) to the EGC folder. 

Convert the raw image to latent space, using `python scripts/extract_feat.py ./data/imagenet/train ./autoencoder_kl.pth ./data/imagenet/imagenet256_features`.

Convert the raw image to latent space, using `python scripts/extract_feat.py ./data/imagenet/val ./autoencoder_kl.pth ./data/imagenet/imagenet256_features_val`.



# Cite
If you find EGC useful for your work, please cite:

```bibtex
@article{guo2023egc,
  title={EGC: Image Generation and Classification via a Single Energy-Based Model},
  author={Guo, Qiushan and Ma, Chuofan and Jiang, Yi and Yuan, Zehuan and Yu, Yizhou and Luo, Ping},
  journal={arXiv preprint arXiv:2304.02012},
  year={2023}
}
```

# Acknowledgement
This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion/), with modifications for energy-based training and sampling and architecture improvements. Thanks for their wonderful works.