from autoencoder import FrozenAutoencoderKL
import os, sys, torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np



ddconfig_f4 = dict(
    double_z=True,
    z_channels=3,
    resolution=256,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 2, 4],
    num_res_blocks=2,
    attn_resolutions=[],
    dropout=0.0
)

ddconfig_f8 = dict(
    double_z=True,
    z_channels=4,
    resolution=256,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    attn_resolutions=[],
    dropout=0.0
)
T = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])

model = FrozenAutoencoderKL(ddconfig_f8, 4, sys.argv[2], 0.18215).cuda()
nlist = sorted(os.listdir(sys.argv[1]))
cnt = 0
for i, dirname in enumerate(nlist):
    subdir = os.path.join(sys.argv[1], dirname)
    for name in sorted(os.listdir(subdir)):
        path = os.path.join(subdir, name)
        img = Image.open(path).convert('RGB')
        img = T(img)
        img = img * 2. - 1
        img = img[None, ...]
        img = img.to('cuda')

        moments = model(img, fn='encode_moments')
        moments = moments.detach().cpu().numpy()
    
        for moment in moments:
            np.save(os.path.join(sys.argv[3], str(cnt)+'.npy'), (moment, np.int64(i)))
            cnt += 1