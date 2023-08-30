import os
import cv2
import sys
import numpy as np
import multiprocessing



def convert(img, i, target_dir):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    new_path = os.path.join(target_dir, f'{i}.png')
    cv2.imwrite(new_path, img)
    return None


if __name__ == "__main__":
    npz = sys.argv[1]
    target_dir = sys.argv[2]

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    m = np.load(npz)
    imgs = np.array(m['arr_0'])

    print(imgs.shape[0])
    print(target_dir)

    pool = multiprocessing.Pool(processes = 64)


    for i in range(m['arr_0'].shape[0]):
        img = imgs[i]
        pool.apply_async(convert, (img, i, target_dir,))

    pool.close()
    pool.join()