import os
import os.path
import sys
from multiprocessing import Pool
import numpy as np
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.progress_bar import ProgressBar
from pathlib import Path


ALLOWED_EXT = [".png",".jpg",".JPG",".bmp"]

def main():
    """A multi-thread tool to crop sub imags."""
    # input_folder = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800'
    # save_folder = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub'
    input_sources = {
        'pristine':'/data/deva/pristine_dataset',
        # 'OST': '/data/deva/OutdoorSceneTrain_v2',
        #'Flickr2k_HR': '/data/deva/Flickr2K/Flickr2K_HR',
    }
    save_folder = Path('/home/deva/pristine_dataset_train_HR_crop_128')

    n_thread = 20
    crop_sz = 128
    step = 64
    thres_sz = 48
    compression_level = 0  # 3 is the default value in cv2
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.


    image_list = []
    for input_src in input_sources:
        print(f"[INFO] {input_src}")
        dir_path = Path(input_sources[input_src])
        if not dir_path.is_dir():
            print(f"[ERROR] {dir_path} does not exists")
            continue
        for ext in ALLOWED_EXT:
            path_list = list(dir_path.glob(f"**/*{ext}"))
            print(f"[INFO] found {len(path_list)} file in {dir_path} with {ext} extension")
            for image_path in path_list:
                image_list.append((input_src,image_path))

    # image_list = image_list[:10]
    print(f"[INFO] initiating processing of {len(image_list)} images")

    save_folder.mkdir(exist_ok=True,parents=True)

    def update(arg):
        pbar.update(arg)
    '''
    pbar = ProgressBar(len(img_list))

    pool = Pool(n_thread)
    for path in img_list:
        pool.apply_async(worker,
            args=(path, save_folder, crop_sz, step, thres_sz, compression_level),
            callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')
    '''
    pbar = ProgressBar(len(image_list))

    pool = Pool(n_thread)
    for img_src in image_list:
        A = worker(img_src, save_folder, crop_sz, step, thres_sz, compression_level)
        update(A)
    print('All subprocesses done.')


def worker(img_src, save_folder, crop_sz, step, thres_sz, compression_level):
    dataset_key,img_path = img_src
    img_name = img_path.with_suffix('').name
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))
    if h < crop_sz or w < crop_sz:
        return ".."

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            # var = np.var(crop_img / 255)
            # if var > 0.008:
            #     print(img_name, index_str, var)
            img_output_path = save_folder/f"{dataset_key}_{img_name}_{crop_sz}_s{index:03d}.png"

            cv2.imwrite(
                str(img_output_path),
                crop_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    return f'Processed {dataset_key} {img_path} ...'


if __name__ == '__main__':
    main()
