"""
Segment input images.
"""

import argparse
import os

import SimpleITK as sitk
import numpy as np
import pandas as pd
from skimage import color, filters
from skimage import io

from segmentation import Morphology
from segmentation import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation Configs Params.')
    parser.add_argument('-c', '--config', dest='config_path', type=str,
                        help='The string path of the config file.')
    args = parser.parse_args()
    configs = utils.config_loader(args.config_path)

    metadata = pd.read_csv(configs['metadata_path'])
    os.makedirs(configs['out_dir'], exist_ok=True)

    # Define a morphology object.
    morphology = Morphology(
        disk_radius=12,
        area_threshold=1024,
        min_size=2048,
        connectivity=2,
        threshold_method=filters.threshold_otsu,
        color_space_converter=color.rgb2hsv,
        channel_number=2,
        num_kmean_clusters=None,
        object_mode='dark'
    )
    region_crop = utils.RegionCrop(
        height=(0, 1400),
        width=(200, 2200)
    )
    composer = utils.Compose([
        morphology.apply_thresholding,
        morphology.apply_opening,
        morphology.apply_rm_small_objects,
        morphology.apply_rm_small_holes
    ])

    os.makedirs(os.path.join(configs['out_dir'], 'images'), exist_ok=True)
    os.makedirs(os.path.join(configs['out_dir'], 'masks'), exist_ok=True)

    segmented_paths = {
        'Image': [],
        'Mask': []
    }
    for i, path in enumerate(metadata['Image'].tolist()):
        image = io.imread(path)
        image = region_crop(image)
        
        mask = composer(image)

        if configs['visualize'] == True:
            utils.masked_image_visualizer(image, mask)
        
        seg_img_pth = os.path.join(configs['out_dir'], 'images',
                         f"{i:0>4}_{os.path.basename(path)[:-4]}{configs['img_ext']}")
        seg_msk_pth = os.path.join(configs['out_dir'], 'masks',
                         f"{i:0>4}_{os.path.basename(path)[:-4]}{configs['msk_ext']}")
        
        io.imsave(seg_img_pth, image, check_contrast=False)
        io.imsave(seg_msk_pth, mask.astype(np.uint8), check_contrast=False)
        
        segmented_paths['Image'].append(seg_img_pth)
        segmented_paths['Mask'].append(seg_msk_pth)
        
        print('Processed: ', path)
    # segmented_paths['Label'] = metadata['Label'].tolist()
    df = pd.DataFrame(segmented_paths)
    df.to_csv(configs['segmented_metadata_path'], index=False)