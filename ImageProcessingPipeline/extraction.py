"""
Extract the color intensity for each contoured image, separately.
The results would be saved inside a json file, group-wise and in a daily bases.
"""
import os 
import argparse
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from skimage import color 
from skimage import io 

from segmentation import RoIExtraction, FeatureExtraction
from segmentation import utils



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation Configs Params.')
    parser.add_argument('-c', '--config', dest='config_path', type=str,
                        help='The string path of the config file.')
    args = parser.parse_args()
    configs = utils.config_loader(args.config_path)
    
    # RoI Extractor.
    roi_extractor = RoIExtraction(
        metadata_path=configs['metadata_path'],
        channel_number=configs['channel_number'],
        include_mirror_obj=configs['include_mirror_obj']
    )
    print('Extracting Regions of Interests...')
    rois = roi_extractor()
    
    # Feature Extractor.
    print('Extracting Texture Features...')
    feature_extractor = FeatureExtraction()
    rois['Color'] = []
    rois['Glcm'] = []
    rois['Gccrop'] = []
    rois['Lbp'] = []
    for real_region in rois['Real']:
        region_gray = (255 * color.rgb2gray(real_region)).astype(np.uint8)

        region_meaned = np.zeros((real_region.shape[0], real_region.shape[1]))
        # get average values, 3 channels into 1 channel
        for i in range(0, real_region.shape[0]):
            for j in range(0, real_region.shape[1]):
                region_meaned[i][j] = ((int(real_region[i, j, 0]) + int(real_region[i, j, 1]) + int(real_region[i, j, 2])) / 3)

        rois['Color'].append(
            feature_extractor.histogram_extractor(
                features=region_meaned
            ).tolist()
        )
        glcm_feat = feature_extractor.glcm(
                image=region_gray,
                distances=[1, 5, 10],
                angles=[0, np.pi/2],
                levels=256
        )

        rois['Glcm'].append(
            glcm_feat.mean(axis=0).flatten().tolist()
        )
        rois['Gccrop'].append(
            feature_extractor.gccrop(
                features=glcm_feat,
                props=['contrast', 'dissimilarity', 'homogeneity', 'energy']
            ).tolist()
        )
        rois['Lbp'].append(
            feature_extractor.lbp(
                image=region_gray,
                p=8,
                r=1
            ).tolist()
        )

    os.makedirs(os.path.join(configs['out_dir'], 'real_regions'), exist_ok=True)
    rois['RoIRegionPath'] = []
    for i, (real_region, path) in enumerate(zip(rois['Real'], rois['Image'])): 
        reg_pth = os.path.join(configs['out_dir'], 'real_regions', 
                               f"{os.path.basename(path)[:-4]}.{configs['out_real_path']}")
        io.imsave(reg_pth, real_region, check_contrast=False)
        rois['RoIRegionPath'].append(reg_pth)
    # Remove extra infos. 
    del rois['Real']
    del rois['Label']
    del rois['Mirror']
    del rois['Image']
    rois['Image'] = rois.pop('RoIRegionPath')


    df = pd.DataFrame()
    df['Color'] = rois['Color']
    df['Glcm'] = rois['Glcm']
    df['Gccrop'] = rois['Gccrop']
    df['Lbp'] = rois['Lbp']
    df['Image'] = rois["Image"]

    print(type(df.iloc[0].Color))
    print(len(df.iloc[0].Color))

    df.to_csv(configs['real_metadata_features_path'], index=False)