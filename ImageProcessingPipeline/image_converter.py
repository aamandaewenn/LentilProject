import os  
import glob

from yaml import CollectionNode 
import numpy as np  
import pandas as pd

from skimage import io  
from typing import List, Tuple, Dict 

import time 


def converter(image_path, out_pth): 
    image = np.load(image_path)                                                                                                                                                                                                                                                                                                
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    np.save(out_pth, image)

def preprocessing(collection: List, out_dir: str, out_ext: str):
    assert out_ext.startswith('.')
    os.makedirs(out_dir, exist_ok=True)
    not_processed = []
    for image_path in collection:
        print('processed: ', image_path)
        os.makedirs(os.path.join(out_dir, *os.path.dirname(image_path).split('/')[-1:]), exist_ok=True)
        try: 
            converter(image_path, os.path.join(out_dir, *image_path.split('/')[-2:]))
        except: 
            print(image_path)

if __name__ == '__main__': 
    dirs = sorted(glob.glob('/Volumes/My Passport/BELT 2022/2020 PGRC/*', recursive=True))
    print(len(dirs))

    for item in dirs: 
        if os.path.isdir(item): 
            collection = sorted(glob.glob(os.path.join(item, '*.npy'), recursive=True))
            if len(collection) > 0: 
                preprocessing(
                    collection=collection,
                    out_dir = 'data/AlejandraVersion/',
                    out_ext='.npy'
                )