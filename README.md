# Classifying Lentil Testa (Seedcoat) Phenotypes using Unsupervised Learning
This project used a dataset of BELT images and unsupervised learning techniques to classify lentils by testa
Required packages to run the source code can be found in requirements.txt
- - -
## Conversion of images from NPY to PNG
+ Code for converting images from npy to png format can be found in convert_lentils.ipynb
+ The pathnames are hardcoded and may need to be changed to reflect the directory that holds your dataset
## Segmentation and Image Processing Feature Extraction
+ Code for segmenting and extracting features can be found in LentilProject/ImageProcessingPipeline/segmentor.py and LentilProject/ImageProcessingPipeline/extraction.py
+ Both have config files to run the code
+ Can be ran from the command line using: 
>python segmentor.py -c configs/segmentor.yaml  
>python extraction.py -c configs/extractor.yaml
## Multi Layer Perceptron Autoencoder

+ The pathnames are hardcoded and may need to be changed to reflect the directory that holds your dataset
## CNN Autoencoders

## K-means Clustering
+ The pathnames are hardcoded and may need to be changed to reflect the directory that holds your dataset
## Completeness Score Calculation
+ item 1
+ item 2
* item 3
- item 4

> The highlighted section.  
> The second one. 
> the third one. 
