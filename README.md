# Classifying Lentil Testa (Seedcoat) Phenotypes using Unsupervised Learning
This project used a dataset of BELT images and unsupervised learning techniques to classify lentils by testa   
Trained weights for machine learning models can be found [here](https://drive.google.com/drive/folders/1wXH5kVpuuVro9x5_Y_LRWXuJ-TFzmt4p)  
Required packages to run the source code can be found in requirements.txt and installed with:
>pip install -r requirements.txt  
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
+ Code for training MLP autoencoder and encoding previously extracted features can be found in LentilProject/LentilAutoEncoder.ipynb
+ The pathnames are hardcoded and may need to be changed to reflect the directory that holds your dataset
## CNN Autoencoders
+ Code for training shallow CNN autoencoder and encoding images can be found in LentilProject/LentilCNNAutoEncoder.ipynb
+ Code for training VGG transfer network autoencoder and encoding images can be found in LentilProject/VGGTransfer-kn.ipynb
+ The pathnames are hardcoded and may need to be changed to reflect the directory that holds your dataset
## K-means Clustering
+ Pipeline for clustering images using a .csv file of extracted features can be found in LentilProject/clustering pipeline.ipynb
+ The pathnames are hardcoded and may need to be changed to reflect the directory that holds your dataset
## Completeness Score Calculation
+ Code for calculating Completeness Score from .csv file of clustering results can be found in LentilProject/CalculateCompleteness.ipynb
+ Original dataset must be labelled, you may need to change code to reflect how your dataset is labelled
+ The pathnames are hardcoded and may need to be changed to reflect the directory that holds your dataset
