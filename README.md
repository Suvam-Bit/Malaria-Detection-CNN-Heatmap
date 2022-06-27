# Malaria-Detection-CNN-Heatmap

## Streamlit App
Web-App Link: https://share.streamlit.io/anubhabpaul99/malaria-detection-cnn-heatmap/app.py


NOTE: The web-app can be tested using the given sample input images

## Dataset URL
https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

## Problem Statement

1. Detect Malaria infected cells


2. Classify the cell images in two classes
    - Parasitized
    - Uninfected


3. Highlight the infected area in the cell

## About Data
- Number of parasitized images: 13,780
- Number of uninfected images: 13,780

## Analysis
1. The analysis has been done on `Google Colab Environment`


2. Train Test Split
    - Size of train data: 80% (parasitized: 11024, uninfected: 11024)
    - Size of test data: 20% (parasitized: 2756, uninfected: 2756)


3. Model
    - Use of Transfer Learning
    - Feature Extractor: InceptionV3 (upto layer 'mixed7')
    - Activation at output layer: Softmax
    - Optimizer: Adam
    - Loss: Categorical Cross Entropy
    - Metric: Accuracy
    - Epochs: 10
    

4. Highlight Infected Area
    - Use of Class Activation Map (CAM)
    - Reference: https://keras.io/examples/vision/grad_cam/
    

5. Accuarcy: 94%

## Deployment
- Deployed the Malaria Cell Detection Model at Streamlit

## Result:
#### Input Image
![alt text](https://github.com/Suvam-Bit/Malaria-Detection-CNN-Heatmap/blob/main/sample_image_inputs/input_02.png?raw=true)

#### Output Image
![alt text](https://github.com/Suvam-Bit/Malaria-Detection-CNN-Heatmap/blob/main/sample_image_outputs/output_02.jpg?raw=true)
