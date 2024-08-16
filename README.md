﻿# lung-cancer-histopathological-images-enhancer

This study aims to enhance the dataset of lung cancer histopathological images. There is already exist CNN architecture for lung cancer classification, but only have accuracy of 71%. The dataset of lung cancer histopathological images suffers from poor color constancy and contrast due to variability in image capture, management, analysis, and data collection processes. To address these issues, this study implements image enhancement techniques, including Weighted Contrast-Limited Adaptive Histogram Equalization (WCLAHE), Reinhard’s Method, and Color Deconvolution Vectors (CDV) Multi Modal. These methods will be applied to the dataset to improve color constancy and contrast. The effectiveness of each method will be evaluated by comparing each Histogram Correlation, Entropy, PSNR, PCC, dan SSIM scores for each image enhancement method and the classification accuracy of the CNN architecture that has been developed using the enhanced images as the training dataset After the evaluation, the image dataset that has been enhanced achieved a high score and The CNN Models has 80%, 83%, and 83% accuracy respectively for WCLAHE, Reinhard’s Method, dan CDV Multi Modal image enhancement method.
