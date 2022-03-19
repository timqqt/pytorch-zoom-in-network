This is the official repository for "[Efficient Classification of Very Large Images with Tiny Objects](https://arxiv.org/abs/2106.02694)".

# Overview
An increasing number of applications in computer vision, specially, in medical imaging and remote sensing, become challenging when the goal is to classify very large images with tiny informative objects. 
Specifically, these classification tasks face two key challenges: $i$) the size of the input image is usually in the order of mega- or giga-pixels, however, existing deep architectures do not easily operate on such big images due to memory constraints, consequently, we seek a memory-efficient method to process these images; and $ii$) only a very small fraction of the input images are informative of the label of interest, resulting in low region of interest (ROI) to image ratio.
However, most of the current convolutional neural networks (CNNs) are designed for image classification datasets that have relatively large ROIs and small image sizes (sub-megapixel).
Existing approaches have addressed these two challenges in isolation.
We present an end-to-end CNN model termed Zoom-In network that leverages hierarchical attention sampling for classification of large images with tiny objects using a single GPU.
We evaluate our method on four large-image histopathology, road-scene and satellite imaging datasets, and one gigapixel pathology dataset.
Experimental results show that our model achieves higher accuracy than existing methods while requiring less memory resources.
## Description
This code file includes the training and evaluating scripts for all experiments described in the paper. 

## Major Dependencies
pytorch-gpu == 1.6.0
torchvision == 0.7.0 

## Training
training on colon cancer dataset:

    python3 main.py /path_to_your_dataset/ /path_to_your_output/ --mode 10CrossValidation
    
## Evaluation

    python3 main.py /path_to_your_dataset/ /path_to_your_output/ --mode Evaluation --load_model /model_name/
    
## Acknowledgemennt
This work was supported by NIH (R44-HL140794), DARPA (FA8650-18-2-7832-P00009-12) and ONR (N00014-18-1-2871-P00002-3).

##### Thanks to the following repositories: 
- https://github.com/idiap/attention-sampling.git
