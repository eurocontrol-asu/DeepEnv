"""
File Name: default_config.py
Author: Gabriel JARRY, Valentin TORDJMAN--LEVAVASSEUR, Philippe VERY
Contact Information: {gabriel.jarry, philippe.very}@eurocontrol.int
Date Created: July 2023
Last Modified : December 2023
Description : 

This Python module part of the DeepContrail library, serves as a configuration hub, specifically designed for managing and 
storing various types of data and model architectures. It defines constants for directories where raw data, gold standard data, model data, 
and evaluation data are stored, streamlining data management and access. The module also enumerates a comprehensive list of available model 
architectures and backbones, encompassing a wide range of models from simple U-Nets to complex architectures like Segformer and DeepLabV3. 
This extensive list includes both traditional architectures and more recent advancements in the field, offering a robust selection for different 
machine learning tasks. The inclusion of various backbone models, including pre-trained architectures and custom models, enhances the flexibility 
and adaptability of the module for different applications.
"""



# Paths to different directories used for storing raw data, gold standard data, model data, and evaluation data.
RAW_DIR = "/data/common/KAGGLE/data/RAW/" # Folder with raw band data
GOLD_DIR = "/data/common/KAGGLE/data/GOLD/" # Folder with ash rgb data
MODEL_DIR = "/data/common/KAGGLE/models/tempo/"
EVALUATION_DIR = "/data/common/KAGGLE/models/evaluated/"

# List of available model architectures for machine learning or data processing tasks.
AVAILABLE_ARCHITECTURES = [
    "TemporalMixing", "Upernet", "OpticalFlowModel", "LightVideoModel", "Unet", "UnetPlusPlus", "MAnet", "Linknet", "FPN", "PSPNet", 
    "DeepLabV3", "DeepLabV3Plus", "PAN", "Unet3d", "DeepLabV3Plus3D", "Segformer", "MaskUnet3D"
]

# List of available backbones, which are pre-trained models or architectures, that can be used in conjunction with the model architectures.
AVAILABLE_BACKBONES = ["tu-vit_base_patch16_224.sam_in1k","tu-maxvit_tiny_tf_384.in1k",
                 "tu-maxvit_xlarge_tf_512.in21k_ft_in1k", "tu-maxvit_base_tf_512.in21k_ft_in1k",
                 "tu-resnest26d", "tu-tf_efficientnetv2_s.in21k_ft_in1k","tu-coat_lite_medium.in1k",
                 "tu-coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",'tu-maxvit_tiny_tf_512.in1k',"tu-efficientnetv2_m","tu-efficientnetv2_s","openmmlab/upernet-swin-small",
                 "openmmlab/upernet-swin-base","openmmlab/upernet-swin-large","openmmlab/upernet-swin-tiny",
                 "openmmlab/upernet-convnext-small","openmmlab/upernet-convnext-base","openmmlab/upernet-convnext-large",
                 "openmmlab/upernet-convnext-tiny","tu-tf_efficientnet_b8","tu-tf_efficientnet_b5","tu-tf_efficientnet_b6",
                 "tu-tf_efficientnet_b7","tu-efficientnet_b5","tu-efficientnet_b5","tu-maxvit_base_tf_512",
                 "tu-seresnextaa101d_32x8d","tu-coatnet_rmlp_2_rw_384",
                 "custom_resnet3d","monai", 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d',
                 'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_32x16d', 'resnext101_32x32d', 
                 'resnext101_32x48d', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 'vgg11',
                 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'senet154',
                 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d',
                 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'inceptionresnetv2', 'inceptionv4',
                 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4',
                 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 'mobilenet_v2', 'xception',
                 'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2', 'timm-efficientnet-b3',
                 'timm-efficientnet-b4', 'timm-efficientnet-b5', 'timm-efficientnet-b6', 'timm-efficientnet-b7',
                 'timm-efficientnet-b8', 'timm-efficientnet-l2', 'timm-tf_efficientnet_lite0',
                 'timm-tf_efficientnet_lite1', 'timm-tf_efficientnet_lite2', 'timm-tf_efficientnet_lite3',
                 'timm-tf_efficientnet_lite4', 'timm-resnest14d', 'timm-resnest26d', 'timm-resnest50d',
                 'timm-resnest101e', 'timm-resnest200e', 'timm-resnest269e', 'timm-resnest50d_4s2x40d', 
                 'timm-resnest50d_1s4x24d', 'timm-res2net50_26w_4s', 'timm-res2net101_26w_4s', 
                 'timm-res2net50_26w_6s', 'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s',
                 'timm-res2net50_14w_8s', 'timm-res2next50', 'timm-regnetx_002', 'timm-regnetx_004',
                 'timm-regnetx_006', 'timm-regnetx_008', 'timm-regnetx_016', 'timm-regnetx_032', 
                 'timm-regnetx_040', 'timm-regnetx_064', 'timm-regnetx_080', 'timm-regnetx_120', 
                 'timm-regnetx_160', 'timm-regnetx_320', 'timm-regnety_002', 'timm-regnety_004', 
                 'timm-regnety_006', 'timm-regnety_008', 'timm-regnety_016', 'timm-regnety_032', 
                 'timm-regnety_040', 'timm-regnety_064', 'timm-regnety_080', 'timm-regnety_120', 
                 'timm-regnety_160', 'timm-regnety_320', 'timm-skresnet18', 'timm-skresnet34', 
                 'timm-skresnext50_32x4d', 'timm-mobilenetv3_large_075', 'timm-mobilenetv3_large_100',
                 'timm-mobilenetv3_large_minimal_100', 'timm-mobilenetv3_small_075', 'timm-mobilenetv3_small_100',
                 'timm-mobilenetv3_small_minimal_100', 'timm-gernet_s', 'timm-gernet_m', 'timm-gernet_l', 
                 'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5', 'mobileone_s0', 'mobileone_s1',
                 'mobileone_s2', 'mobileone_s3', 'mobileone_s4',"resnet3d","nvidia/segformer-b0-finetuned-ade-512-512","nvidia/segformer-b1-finetuned-ade-512-512","nvidia/segformer-b2-finetuned-ade-512-512","nvidia/segformer-b3-finetuned-ade-512-512","nvidia/segformer-b4-finetuned-ade-512-512","nvidia/segformer-b5-finetuned-ade-512-512"]
                 
    