"""
File Name: image_preprocessing.py
Author: Gabriel JARRY, Valentin TORDJMAN--LEVAVASSEUR, Philippe VERY
Contact Information: {gabriel.jarry, philippe.very}@eurocontrol.int
Date Created: July 2023
Last Modified : December 2023
Description : 

This Python module offers a comprehensive suite of functions for image preprocessing and data manipulation, 
particularly tailored for handling both individual images and video sequences. The module leverages libraries 
such as OpenCV (cv2), NumPy, PyWavelets (pywt), Torch, and Joblib to perform a variety of tasks, including image 
shifting, concatenation, normalization, false color image generation, resizing with Fourier and wavelet transforms, 
and loading and preprocessing of image data.
"""

import importlib

import os
import cv2
import torch
import pywt
import numpy as np

from typing import List, Union, Tuple

from joblib import delayed, Parallel

from DeepContrail.config.default_config import RAW_DIR, GOLD_DIR


def shift_bottom_right(image: Union[np.ndarray, List[np.ndarray]], 
                    is_video: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Applies a bottom-right shift to an image or a sequence of images.

    This function shifts the image(s) slightly to the bottom right using affine transformation.
    It can handle both individual images and a sequence of images (in case of videos).

    Parameters:
    image (Union[np.ndarray, List[np.ndarray]]): The image or sequence of images to be shifted.
    is_video (bool): A flag indicating whether the input is a video (sequence of images). 
                    Defaults to False.

    Returns:
    Union[np.ndarray, List[np.ndarray]]: The shifted image or sequence of images.
    """
    # Define affine transformation matrix for bottom-right shift
    affine_matrix = np.array([[1, 0.0, 0.5], [0, 1, 0.5]], dtype=np.float64)

    # Process a sequence of images if it's a video
    if is_video: 
        for i in range(image.shape[0]):
            image[i] = cv2.warpAffine(
                image[i],
                affine_matrix,
                (image.shape[-2], image.shape[-2]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
    # Process a single image
    else: 
        image = cv2.warpAffine(
            image,
            affine_matrix,
            (image.shape[-2], image.shape[-2]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return image


def concat_panel_images(image_sequence: List[torch.Tensor]) -> torch.Tensor:
    """
    Concatenates a sequence of images into a single image.

    This function concatenates images both horizontally and vertically to form a single composite image.

    Parameters:
    image_sequence (List[torch.Tensor]): A list of image tensors to be concatenated.

    Returns:
    torch.Tensor: The resulting concatenated image tensor.
    """
    # Concatenate first two images in the sequence horizontally
    horizontal_concat_top = torch.cat((image_sequence[:, 0], image_sequence[:, 1]), axis=-1)

    # Concatenate next two images in the sequence horizontally
    horizontal_concat_bottom = torch.cat((image_sequence[:, 2], image_sequence[:, 3]), axis=-1)

    # Concatenate the above two horizontal concatenations vertically
    vertical_concat = torch.cat((horizontal_concat_top, horizontal_concat_bottom), axis=-2)

    return vertical_concat


def normalize(band: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    """
    Normalize an input array based on given bounds.

    Parameters:
        band (np.ndarray): Input array to be normalized.
        bounds (Tuple[float, float]): Lower and upper bounds for normalization.

    Returns:
        np.ndarray: Normalized array.
    """
    return (band - bounds[0]) / (bounds[1] - bounds[0])   


def false_color(band11: np.ndarray, band14: np.ndarray, band15: np.ndarray, axis: int = 2) -> np.ndarray:
    """
    Generate false-color image using three input bands.

    Parameters:
        band11 (np.ndarray): Input array for band 11.
        band14 (np.ndarray): Input array for band 14.
        band15 (np.ndarray): Input array for band 15.
        axis (int, optional): Axis along which the arrays are stacked. Defaults to 2.

    Returns:
        np.ndarray: False-color image array.
    """
    # Bounds for normalization
    _T11_BOUNDS = (243, 303)
    _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
    _TDIFF_BOUNDS = (-4, 2)

    # Calculate the normalized values for each color channel
    r = normalize(band15 - band14, _TDIFF_BOUNDS)
    g = normalize(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize(band14, _T11_BOUNDS)

    # Clip the values between 0 and 1 and stack the channels
    return np.clip(np.stack([r, g, b], axis=axis), 0, 1)


def resize_fft(image: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Resizes an image using Fourier Transform based padding.

    This function increases the size of an image to the target size by applying 
    padding in the frequency domain using the Fourier Transform. The image is first
    transformed into the frequency domain, padded, and then transformed back.

    Parameters:
        image (torch.Tensor): The input image tensor.
        target_size (int): The target size to resize the image to.

    Returns:
        torch.Tensor: The resized image tensor.
    """

    # Calculate the amount of padding needed for each side
    padding_amount = (target_size - image.shape[-1]) // 2

    # Apply Fourier Transform to the image
    fft_image = torch.fft.fftshift(torch.fft.fft2(image, norm="forward"))

    # Separate the real and imaginary parts
    fft_real = fft_image.real
    fft_imag = fft_image.imag

    # Apply padding to the real and imaginary parts
    fft_real_padded = torch.nn.functional.pad(fft_real, (padding_amount, padding_amount, padding_amount, padding_amount), mode='constant', value=0)
    fft_imag_padded = torch.nn.functional.pad(fft_imag, (padding_amount, padding_amount, padding_amount, padding_amount), mode='constant', value=0)

    # Combine the real and imaginary parts and apply inverse Fourier Shift
    fft_padded = torch.fft.ifftshift(torch.complex(fft_real_padded, fft_imag_padded))

    # Apply inverse Fourier Transform and return the real part
    return torch.fft.ifft2(fft_padded, norm="forward").real


def resize_wavelet(image: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resizes an image using wavelet transform and sharpens it.

    This function resizes the input image to a specified target size and then applies
    a wavelet transform to sharpen the image. The process involves resizing, sharpening 
    using a kernel, and applying discrete wavelet transform (DWT) followed by an inverse 
    DWT for each color channel.

    Parameters:
        image (np.ndarray): The input image as a NumPy array.
        target_size (int): The target size to resize the image to.

    Returns:
        np.ndarray: The resized and wavelet-transformed image.
    """

    # Initialize empty arrays for the resized and sharpened image and the wavelet-processed image
    sharpened_resized_image = np.empty((target_size, target_size, 3))
    wavelet_processed_image = np.empty((target_size, target_size, 3))

    # Define a sharpening kernel
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Resize the image to the target size
    resized_image = cv2.resize(image, (target_size, target_size))

    # Process each color channel separately
    for channel in range(3):
        # Sharpen the resized image using the kernel
        sharpened_channel = cv2.filter2D(resized_image[:, :, channel], -1, sharpening_kernel)

        # Perform discrete wavelet transform
        coeffs = pywt.dwt2(sharpened_channel, 'db1')
        cA, (cH, cV, cD) = coeffs

        # Perform inverse discrete wavelet transform
        wavelet_reconstructed = pywt.idwt2((cA, (cH, cV, cD)), 'db1')

        # Store the wavelet processed channel in the output image
        wavelet_processed_image[:, :, channel] = wavelet_reconstructed

    return wavelet_processed_image

def load_image(img: str, directory_path: str, is_video = False) -> np.ndarray:
    """
    Load image bands and generate false-color image.

    Parameters:
        img (str): Image name.
        directory_path (str): Path to the directory containing the image bands.

    Returns:
        np.ndarray: False-color image array.
    """
    bands = []
    for band in ["11", "14", "15"]:
        # Load and append the band array
        if is_video :
            images = np.load(directory_path + img + "/band_%s.npy" % band)
            images = np.moveaxis(images, -1 ,0 )
            bands.append(images)
        else :
            bands.append(np.load(directory_path + img + "/band_%s.npy" % band)[:,:,4])

    # Generate false-color image using the loaded bands
    return false_color(*bands, 3 if is_video else 2)


def load_image_full(img: str, directory_path: str, is_video = False) -> np.ndarray:
    """
    Load all image bands and stack them into a single array.

    Parameters:
        img (str): Image name.
        directory_path (str): Path to the directory containing the image bands.

    Returns:
        np.ndarray: Stacked array of all image bands.
    """
    bands = []
    if not is_video: 
        band_14 = np.load(directory_path + img + "/band_%s.npy" % '14')[:,:,4]

    for band in ["08", "09", "10", "11", "12", "13", "14", "15", "16"]:
        # Load and append the band array
        if is_video : 
            images = np.load(directory_path + img + "/band_%s.npy" % band)
            images = np.moveaxis(images, -1,0)
            bands.append(images)
        else:
            if band == '14':
                bands.append(band_14)
            else:
                bands.append(np.load(directory_path + img + "/band_%s.npy" % band)[:,:,4] - band_14)
    # Stack all bands along the third axis
    stacked_bands = np.stack(bands, axis=3 if is_video else 2)
    return stacked_bands


def preprocess(image_name: str, 
            source_directory: 
            str, destination_directory: str, 
            process_video: bool = False) -> None:
    """
    Preprocesses an image or a video and saves the output in a specified directory.

    This function loads an image or a video from a source directory, preprocesses it, 
    and then saves the preprocessed data as a NumPy array file in the destination directory. 
    The function handles images and videos differently based on the 'process_video' flag.

    Parameters:
        image_name (str): The name of the image or video file to preprocess.
        source_directory (str): The directory where the original image or video is located.
        destination_directory (str): The directory where the preprocessed file will be saved.
        process_video (bool): Flag to indicate if the file is a video. Defaults to False.
    """

    # Load the image or video from the source directory
    preprocessed_data = load_image(image_name, source_directory, process_video)

    # Save the preprocessed data as a numpy file in the destination directory
    # Differentiate the file extension based on whether it's a video
    if process_video:
        np.save(destination_directory + image_name + "_video.npy", preprocessed_data)
    else:
        np.save(destination_directory + image_name + ".npy", preprocessed_data)



def preprocess_all(raw_directory_path: str = RAW_DIR,
                gold_directory_path: str = GOLD_DIR,
                process_video: bool = False) -> None:
    """
    Preprocesses all images in the specified raw directory and saves them to the gold directory.

    This function goes through each image file in the TRAIN and VALIDATION subdirectories of
    the raw directory, applies a preprocessing function, and saves the processed images to
    the corresponding subdirectories in the gold directory. It supports parallel processing
    for efficiency.

    Parameters:
        raw_directory_path (str): Path to the raw data directory. Defaults to RAW_DIR.
        gold_directory_path (str): Path to the gold standard data directory. Defaults to GOLD_DIR.
        process_video (bool): Flag to indicate if the data is video. Defaults to False.
    """

    # Iterate through TRAIN and VALIDATION subdirectories
    for subdirectory in ["TRAIN/", "VALIDATION/"]:
        # Define source and destination directories for each case
        source_directory = raw_directory_path + subdirectory
        destination_directory = gold_directory_path + subdirectory

        # List all image files (excluding .json files) in the source directory
        image_files = [file for file in os.listdir(source_directory) if ".json" not in file]

        # Process each image in parallel using the preprocess function
        Parallel(n_jobs=36)(delayed(preprocess)(image_file, source_directory, destination_directory, process_video) for image_file in image_files)


