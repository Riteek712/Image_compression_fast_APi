import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim

def compress_image_k(image_array, k):
    """Compress the image using K-means clustering."""
    # Reshape the image array to a 2D array of pixels
    h, w, d = image_array.shape
    image_array_2d = image_array.reshape((h * w, d))

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(image_array_2d)
    compressed_colors = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Replace each pixel with the centroid of its cluster
    compressed_image = compressed_colors[labels].reshape((h, w, d))
    return compressed_image, compressed_colors
def decompress_image_k(compressed_image, compressed_colors=None):
    """Decompress the image using the compressed colors."""
    h, w, d = compressed_image.shape
    flat_image = compressed_image.reshape((h * w, d))
    if compressed_colors is not None:
        compressed_colors_array = np.array(compressed_colors)
        indices = np.argmin(np.linalg.norm(flat_image[:, np.newaxis] - compressed_colors_array, axis=2), axis=1)
    else:
        indices = flat_image
    decompressed_image = indices.reshape((h, w, d))
    return decompressed_image

def calculate_image_size(image_array):
    """Calculate the size of the image in bits."""
    h, w, d = image_array.shape
    bits_per_pixel = d * 8  # Assuming 8 bits per channel (RGB)
    image_size_bits = h * w * bits_per_pixel
    return image_size_bits

def calculate_psnr(original, compressed):
    """Calculate Peak Signal-to-Noise Ratio (PSNR)."""
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ncc(original, compressed):
    """Calculate Normalized Cross-Correlation (NCC)."""
    original_mean = np.mean(original)
    compressed_mean = np.mean(compressed)
    numerator = np.sum((original - original_mean) * (compressed - compressed_mean))
    denominator = np.sqrt(np.sum((original - original_mean) * 2) * np.sum((compressed - compressed_mean) * 2))
    ncc = numerator / denominator if denominator != 0 else 0
    return ncc

