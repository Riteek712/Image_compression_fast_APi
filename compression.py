import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
import heapq
from io import BytesIO
import base64
from skimage.metrics import structural_similarity as ssim

def huffman_encoding(data):
    """Perform Huffman encoding on the input data."""
    frequency = Counter(data)
    heap = []
    for value, freq in frequency.items():
        heapq.heappush(heap, (freq, len(heap), str(value)))  # Convert keys to strings

    while len(heap) > 1:
        freq1, _count1, left = heapq.heappop(heap)
        freq2, _count2, right = heapq.heappop(heap)
        heapq.heappush(heap, (freq1 + freq2, max(_count1, _count2), {0: left, 1: right}))

    huffman_tree = {}
    stack = [(heap[0][-1], '')]
    while stack:
        node, code = stack.pop()
        if isinstance(node, dict):
            stack.append((node[0], code + '0'))
            stack.append((node[1], code + '1'))
        else:
            huffman_tree[node] = code

    encoded_data = ''.join(huffman_tree[str(value)] for value in data)  # Use string keys

    return huffman_tree, encoded_data

def apply_kmeans(image_array, max_k):
    """Apply K-means clustering to reduce the number of colors in the image."""
    shape = image_array.shape
    flat_image_array = image_array.reshape(-1, shape[-1])
    unique_colors = np.unique(flat_image_array, axis=0)
    k = min(max_k, len(unique_colors))  # Ensure k does not exceed the number of unique colors
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(flat_image_array)
    labels = kmeans.labels_
    centers = np.uint8(kmeans.cluster_centers_)
    quantized_image = centers[labels].reshape(shape)
    return quantized_image

def calculate_psnr(original, decompressed):
    error = np.random.normal(0, 0.01, original.shape)
    decompressed_with_error = decompressed + error
    decompressed_with_error = np.clip(decompressed_with_error, 0, 255)
    mse = np.mean((original - decompressed_with_error) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_normalized_cross_correlation(original, decompressed):
    error = np.random.normal(0, 0.01, original.shape)
    decompressed_with_error = decompressed + error
    decompressed_with_error = np.clip(decompressed_with_error, 0, 255)

    original = original.astype(float)
    decompressed_with_error = decompressed_with_error.astype(float)
    original_mean = np.mean(original)
    decompressed_mean = np.mean(decompressed_with_error)
    numerator = np.sum((original - original_mean) * (decompressed_with_error - decompressed_mean))
    denominator = np.sqrt(np.sum((original - original_mean) ** 2) * np.sum((decompressed_with_error - decompressed_mean) ** 2))
    return numerator / denominator if denominator != 0 else 0



def compress_image(image_array, max_k=16):
    """Compress the input image array using K-means clustering and Huffman encoding."""
    print("entered the code 2")

    quantized_image = apply_kmeans(image_array, max_k)

    # Flatten quantized image for Huffman encoding
    data = quantized_image.flatten()

    # Perform Huffman encoding
    huffman_tree, encoded_data = huffman_encoding(data)

    # Convert compressed image to base64 string
    buffer = BytesIO()
    Image.fromarray(quantized_image).save(buffer, format='PNG')
    compressed_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    original_size = image_array.size
    compressed_size = len(compressed_image_base64)
    psnr = calculate_psnr(image_array, quantized_image)
    compression_ratio = original_size / compressed_size
    ncc = calculate_normalized_cross_correlation(image_array, quantized_image)
    # ssim_value = calculate_ssim(image_array, quantized_image)
    # compressed_image_base64, encoded_data, huffman_tree,
    return  original_size, compressed_size, psnr, compression_ratio, ncc

def decompress_image(compressed_image_base64, encoded_data, huffman_tree):
    """Decompress the base64-encoded compressed image using the Huffman tree."""
    # Decode base64 and load image
    compressed_data = base64.b64decode(compressed_image_base64)
    buffer = BytesIO(compressed_data)
    compressed_image = Image.open(buffer)
    compressed_image_array = np.array(compressed_image)

    # Perform Huffman decoding
    decoded_data = []
    current_code = ''
    for bit in encoded_data:
        current_code += bit
        if current_code in huffman_tree:
            decoded_data.append(huffman_tree[current_code])
            current_code = ''

    decoded_data = np.array(decoded_data)

    # Reshape to original image dimensions
    decompressed_image = decoded_data.reshape(compressed_image_array.shape)

    return decompressed_image
