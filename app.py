import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from compression import compress_image
from work import compress_image_k, decompress_image_k
from io import BytesIO
from PIL import Image
import base64
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to allow specific origins instead of "*"
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
async def hello_world():
    return {"message": "Hello World"}

@app.post("/compress")
async def compress_image_endpoint(file: UploadFile = File(...), k: int = 16):
    print("entered the code")
    file_contents = await file.read()
    image = Image.open(BytesIO(file_contents))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_array = np.array(image)
    # compressed_image_base64, encoded_data, huffman_tree, 
    original_size, compressed_size, psnr, compression_ratio, ncc = compress_image(image_array, k)

    print("results")
    print("--------------------------------------------------------------")
    return {
        # 'compressed_image_base64': compressed_image_base64,
        # 'encoded_data': encoded_data,
        # 'huffman_tree': json.dumps(huffman_tree),
        'original_size': original_size,
        'compressed_size': compressed_size,
        'psnr': psnr,
        'compression_ratio': compression_ratio,
        'ncc': ncc,
        # 'ssim_value': ssim_value
    }

@app.post("/decompress")
async def decompress_image_endpoint(file: UploadFile = File(...)):
    # Read the file content
    print("entered the code")
    file_content = await file.read()
    if (file_content):
        print("file read")
    # Decode the file content as JSON
    json_data = file_content.decode('utf-8')
    json_dict = json.loads(json_data)  # Parse the JSON data

    # Extract data from the JSON dictionary
    compressed_image_base64 = json_dict.get('compressed_image_base64')
    encoded_data = json_dict.get('encoded_data')
    huffman_tree_str = json_dict.get('huffman_tree')

    # Convert the huffman_tree from JSON string to dictionary
    huffman_tree = json.loads(huffman_tree_str)
    print("entered the code2")
    compressed_data = base64.b64decode(compressed_image_base64)
    buffer = BytesIO(compressed_data)
    compressed_image = Image.open(buffer)
    compressed_image_array = np.array(compressed_image)

    print("entered the code3")

    decoded_data = []
    current_code = ''
    for bit in encoded_data:
        current_code += bit
        if current_code in huffman_tree:
            decoded_data.append(huffman_tree[current_code])
            current_code = ''
    print("entered the code4")
    decoded_data = np.array(decoded_data)
    decompressed_image = decoded_data.reshape(compressed_image_array.shape)

    with BytesIO() as output:
        Image.fromarray(decompressed_image).save(output, format="PNG")
        decompressed_image_base64 = base64.b64encode(output.getvalue()).decode()

    return {'decompressed_image_base64': decompressed_image_base64}


@app.post("/compressKmeans")
async def compress_kmeans(file: UploadFile = File(...), k: int = 16):
    try:
        # Read the uploaded image file
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        image_array = np.array(image)

        # Compress the image using K-means clustering
        compressed_image, compressed_colors = compress_image_k(image_array, k)
        compressed_image_list = compressed_image.tolist()
        compressed_colors_list = compressed_colors.tolist()

        return JSONResponse(
            status_code=200,
            content={
                "compressed_image": compressed_image_list,
                "compressed_colors": compressed_colors_list
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during compression: {e}")

@app.post("/decompressKmeans")
async def decompress_kmeans(file: UploadFile = File(...), compressed_colors: list = None):
    try:
        # Read the uploaded compressed image file as binary data
        file_content = await file.read()
        json_data = file_content.decode('utf-8')
        json_dict = json.loads(json_data)  # Parse the JSON data
        print("okkokokokokkokokokok")
    # Extract data from the JSON dictionary
        compressed_image_bytes = json_dict.get('compressed_image')
        print("okkokokokokkokokokok")
        # Open the binary data as an image using PIL
        compressed_image = Image.open(BytesIO(compressed_image_bytes))
        print("okkokokokokkokokokok")
        compressed_image_array = np.array(compressed_image)
        print("okkokokokokkokokokok")
        # Convert compressed colors list back to NumPy array if provided
        if compressed_colors is not None:
            compressed_colors_array = np.array(compressed_colors)
        else:
            compressed_colors_array = None
        print("okkokokokokkokokokok")
        # Decompress the image using the compressed colors
        decompressed_image = decompress_image_k(compressed_image_array, compressed_colors_array)

        # Convert decompressed image to list for JSON serialization
        decompressed_image_list = decompressed_image.tolist()

        return JSONResponse(
            status_code=200,
            content={
                "decompressed_image": decompressed_image_list
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during decompression: {e}")
