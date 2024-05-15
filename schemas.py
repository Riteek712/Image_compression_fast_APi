from pydantic import BaseModel

class ImageCompressionResponse(BaseModel):
    original_size: int
    compressed_size: int
    psnr: float
    compression_ratio: float
    ncc: float
    ssim_value: float
    decompressed_image: str

