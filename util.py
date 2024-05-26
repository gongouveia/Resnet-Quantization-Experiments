import torch
import random

def quantize_to_random_float8(tensor):
    # Define the float8 range
    float8_max = 127.0
    float8_min = -128.0
    
    # Define the quantization grid size (assuming 256 levels for 8-bit)
    grid_size = (float8_max - float8_min) / 255.0
    
    # Clamp the values to be within the range of float8
    clamped_tensor = torch.clamp(tensor, float8_min, float8_max)
    
    # Compute the floor and ceiling quantized values
    floor_quantized = torch.floor((clamped_tensor - float8_min) / grid_size) * grid_size + float8_min
    ceil_quantized = torch.ceil((clamped_tensor - float8_min) / grid_size) * grid_size + float8_min
    
    # Create a random mask for choosing between floor and ceiling
    random_mask = torch.rand_like(clamped_tensor) < 0.5
    
    # Apply the mask to choose between floor and ceiling quantized values
    quantized_tensor = torch.where(random_mask, floor_quantized, ceil_quantized)
    
    return quantized_tensor



#  quantizes a given float32 tensor to a simulated float8 format, rounding up to the nearest value on a quantization grid.
def quantize_to_float8_ceil(tensor):
    # Define the float8 range
    float8_max = 127.0
    float8_min = -128.0
    
    # Define the quantization grid size (assuming 256 levels for 8-bit)
    grid_size = (float8_max - float8_min) / 255.0
    
    # Clamp the values to be within the range of float8
    clamped_tensor = torch.clamp(tensor, float8_min, float8_max)
    
    # Quantize to the nearest float8 value by ceiling
    quantized_tensor = torch.ceil((clamped_tensor - float8_min) / grid_size) * grid_size + float8_min
    
    return quantized_tensor


#  quantizes a given float32 tensor to a simulated float8 format, rounding down to the nearest value on a quantization grid.
def quantize_to_float8_floor(tensor):
    # Define the float8 range
    float8_max = 127.0
    float8_min = -128.0
    
    # Define the quantization grid size (assuming 256 levels for 8-bit)
    grid_size = (float8_max - float8_min) / 255.0
    
    # Clamp the values to be within the range of float8
    clamped_tensor = torch.clamp(tensor, float8_min, float8_max)
    
    # Quantize to the nearest float8 value by flooring
    quantized_tensor = torch.floor((clamped_tensor - float8_min) / grid_size) * grid_size + float8_min
    
    return quantized_tensor


#  quantizes a given float32 tensor to a simulated float8 format, rounding to the nearest value on a quantization grid.
def quantize_to_nearest_float8(tensor):
    # Define the float8 range
    float8_max = 127.0
    float8_min = -128.0
    
    # Define the quantization grid size (assuming 256 levels for 8-bit)
    grid_size = (float8_max - float8_min) / 255.0
    
    # Clamp the values to be within the range of float8
    clamped_tensor = torch.clamp(tensor, float8_min, float8_max)
    
    # Quantize to the nearest float8 value by rounding
    quantized_tensor = torch.round((clamped_tensor - float8_min) / grid_size) * grid_size + float8_min
    
    return quantized_tensor


tensor = torch.tensor([[1.2, 3.5, 128.5], [-130.0, 0.0, 64.2]], dtype=torch.float32)
quantized_tensor = quantize_to_random_float8(tensor)

