import torch
import time
from torchvision import transforms
from PIL import Image
import numpy as np

# Function to replace chroma key background
def replace_chroma_key(src, bg, chroma_key, threshold):
    diff = torch.abs(src - chroma_key).sum(dim=2)
    mask = diff < threshold
    dst = torch.where(mask.unsqueeze(2), bg, src)
    return dst

# Load images using PIL and convert to tensors
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    image_tensor = transform(image).cuda()
    return image_tensor

# Convert tensor to PIL image for visualization if needed
def tensor_to_image(tensor):
    transform = transforms.ToPILImage()
    image = transform(tensor.cpu())
    return image

# Main function
def main():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your PyTorch installation.")
        return

    # Load images
    src_image_path = 'image-640x480case2chr.jpg'
    bg_image_path = 'case2bgimage-640x480.jpg'
    src = load_image(src_image_path).permute(1, 2, 0)  
    bg = load_image(bg_image_path).permute(1, 2, 0)    

    # Define chroma key and threshold
    chroma_key = torch.tensor([0.0, 1.0, 0.0]).cuda()  # Example: green screen (normalized [0,1] range)
    threshold = 0.3  

    # Perform chroma key replacement
    start_time = time.time()
    dst = replace_chroma_key(src, bg, chroma_key, threshold)
    end_time = time.time()

    # Calculate execution time
    execution_time = end_time - start_time
    print(f"Execution Time on GPU (using PyTorch): {execution_time:.6f} seconds")

    # Optionally save the result image
    result_image = tensor_to_image(dst.permute(2, 0, 1))  # Convert back to CHW format
    result_image.save('result_image_pytorch_700x900.png')

if __name__ == '__main__':
    main()
