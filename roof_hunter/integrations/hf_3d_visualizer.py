import os
import requests
import base64
from typing import Optional

def generate_3d_from_2d(image_path: str, output_path: str, model_id: str = "stabilityai/stable-fast-3d") -> bool:
    """
    Generate a 3D model (GLB/OBJ) from a 2D roof image using Hugging Face Inference API.
    
    Args:
        image_path: Path to the input 2D roof image (e.g., satellite crop).
        output_path: Path to save the resulting 3D file.
        model_id: HuggingFace model ID for image-to-3d (e.g., stabilityai/stable-fast-3d or TencentARC/InstantMesh).
        
    Returns:
        bool: True if successful, False otherwise.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("[HF 3D] Error: HF_TOKEN environment variable not set.")
        return False
        
    if not os.path.exists(image_path):
        print(f"[HF 3D] Error: Input image {image_path} not found.")
        return False

    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    print(f"[HF 3D] Sending {image_path} to {model_id} for 3D generation...")
    
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
            
        response = requests.post(api_url, headers=headers, data=image_data, timeout=120)
        
        if response.status_code == 200:
            with open(output_path, "wb") as f_out:
                f_out.write(response.content)
            print(f"[HF 3D] Successfully generated 3D model at {output_path}")
            return True
        else:
            print(f"[HF 3D] API Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"[HF 3D] Exception during generation: {e}")
        return False

if __name__ == "__main__":
    # Example usage:
    # Set HF_TOKEN=...
    # generate_3d_from_2d("roof_sample.png", "roof_3d.glb")
    pass
