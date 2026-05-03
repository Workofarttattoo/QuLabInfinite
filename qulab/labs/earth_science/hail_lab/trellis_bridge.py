from gradio_client import Client, handle_file
from typing import Optional

class TrellisBridge:
    def __init__(self, space_id: str = "microsoft/TRELLIS.2"):
        self.space_id = space_id

    def generate_3d_roof(self, image_path: str) -> Optional[str]:
        """
        Connects to the TRELLIS.2 backend and generates a 3D model (GLB) from a satellite image.
        """
        try:
            client = Client(self.space_id)

            # 1. Generate the Asset
            # Note: The actual parameters might vary slightly depending on the Space's API
            # ss_guidance_strength, ss_sampling_steps, etc. are specific to TRELLIS architecture.
            _ = client.predict(
                image=handle_file(image_path),
                seed=0,
                randomize_seed=True,
                ss_guidance_strength=7.5,
                ss_sampling_steps=12,
                slat_guidance_strength=3.0,
                slat_sampling_steps=12,
                api_name="/image_to_3d"
            )

            # 2. Extract the GLB (The actual 3D file)
            glb_path = client.predict(api_name="/extract_glb")
            return glb_path
        except Exception as e:
            print(f"Error in TRELLIS bridge: {e}")
            return None
