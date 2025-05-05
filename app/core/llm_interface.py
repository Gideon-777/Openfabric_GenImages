import os
import base64
import sqlite3
import requests
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import time
from PIL import Image
from rembg import remove

from .db_manager import db_manager
from .stub import Stub

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreativeEngine:
    def __init__(self):
        logger.info("Initializing CreativeEngine...")
        
        # Initialize LLM model and tokenizer
        self.model_name = "google/gemma-2b-it"
        logger.info(f"Loading model: {self.model_name}")
        
        # Define app IDs
        self.text2img_app_id = "f0997a01-d6d3-a5fe-53d8-561300318557"
        self.img2model_app_id = "69543f29-4d41-4afc-7f29-3d51591f11eb"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("Tokenizer loaded successfully")
            
            # Use CPU for inference
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            logger.info("Model loaded successfully on CPU")

            # Initialize Hunyuan3D pipeline
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                'tencent/Hunyuan3D-2',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            logger.info("Hunyuan3D pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
        
        # Set up storage paths
        self.image_dir = os.path.join(os.path.dirname(__file__), '..', 'datastore', 'images')
        self.model_dir = os.path.join(os.path.dirname(__file__), '..', 'datastore', 'models')
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"Storage paths initialized. Images: {self.image_dir}, Models: {self.model_dir}")
        
    def enhance_prompt(self, prompt: str) -> str:
        """Enhance the user prompt using the Gemma model"""
        try:

            full_prompt = f'''High-resolution front-facing orthographic view of {prompt}, centered, isolated on a plain white background, no shadows, no reflections, evenly lit, symmetrical, neutral lighting, no text, no watermarks, realistic textures, clearly defined geometry, sharp details, professional studio lighting, product-style render, single object in frame, full-body view, 3D model reference style, photorealistic, 8k resolution.
            Enhanced prompt:"
            '''
            
            # Tokenize input
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            
            # Generate enhanced prompt
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=256,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and clean up the response
            enhanced = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            enhanced = enhanced.split("Enhanced prompt")[-1].strip()
            if enhanced.startswith(":"): 
                enhanced = enhanced[1:].strip()
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}")
            # Return original prompt if enhancement fails
            return prompt + " Set against a pure white background."
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract meaningful tags from the text for better searchability"""
        logger.info(f"Extracting tags from text: {text}")
        
        # Simple implementation - split on spaces and filter
        words = text.lower().split()
        # Filter common words and keep meaningful ones (basic stopwords filtering)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
        meaningful_words = [word for word in words if len(word) > 3 and word not in stopwords]
        return list(set(meaningful_words))  # Remove duplicates
    
    def create_gradio_interface(self):
        """Create and launch the Gradio interface"""
        logger.info("Creating Gradio interface...")
        
        with gr.Blocks(title="Creative AI Engine") as interface:
            gr.Markdown("# 🎨 Creative AI Engine")
            gr.Markdown("Transform your ideas into enhanced, detailed prompts and images using AI.")
            
            with gr.Row():
                with gr.Column():
                    input_prompt = gr.Textbox(
                        label="Your Prompt",
                        placeholder="Describe what you want to create...",
                        lines=3
                    )
                    enhance_btn = gr.Button("✨ Enhance & Generate", variant="primary")
                
                with gr.Column():
                    output_prompt = gr.Textbox(
                        label="Enhanced Prompt",
                        lines=3,
                        interactive=False
                    )
                    output_image = gr.Image(
                        label="Generated Image",
                        type="filepath"
                    )
                    output_model = gr.Model3D(
                        label="Generated 3D Model",
                        clear_color=[1, 1, 1, 1]
                    )
            
            with gr.Accordion("Recent Creations", open=False):
                history = gr.DataFrame(
                    headers=["Original Prompt", "Enhanced Prompt", "Created At"],
                    datatype=["str", "str", "datetime"],
                    label="Creation History"
                )
                refresh_btn = gr.Button("🔄 Refresh History")
                
            with gr.Accordion("Search Past Creations", open=False):
                search_input = gr.Textbox(
                    label="Search prompts or tags",
                    placeholder="e.g., dragon, sunset, fantasy...",
                )
                search_btn = gr.Button("🔍 Search")
                search_results = gr.DataFrame(
                    headers=["Original Prompt", "Enhanced Prompt", "Created At"],
                    datatype=["str", "str", "datetime"],
                    label="Search Results"
                )
            
            def process_prompt(prompt: str):
                try:
                    # Generate image and get enhanced prompt
                    enhanced_prompt, image_path, model_path = self.process_prompt(prompt)
                    return enhanced_prompt, image_path, model_path
                except Exception as e:
                    logger.error(f"Error in process_prompt: {e}")
                    return str(e), None, None
            
            def update_history():
                logger.info("Updating history...")
                generations = db_manager.get_recent_generations()
                return [[g[1], g[2], g[5]] for g in generations]
            
            def search_creations(query):
                logger.info(f"Searching creations with query: {query}")
                results = db_manager.search_generations(query)
                return [[r[1], r[2], r[5]] for r in results]
            
            enhance_btn.click(
                fn=process_prompt,
                inputs=[input_prompt],
                outputs=[output_prompt, output_image, output_model]
            ).success(
                fn=update_history,
                inputs=[],
                outputs=[history]
            )
            
            refresh_btn.click(
                fn=update_history,
                inputs=[],
                outputs=[history]
            )
            
            search_btn.click(
                fn=search_creations,
                inputs=[search_input],
                outputs=[search_results]
            )
            
            logger.info("Launching Gradio interface on port 7860...")
            interface.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=True,
                auth=None,
                show_error=True
            )
            
        return interface

    def generate_image(self, prompt: str) -> Tuple[str, str]:
        """Generate an image from a prompt using the text-to-image app
        Returns: (enhanced_prompt, image_path)
        """
        try:
            # First enhance the prompt
            enhanced_prompt = self.enhance_prompt(prompt)
            
            # Extract tags for searchability
            tags = self._extract_tags(enhanced_prompt)
            
            # Save to database and get generation ID
            generation_id = db_manager.save_generation(prompt, enhanced_prompt, tags)
            
            # Initialize Openfabric app with correct endpoint
            endpoint = 'c25dcd829d134ea98f5ae4dd311d13bc.node3.openfabric.network'
            base_url = f"https://{endpoint}"
            
            try:
                # Step 1: Submit the prompt to get resource ID
                logger.info(f"Submitting prompt to text-to-image API: {enhanced_prompt[:100]}...")
                response = requests.post(
                    f"{base_url}/execution",
                    headers={'Content-Type': 'application/json'},
                    json={'prompt': enhanced_prompt},
                    timeout=300
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Execution response: {result}")
                
                if result and 'result' in result:
                    # Extract resource ID and ensure it ends with /resources
                    reid = result['result']
                    if not reid.endswith('/resources'):
                        reid = f"{reid}/resources"
                    logger.info(f"Using resource ID: {reid}")
                    
                    # Step 2: Retrieve the generated image using resource endpoint
                    logger.info(f"Fetching image with resource ID: {reid}")
                    image_response = requests.get(
                        f"{base_url}/resource",
                        params={'reid': reid},
                        timeout=300
                    )
                    image_response.raise_for_status()
                    
                    # Save the original image
                    image_filename = f"generation_{generation_id}_image.png"
                    image_path = os.path.join(os.path.dirname(__file__), '..', 'datastore', 'images', image_filename)
                    
                    # Save image data to file with proper permissions
                    with open(image_path, 'wb') as f:
                        f.write(image_response.content)
                    os.chmod(image_path, 0o644)  # rw-r--r--
                    
                    # Create a background-removed version for 3D generation
                    bg_removed_filename = f"generation_{generation_id}_image_nobg.png"
                    bg_removed_path = os.path.join(os.path.dirname(__file__), '..', 'datastore', 'images', bg_removed_filename)
                    
                    # Remove background using rembg
                    input_image = Image.open(image_path)
                    output_image = remove(input_image)
                    output_image.save(bg_removed_path)
                    os.chmod(bg_removed_path, 0o644)  # rw-r--r--
                    
                    # Update database with image paths
                    db_manager.update_paths(generation_id, image_path=image_path, bg_removed_path=bg_removed_path)
                    logger.info(f"Successfully saved images to {image_path} and {bg_removed_path}")
                    
                    # Return absolute path for Gradio
                    absolute_path = os.path.abspath(image_path)
                    logger.info(f"Returning absolute image path: {absolute_path}")
                    return enhanced_prompt, absolute_path
                    
                else:
                    logger.error("No result in API response")
                    
            except Exception as e:
                logger.error(f"Error calling API: {e}")
                logger.error(f"Response content: {getattr(response, 'text', 'No response text')}")
                
        except Exception as e:
            logger.error(f"Error generating image: {e}")
        
        return enhanced_prompt, None

    def generate_3d_model(self, image_path: str, generation_id: int) -> Optional[str]:
        """Generate a 3D model from an image using Hunyuan3D
        Returns: Path to the generated 3D model file
        """
        try:
            logger.info(f"Starting 3D model generation for image: {image_path} (ID: {generation_id})")
            start_time = time.time()

            # Check if there's a background-removed version of the image
            bg_removed_path = os.path.join(os.path.dirname(image_path), f"generation_{generation_id}_image_nobg.png")
            if os.path.exists(bg_removed_path):
                logger.info(f"Using background-removed image for 3D generation: {bg_removed_path}")
                source_image_path = bg_removed_path
            else:
                logger.info("No background-removed image found, using original image")
                source_image_path = image_path

            # --- Add detailed logging and error handling around the pipeline call ---
            try:
                logger.info("Calling Hunyuan3D pipeline...")
                mesh = self.pipeline(image=source_image_path)[0]
                logger.info("Hunyuan3D pipeline call completed.")
            except Exception as pipeline_error:
                logger.exception(f"Error during Hunyuan3D pipeline execution: {pipeline_error}")
                return None
            # --- End of added block ---

            # Save the mesh to a file
            model_filename = f"model_{generation_id}.glb"
            model_path = os.path.join(self.model_dir, model_filename)

            # --- Add logging for export ---
            try:
                logger.info(f"Exporting mesh to {model_path}...")
                mesh.export(model_path)
                logger.info(f"Mesh successfully exported.")
            except Exception as export_error:
                logger.exception(f"Error exporting mesh to {model_path}: {export_error}")
                return None
            # --- End of added block ---

            end_time = time.time()
            logger.info(f"3D model generated successfully: {model_path} in {end_time - start_time:.2f} seconds")
            return model_path

        except Exception as e:
            logger.exception(f"Unexpected error in generate_3d_model: {e}")
            return None

    def process_prompt(self, prompt: str) -> Tuple[str, str, str]:
        """Process a prompt to generate both image and 3D model
        Returns: (enhanced_prompt, image_path, model_path) - model_path is absolute
        """
        logger.info(f"process_prompt started for prompt: '{prompt[:50]}...'")
        try:
            # Generate image from prompt
            logger.info("Calling generate_image...")
            enhanced_prompt, image_path = self.generate_image(prompt)
            logger.info(f"generate_image completed. Image path: {image_path}")

            if not image_path:
                logger.error("Image generation failed or returned no path.")
                return enhanced_prompt, None, None

            # Get the generation ID from the image path
            try:
                generation_id = int(os.path.basename(image_path).split('_')[1])
                logger.info(f"Extracted generation ID: {generation_id}")
            except (IndexError, ValueError) as e:
                logger.error(f"Could not extract generation ID from image path {image_path}: {e}")
                return enhanced_prompt, image_path, None

            # Generate 3D model using Hunyuan3D
            logger.info("Calling generate_3d_model...")
            model_path = self.generate_3d_model(image_path, generation_id)
            logger.info(f"generate_3d_model completed. Model path: {model_path}")


            if model_path:
                # Ensure the returned path is absolute
                abs_model_path = os.path.abspath(model_path)
                logger.info(f"Returning absolute model path: {abs_model_path}")
                # Update database with model path
                db_manager.update_paths(generation_id, model_path=abs_model_path)
                logger.info(f"process_prompt finished successfully.")
                return enhanced_prompt, image_path, abs_model_path
            else:
                logger.error("3D model generation failed or returned no path.")
                logger.info(f"process_prompt finished with image only.")
                return enhanced_prompt, image_path, None

        except Exception as e:
            logger.exception(f"Error in process_prompt: {e}") # Use exception for full traceback
            logger.info(f"process_prompt finished with error.")
            return str(e), None, None

# Create singleton instance
creative_engine = CreativeEngine()