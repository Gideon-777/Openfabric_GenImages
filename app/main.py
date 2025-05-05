import logging
import os
from typing import Dict
import threading
import time

from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from core.openfabric_wrapper import AppModel, State
from core.llm_interface import creative_engine
from core.stub import Stub
from core.db_manager import db_manager

logger = logging.getLogger(__name__)

# Configurations for the app
configurations: Dict[str, ConfigClass] = dict()

def launch_gradio():
    """Launch Gradio interface in a separate thread"""
    logger.info("Starting Gradio interface thread...")
    try:
        interface = creative_engine.create_gradio_interface()
    except Exception as e:
        logger.error(f"Error starting Gradio interface: {e}")
        raise

############################################################
# Config callback function
############################################################
def config(configuration: Dict[str, ConfigClass], state: State) -> None:
    """
    Stores user-specific configuration data.

    Args:
        configuration (Dict[str, ConfigClass]): A mapping of user IDs to configuration objects.
        state (State): The current state of the application.
    """
    for uid, conf in configuration.items():
        logger.info(f"Saving new config for user with id:'{uid}'")
        configurations[uid] = conf

############################################################
# Execution callback function
############################################################
def execute(model: AppModel) -> None:
    """
    Main execution entry point for handling a model pass.

    Args:
        model (AppModel): The model object containing request and response structures.
    """
    # Initialize response
    response: OutputClass = OutputClass()
    model.response = response
    
    try:
        # Retrieve input
        request: InputClass = model.request
        prompt = request.prompt
        
        logger.info(f"Received prompt: {prompt}")
        
        # Check if the prompt references past creations
        references = db_manager.search_generations(prompt)
        if references:
            logger.info(f"Found {len(references)} relevant past creations")
            # Add context from past creations to the prompt
            context = "Previous relevant creations:\n"
            for ref in references[:2]:  # Limit to 2 most recent relevant items
                context += f"- {ref[1]}\n"  # Add original prompts
            prompt = context + "\nNew request: " + prompt
        
        # Use CreativeEngine to enhance the prompt
        enhanced_prompt = creative_engine.enhance_prompt(prompt)
        response.enhanced_prompt = enhanced_prompt
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
        
        # Initialize Openfabric apps
        app_ids = [
            "f0997a01-d6d3-a5fe-53d8-561300318557",  # Text-to-Image app
            "69543f29-4d41-4afc-7f29-3d51591f11eb"   # Image-to-3D app
        ]
        
        # Initialize the Stub with app IDs
        stub = Stub(app_ids)
        
        # Get the current generation ID
        cursor = db_manager.conn.cursor()
        cursor.execute('SELECT last_insert_rowid()')
        generation_id = cursor.fetchone()[0]
        
        # Generate image from enhanced prompt
        image_path = stub.generate_image(enhanced_prompt, generation_id)
        if image_path:
            response.image_url = f"file://{image_path}"
            logger.info(f"Image generated: {image_path}")
            
            # Generate 3D model from the image
            model_path = stub.generate_3d_model(image_path, generation_id)
            if model_path:
                response.model_url = f"file://{model_path}"
                logger.info(f"3D model generated: {model_path}")
                response.message = "Successfully created image and 3D model"
            else:
                response.message = "Created image but failed to generate 3D model"
        else:
            response.message = "Failed to generate image"
            logger.error("Image generation failed")
        
    except Exception as e:
        error_msg = f"Error during execution: {str(e)}"
        logger.error(error_msg)
        response.message = error_msg

# Launch Gradio interface when the module is imported
threading.Thread(target=launch_gradio, daemon=True).start()