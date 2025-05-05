import json
import logging
import os
import base64
from typing import Any, Dict, List, Literal, Tuple, Optional
import requests
from marshmallow import Schema, fields

from core.remote import Remote
from .db_manager import db_manager

logger = logging.getLogger(__name__)

def has_resource_fields(schema: Schema) -> bool:
    """Check if a schema has resource fields"""
    return any(
        isinstance(field, fields.Raw) and getattr(field, 'resource', False)
        for field in schema.fields.values()
    )

def resolve_resources(url_template: str, data: Any, schema: Schema) -> Any:
    """Resolve resource references in the data"""
    if isinstance(data, dict):
        resolved = {}
        for key, value in data.items():
            field = schema.fields.get(key)
            if field and getattr(field, 'resource', False) and isinstance(value, str):
                try:
                    response = requests.get(url_template.format(reid=value))
                    resolved[key] = response.content
                except Exception as e:
                    logging.error(f"Failed to resolve resource {value}: {e}")
                    resolved[key] = value
            else:
                resolved[key] = resolve_resources(url_template, value, schema)
        return resolved
    elif isinstance(data, list):
        return [resolve_resources(url_template, item, schema) for item in data]
    return data

class Stub:
    def __init__(self, app_ids: list):
        """Initialize Stub with app IDs for text-to-image and image-to-3D"""
        self.text2img_app_id = app_ids[0]
        self.img2model_app_id = app_ids[1]
        
        # Set up storage paths
        self.image_dir = os.path.join(os.path.dirname(__file__), '..', 'datastore', 'images')
        self.model_dir = os.path.join(os.path.dirname(__file__), '..', 'datastore', 'models')
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load API tokens
        token_path = os.path.join(os.path.dirname(__file__), '..', 'datastore', 'tokens.json')
        try:
            with open(token_path, 'r') as f:
                self.tokens = json.load(f)
        except FileNotFoundError:
            logger.warning("No tokens.json found. Using default configuration.")
            self.tokens = {}
    
    def call_app(self, app_id: str, data: Dict[str, Any]) -> Optional[Dict]:
        """Call an Openfabric app using WebSocket connection"""
        try:
            # WebSocket implementation would go here
            # For now, fall back to REST API
            return None
        except Exception as e:
            logger.error(f"WebSocket call failed: {e}")
            return None
    
    def call_rest(self, endpoint: str, data: Dict[str, Any]) -> Optional[Dict]:
        """Call an Openfabric app using REST API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.tokens.get("api_key", "")}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f"https://{endpoint}/execute",
                headers=headers,
                json=data,
                timeout=300  # 5-minute timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"REST API call failed: {e}")
            return None
    
    def generate_image(self, prompt: str, generation_id: int) -> Optional[str]:
        """Generate an image from a text prompt using the Text-to-Image app"""
        try:
            # Use the example endpoint from readme
            endpoint = 'c25dcd829d134ea98f5ae4dd311d13bc.node3.openfabric.network'
            result = self.call(endpoint, {'prompt': prompt}, 'super-user')
            
            if result and 'result' in result:
                # Save the image
                image_filename = f"generation_{generation_id}_image.png"
                image_path = os.path.join(self.image_dir, image_filename)
                
                try:
                    # Get image data from result
                    image_data = result['result']
                    if isinstance(image_data, str):
                        # Handle base64 encoded data
                        image_data = base64.b64decode(image_data)
                    
                    # Save to file
                    with open(image_path, 'wb') as f:
                        f.write(image_data)
                    
                    # Update the generation record
                    db_manager.update_paths(generation_id, image_path=image_path)
                    return image_path
                    
                except Exception as e:
                    logger.error(f"Failed to save generated image: {e}")
                    return None
            else:
                logger.error("No result in API response")
                return None
                    
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return None
    
    def call(self, endpoint: str, data: Dict[str, Any], user: str = 'super-user') -> Optional[Dict]:
        """Call an Openfabric app using direct endpoint as shown in readme example"""
        try:
            logger.info(f"Calling endpoint {endpoint} with data: {data}")
            
            # Format the API endpoint correctly
            if not endpoint.startswith('http'):
                endpoint = f"https://{endpoint}"
            
            # Use /execution endpoint as specified in documentation
            if not endpoint.endswith('/execution'):
                endpoint = f"{endpoint}/execution"
            
            response = requests.post(
                endpoint,
                headers={'accept': 'application/json', 'Content-Type': 'application/json'},
                json=data,  # Send data directly as specified in documentation
                timeout=300
            )
            response.raise_for_status()
            
            # Log the raw response for debugging
            logger.info(f"Raw API response: {response.text[:200]}...")
            
            # Return the parsed JSON response
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to call endpoint {endpoint}: {e}")
            logger.error(f"Response content: {getattr(response, 'text', 'No response text')}")
            return None
    
    def generate_3d_model(self, image_path: str, generation_id: int) -> Optional[str]:
        """Generate a 3D model from an image using the Image-to-3D app"""
        try:
            logger.info(f"Reading image from: {image_path}")
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Convert image to base64 if needed
            if isinstance(image_data, bytes):
                image_data = base64.b64encode(image_data).decode('utf-8')
            
            data = {"image": image_data}
            logger.info("Preparing to call Image-to-3D app...")
            
            # Try WebSocket connection first
            result = None
            try:
                result = self.call_app(self.img2model_app_id, data)
                if result:
                    logger.info("Successfully called Image-to-3D app via WebSocket")
            except Exception as e:
                logger.warning(f"WebSocket call failed, falling back to REST: {e}")
            
            # Fall back to REST if WebSocket failed
            if not result:
                try:
                    endpoint = f"{self.img2model_app_id}.node3.openfabric.network"
                    logger.info(f"Calling Image-to-3D app via REST at endpoint: {endpoint}")
                    result = self.call_rest(endpoint, data)
                except Exception as e:
                    logger.error(f"REST call failed with error: {e}")
                    if hasattr(e, 'response'):
                        logger.error(f"Response status: {e.response.status_code}")
                        logger.error(f"Response headers: {e.response.headers}")
                        logger.error(f"Response content: {e.response.text}")
            
            if result and "result" in result:
                # Save the 3D model
                model_filename = f"generation_{generation_id}_model.glb"
                model_path = os.path.join(self.model_dir, model_filename)
                
                try:
                    # Handle base64 encoded model data
                    if isinstance(result["result"], str):
                        model_data = base64.b64decode(result["result"])
                    else:
                        model_data = result["result"]
                    
                    with open(model_path, 'wb') as f:
                        f.write(model_data)
                    
                    # Update the generation record
                    db_manager.update_paths(generation_id, model_path=model_path)
                    logger.info(f"Successfully saved 3D model to: {model_path}")
                    
                    return model_path
                    
                except Exception as e:
                    logger.error(f"Failed to save 3D model: {e}")
                    return None
            else:
                logger.error("No result in API response for 3D model generation")
                return None
            
        except Exception as e:
            logger.error(f"Error in generate_3d_model: {e}")
            return None
        
        return None

    def manifest(self, app_id: str) -> dict:
        """Get the manifest for an app"""
        return self._manifest.get(app_id, {})

    def schema(self, app_id: str, type: Literal['input', 'output']) -> dict:
        """Get the input or output schema for an app"""
        _input, _output = self._schema.get(app_id, (None, None))

        if type == 'input':
            if _input is None:
                raise ValueError(f"Input schema not found for app ID: {app_id}")
            return _input
        elif type == 'output':
            if _output is None:
                raise ValueError(f"Output schema not found for app ID: {app_id}")
            return _output
        else:
            raise ValueError("Type must be either 'input' or 'output'")
