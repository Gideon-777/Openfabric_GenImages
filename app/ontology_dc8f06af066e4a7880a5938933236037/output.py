from decimal import *
from datetime import *
from typing import *
from dataclasses import dataclass
from marshmallow import Schema, fields, post_load
from openfabric_pysdk.utility import SchemaUtil

class OutputClass:
    def __init__(self):
        self.message: str = None  # Status message or error
        self.enhanced_prompt: str = None  # AI-enhanced creative prompt
        self.image_url: str = None  # URL to the generated image
        self.model_url: str = None  # URL to the generated 3D model

class OutputClassSchema(Schema):
    message = fields.String(allow_none=True)
    enhanced_prompt = fields.String(allow_none=True)
    image_url = fields.String(allow_none=True)
    model_url = fields.String(allow_none=True)
    
    @post_load
    def create(self, data, **kwargs):
        return SchemaUtil.create(OutputClass(), data)
