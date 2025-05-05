"""
A minimal wrapper to provide openfabric_pysdk functionality
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Api
from flask_socketio import SocketIO
from gevent.pywsgi import WSGIServer
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass, InputClassSchema
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass, OutputClassSchema

logger = logging.getLogger(__name__)

@dataclass
class AppModel:
    request: Any
    response: Any

class Starter:
    @staticmethod
    def ignite(debug: bool = False, host: str = "0.0.0.0", port: int = 8888):
        app = Flask(__name__)
        CORS(app)
        api = Api(app)
        socketio = SocketIO(app)
        input_schema = InputClassSchema()
        output_schema = OutputClassSchema()
        
        @app.route('/execute', methods=['POST'])
        def execute():
            try:
                data = request.json
                from main import execute
                
                # Deserialize and validate input data
                input_data = input_schema.load(data.get('request', {}))
                
                # Create AppModel instance with proper input and empty output
                model = AppModel(request=input_data, response=OutputClass())
                
                # Execute the request
                execute(model)
                
                # Serialize the response using schema
                response_data = output_schema.dump(model.response)
                
                return jsonify({
                    'status': 'success',
                    'response': response_data
                })
            except Exception as e:
                logger.error(f"Error during execution: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @app.route('/config', methods=['POST'])
        def config():
            try:
                data = request.json
                from main import config
                
                # Create State instance
                state = State()
                
                # Execute config
                config(data.get('configuration', {}), state)
                
                return jsonify({
                    'status': 'success'
                })
            except Exception as e:
                logger.error(f"Error during config: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        if debug:
            logger.info(f"Starting server in debug mode on {host}:{port}")
            socketio.run(app, host=host, port=port, debug=True)
        else:
            logger.info(f"Starting production server on {host}:{port}")
            http_server = WSGIServer((host, port), app)
            http_server.serve_forever()

class State:
    def __init__(self):
        self._state = {}
    
    def get(self, key: str) -> Optional[Any]:
        return self._state.get(key)
    
    def set(self, key: str, value: Any):
        self._state[key] = value