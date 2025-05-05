import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, Optional
from socketio import Client

class Remote:
    def __init__(self, uri: str, name: str = None):
        self.uri = uri
        self.name = name or str(uuid.uuid4())
        self.client = Client()
        self.handlers: Dict[str, Dict[str, Any]] = {}
        self._setup_handlers()
    
    def _setup_handlers(self):
        @self.client.on('connect')
        def on_connect():
            logging.info(f"[{self.name}] Connected to {self.uri}")
        
        @self.client.on('disconnect')
        def on_disconnect():
            logging.info(f"[{self.name}] Disconnected from {self.uri}")
        
        @self.client.on('response')
        def on_response(data):
            handler_id = data.get('handler')
            if handler_id in self.handlers:
                self.handlers[handler_id]['result'] = data.get('result')
                self.handlers[handler_id]['status'] = 'completed'
                logging.info(f"[{self.name}] Handler {handler_id} completed")
    
    def connect(self) -> 'Remote':
        """Connect to the WebSocket server"""
        try:
            self.client.connect(self.uri)
            return self
        except Exception as e:
            logging.error(f"[{self.name}] Connection failed: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from the WebSocket server"""
        try:
            self.client.disconnect()
        except Exception as e:
            logging.error(f"[{self.name}] Disconnection failed: {e}")
    
    def execute(self, data: Any, uid: str = 'super-user', timeout: int = 60) -> str:
        """Execute a request and return a handler ID"""
        handler_id = str(uuid.uuid4())
        self.handlers[handler_id] = {
            'status': 'pending',
            'result': None,
            'start_time': time.time(),
            'timeout': timeout
        }
        
        try:
            self.client.emit('execute', {
                'handler': handler_id,
                'uid': uid,
                'data': data
            })
            logging.info(f"[{self.name}] Request sent with handler {handler_id}")
            return handler_id
        except Exception as e:
            logging.error(f"[{self.name}] Execution failed: {e}")
            del self.handlers[handler_id]
            raise
    
    def get_response(self, handler_id: str) -> Optional[Dict[str, Any]]:
        """Get the response for a given handler ID"""
        if handler_id not in self.handlers:
            logging.error(f"[{self.name}] Handler {handler_id} not found")
            return None
        
        handler = self.handlers[handler_id]
        timeout = handler['timeout']
        start_time = handler['start_time']
        
        while handler['status'] == 'pending':
            if time.time() - start_time > timeout:
                logging.error(f"[{self.name}] Handler {handler_id} timed out")
                del self.handlers[handler_id]
                return None
            time.sleep(0.1)
        
        result = handler['result']
        del self.handlers[handler_id]
        return result
    
    def __del__(self):
        if hasattr(self, 'client') and self.client.connected:
            self.disconnect()
