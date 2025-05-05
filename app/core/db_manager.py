import sqlite3
import os
import logging
from typing import List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DBManager:
    def __init__(self):
        """Initialize database connection and create tables if they don't exist"""
        db_path = os.path.join(os.path.dirname(__file__), '..', 'datastore', 'generations.db')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        
    def _create_tables(self):
        """Create necessary database tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Create generations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_prompt TEXT NOT NULL,
                enhanced_prompt TEXT NOT NULL,
                image_path TEXT,
                bg_removed_path TEXT,
                model_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create tags table for searchability
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation_id INTEGER,
                tag TEXT NOT NULL,
                FOREIGN KEY (generation_id) REFERENCES generations(id),
                UNIQUE(generation_id, tag)
            )
        ''')
        
        self.conn.commit()
    
    def save_generation(self, original_prompt: str, enhanced_prompt: str, tags: List[str] = None) -> int:
        """Save a new generation and return its ID"""
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO generations (original_prompt, enhanced_prompt) VALUES (?, ?)',
            (original_prompt, enhanced_prompt)
        )
        generation_id = cursor.lastrowid
        
        # Save tags if provided
        if tags:
            self.save_tags(generation_id, tags)
            
        self.conn.commit()
        return generation_id
    
    def save_tags(self, generation_id: int, tags: List[str]):
        """Save tags for a generation"""
        cursor = self.conn.cursor()
        for tag in tags:
            try:
                cursor.execute(
                    'INSERT INTO tags (generation_id, tag) VALUES (?, ?)',
                    (generation_id, tag)
                )
            except sqlite3.IntegrityError:
                # Skip duplicate tags
                continue
        self.conn.commit()
    
    def update_paths(self, generation_id: int, image_path: Optional[str] = None, 
                    bg_removed_path: Optional[str] = None, model_path: Optional[str] = None):
        """Update the file paths for a generation"""
        cursor = self.conn.cursor()
        updates = []
        params = []
        
        if image_path:
            updates.append("image_path = ?")
            params.append(image_path)
        if bg_removed_path:
            updates.append("bg_removed_path = ?")
            params.append(bg_removed_path)
        if model_path:
            updates.append("model_path = ?")
            params.append(model_path)
            
        if updates:
            query = f"UPDATE generations SET {', '.join(updates)} WHERE id = ?"
            params.append(generation_id)
            cursor.execute(query, tuple(params))
            self.conn.commit()
    
    def get_recent_generations(self, limit: int = 10) -> List[Tuple]:
        """Get the most recent generations"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, original_prompt, enhanced_prompt, image_path, bg_removed_path, model_path, created_at
            FROM generations 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        return cursor.fetchall()
    
    def search_generations(self, query: str) -> List[Tuple]:
        """Search generations by prompt text or tags"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT DISTINCT g.id, g.original_prompt, g.enhanced_prompt, 
                   g.image_path, g.bg_removed_path, g.model_path, g.created_at
            FROM generations g
            LEFT JOIN tags t ON g.id = t.generation_id
            WHERE g.original_prompt LIKE ? 
               OR g.enhanced_prompt LIKE ?
               OR t.tag LIKE ?
            ORDER BY g.created_at DESC
            LIMIT 10
        ''', (f'%{query}%', f'%{query}%', f'%{query}%'))
        return cursor.fetchall()
    
    def __del__(self):
        """Close the database connection when the object is destroyed"""
        if hasattr(self, 'conn'):
            self.conn.close()

# Create singleton instance
db_manager = DBManager()