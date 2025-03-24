"""
History manager for tracking and managing code generation history.
"""

import os
import json
import time
import logging
import uuid
from typing import Dict, List, Any, Optional

# Logger for this module
logger = logging.getLogger(__name__)

class HistoryManager:
    """Manages history entries for code generation"""
    
    def __init__(self, history_path: str = None, max_entries: int = 100):
        """
        Initialize the history manager
        
        Args:
            history_path: Path to the history file
            max_entries: Maximum number of history entries to keep
        """
        self.history_path = history_path
        self.max_entries = max_entries
        
        # Dictionary to store history entries: entry_id -> entry
        self.entries: Dict[str, Dict[str, Any]] = {}
        
        # Load history from file if it exists
        if self.history_path and os.path.exists(self.history_path):
            self.load_history()
        
        logger.info(f"History manager initialized with {len(self.entries)} entries")
    
    def set_history_path(self, path: str) -> None:
        """
        Set the path to the history file
        
        Args:
            path: Path to the history file
        """
        self.history_path = path
        
        # Load history from new path if it exists
        if os.path.exists(path):
            self.load_history()
    
    def set_max_entries(self, max_entries: int) -> None:
        """
        Set the maximum number of history entries
        
        Args:
            max_entries: Maximum number of entries to keep
        """
        self.max_entries = max_entries
        
        # Trim history if needed
        if len(self.entries) > self.max_entries:
            self.trim_history()
    
    def load_history(self) -> bool:
        """
        Load history from file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.history_path or not os.path.exists(self.history_path):
                logger.warning("History file not found")
                return False
            
            with open(self.history_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Validate the data structure
                if not isinstance(data, dict) or "entries" not in data:
                    logger.warning("Invalid history file format")
                    return False
                
                # Update entries dictionary
                self.entries = data["entries"]
            
            logger.info(f"Loaded {len(self.entries)} history entries from {self.history_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading history: {str(e)}")
            return False
    
    def save_history(self) -> bool:
        """
        Save history to file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.history_path:
                logger.warning("No history file path specified")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.history_path)), exist_ok=True)
            
            # Prepare data for saving
            data = {
                "version": 1,
                "timestamp": time.time(),
                "entries": self.entries
            }
            
            # Save to file
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.entries)} history entries to {self.history_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving history: {str(e)}")
            return False
    
    def add_entry(self, entry: Dict[str, Any]) -> str:
        """
        Add a new entry to the history
        
        Args:
            entry: Dictionary with entry data
        
        Returns:
            str: ID of the new entry
        """
        try:
            # Generate a unique ID for the entry
            entry_id = str(uuid.uuid4())
            
            # Ensure required fields
            if "timestamp" not in entry:
                entry["timestamp"] = time.time()
            
            if "title" not in entry:
                # Create a title from prompt
                if "prompt" in entry:
                    prompt = entry["prompt"]
                    entry["title"] = prompt[:50] + "..." if len(prompt) > 50 else prompt
                else:
                    entry["title"] = f"Entry {entry_id[:8]}"
            
            # Add to entries dictionary
            self.entries[entry_id] = entry
            
            # Trim history if needed
            if len(self.entries) > self.max_entries:
                self.trim_history()
            
            # Auto-save
            self.save_history()
            
            logger.debug(f"Added history entry: {entry_id}")
            return entry_id
            
        except Exception as e:
            logger.error(f"Error adding history entry: {str(e)}")
            return ""
    
    def remove_entry(self, entry_id: str) -> bool:
        """
        Remove an entry from the history
        
        Args:
            entry_id: ID of the entry to remove
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if entry_id in self.entries:
                del self.entries[entry_id]
                
                # Auto-save
                self.save_history()
                
                logger.debug(f"Removed history entry: {entry_id}")
                return True
            else:
                logger.warning(f"Entry not found: {entry_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing history entry: {str(e)}")
            return False
    
    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entry from the history
        
        Args:
            entry_id: ID of the entry to get
        
        Returns:
            dict: Entry data, or None if not found
        """
        return self.entries.get(entry_id, None)
    
    def update_entry(self, entry_id: str, entry: Dict[str, Any]) -> bool:
        """
        Update an existing entry
        
        Args:
            entry_id: ID of the entry to update
            entry: Updated entry data
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if entry_id in self.entries:
                # Update the entry
                self.entries[entry_id] = entry
                
                # Auto-save
                self.save_history()
                
                logger.debug(f"Updated history entry: {entry_id}")
                return True
            else:
                logger.warning(f"Entry not found: {entry_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating history entry: {str(e)}")
            return False
    
    def get_all_entries(self) -> List[Dict[str, Any]]:
        """
        Get all history entries as a list, sorted by timestamp (newest first)
        
        Returns:
            list: List of entries with their IDs
        """
        try:
            # Convert entries to list with ID included
            entries_list = []
            for entry_id, entry in self.entries.items():
                entry_with_id = entry.copy()
                entry_with_id["id"] = entry_id
                entries_list.append(entry_with_id)
            
            # Sort by timestamp (newest first)
            return sorted(entries_list, key=lambda x: x.get("timestamp", 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting all entries: {str(e)}")
            return []
    
    def search_entries(self, query: str = None, language: str = None, 
                     tag: str = None) -> List[Dict[str, Any]]:
        """
        Search history entries based on criteria
        
        Args:
            query: Search query (searches in title, prompt, and code)
            language: Filter by programming language
            tag: Filter by tag
        
        Returns:
            list: List of matching entries
        """
        try:
            # Get all entries as a starting point
            results = self.get_all_entries()
            
            # Apply filters
            if query:
                query = query.lower()
                results = [
                    entry for entry in results if
                    (entry.get("title", "").lower().find(query) >= 0 or
                     entry.get("prompt", "").lower().find(query) >= 0 or
                     entry.get("code", "").lower().find(query) >= 0)
                ]
            
            if language:
                results = [
                    entry for entry in results if
                    entry.get("language", "").lower() == language.lower()
                ]
            
            if tag:
                results = [
                    entry for entry in results if
                    tag.lower() in [t.lower() for t in entry.get("tags", [])]
                ]
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching entries: {str(e)}")
            return []
    
    def clear_history(self) -> bool:
        """
        Clear all history entries
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.entries = {}
            
            # Auto-save
            self.save_history()
            
            logger.info("History cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing history: {str(e)}")
            return False
    
    def trim_history(self) -> None:
        """
        Trim history to the maximum number of entries (oldest first)
        """
        try:
            if len(self.entries) <= self.max_entries:
                return
            
            # Get all entries sorted by timestamp (oldest first)
            all_entries = [(entry_id, entry.get("timestamp", 0))
                         for entry_id, entry in self.entries.items()]
            all_entries.sort(key=lambda x: x[1])
            
            # Calculate how many to remove
            to_remove = len(all_entries) - self.max_entries
            
            # Remove oldest entries
            for i in range(to_remove):
                entry_id = all_entries[i][0]
                del self.entries[entry_id]
            
            logger.debug(f"Trimmed {to_remove} history entries")
            
        except Exception as e:
            logger.error(f"Error trimming history: {str(e)}")