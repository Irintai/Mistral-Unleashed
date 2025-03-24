"""
Template manager for handling code generation templates.
"""

import os
import json
import logging
import uuid
import re
from typing import Dict, List, Any, Optional

from constants import DEFAULT_TEMPLATE_CATEGORIES, DEFAULT_PROMPT_TEMPLATES

# Logger for this module
logger = logging.getLogger(__name__)

class TemplateManager:
    """Manages templates for code generation"""
    
    def __init__(self, templates_path: str = None):
        """
        Initialize the template manager
        
        Args:
            templates_path: Path to the templates file
        """
        self.templates_path = templates_path
        
        # Dictionary to store templates: template_id -> template
        self.templates: Dict[str, Dict[str, Any]] = {}
        
        # Try to load templates from file
        if self.templates_path and os.path.exists(self.templates_path):
            self.load_templates()
        
        # If no templates were loaded, use default templates
        if not self.templates:
            self.init_default_templates()
        
        logger.info(f"Template manager initialized with {len(self.templates)} templates")
    
    def set_templates_path(self, path: str) -> None:
        """
        Set the path to the templates file
        
        Args:
            path: Path to the templates file
        """
        self.templates_path = path
        
        # Load templates from new path if it exists
        if os.path.exists(path):
            self.load_templates()
    
    def init_default_templates(self) -> None:
        """Initialize with default templates from constants"""
        # Clear existing templates
        self.templates = {}
        
        # Add default templates
        for template in DEFAULT_PROMPT_TEMPLATES:
            template_id = str(uuid.uuid4())
            self.templates[template_id] = template
        
        logger.info(f"Initialized {len(self.templates)} default templates")
    
    def load_templates(self) -> bool:
        """
        Load templates from file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.templates_path or not os.path.exists(self.templates_path):
                logger.warning("Templates file not found")
                return False
            
            with open(self.templates_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Validate the data structure
                if not isinstance(data, dict) or "templates" not in data:
                    logger.warning("Invalid templates file format")
                    return False
                
                # Update templates dictionary
                self.templates = data["templates"]
            
            logger.info(f"Loaded {len(self.templates)} templates from {self.templates_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading templates: {str(e)}")
            return False
    
    def save_templates(self) -> bool:
        """
        Save templates to file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.templates_path:
                logger.warning("No templates file path specified")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.templates_path)), exist_ok=True)
            
            # Prepare data for saving
            data = {
                "version": 1,
                "templates": self.templates
            }
            
            # Save to file
            with open(self.templates_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.templates)} templates to {self.templates_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving templates: {str(e)}")
            return False
    
    def add_template(self, template: Dict[str, Any]) -> str:
        """
        Add a new template
        
        Args:
            template: Dictionary with template data
        
        Returns:
            str: ID of the new template
        """
        try:
            # Validate template structure
            if not self.validate_template(template):
                logger.warning("Invalid template structure")
                return ""
            
            # Generate a unique ID for the template
            template_id = str(uuid.uuid4())
            
            # Add to templates dictionary
            self.templates[template_id] = template
            
            # Auto-save
            self.save_templates()
            
            logger.debug(f"Added template: {template_id}")
            return template_id
            
        except Exception as e:
            logger.error(f"Error adding template: {str(e)}")
            return ""
    
    def remove_template(self, template_id: str) -> bool:
        """
        Remove a template
        
        Args:
            template_id: ID of the template to remove
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if template_id in self.templates:
                del self.templates[template_id]
                
                # Auto-save
                self.save_templates()
                
                logger.debug(f"Removed template: {template_id}")
                return True
            else:
                logger.warning(f"Template not found: {template_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing template: {str(e)}")
            return False
    
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a template
        
        Args:
            template_id: ID of the template to get
        
        Returns:
            dict: Template data, or None if not found
        """
        return self.templates.get(template_id, None)
    
    def update_template(self, template_id: str, template: Dict[str, Any]) -> bool:
        """
        Update an existing template
        
        Args:
            template_id: ID of the template to update
            template: Updated template data
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate template structure
            if not self.validate_template(template):
                logger.warning("Invalid template structure")
                return False
            
            if template_id in self.templates:
                # Update the template
                self.templates[template_id] = template
                
                # Auto-save
                self.save_templates()
                
                logger.debug(f"Updated template: {template_id}")
                return True
            else:
                logger.warning(f"Template not found: {template_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating template: {str(e)}")
            return False
    
    def get_all_templates(self) -> List[Dict[str, Any]]:
        """
        Get all templates as a list
        
        Returns:
            list: List of templates with their IDs
        """
        try:
            # Convert templates to list with ID included
            templates_list = []
            for template_id, template in self.templates.items():
                template_with_id = template.copy()
                template_with_id["id"] = template_id
                templates_list.append(template_with_id)
            
            # Sort by name
            return sorted(templates_list, key=lambda x: x.get("name", ""))
            
        except Exception as e:
            logger.error(f"Error getting all templates: {str(e)}")
            return []
    
    def get_templates_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get templates filtered by category
        
        Args:
            category: Category to filter by
        
        Returns:
            list: List of matching templates
        """
        try:
            # Get all templates
            all_templates = self.get_all_templates()
            
            # Filter by category
            return [
                template for template in all_templates
                if template.get("category", "").lower() == category.lower()
            ]
            
        except Exception as e:
            logger.error(f"Error getting templates by category: {str(e)}")
            return []
    
    def get_templates_by_language(self, language: str) -> List[Dict[str, Any]]:
        """
        Get templates filtered by language
        
        Args:
            language: Language to filter by
        
        Returns:
            list: List of matching templates
        """
        try:
            # Get all templates
            all_templates = self.get_all_templates()
            
            # Filter by language
            return [
                template for template in all_templates
                if template.get("language", "").lower() == language.lower()
            ]
            
        except Exception as e:
            logger.error(f"Error getting templates by language: {str(e)}")
            return []
    
    def get_categories(self) -> List[str]:
        """
        Get all unique categories
        
        Returns:
            list: List of categories
        """
        try:
            # Get categories from templates
            categories = set()
            for template in self.templates.values():
                if "category" in template and template["category"]:
                    categories.add(template["category"])
            
            # Include default categories if they're not already there
            for category in DEFAULT_TEMPLATE_CATEGORIES:
                categories.add(category)
            
            # Sort and return
            return sorted(list(categories))
            
        except Exception as e:
            logger.error(f"Error getting categories: {str(e)}")
            return DEFAULT_TEMPLATE_CATEGORIES
    
    def validate_template(self, template: Dict[str, Any]) -> bool:
        """
        Validate template structure
        
        Args:
            template: Template to validate
        
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required fields
            if "name" not in template or not template["name"]:
                logger.warning("Template missing required field: name")
                return False
            
            if "template" not in template or not template["template"]:
                logger.warning("Template missing required field: template")
                return False
            
            # Validate placeholders
            if "placeholders" in template:
                if not isinstance(template["placeholders"], list):
                    logger.warning("Template placeholders must be a list")
                    return False
                
                # Check each placeholder
                for placeholder in template["placeholders"]:
                    if not isinstance(placeholder, dict):
                        logger.warning("Each placeholder must be a dictionary")
                        return False
                    
                    if "name" not in placeholder or not placeholder["name"]:
                        logger.warning("Placeholder missing required field: name")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating template: {str(e)}")
            return False
    
    def apply_template(self, template_id: str, values: Dict[str, str]) -> str:
        """
        Apply values to a template
        
        Args:
            template_id: ID of the template to use
            values: Dictionary of placeholder values
        
        Returns:
            str: Filled template, or empty string if error
        """
        try:
            # Get template
            template = self.get_template(template_id)
            if not template:
                logger.warning(f"Template not found: {template_id}")
                return ""
            
            # Get template text
            template_text = template["template"]
            
            # Replace placeholders with values
            result = template_text
            
            # Use regex to find all placeholders of the form {{name}}
            placeholders = re.findall(r'{{([^}]+)}}', template_text)
            
            for placeholder in placeholders:
                placeholder = placeholder.strip()
                if placeholder in values:
                    result = result.replace(f"{{{{{placeholder}}}}}", values[placeholder])
                else:
                    # Find default value for this placeholder
                    default_value = ""
                    if "placeholders" in template:
                        for ph in template["placeholders"]:
                            if ph.get("name") == placeholder and "default" in ph:
                                default_value = ph["default"]
                                break
                    
                    # Replace with default value
                    result = result.replace(f"{{{{{placeholder}}}}}", default_value)
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying template: {str(e)}")
            return ""
    
    def get_template_placeholders(self, template_id: str) -> List[Dict[str, Any]]:
        """
        Get placeholders for a template
        
        Args:
            template_id: ID of the template
        
        Returns:
            list: List of placeholder dictionaries
        """
        try:
            # Get template
            template = self.get_template(template_id)
            if not template:
                logger.warning(f"Template not found: {template_id}")
                return []
            
            # Extract defined placeholders
            defined_placeholders = template.get("placeholders", [])
            
            # Find all placeholders in the template text using regex
            template_text = template["template"]
            placeholder_names = set(re.findall(r'{{([^}]+)}}', template_text))
            
            # Create result list
            result = []
            
            # Add defined placeholders first
            for placeholder in defined_placeholders:
                if placeholder["name"] in placeholder_names:
                    result.append(placeholder)
                    placeholder_names.remove(placeholder["name"])
            
            # Add any undefined placeholders found in the template text
            for name in placeholder_names:
                name = name.strip()
                if name:
                    result.append({
                        "name": name,
                        "description": name.replace("_", " ").capitalize(),
                        "default": ""
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting template placeholders: {str(e)}")
            return []