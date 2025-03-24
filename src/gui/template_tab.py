"""
Template tab for the Advanced Code Generator.
Provides UI for managing and using code generation templates.
"""

import os
import logging
import json
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                           QComboBox, QPushButton, QGroupBox, QFormLayout, 
                           QListWidget, QListWidgetItem, QSplitter, QTextEdit,
                           QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
                           QMenu, QAction, QMessageBox, QFileDialog, QDialog,
                           QDialogButtonBox, QRadioButton, QSpinBox, QDoubleSpinBox,
                           QScrollArea, QFrame, QStackedWidget)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QIcon, QFont
import torch
import time

from constants import SUPPORTED_LANGUAGES, DEFAULT_TEMPLATE_CATEGORIES

# Logger for this module
logger = logging.getLogger(__name__)

class TemplateEditDialog(QDialog):
    """Dialog for editing templates"""
    
    def __init__(self, template=None, parent=None):
        super().__init__(parent)
        
        self.template = template or {}
        self.setWindowTitle("Edit Template" if template else "New Template")
        self.resize(600, 500)
        
        self.init_ui()
        self.load_template()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        
        # Create scroll area for the form
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Create container for the form
        form_container = QWidget()
        form_layout = QVBoxLayout()
        
        # Basic info group
        basic_group = QGroupBox("Basic Information")
        basic_layout = QFormLayout()
        
        # Template name
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter template name")
        basic_layout.addRow("Name:", self.name_edit)
        
        # Language
        self.language_combo = QComboBox()
        self.language_combo.addItems(SUPPORTED_LANGUAGES)
        basic_layout.addRow("Language:", self.language_combo)
        
        # Category
        self.category_combo = QComboBox()
        self.category_combo.setEditable(True)
        self.category_combo.addItems(DEFAULT_TEMPLATE_CATEGORIES)
        basic_layout.addRow("Category:", self.category_combo)
        
        basic_group.setLayout(basic_layout)
        form_layout.addWidget(basic_group)
        
        # Template content group
        content_group = QGroupBox("Template Content")
        content_layout = QVBoxLayout()
        
        # Template text editor
        self.template_editor = QTextEdit()
        self.template_editor.setPlaceholderText("Enter template content with {{placeholders}}")
        self.template_editor.setMinimumHeight(150)
        
        content_layout.addWidget(QLabel("Template Text:"))
        content_layout.addWidget(self.template_editor)
        
        # Help text
        help_label = QLabel(
            "Use {{placeholder}} syntax for variables. Example: \"Create a function named {{function_name}} that {{purpose}}.\""
        )
        help_label.setWordWrap(True)
        content_layout.addWidget(help_label)
        
        content_group.setLayout(content_layout)
        form_layout.addWidget(content_group)
        
        # Placeholders group
        placeholder_group = QGroupBox("Placeholders")
        placeholder_layout = QVBoxLayout()
        
        # Placeholder table
        self.placeholder_table = QTableWidget()
        self.placeholder_table.setColumnCount(3)
        self.placeholder_table.setHorizontalHeaderLabels(["Name", "Description", "Default Value"])
        self.placeholder_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.placeholder_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.placeholder_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        
        placeholder_layout.addWidget(self.placeholder_table)
        
        # Add/remove placeholder buttons
        ph_buttons_layout = QHBoxLayout()
        
        self.add_placeholder_button = QPushButton("Add Placeholder")
        self.add_placeholder_button.clicked.connect(self.add_placeholder)
        ph_buttons_layout.addWidget(self.add_placeholder_button)
        
        self.remove_placeholder_button = QPushButton("Remove Selected")
        self.remove_placeholder_button.clicked.connect(self.remove_placeholder)
        ph_buttons_layout.addWidget(self.remove_placeholder_button)
        
        self.detect_button = QPushButton("Detect Placeholders")
        self.detect_button.clicked.connect(self.detect_placeholders)
        ph_buttons_layout.addWidget(self.detect_button)
        
        placeholder_layout.addLayout(ph_buttons_layout)
        
        placeholder_group.setLayout(placeholder_layout)
        form_layout.addWidget(placeholder_group)
        
        # Set container layout
        form_container.setLayout(form_layout)
        scroll_area.setWidget(form_container)
        layout.addWidget(scroll_area)
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def load_template(self):
        """Load template data into UI"""
        if not self.template:
            return
        
        # Basic info
        self.name_edit.setText(self.template.get("name", ""))
        
        # Language
        language = self.template.get("language", "python")
        index = self.language_combo.findText(language)
        if index >= 0:
            self.language_combo.setCurrentIndex(index)
        
        # Category
        category = self.template.get("category", "")
        index = self.category_combo.findText(category)
        if index >= 0:
            self.category_combo.setCurrentIndex(index)
        else:
            self.category_combo.setEditText(category)
        
        # Template text
        self.template_editor.setText(self.template.get("template", ""))
        
        # Load placeholders
        self.load_placeholders()
    
    def load_placeholders(self):
        """Load placeholders into table"""
        placeholders = self.template.get("placeholders", [])
        
        # Clear table
        self.placeholder_table.setRowCount(0)
        
        # Add placeholders to table
        for ph in placeholders:
            row_position = self.placeholder_table.rowCount()
            self.placeholder_table.insertRow(row_position)
            
            # Name
            name_item = QTableWidgetItem(ph.get("name", ""))
            self.placeholder_table.setItem(row_position, 0, name_item)
            
            # Description
            desc_item = QTableWidgetItem(ph.get("description", ""))
            self.placeholder_table.setItem(row_position, 1, desc_item)
            
            # Default value
            default_item = QTableWidgetItem(str(ph.get("default", "")))
            self.placeholder_table.setItem(row_position, 2, default_item)
    
    def add_placeholder(self):
        """Add a new placeholder row"""
        row_position = self.placeholder_table.rowCount()
        self.placeholder_table.insertRow(row_position)
        
        # Add empty items
        self.placeholder_table.setItem(row_position, 0, QTableWidgetItem(""))
        self.placeholder_table.setItem(row_position, 1, QTableWidgetItem(""))
        self.placeholder_table.setItem(row_position, 2, QTableWidgetItem(""))
    
    def remove_placeholder(self):
        """Remove selected placeholder row"""
        selected_rows = self.placeholder_table.selectedItems()
        
        if not selected_rows:
            return
        
        # Get unique rows
        rows = set()
        for item in selected_rows:
            rows.add(item.row())
        
        # Remove rows in reverse order
        for row in sorted(rows, reverse=True):
            self.placeholder_table.removeRow(row)
    
    def detect_placeholders(self):
        """Detect placeholders from template text"""
        template_text = self.template_editor.toPlainText()
        
        # Use regex to find placeholders
        import re
        placeholders = re.findall(r'{{([^}]+)}}', template_text)
        
        # Create a set of unique placeholders
        unique_placeholders = set()
        for ph in placeholders:
            ph = ph.strip()
            if ph:
                unique_placeholders.add(ph)
        
        # Get existing placeholders
        existing_names = set()
        for row in range(self.placeholder_table.rowCount()):
            name = self.placeholder_table.item(row, 0).text().strip()
            if name:
                existing_names.add(name)
        
        # Add new placeholders
        for ph in unique_placeholders:
            if ph not in existing_names:
                row_position = self.placeholder_table.rowCount()
                self.placeholder_table.insertRow(row_position)
                
                # Add items
                self.placeholder_table.setItem(row_position, 0, QTableWidgetItem(ph))
                
                # Generate description from name
                desc = ph.replace("_", " ").capitalize()
                self.placeholder_table.setItem(row_position, 1, QTableWidgetItem(desc))
                
                # Empty default value
                self.placeholder_table.setItem(row_position, 2, QTableWidgetItem(""))
    
    def get_template_data(self):
        """Get template data from UI"""
        template = {}
        
        # Basic info
        template["name"] = self.name_edit.text().strip()
        template["language"] = self.language_combo.currentText()
        template["category"] = self.category_combo.currentText()
        
        # Template text
        template["template"] = self.template_editor.toPlainText()
        
        # Placeholders
        placeholders = []
        for row in range(self.placeholder_table.rowCount()):
            name = self.placeholder_table.item(row, 0).text().strip()
            
            if name:  # Only add if name is not empty
                desc = self.placeholder_table.item(row, 1).text().strip()
                default = self.placeholder_table.item(row, 2).text()
                
                placeholders.append({
                    "name": name,
                    "description": desc,
                    "default": default
                })
        
        template["placeholders"] = placeholders
        
        return template
    
    def validate(self):
        """Validate template data"""
        # Check name
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Validation Error", "Template name cannot be empty")
            return False
        
        # Check template text
        if not self.template_editor.toPlainText().strip():
            QMessageBox.warning(self, "Validation Error", "Template text cannot be empty")
            return False
        
        # Check placeholders
        for row in range(self.placeholder_table.rowCount()):
            name = self.placeholder_table.item(row, 0).text().strip()
            
            if not name:
                QMessageBox.warning(
                    self, 
                    "Validation Error", 
                    f"Placeholder name in row {row+1} cannot be empty"
                )
                return False
        
        return True
    
    def accept(self):
        """Handle dialog acceptance"""
        if not self.validate():
            return
        
        # Call parent accept
        super().accept()


class TemplateApplyDialog(QDialog):
    """Dialog for applying a template with placeholder values"""
    
    def __init__(self, template, parent=None):
        super().__init__(parent)
        
        self.template = template
        self.setWindowTitle(f"Apply Template: {template.get('name', '')}")
        self.resize(500, 400)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        
        # Create scroll area for the form
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Create container for the form
        form_container = QWidget()
        form_layout = QVBoxLayout()
        
        # Template info
        info_text = (
            f"<b>Template:</b> {self.template.get('name', '')}<br>"
            f"<b>Language:</b> {self.template.get('language', '')}<br>"
            f"<b>Category:</b> {self.template.get('category', '')}"
        )
        info_label = QLabel(info_text)
        form_layout.addWidget(info_label)
        
        # Placeholder values group
        values_group = QGroupBox("Placeholder Values")
        values_layout = QFormLayout()
        
        # Add fields for each placeholder
        self.value_fields = {}
        placeholders = self.template.get("placeholders", [])
        
        if not placeholders:
            values_layout.addRow(QLabel("No placeholders in this template"))
        
        for ph in placeholders:
            name = ph.get("name", "")
            description = ph.get("description", "")
            default = ph.get("default", "")
            
            # Create field
            field = QLineEdit()
            field.setText(default)
            field.setPlaceholderText(description)
            
            # Add to form
            label_text = f"{name}:"
            if description:
                label_text = f"{name} ({description}):"
            
            values_layout.addRow(label_text, field)
            
            # Store field reference
            self.value_fields[name] = field
        
        values_group.setLayout(values_layout)
        form_layout.addWidget(values_group)
        
        # Preview group
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_editor = QTextEdit()
        self.preview_editor.setReadOnly(True)
        preview_layout.addWidget(self.preview_editor)
        
        preview_button = QPushButton("Generate Preview")
        preview_button.clicked.connect(self.generate_preview)
        preview_layout.addWidget(preview_button)
        
        preview_group.setLayout(preview_layout)
        form_layout.addWidget(preview_group)
        
        # Set container layout
        form_container.setLayout(form_layout)
        scroll_area.setWidget(form_container)
        layout.addWidget(scroll_area)
        
        # Button box
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        # Rename Ok button to "Apply Template"
        self.button_box.button(QDialogButtonBox.Ok).setText("Apply Template")
        
        layout.addWidget(self.button_box)
        
        self.setLayout(layout)
        
        # Generate initial preview
        QTimer = QTimer.singleShot(100, self.generate_preview)
    
    def generate_preview(self):
        """Generate preview with current placeholder values"""
        # Get template text
        template_text = self.template.get("template", "")
        
        # Get current values
        values = {}
        for name, field in self.value_fields.items():
            values[name] = field.text()
        
        # Replace placeholders
        preview = template_text
        for name, value in values.items():
            preview = preview.replace(f"{{{{{name}}}}}", value)
        
        # Update preview
        self.preview_editor.setText(preview)
    
    def get_values(self):
        """Get current placeholder values"""
        values = {}
        for name, field in self.value_fields.items():
            values[name] = field.text()
        return values
    
    def get_filled_template(self):
        """Get template filled with current values"""
        # Get template text
        template_text = self.template.get("template", "")
        
        # Get current values
        values = self.get_values()
        
        # Replace placeholders
        filled_template = template_text
        for name, value in values.items():
            filled_template = filled_template.replace(f"{{{{{name}}}}}", value)
        
        return filled_template


class TemplateTab(QWidget):
    """Tab for managing templates"""
    
    # Signals
    template_applied = pyqtSignal(str, str)  # prompt, language
    
    def __init__(self, template_manager, parent=None):
        super().__init__(parent)
        
        self.template_manager = template_manager
        self.current_template_id = None
        
        # Initialize UI
        self.init_ui()
        
        # Load templates
        self.load_templates()
    
    def init_ui(self):
        """Initialize the UI components"""
        # Main layout
        main_layout = QVBoxLayout()
        
        # Top toolbar with actions
        toolbar_layout = QHBoxLayout()
        
        # Category filter
        self.category_filter = QComboBox()
        self.category_filter.addItem("All Categories")
        self.category_filter.currentTextChanged.connect(self.filter_templates)
        toolbar_layout.addWidget(QLabel("Category:"))
        toolbar_layout.addWidget(self.category_filter)
        
        # Language filter
        self.language_filter = QComboBox()
        self.language_filter.addItem("All Languages")
        self.language_filter.addItems(SUPPORTED_LANGUAGES)
        self.language_filter.currentTextChanged.connect(self.filter_templates)
        toolbar_layout.addWidget(QLabel("Language:"))
        toolbar_layout.addWidget(self.language_filter)
        
        # Spacer
        toolbar_layout.addStretch()
        
        # New template button
        self.new_template_button = QPushButton("New Template")
        self.new_template_button.clicked.connect(self.create_template)
        toolbar_layout.addWidget(self.new_template_button)
        
        main_layout.addLayout(toolbar_layout)
        
        # Main splitter
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Templates list on left
        templates_group = QGroupBox("Templates")
        templates_layout = QVBoxLayout()
        
        self.templates_list = QListWidget()
        self.templates_list.itemSelectionChanged.connect(self.on_template_selected)
        self.templates_list.doubleClicked.connect(self.apply_template)
        self.templates_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.templates_list.customContextMenuRequested.connect(self.show_context_menu)
        
        templates_layout.addWidget(self.templates_list)
        
        templates_group.setLayout(templates_layout)
        self.splitter.addWidget(templates_group)
        
        # Template details on right
        details_group = QGroupBox("Template Details")
        details_layout = QVBoxLayout()
        
        # Template info
        info_layout = QFormLayout()
        
        self.template_name_label = QLabel("")
        info_layout.addRow("Name:", self.template_name_label)
        
        self.template_language_label = QLabel("")
        info_layout.addRow("Language:", self.template_language_label)
        
        self.template_category_label = QLabel("")
        info_layout.addRow("Category:", self.template_category_label)
        
        details_layout.addLayout(info_layout)
        
        # Template content
        details_layout.addWidget(QLabel("Template:"))
        
        self.template_content = QTextEdit()
        self.template_content.setReadOnly(True)
        details_layout.addWidget(self.template_content)
        
        # Placeholders
        details_layout.addWidget(QLabel("Placeholders:"))
        
        self.placeholders_list = QListWidget()
        details_layout.addWidget(self.placeholders_list)
        
        # Action buttons
        actions_layout = QHBoxLayout()
        
        self.apply_button = QPushButton("Apply Template")
        self.apply_button.clicked.connect(self.apply_template)
        self.apply_button.setEnabled(False)
        actions_layout.addWidget(self.apply_button)
        
        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self.edit_template)
        self.edit_button.setEnabled(False)
        actions_layout.addWidget(self.edit_button)
        
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_template)
        self.delete_button.setEnabled(False)
        actions_layout.addWidget(self.delete_button)
        
        details_layout.addLayout(actions_layout)
        
        details_group.setLayout(details_layout)
        self.splitter.addWidget(details_group)
        
        # Set initial splitter sizes (40% list, 60% details)
        self.splitter.setSizes([200, 300])
        
        main_layout.addWidget(self.splitter)
        
        # Status label
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
        # Set the main layout
        self.setLayout(main_layout)
    
    def load_templates(self):
        """Load templates from manager"""
        try:
            # Update category filter
            current_category = self.category_filter.currentText()
            self.category_filter.clear()
            
            categories = ["All Categories"] + self.template_manager.get_categories()
            self.category_filter.addItems(categories)
            
            # Try to restore previous selection
            index = self.category_filter.findText(current_category)
            if index >= 0:
                self.category_filter.setCurrentIndex(index)
            
            # Filter templates
            self.filter_templates()
            
        except Exception as e:
            logger.error(f"Error loading templates: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
    
    def filter_templates(self):
        """Filter templates based on category and language"""
        try:
            # Clear list
            self.templates_list.clear()
            
            # Get all templates
            templates = self.template_manager.get_all_templates()
            
            # Get filters
            category = self.category_filter.currentText()
            language = self.language_filter.currentText()
            
            # Apply filters
            filtered_templates = []
            for template in templates:
                # Apply category filter
                if category != "All Categories" and template.get("category") != category:
                    continue
                
                # Apply language filter
                if language != "All Languages" and template.get("language") != language:
                    continue
                
                filtered_templates.append(template)
            
            # Sort templates by name
            filtered_templates.sort(key=lambda x: x.get("name", ""))
            
            # Add to list
            for template in filtered_templates:
                item = QListWidgetItem(template.get("name", ""))
                item.setData(Qt.UserRole, template.get("id"))  # Store ID in item data
                self.templates_list.addItem(item)
            
            # Update status
            if len(filtered_templates) != len(templates):
                self.status_label.setText(f"Showing {len(filtered_templates)} of {len(templates)} templates")
            else:
                self.status_label.setText(f"Total templates: {len(templates)}")
            
            # Clear template details
            self.clear_template_details()
            
        except Exception as e:
            logger.error(f"Error filtering templates: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
    
    def on_template_selected(self):
        """Handle template selection"""
        selected_items = self.templates_list.selectedItems()
        
        if not selected_items:
            # Clear details
            self.clear_template_details()
            self.apply_button.setEnabled(False)
            self.edit_button.setEnabled(False)
            self.delete_button.setEnabled(False)
            return
        
        # Get template ID
        template_id = selected_items[0].data(Qt.UserRole)
        
        # Load and display template
        self.load_template_details(template_id)
        
        # Enable actions
        self.apply_button.setEnabled(True)
        self.edit_button.setEnabled(True)
        self.delete_button.setEnabled(True)
    
    def load_template_details(self, template_id):
        """Load and display template details"""
        try:
            # Get template
            template = self.template_manager.get_template(template_id)
            
            if not template:
                self.clear_template_details()
                logger.warning(f"Template not found: {template_id}")
                return
            
            # Store current template ID
            self.current_template_id = template_id
            
            # Update info
            self.template_name_label.setText(template.get("name", ""))
            self.template_language_label.setText(template.get("language", ""))
            self.template_category_label.setText(template.get("category", ""))
            
            # Update template content
            self.template_content.setText(template.get("template", ""))
            
            # Update placeholders
            self.placeholders_list.clear()
            
            placeholders = template.get("placeholders", [])
            for ph in placeholders:
                name = ph.get("name", "")
                description = ph.get("description", "")
                default = ph.get("default", "")
                
                item_text = name
                if description:
                    item_text += f" - {description}"
                if default:
                    item_text += f" (Default: {default})"
                
                self.placeholders_list.addItem(item_text)
            
        except Exception as e:
            logger.error(f"Error loading template details: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
    
    def clear_template_details(self):
        """Clear template details display"""
        self.current_template_id = None
        self.template_name_label.clear()
        self.template_language_label.clear()
        self.template_category_label.clear()
        self.template_content.clear()
        self.placeholders_list.clear()
    
    def create_template(self):
        """Create a new template"""
        dialog = TemplateEditDialog(parent=self)
        
        if dialog.exec_() == QDialog.Accepted:
            try:
                # Get template data
                template_data = dialog.get_template_data()
                
                # Add to manager
                template_id = self.template_manager.add_template(template_data)
                
                if template_id:
                    self.status_label.setText(f"Template created: {template_data['name']}")
                    
                    # Reload templates
                    self.load_templates()
                    
                    # Select the new template
                    for i in range(self.templates_list.count()):
                        item = self.templates_list.item(i)
                        if item.data(Qt.UserRole) == template_id:
                            self.templates_list.setCurrentItem(item)
                            break
                else:
                    QMessageBox.warning(
                        self,
                        "Creation Failed",
                        "Failed to create template. See log for details."
                    )
                    
            except Exception as e:
                logger.error(f"Error creating template: {str(e)}")
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Error creating template: {str(e)}"
                )
    
    def edit_template(self):
        """Edit the current template"""
        if not self.current_template_id:
            return
        
        try:
            # Get template
            template = self.template_manager.get_template(self.current_template_id)
            
            if not template:
                logger.warning(f"Template not found for editing: {self.current_template_id}")
                return
            
            # Create dialog with template data
            dialog = TemplateEditDialog(template, parent=self)
            
            if dialog.exec_() == QDialog.Accepted:
                # Get updated template data
                template_data = dialog.get_template_data()
                
                # Update in manager
                success = self.template_manager.update_template(self.current_template_id, template_data)
                
                if success:
                    self.status_label.setText(f"Template updated: {template_data['name']}")
                    
                    # Reload templates
                    self.load_templates()
                    
                    # Try to re-select the template
                    for i in range(self.templates_list.count()):
                        item = self.templates_list.item(i)
                        if item.data(Qt.UserRole) == self.current_template_id:
                            self.templates_list.setCurrentItem(item)
                            break
                else:
                    QMessageBox.warning(
                        self,
                        "Update Failed",
                        "Failed to update template. See log for details."
                    )
                    
        except Exception as e:
            logger.error(f"Error editing template: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"Error editing template: {str(e)}"
            )
    
    def delete_template(self):
        """Delete the current template"""
        if not self.current_template_id:
            return
        
        # Confirm deletion
        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            "Are you sure you want to delete this template?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.No:
            return
        
        try:
            # Get template name for status message
            template = self.template_manager.get_template(self.current_template_id)
            template_name = template.get("name", "") if template else "Template"
            
            # Delete the template
            success = self.template_manager.remove_template(self.current_template_id)
            
            if success:
                # Update UI
                self.status_label.setText(f"Template deleted: {template_name}")
                self.clear_template_details()
                
                # Reload templates
                self.load_templates()
            else:
                QMessageBox.warning(
                    self,
                    "Deletion Failed",
                    "Failed to delete the template. See log for details."
                )
                
        except Exception as e:
            logger.error(f"Error deleting template: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"Error deleting template: {str(e)}"
            )
    
    def apply_template(self):
        """Apply the current template"""
        if not self.current_template_id:
            return
        
        try:
            # Get template
            template = self.template_manager.get_template(self.current_template_id)
            
            if not template:
                logger.warning(f"Template not found for application: {self.current_template_id}")
                return
            
            # Create dialog
            dialog = TemplateApplyDialog(template, parent=self)
            
            if dialog.exec_() == QDialog.Accepted:
                # Get filled template
                filled_template = dialog.get_filled_template()
                language = template.get("language", "python")
                
                # Emit signal with result
                self.template_applied.emit(filled_template, language)
                
                self.status_label.setText(f"Template applied: {template.get('name', '')}")
                
        except Exception as e:
            logger.error(f"Error applying template: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"Error applying template: {str(e)}"
            )
    
    def show_context_menu(self, position):
        """Show context menu for templates list"""
        if not self.templates_list.selectedItems():
            return
        
        # Create menu
        context_menu = QMenu(self)
        
        # Add actions
        apply_action = QAction("Apply Template", self)
        apply_action.triggered.connect(self.apply_template)
        context_menu.addAction(apply_action)
        
        edit_action = QAction("Edit Template", self)
        edit_action.triggered.connect(self.edit_template)
        context_menu.addAction(edit_action)
        
        context_menu.addSeparator()
        
        delete_action = QAction("Delete Template", self)
        delete_action.triggered.connect(self.delete_template)
        context_menu.addAction(delete_action)
        
        # Show the menu
        context_menu.exec_(self.templates_list.mapToGlobal(position))
    
    def import_templates(self):
        """Import templates from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Templates",
            os.path.expanduser("~"),
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Read templates from file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure
            if not isinstance(data, dict) or "templates" not in data:
                QMessageBox.warning(
                    self,
                    "Import Failed",
                    "Invalid template file format"
                )
                return
            
            # Get templates
            imported_templates = data["templates"]
            
            # Count imported
            count = 0
            
            # Add each template
            for template_id, template in imported_templates.items():
                # Validate template
                if not self.template_manager.validate_template(template):
                    continue
                
                # Add template
                new_id = self.template_manager.add_template(template)
                
                if new_id:
                    count += 1
            
            # Update UI
            self.load_templates()
            
            # Show result
            QMessageBox.information(
                self,
                "Import Successful",
                f"Imported {count} templates from {file_path}"
            )
            
            self.status_label.setText(f"Imported {count} templates")
            
        except Exception as e:
            logger.error(f"Error importing templates: {str(e)}")
            QMessageBox.critical(
                self,
                "Import Error",
                f"Error importing templates: {str(e)}"
            )
    
    def export_templates(self):
        """Export templates to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Templates",
            os.path.expanduser("~/templates.json"),
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Get templates
            templates = self.template_manager.get_all_templates()
            
            # Prepare data for saving
            data = {
                "version": 1,
                "timestamp": time.time(),
                "templates": {}
            }
            
            # Add templates (without IDs in the export)
            for template in templates:
                template_id = template.pop("id", None)
                data["templates"][template_id] = template
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            self.status_label.setText(f"Exported {len(templates)} templates to {file_path}")
            
            # Show result
            QMessageBox.information(
                self,
                "Export Successful",
                f"Exported {len(templates)} templates to {file_path}"
            )
            
        except Exception as e:
            logger.error(f"Error exporting templates: {str(e)}")
            QMessageBox.critical(
                self,
                "Export Error",
                f"Error exporting templates: {str(e)}"
            )