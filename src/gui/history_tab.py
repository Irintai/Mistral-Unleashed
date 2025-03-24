"""
History tab for the Advanced Code Generator.
Displays and manages code generation history.
"""

import os
import logging
import time
from datetime import datetime
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                           QComboBox, QPushButton, QGroupBox, QFormLayout, 
                           QListWidget, QListWidgetItem, QSplitter, QTextEdit,
                           QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
                           QMenu, QAction, QMessageBox, QFileDialog, QRadioButton, QDialog,
                           QDialogButtonBox, QSpacerItem, QSizePolicy)  
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QFont

# Logger for this module
logger = logging.getLogger(__name__)

class HistoryTab(QWidget):
    """Tab for managing generation history"""
    
    # Signals
    entry_selected = pyqtSignal(dict)  # Entry data
    entry_deleted = pyqtSignal(str)    # Entry ID
    
    def __init__(self, history_manager, parent=None):
        super().__init__(parent)
        
        self.history_manager = history_manager
        self.current_entry_id = None
        
        # Initialize UI
        self.init_ui()
        
        # Load history entries
        self.load_history_entries()
    
    def init_ui(self):
        """Initialize the UI components"""
        # Main layout
        main_layout = QVBoxLayout()
        
        # Top toolbar with actions
        self.toolbar = QHBoxLayout()
        
        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search history...")
        self.search_input.textChanged.connect(self.on_search_changed)
        self.toolbar.addWidget(self.search_input)
        
        # Language filter
        self.language_filter = QComboBox()
        self.language_filter.addItem("All Languages")
        self.language_filter.currentTextChanged.connect(self.on_filter_changed)
        self.toolbar.addWidget(self.language_filter)
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.load_history_entries)
        self.toolbar.addWidget(self.refresh_button)
        
        # Export button
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export_entry)
        self.export_button.setEnabled(False)
        self.toolbar.addWidget(self.export_button)
        
        # Delete button
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_entry)
        self.delete_button.setEnabled(False)
        self.toolbar.addWidget(self.delete_button)
        
        main_layout.addLayout(self.toolbar)
        
        # Main splitter
        self.splitter = QSplitter(Qt.Vertical)
        
        # History table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["Title", "Language", "Model", "Date", "ID"])
        self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.setSelectionMode(QTableWidget.SingleSelection)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.history_table.customContextMenuRequested.connect(self.show_context_menu)
        self.history_table.itemSelectionChanged.connect(self.on_selection_changed)
        
        self.splitter.addWidget(self.history_table)
        
        # Entry details
        self.details_tabs = QTabWidget()
        
        # Prompt tab
        self.prompt_tab = QWidget()
        prompt_layout = QVBoxLayout()
        self.prompt_editor = QTextEdit()
        self.prompt_editor.setReadOnly(True)
        prompt_layout.addWidget(self.prompt_editor)
        self.prompt_tab.setLayout(prompt_layout)
        
        # Code tab
        self.code_tab = QWidget()
        code_layout = QVBoxLayout()
        self.code_editor = QTextEdit()
        self.code_editor.setReadOnly(True)
        self.code_editor.setFont(QFont("Consolas" if os.name == "nt" else "Menlo", 10))
        code_layout.addWidget(self.code_editor)
        self.code_tab.setLayout(code_layout)
        
        # Info tab
        self.info_tab = QWidget()
        info_layout = QFormLayout()
        
        self.info_model = QLabel("")
        info_layout.addRow("Model:", self.info_model)
        
        self.info_language = QLabel("")
        info_layout.addRow("Language:", self.info_language)
        
        self.info_date = QLabel("")
        info_layout.addRow("Date:", self.info_date)
        
        self.info_params = QLabel("")
        info_layout.addRow("Parameters:", self.info_params)
        
        self.info_id = QLabel("")
        info_layout.addRow("ID:", self.info_id)
        
        # Add spacer
        info_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        self.info_tab.setLayout(info_layout)
        
        # Add tabs
        self.details_tabs.addTab(self.prompt_tab, "Prompt")
        self.details_tabs.addTab(self.code_tab, "Generated Code")
        self.details_tabs.addTab(self.info_tab, "Details")
        
        self.splitter.addWidget(self.details_tabs)
        
        # Set initial splitter sizes (60% list, 40% details)
        self.splitter.setSizes([600, 400])
        
        main_layout.addWidget(self.splitter)
        
        # Status label
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
        # Set the main layout
        self.setLayout(main_layout)
    
    def load_history_entries(self):
        """Load history entries into the table"""
        try:
            # Get all entries
            entries = self.history_manager.get_all_entries()
            
            # Clear table
            self.history_table.setRowCount(0)
            
            # Collect all languages for filter
            languages = set(["All Languages"])
            
            # Add entries to table
            for i, entry in enumerate(entries):
                # Add language to filter options
                if "language" in entry and entry["language"]:
                    languages.add(entry["language"])
                
                # Apply current filter
                if self.language_filter.currentText() != "All Languages" and \
                entry.get("language", "") != self.language_filter.currentText():
                    continue
                
                # Apply search filter if any
                search_text = self.search_input.text().lower()
                if search_text:
                    title = entry.get("title", "").lower()
                    prompt = entry.get("prompt", "").lower()
                    code = entry.get("code", "").lower()
                    
                    if title.find(search_text) < 0 and \
                    prompt.find(search_text) < 0 and \
                    code.find(search_text) < 0:
                        continue
                
                # Add row to table
                row_position = self.history_table.rowCount()
                self.history_table.insertRow(row_position)
                
                # Set title
                title_item = QTableWidgetItem(entry.get("title", ""))
                title_item.setData(Qt.UserRole, entry["id"])  # Store ID in item data
                self.history_table.setItem(row_position, 0, title_item)
                
                # Set language
                language_item = QTableWidgetItem(entry.get("language", ""))
                self.history_table.setItem(row_position, 1, language_item)
                
                # Set model
                model_item = QTableWidgetItem(entry.get("model", ""))
                self.history_table.setItem(row_position, 2, model_item)
                
                # Set date (formatted)
                timestamp = entry.get("timestamp", 0)
                date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
                date_item = QTableWidgetItem(date_str)
                date_item.setData(Qt.UserRole, timestamp)  # Store timestamp for sorting
                self.history_table.setItem(row_position, 3, date_item)
                
                # Set ID (shortened)
                id_item = QTableWidgetItem(entry["id"][:8] + "...")
                self.history_table.setItem(row_position, 4, id_item)
            
            # Update language filter while preserving current selection
            current_language = self.language_filter.currentText()
            
            # Temporarily block signals to prevent recursion
            self.language_filter.blockSignals(True)
            self.language_filter.clear()
            self.language_filter.addItems(sorted(list(languages)))
            
            # Try to restore previous selection
            index = self.language_filter.findText(current_language)
            if index >= 0:
                self.language_filter.setCurrentIndex(index)
                
            # Restore signals
            self.language_filter.blockSignals(False)
            
            # Update status
            visible_entries = self.history_table.rowCount()
            total_entries = len(entries)
            
            if visible_entries < total_entries:
                self.status_label.setText(f"Showing {visible_entries} of {total_entries} entries")
            else:
                self.status_label.setText(f"Total entries: {total_entries}")
                
        except Exception as e:
            logger.error(f"Error loading history entries: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")

    def on_filter_changed(self, text):
        """Handle filter combo change"""
        self.load_history_entries()

    def on_search_changed(self, text):
        """Handle search text change"""
        self.load_history_entries()
    
    def on_selection_changed(self):
        """Handle table selection change"""
        selected_items = self.history_table.selectedItems()
        
        if not selected_items:
            # Clear details
            self.clear_entry_details()
            self.export_button.setEnabled(False)
            self.delete_button.setEnabled(False)
            return
        
        # Get entry ID from the selected row
        row = selected_items[0].row()
        entry_id = self.history_table.item(row, 0).data(Qt.UserRole)
        
        # Load and display entry
        self.load_entry_details(entry_id)
        
        # Enable actions
        self.export_button.setEnabled(True)
        self.delete_button.setEnabled(True)
    
    def load_entry_details(self, entry_id):
        """Load and display entry details"""
        try:
            # Get entry
            entry = self.history_manager.get_entry(entry_id)
            
            if not entry:
                self.clear_entry_details()
                logger.warning(f"Entry not found: {entry_id}")
                return
            
            # Store current entry ID
            self.current_entry_id = entry_id
            
            # Update prompt
            prompt_text = entry.get("prompt", "")
            self.prompt_editor.setText(prompt_text)
            
            # Update code
            code_text = entry.get("code", "")
            self.code_editor.setText(code_text)
            
            # Update info
            self.info_model.setText(entry.get("model", "Unknown"))
            self.info_language.setText(entry.get("language", "Unknown"))
            
            # Format date
            timestamp = entry.get("timestamp", 0)
            date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            self.info_date.setText(date_str)
            
            # Format parameters
            params = entry.get("parameters", {})
            params_text = ""
            for key, value in params.items():
                params_text += f"{key}: {value}\n"
            self.info_params.setText(params_text)
            
            # Display ID
            self.info_id.setText(entry_id)
            
            # Emit signal
            self.entry_selected.emit(entry)
            
        except Exception as e:
            logger.error(f"Error loading entry details: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
    
    def clear_entry_details(self):
        """Clear entry details display"""
        self.current_entry_id = None
        self.prompt_editor.clear()
        self.code_editor.clear()
        self.info_model.clear()
        self.info_language.clear()
        self.info_date.clear()
        self.info_params.clear()
        self.info_id.clear()
    
    def delete_entry(self):
        """Delete the selected entry"""
        if not self.current_entry_id:
            return
        
        # Confirm deletion
        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            "Are you sure you want to delete this history entry?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.No:
            return
        
        try:
            # Delete the entry
            success = self.history_manager.remove_entry(self.current_entry_id)
            
            if success:
                # Update UI
                self.status_label.setText(f"Entry deleted: {self.current_entry_id[:8]}...")
                self.clear_entry_details()
                self.load_history_entries()
                
                # Emit signal
                self.entry_deleted.emit(self.current_entry_id)
            else:
                QMessageBox.warning(
                    self,
                    "Deletion Failed",
                    "Failed to delete the entry. See log for details."
                )
                
        except Exception as e:
            logger.error(f"Error deleting entry: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"Error deleting entry: {str(e)}"
            )
    
    def export_entry(self):
        """Export the selected entry to file"""
        if not self.current_entry_id:
            return
        
        try:
            # Get entry
            entry = self.history_manager.get_entry(self.current_entry_id)
            
            if not entry:
                logger.warning(f"Entry not found for export: {self.current_entry_id}")
                return
            
            # Ask what to export
            export_dialog = QDialog(self)
            export_dialog.setWindowTitle("Export Options")
            dialog_layout = QVBoxLayout()
            
            # Options
            options_group = QGroupBox("What to export")
            options_layout = QVBoxLayout()
            
            code_radio = QRadioButton("Generated Code Only")
            prompt_radio = QRadioButton("Prompt Only")
            both_radio = QRadioButton("Both Prompt and Code")
            full_radio = QRadioButton("Full Entry Data (JSON)")
            
            code_radio.setChecked(True)  # Default selection
            
            options_layout.addWidget(code_radio)
            options_layout.addWidget(prompt_radio)
            options_layout.addWidget(both_radio)
            options_layout.addWidget(full_radio)
            
            options_group.setLayout(options_layout)
            dialog_layout.addWidget(options_group)
            
            # Buttons
            from PyQt5.QtWidgets import QDialogButtonBox
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(export_dialog.accept)
            button_box.rejected.connect(export_dialog.reject)
            dialog_layout.addWidget(button_box)
            
            export_dialog.setLayout(dialog_layout)
            
            # Show dialog
            result = export_dialog.exec_()
            
            if result != QDialog.Accepted:
                return
            
            # Determine export type
            export_type = "code"
            if prompt_radio.isChecked():
                export_type = "prompt"
            elif both_radio.isChecked():
                export_type = "both"
            elif full_radio.isChecked():
                export_type = "json"
            
            # Get file extension
            extension = ".txt"
            if export_type == "code":
                language = entry.get("language", "").lower()
                if language == "python":
                    extension = ".py"
                elif language == "javascript":
                    extension = ".js"
                elif language == "typescript":
                    extension = ".ts"
                elif language == "c++":
                    extension = ".cpp"
                elif language == "java":
                    extension = ".java"
                # Add more as needed
            elif export_type == "json":
                extension = ".json"
            
            # Ask for save location
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Entry",
                os.path.expanduser(f"~/exported_entry{extension}"),
                f"Text Files (*{extension});;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Prepare content
            content = ""
            
            if export_type == "code":
                content = entry.get("code", "")
            elif export_type == "prompt":
                content = entry.get("prompt", "")
            elif export_type == "both":
                content = f"PROMPT:\n\n{entry.get('prompt', '')}\n\n"
                content += f"GENERATED CODE:\n\n{entry.get('code', '')}"
            elif export_type == "json":
                import json
                content = json.dumps(entry, indent=2)
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.status_label.setText(f"Entry exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting entry: {str(e)}")
            QMessageBox.critical(
                self,
                "Export Error",
                f"Error exporting entry: {str(e)}"
            )
    
    def show_context_menu(self, position):
        """Show context menu for history table"""
        if not self.history_table.selectedItems():
            return
        
        # Create menu
        context_menu = QMenu(self)
        
        # Add actions
        load_action = QAction("Open in Generator", self)
        load_action.triggered.connect(self.load_to_generator)
        context_menu.addAction(load_action)
        
        export_action = QAction("Export...", self)
        export_action.triggered.connect(self.export_entry)
        context_menu.addAction(export_action)
        
        context_menu.addSeparator()
        
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(self.delete_entry)
        context_menu.addAction(delete_action)
        
        # Show the menu
        context_menu.exec_(self.history_table.mapToGlobal(position))
    
    def load_to_generator(self):
        """Load selected entry to code generator tab"""
        if not self.current_entry_id:
            return
        
        try:
            # Get entry
            entry = self.history_manager.get_entry(self.current_entry_id)
            
            if not entry:
                logger.warning(f"Entry not found: {self.current_entry_id}")
                return
            
            # Find the code generation tab
            main_window = self.window()
            tabs = None
            
            # Try to find tab widget
            for widget in main_window.findChildren(QTabWidget):
                if hasattr(widget, 'findChild'):
                    code_tab = None
                    for i in range(widget.count()):
                        if isinstance(widget.widget(i), QWidget) and \
                           widget.tabText(i) == "Code Generation":
                            code_tab = widget.widget(i)
                            tabs = widget
                            break
                    if code_tab:
                        break
            
            if not tabs or not code_tab:
                QMessageBox.warning(
                    self,
                    "Tab Not Found",
                    "Could not find the Code Generation tab."
                )
                return
            
            # Switch to the code generation tab
            tabs.setCurrentWidget(code_tab)
            
            # Set prompt and language
            if hasattr(code_tab, 'prompt_editor'):
                code_tab.prompt_editor.setText(entry.get("prompt", ""))
            
            if hasattr(code_tab, 'language_combo'):
                language = entry.get("language", "")
                index = code_tab.language_combo.findText(language)
                if index >= 0:
                    code_tab.language_combo.setCurrentIndex(index)
            
            # Set parameters if available
            params = entry.get("parameters", {})
            
            if hasattr(code_tab, 'max_length_spin') and "max_length" in params:
                code_tab.max_length_spin.setValue(params["max_length"])
            
            if hasattr(code_tab, 'temperature_spin') and "temperature" in params:
                code_tab.temperature_spin.setValue(params["temperature"])
            
            if hasattr(code_tab, 'top_p_spin') and "top_p" in params:
                code_tab.top_p_spin.setValue(params["top_p"])
            
            self.status_label.setText(f"Entry loaded to generator: {self.current_entry_id[:8]}...")
            
        except Exception as e:
            logger.error(f"Error loading entry to generator: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"Error loading entry to generator: {str(e)}"
            )
    
    def refresh(self):
        """Refresh the history display"""
        self.load_history_entries()