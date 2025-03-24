"""
Conversation module for Advanced Code Generator.
Allows conversational interaction with the model.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
                           QLineEdit, QPushButton, QLabel, QSplitter, 
                           QScrollArea, QFrame, QSizePolicy, QMenu, QAction,
                           QInputDialog, QMessageBox, QFileDialog, QDialog, QToolBar,
                           QProgressBar, QComboBox, QCheckBox, QGroupBox, QFormLayout,
                           QSlider, QDoubleSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QSize
from PyQt5.QtGui import QFont, QTextCursor, QIcon
from PyQt5.QtWidgets import QApplication

# Import custom modules
from src.gui.conversation_streaming import StreamingConversationGenerator
from src.gui.message_widget import Message, MessageWidget
from src.gui.message_formatting import create_formatted_message_widget
from src.gui.markdown_renderer import MarkdownTextEdit, MarkdownRenderer

# Logger for this module
logger = logging.getLogger(__name__)

class ConversationTab(QWidget):
    """Tab for having conversations with the model."""
    
    def __init__(self, parent=None):
        """Initialize the conversation tab."""
        super().__init__(parent)
        self.parent = parent
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.model_name = None
        
        # Initialize conversation history
        self.conversation_history = []
        self.current_conversation_id = None
        
        # Initialize streaming generator and keep a strong reference
        self._streaming_generator = StreamingConversationGenerator(self)
        self.streaming_generator = self._streaming_generator  # Keep a reference to prevent garbage collection
        
        # Connect streaming signals
        self.streaming_generator.token_generated.connect(self.on_token_generated)
        self.streaming_generator.generation_started.connect(self.on_generation_started)
        self.streaming_generator.generation_finished.connect(self.on_generation_finished)
        self.streaming_generator.generation_error.connect(self.on_generation_error)
        self.streaming_generator.progress_updated.connect(self.on_progress_updated)
        
        # Initialize response being generated flag
        self.is_generating = False
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the conversation tab UI."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Controls area - for parameters, model selection, etc.
        controls_layout = QHBoxLayout()
        
        # Parameter controls group
        params_group = QGroupBox("Generation Parameters")
        params_layout = QFormLayout(params_group)
        
        # Temperature slider
        self.temperature_label = QLabel("Temperature: 0.7")
        self.temperature_slider = QSlider(Qt.Horizontal)
        self.temperature_slider.setRange(0, 100)  # 0.0 to 1.0
        self.temperature_slider.setValue(70)      # Default 0.7
        self.temperature_slider.valueChanged.connect(self.update_temperature_label)
        params_layout.addRow(self.temperature_label, self.temperature_slider)
        
        # Top-p slider
        self.top_p_label = QLabel("Top-p: 0.9")
        self.top_p_slider = QSlider(Qt.Horizontal)
        self.top_p_slider.setRange(0, 100)  # 0.0 to 1.0
        self.top_p_slider.setValue(90)      # Default 0.9
        self.top_p_slider.valueChanged.connect(self.update_top_p_label)
        params_layout.addRow(self.top_p_label, self.top_p_slider)
        
        # Max length control
        self.max_length_label = QLabel("Max Length:")
        self.max_length_spin = QDoubleSpinBox()
        self.max_length_spin.setRange(50, 2000)
        self.max_length_spin.setValue(500)
        self.max_length_spin.setSingleStep(50)
        params_layout.addRow(self.max_length_label, self.max_length_spin)
        
        # Add parameters group to controls layout
        controls_layout.addWidget(params_group)
        
        # Status label for model info
        self.model_info_label = QLabel("No model loaded")
        self.model_info_label.setStyleSheet("color: red;")
        controls_layout.addWidget(self.model_info_label)
        
        # Add controls to main layout
        layout.addLayout(controls_layout)
        
        # Messages area
        self.messages_scroll = QScrollArea()
        self.messages_scroll.setWidgetResizable(True)
        self.messages_scroll.setFrameShape(QFrame.NoFrame)
        
        # Messages container
        self.messages_container = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_container)
        self.messages_layout.setAlignment(Qt.AlignTop)
        self.messages_layout.setSpacing(10)
        self.messages_scroll.setWidget(self.messages_container)
        
        # Message input area
        input_layout = QHBoxLayout()
        
        # Text input
        self.message_input = QTextEdit()
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.setMinimumHeight(100)
        self.message_input.setMaximumHeight(200)
        input_layout.addWidget(self.message_input)
        
        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.setIcon(QIcon("resources/icons/send.png"))
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        # Progress bar for generation
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        
        # Stop generation button
        self.stop_button = QPushButton("Stop Generation")
        self.stop_button.setIcon(QIcon("resources/icons/stop.png"))
        self.stop_button.clicked.connect(self.stop_generation)
        self.stop_button.setVisible(False)
        
        # Add to main layout
        layout.addWidget(self.messages_scroll)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.stop_button)
        layout.addLayout(input_layout)
        
        # Connect signals
        self.message_input.textChanged.connect(self.adjust_input_height)
        
        # Markdown renderer
        self.markdown_renderer = MarkdownRenderer()
    
    def update_temperature_label(self, value):
        """Update the temperature label when slider value changes."""
        temperature = value / 100.0
        self.temperature_label.setText(f"Temperature: {temperature:.1f}")
    
    def update_top_p_label(self, value):
        """Update the top-p label when slider value changes."""
        top_p = value / 100.0
        self.top_p_label.setText(f"Top-p: {top_p:.1f}")
    
    def adjust_input_height(self):
        """Adjust the height of the input field based on content."""
        doc_height = self.message_input.document().size().height()
        # Cap the height to avoid taking too much space
        if doc_height < 200:
            self.message_input.setFixedHeight(max(100, int(doc_height) + 20))
    
    def send_message(self):
        """Send a message and generate a response."""
        # Check if a generation is already in progress
        if self.is_generating:
            return
        
        # Get message text
        message_text = self.message_input.toPlainText().strip()
        if not message_text:
            return
        
        # Clear input
        self.message_input.clear()
        
        # Create user message
        user_message = Message(message_text, role="user")
        
        # Add to conversation history
        self.conversation_history.append(user_message)
        
        # Display user message
        self.add_message_widget(user_message)
        
        # Start generating response
        self.generate_response()
    
    def generate_response(self):
        """Generate a response based on conversation history."""
        # Check if model is loaded
        if not self.model or not self.tokenizer:
            self.show_error_message("No model loaded", "Please load a model first.")
            return
        
        # Check if already generating
        if self.is_generating:
            return
            
        # Set generating flag
        self.is_generating = True
        
        # Show progress bar and stop button
        self.progress_bar.setVisible(True)
        self.stop_button.setVisible(True)
        
        # Disable send button
        self.send_button.setEnabled(False)
        
        # Create assistant message widget (initially empty)
        assistant_message = Message("", role="assistant")
        self.conversation_history.append(assistant_message)
        message_widget = self.add_message_widget(assistant_message)
        
        try:
            # Build the conversation prompt
            prompt = self.build_conversation_prompt()
            
            # Log the prompt for debugging
            logger.info(f"Prompt length: {len(prompt)}")
            logger.debug(f"Prompt: {prompt[:100]}...")
            
            # Set parameters
            params = {
                "temperature": self.temperature_slider.value() / 100.0,
                "top_p": self.top_p_slider.value() / 100.0,
                "max_length": int(self.max_length_spin.value()),
                "stream_interval": 0.05  # 50ms interval for smooth streaming
            }
            
            # Set up streaming generator
            self.streaming_generator.setup(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                params=params
            )
            
            # Start generation
            logger.info("Starting generation process...")
            self.streaming_generator.start_generation()
            
        except Exception as e:
            logger.error(f"Error starting generation: {str(e)}")
            self.show_error_message("Generation Error", str(e))
            self.is_generating = False
            self.send_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.stop_button.setVisible(False)
    
    def stop_generation(self):
        """Stop ongoing response generation."""
        if self.is_generating:
            self.streaming_generator.stop()
            self.on_generation_finished()
    
    @pyqtSlot(str)
    def on_token_generated(self, token):
        """Handle newly generated token."""
        logger.debug(f"Token received: {token}")
        
        # Update the last assistant message
        if self.conversation_history and self.conversation_history[-1].role == "assistant":
            assistant_message = self.conversation_history[-1]
            assistant_message.content += token
            
            # Find assistant message widget and update it
            for i in range(self.messages_layout.count()):
                widget = self.messages_layout.itemAt(i).widget()
                if isinstance(widget, MessageWidget) and widget.message == assistant_message:
                    logger.debug(f"Updating message widget with new content")
                    widget.update_message(assistant_message)
                    break
    
    @pyqtSlot()
    def on_generation_started(self):
        """Handle generation start event."""
        self.is_generating = True
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.stop_button.setVisible(True)
        self.send_button.setEnabled(False)
    
    @pyqtSlot()
    def on_generation_finished(self):
        """Handle generation finished event."""
        self.is_generating = False
        self.progress_bar.setVisible(False)
        self.stop_button.setVisible(False)
        self.send_button.setEnabled(True)
        self.progress_bar.setValue(100)
        
        # Scroll to bottom to show new message
        self.messages_scroll.verticalScrollBar().setValue(
            self.messages_scroll.verticalScrollBar().maximum()
        )
    
    @pyqtSlot(str)
    def on_generation_error(self, error_message):
        """Handle generation error event."""
        logger.error(f"Generation error: {error_message}")
        self.show_error_message("Generation Error", error_message)
        self.is_generating = False
        self.progress_bar.setVisible(False)
        self.stop_button.setVisible(False)
        self.send_button.setEnabled(True)
    
    @pyqtSlot(int)
    def on_progress_updated(self, progress):
        """Handle progress update event."""
        self.progress_bar.setValue(progress)
    
    def build_conversation_prompt(self):
        """Build conversation prompt from history."""
        prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\n"
        
        for message in self.conversation_history:
            if message.role == "user":
                prompt += f"Human: {message.content}\n"
            elif message.role == "assistant":
                # Only include completed assistant messages (not the one being generated)
                if message != self.conversation_history[-1]:
                    prompt += f"Assistant: {message.content}\n"
        
        # Add the last prompt without content
        prompt += "Assistant: "
        
        return prompt
    
    def add_message_widget(self, message):
        """Add a message widget to the conversation."""
        widget = MessageWidget(
            message,
            on_copy=self.copy_message,
            on_edit=self.edit_message if message.role == "user" else None,
            on_delete=self.delete_message
        )
        
        # Connect reaction signal
        widget.reaction_clicked.connect(self.handle_message_reaction)
        
        # Add to layout
        self.messages_layout.addWidget(widget)
        
        # Scroll to bottom
        QTimer.singleShot(50, lambda: self.messages_scroll.verticalScrollBar().setValue(
            self.messages_scroll.verticalScrollBar().maximum()
        ))
        
        return widget
    
    def copy_message(self, message):
        """Copy message content to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(message.content)
    
    def edit_message(self, message):
        """Edit a user message."""
        # Can only edit if not generating
        if self.is_generating:
            return
        
        # Get the index of the message in history
        try:
            index = next(i for i, msg in enumerate(self.conversation_history) 
                       if msg.message_id == message.message_id)
        except StopIteration:
            return
        
        # Set the input text
        self.message_input.setPlainText(message.content)
        
        # Remove this message and all messages after it
        self.conversation_history = self.conversation_history[:index]
        
        # Remove corresponding widgets
        for i in reversed(range(index, self.messages_layout.count())):
            widget = self.messages_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
    
    def delete_message(self, message):
        """Delete a message."""
        # Can only delete if not generating
        if self.is_generating:
            return
        
        # Confirm deletion
        confirm = QMessageBox.question(
            self, "Confirm Deletion", 
            "Are you sure you want to delete this message?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if confirm != QMessageBox.Yes:
            return
        
        # Get the index of the message in history
        try:
            index = next(i for i, msg in enumerate(self.conversation_history) 
                       if msg.message_id == message.message_id)
        except StopIteration:
            return
        
        # Remove this message and all messages after it
        self.conversation_history = self.conversation_history[:index]
        
        # Remove corresponding widgets
        for i in reversed(range(index, self.messages_layout.count())):
            widget = self.messages_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
    
    def handle_message_reaction(self, message_id, action, data):
        """Handle message reactions like thumbs up/down."""
        # Find the message
        try:
            message = next(msg for msg in self.conversation_history 
                         if msg.message_id == message_id)
        except StopIteration:
            return
        
        # Handle thumbs up/down
        if action == "thumbs_up":
            message.add_reaction("thumbs_up")
            message.remove_reaction("thumbs_down")
        elif action == "thumbs_down":
            message.add_reaction("thumbs_down")
            message.remove_reaction("thumbs_up")
        
        # Save conversation if needed
        if self.current_conversation_id:
            self.save_conversation()
    
    def save_conversation(self):
        """Save the current conversation to a file."""
        if not self.current_conversation_id:
            return
        
        # Convert conversation to JSON
        conversation_data = {
            "id": self.current_conversation_id,
            "title": f"Conversation {self.current_conversation_id}",
            "messages": [msg.to_dict() for msg in self.conversation_history],
            "timestamp": datetime.now().isoformat()
        }
        
        # Create conversations directory if it doesn't exist
        os.makedirs("conversations", exist_ok=True)
        
        # Save to file
        filename = f"conversations/{self.current_conversation_id}.json"
        with open(filename, "w") as f:
            json.dump(conversation_data, f, indent=2)
        
        logger.info(f"Saved conversation to {filename}")
    
    def load_conversation(self, conversation_id):
        """Load a conversation from a file."""
        filename = f"conversations/{conversation_id}.json"
        
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            
            # Clear current conversation
            self.clear_conversation()
            
            # Set conversation ID
            self.current_conversation_id = conversation_id
            
            # Load messages
            for msg_data in data.get("messages", []):
                message = Message.from_dict(msg_data)
                self.conversation_history.append(message)
                self.add_message_widget(message)
            
            logger.info(f"Loaded conversation {conversation_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading conversation: {str(e)}")
            return False
    
    def new_conversation(self):
        """Start a new conversation."""
        # Check if already generating
        if self.is_generating:
            return
        
        # Clear the current conversation
        self.clear_conversation()
        
        # Generate a new conversation ID
        self.current_conversation_id = f"conv_{int(time.time())}"
        
        logger.info(f"Started new conversation {self.current_conversation_id}")
    
    def clear_conversation(self):
        """Clear the current conversation."""
        # Clear conversation history
        self.conversation_history = []
        
        # Clear conversation ID
        self.current_conversation_id = None
        
        # Clear UI
        for i in reversed(range(self.messages_layout.count())):
            widget = self.messages_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
    
    def show_error_message(self, title, message):
        """Show error message dialog."""
        QMessageBox.critical(self, title, message)
    
    def closeEvent(self, event):
        """Handle tab closing - clean up resources."""
        if self.streaming_generator:
            logger.info("Cleaning up streaming generator")
            self.streaming_generator.cleanup()
            
        # Unload model from this tab
        if self.model:
            logger.info("Model unloaded from conversation tab")
            self.model = None
            self.tokenizer = None
        
        # Call parent class method
        super().closeEvent(event)
    
    def on_model_loaded(self, model_id, tokenizer, model):
        """Handle event when a model is loaded."""
        logger.info(f"Model loaded in conversation tab: {model_id}")
        
        # Store model and tokenizer references
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_id
        
        # Update model info label
        self.model_info_label.setText(f"Model: {model_id}")
        
        # Enable the send button now that a model is loaded
        self.send_button.setEnabled(True)
        
        # Enable model-specific features if needed
        self.message_input.setEnabled(True)
        
        # Check if model is on GPU
        if next(model.parameters()).is_cuda:
            logger.info("Model is using GPU acceleration")
            self.model_info_label.setText(f"Model: {model_id} (GPU)")
        else:
            logger.info("Model is using CPU")
            self.model_info_label.setText(f"Model: {model_id} (CPU)")
            
        # Setup the streaming generator with the new model
        self.streaming_generator.setup(model=model, tokenizer=tokenizer)
    
    def on_model_unloaded(self):
        """Handle model unloaded event."""
        self.model = None
        self.tokenizer = None
        
        # Update status in UI if needed
        if hasattr(self, 'model_info_label'):
            self.model_info_label.setText("No model loaded")
            self.model_info_label.setStyleSheet("color: red;")
        
        # Disable send button when no model is loaded
        self.send_button.setEnabled(False)
        
        logger.info("Model unloaded from conversation tab")