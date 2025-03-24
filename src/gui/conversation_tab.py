"""
Conversation module for Advanced Code Generator.
Allows for natural language conversation in addition to code generation.
"""

import os
import sys
import time
import json
import logging
import torch
import re
from typing import Dict, List, Tuple, Any, Optional
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
                           QLineEdit, QPushButton, QLabel, QSplitter, 
                           QScrollArea, QFrame, QSizePolicy, QMenu, QAction)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QSize
from PyQt5.QtGui import QFont, QTextCursor, QIcon, QTextCharFormat, QColor

# Logger for this module
logger = logging.getLogger(__name__)

class Message:
    """Class representing a conversation message"""
    
    def __init__(self, text: str, is_user: bool, timestamp: float = None):
        self.text = text
        self.is_user = is_user
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            "text": self.text,
            "is_user": self.is_user,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        return cls(
            text=data["text"],
            is_user=data["is_user"],
            timestamp=data["timestamp"]
        )

class Conversation:
    """Class representing a full conversation"""
    
    def __init__(self, title: str = None, messages: List[Message] = None):
        self.title = title or f"Conversation {time.strftime('%Y-%m-%d %H:%M')}"
        self.messages = messages or []
        self.created_at = time.time()
        self.updated_at = time.time()
        self.id = f"conv_{int(self.created_at)}"
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation"""
        self.messages.append(message)
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary for serialization"""
        return {
            "id": self.id,
            "title": self.title,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create conversation from dictionary"""
        conv = cls(
            title=data["title"],
            messages=[Message.from_dict(msg) for msg in data["messages"]]
        )
        conv.id = data.get("id", conv.id)
        conv.created_at = data.get("created_at", conv.created_at)
        conv.updated_at = data.get("updated_at", conv.updated_at)
        return conv

class ConversationManager:
    """Manages conversations and handles persistence"""
    
    def __init__(self, storage_path: str = None):
        """Initialize conversation manager"""
        self.storage_path = storage_path or self._get_default_storage_path()
        self.conversations: Dict[str, Conversation] = {}
        self.current_conversation_id: Optional[str] = None
        
        # Ensure storage directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Load existing conversations
        self.load_conversations()
    
    def _get_default_storage_path(self) -> str:
        """Get default storage path for conversations"""
        if os.name == 'nt':  # Windows
            app_data = os.environ.get('APPDATA', os.path.expanduser('~'))
            app_dir = os.path.join(app_data, "AdvancedCodeGenerator")
        elif os.name == 'posix':  # macOS/Linux
            if os.path.exists('/Applications'):  # macOS
                app_data = os.path.expanduser('~/Library/Application Support')
                app_dir = os.path.join(app_data, "AdvancedCodeGenerator")
            else:  # Linux
                app_data = os.path.expanduser('~/.local/share')
                app_dir = os.path.join(app_data, "advancedcodegenerator")
        else:  # Fallback
            app_dir = os.path.expanduser(f'~/.advancedcodegenerator')
        
        os.makedirs(app_dir, exist_ok=True)
        return os.path.join(app_dir, "conversations.json")
    
    def load_conversations(self) -> None:
        """Load conversations from storage"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load conversations
                self.conversations = {
                    conv_id: Conversation.from_dict(conv_data)
                    for conv_id, conv_data in data.get("conversations", {}).items()
                }
                
                # Set current conversation
                self.current_conversation_id = data.get("current_conversation_id")
                
                logger.info(f"Loaded {len(self.conversations)} conversations")
        except Exception as e:
            logger.error(f"Error loading conversations: {str(e)}")
    
    def save_conversations(self) -> None:
        """Save conversations to storage"""
        try:
            # Prepare data for serialization
            data = {
                "conversations": {
                    conv_id: conv.to_dict()
                    for conv_id, conv in self.conversations.items()
                },
                "current_conversation_id": self.current_conversation_id
            }
            
            # Save to file
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Conversations saved")
        except Exception as e:
            logger.error(f"Error saving conversations: {str(e)}")
    
    def create_conversation(self, title: str = None) -> Conversation:
        """Create a new conversation"""
        conversation = Conversation(title=title)
        self.conversations[conversation.id] = conversation
        self.current_conversation_id = conversation.id
        self.save_conversations()
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID"""
        return self.conversations.get(conversation_id)
    
    def get_current_conversation(self) -> Optional[Conversation]:
        """Get the current conversation"""
        if self.current_conversation_id:
            return self.conversations.get(self.current_conversation_id)
        return None
    
    def set_current_conversation(self, conversation_id: str) -> bool:
        """Set the current conversation"""
        if conversation_id in self.conversations:
            self.current_conversation_id = conversation_id
            self.save_conversations()
            return True
        return False
    
    def add_message(self, text: str, is_user: bool) -> Optional[Message]:
        """Add a message to the current conversation"""
        # Get or create current conversation
        conversation = self.get_current_conversation()
        if not conversation:
            conversation = self.create_conversation()
        
        # Create and add message
        message = Message(text=text, is_user=is_user)
        conversation.add_message(message)
        
        # Save conversations
        self.save_conversations()
        
        return message
    
    def get_all_conversations(self) -> List[Conversation]:
        """Get all conversations sorted by updated_at (newest first)"""
        return sorted(
            self.conversations.values(),
            key=lambda conv: conv.updated_at,
            reverse=True
        )
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            
            # Update current conversation if deleted
            if self.current_conversation_id == conversation_id:
                # Set to newest conversation or None
                all_convs = self.get_all_conversations()
                self.current_conversation_id = all_convs[0].id if all_convs else None
            
            self.save_conversations()
            return True
        return False
    
    def rename_conversation(self, conversation_id: str, new_title: str) -> bool:
        """Rename a conversation"""
        conversation = self.get_conversation(conversation_id)
        if conversation:
            conversation.title = new_title
            conversation.updated_at = time.time()
            self.save_conversations()
            return True
        return False

class MessageWidget(QFrame):
    """Widget for displaying a single message"""
    
    def __init__(self, message: Message, parent=None):
        super().__init__(parent)
        self.message = message
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        # Set frame style
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(1)
        
        # Set background color based on message type
        if self.message.is_user:
            self.setStyleSheet("background-color: #E1F5FE; border-radius: 8px;")
        else:
            self.setStyleSheet("background-color: #F5F5F5; border-radius: 8px;")
        
        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Create message header
        header_layout = QHBoxLayout()
        
        # Sender label
        sender_label = QLabel("You:" if self.message.is_user else "Assistant:")
        sender_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(sender_label)
        
        # Timestamp label
        time_str = time.strftime("%I:%M %p", time.localtime(self.message.timestamp))
        time_label = QLabel(time_str)
        time_label.setStyleSheet("color: #757575; font-size: 10px;")
        header_layout.addWidget(time_label, alignment=Qt.AlignRight)
        
        layout.addLayout(header_layout)
        
        # Message text
        message_text = QTextEdit()
        message_text.setReadOnly(True)
        message_text.setPlainText(self.message.text)
        message_text.setFrameStyle(QFrame.NoFrame)
        message_text.setStyleSheet("background-color: transparent;")
        message_text.document().setDocumentMargin(0)
        
        # Adjust height based on content
        doc_height = message_text.document().size().height()
        message_text.setFixedHeight(min(doc_height + 5, 300))
        
        layout.addWidget(message_text)
        
        self.setLayout(layout)

class ConversationTab(QWidget):
    """Tab for conversation interface"""
    
    # Signals
    message_sent = pyqtSignal(str)
    
    def __init__(self, parent=None, model_manager=None):
        super().__init__(parent)
        
        # Store model manager for generating responses
        self.model_manager = model_manager
        
        # Create conversation manager
        self.conversation_manager = ConversationManager()
        
        # Initialize UI
        self.init_ui()
        
        # Load conversation if any
        self.load_current_conversation()
    
    def init_ui(self):
        """Initialize the UI"""
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        
        # Splitter for conversation list and chat area
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Conversation list section
        conversations_widget = QWidget()
        conversations_layout = QVBoxLayout()
        conversations_layout.setContentsMargins(0, 0, 0, 0)
        
        # Conversation list header
        list_header = QLabel("Conversations")
        list_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        conversations_layout.addWidget(list_header)
        
        # New conversation button
        new_conv_button = QPushButton("New Conversation")
        new_conv_button.clicked.connect(self.new_conversation)
        conversations_layout.addWidget(new_conv_button)
        
        # Conversation list
        self.conversation_list = QScrollArea()
        self.conversation_list.setWidgetResizable(True)
        self.conversation_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.conversation_list.setFrameShape(QFrame.NoFrame)
        
        # Container for conversation items
        self.conversation_list_container = QWidget()
        self.conversation_list_layout = QVBoxLayout()
        self.conversation_list_layout.setContentsMargins(0, 0, 0, 0)
        self.conversation_list_layout.setSpacing(5)
        self.conversation_list_layout.addStretch()
        
        self.conversation_list_container.setLayout(self.conversation_list_layout)
        self.conversation_list.setWidget(self.conversation_list_container)
        
        conversations_layout.addWidget(self.conversation_list)
        conversations_widget.setLayout(conversations_layout)
        
        # Add to splitter
        self.splitter.addWidget(conversations_widget)
        
        # Chat area
        chat_widget = QWidget()
        chat_layout = QVBoxLayout()
        chat_layout.setContentsMargins(0, 0, 0, 0)
        
        # Conversation title
        self.conversation_title = QLabel("New Conversation")
        self.conversation_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        chat_layout.addWidget(self.conversation_title)
        
        # Messages area
        self.messages_scroll = QScrollArea()
        self.messages_scroll.setWidgetResizable(True)
        self.messages_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.messages_scroll.setFrameShape(QFrame.NoFrame)
        
        # Container for messages
        self.messages_container = QWidget()
        self.messages_layout = QVBoxLayout()
        self.messages_layout.setContentsMargins(0, 0, 0, 0)
        self.messages_layout.setSpacing(10)
        self.messages_layout.addStretch()
        
        self.messages_container.setLayout(self.messages_layout)
        self.messages_scroll.setWidget(self.messages_container)
        
        chat_layout.addWidget(self.messages_scroll)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.message_input = QTextEdit()
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.setAcceptRichText(False)
        self.message_input.setFixedHeight(80)
        
        # Set up key event for sending message with Ctrl+Enter
        self.message_input.keyPressEvent = self.input_key_press
        
        input_layout.addWidget(self.message_input)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setFixedWidth(80)
        input_layout.addWidget(self.send_button)
        
        chat_layout.addLayout(input_layout)
        chat_widget.setLayout(chat_layout)
        
        # Add to splitter
        self.splitter.addWidget(chat_widget)
        
        # Set initial splitter sizes (30% list, 70% chat)
        self.splitter.setSizes([300, 700])
        
        main_layout.addWidget(self.splitter)
        
        # Status label
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
        self.setLayout(main_layout)
    
    def on_model_loaded(self, model_id, tokenizer, model):
        """Handle model loaded event"""
        # Update status
        self.status_label.setText(f"Model {model_id} loaded and ready")
        
        # Inform the user if they're in this tab
        if self.isVisible():
            self.add_assistant_message(
                f"Model {model_id} has been loaded and is ready for conversation. Feel free to start chatting!"
            )
    
    def on_model_unloaded(self):
        """Handle model unloaded event"""
        # Reset model references
        self.status_label.setText("Model unloaded")
        
        # Inform the user if they're in this tab
        if self.isVisible():
            self.add_assistant_message(
                "The model has been unloaded. Please load a model from the Model Selection tab to continue the conversation."
            )
    
    def generate_response(self, user_message):
        """Generate a response to the user message"""
        if not self.model_manager or not self.model_manager.is_model_loaded():
            # No model loaded
            self.add_assistant_message(
                "I'm sorry, but no language model is currently loaded. "
                "Please load a model from the Model Selection tab first."
            )
            return
    
    def generate_response(self, user_message):
        """Generate a response to the user message"""
        if not self.model_manager or not self.model_manager.is_model_loaded():
            # No model loaded
            self.add_assistant_message(
                "I'm sorry, but no language model is currently loaded. "
                "Please load a model from the Model Selection tab first."
            )
            return
    def input_key_press(self, event):
        """Handle key press events in the message input"""
        if event.key() == Qt.Key_Return and event.modifiers() & Qt.ControlModifier:
            self.send_message()
        else:
            # Pass event to default handler
            QTextEdit.keyPressEvent(self.message_input, event)
    
    def load_current_conversation(self):
        """Load the current conversation if any"""
        self.update_conversation_list()
        
        current_conv = self.conversation_manager.get_current_conversation()
        if current_conv:
            self.display_conversation(current_conv)
        else:
            # Create a new conversation if none exists
            self.new_conversation()
    
    def update_conversation_list(self):
        """Update the conversation list"""
        # Clear existing items
        for i in reversed(range(self.conversation_list_layout.count() - 1)):
            item = self.conversation_list_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
        
        # Get all conversations
        conversations = self.conversation_manager.get_all_conversations()
        
        # Add conversation items
        current_id = self.conversation_manager.current_conversation_id
        for conv in conversations:
            item = QPushButton(conv.title)
            item.setProperty("conversation_id", conv.id)
            item.clicked.connect(lambda checked, cid=conv.id: self.select_conversation(cid))
            
            # Style for current conversation
            if conv.id == current_id:
                item.setStyleSheet("background-color: #E1F5FE;")
            
            # Add context menu
            item.setContextMenuPolicy(Qt.CustomContextMenu)
            item.customContextMenuRequested.connect(
                lambda pos, cid=conv.id: self.show_conversation_menu(pos, cid)
            )
            
            self.conversation_list_layout.insertWidget(0, item)
    
    def show_conversation_menu(self, position, conversation_id):
        """Show context menu for conversation item"""
        menu = QMenu()
        
        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(lambda: self.rename_conversation(conversation_id))
        menu.addAction(rename_action)
        
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self.delete_conversation(conversation_id))
        menu.addAction(delete_action)
        
        # Get widget that triggered the menu
        sender = self.sender()
        
        # Show menu at position
        global_pos = sender.mapToGlobal(position)
        menu.exec_(global_pos)
    
    def rename_conversation(self, conversation_id):
        """Rename a conversation"""
        from PyQt5.QtWidgets import QInputDialog
        
        conversation = self.conversation_manager.get_conversation(conversation_id)
        if not conversation:
            return
        
        # Show input dialog
        new_title, ok = QInputDialog.getText(
            self,
            "Rename Conversation",
            "Enter new title:",
            text=conversation.title
        )
        
        if ok and new_title:
            # Rename conversation
            self.conversation_manager.rename_conversation(conversation_id, new_title)
            
            # Update UI
            self.update_conversation_list()
            
            # Update title if current conversation
            if conversation_id == self.conversation_manager.current_conversation_id:
                self.conversation_title.setText(new_title)
    
    def delete_conversation(self, conversation_id):
        """Delete a conversation"""
        from PyQt5.QtWidgets import QMessageBox
        
        # Confirm deletion
        confirm = QMessageBox.question(
            self,
            "Delete Conversation",
            "Are you sure you want to delete this conversation?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            # Delete conversation
            self.conversation_manager.delete_conversation(conversation_id)
            
            # Update UI
            self.update_conversation_list()
            
            # Load current conversation
            current_conv = self.conversation_manager.get_current_conversation()
            if current_conv:
                self.display_conversation(current_conv)
            else:
                # Create a new conversation if none exists
                self.new_conversation()
    
    def select_conversation(self, conversation_id):
        """Select a conversation to display"""
        conversation = self.conversation_manager.get_conversation(conversation_id)
        if conversation:
            # Set as current conversation
            self.conversation_manager.set_current_conversation(conversation_id)
            
            # Display conversation
            self.display_conversation(conversation)
            
            # Update conversation list
            self.update_conversation_list()
    
    def display_conversation(self, conversation):
        """Display a conversation"""
        # Clear existing messages
        for i in reversed(range(self.messages_layout.count() - 1)):
            item = self.messages_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
        
        # Set conversation title
        self.conversation_title.setText(conversation.title)
        
        # Add messages
        for message in conversation.messages:
            self.add_message_widget(message)
        
        # Scroll to bottom
        QTimer.singleShot(100, self.scroll_to_bottom)
    
    def add_message_widget(self, message):
        """Add a message widget to the messages area"""
        message_widget = MessageWidget(message)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, message_widget)
    
    def scroll_to_bottom(self):
        """Scroll to the bottom of the messages area"""
        self.messages_scroll.verticalScrollBar().setValue(
            self.messages_scroll.verticalScrollBar().maximum()
        )
    
    def new_conversation(self):
        """Create a new conversation"""
        # Create new conversation
        conversation = self.conversation_manager.create_conversation()
        
        # Update UI
        self.display_conversation(conversation)
        self.update_conversation_list()
        
        # Set focus to input
        self.message_input.setFocus()
    
    def send_message(self):
        """Send a message"""
        # Get message text
        text = self.message_input.toPlainText().strip()
        
        if not text:
            return
        
        # Clear input
        self.message_input.clear()
        
        # Add user message
        self.add_user_message(text)
        
        # Emit message sent signal
        self.message_sent.emit(text)
        
        # Generate response
        self.generate_response(text)
    
    def add_user_message(self, text):
        """Add a user message to the current conversation"""
        # Add message to conversation manager
        message = self.conversation_manager.add_message(text, is_user=True)
        
        # Add message widget
        self.add_message_widget(message)
        
        # Scroll to bottom
        self.scroll_to_bottom()
        
        # Update conversation title if first message
        current_conv = self.conversation_manager.get_current_conversation()
        if current_conv and len(current_conv.messages) == 1:
            # Use first line of message as title
            first_line = text.split('\n', 1)[0]
            title = first_line[:30] + "..." if len(first_line) > 30 else first_line
            self.conversation_manager.rename_conversation(current_conv.id, title)
            self.conversation_title.setText(title)
            
            # Update conversation list
            self.update_conversation_list()
    
    def add_assistant_message(self, text):
        """Add an assistant message to the current conversation"""
        # Add message to conversation manager
        message = self.conversation_manager.add_message(text, is_user=False)
        
        # Add message widget
        self.add_message_widget(message)
        
        # Scroll to bottom
        self.scroll_to_bottom()
    
    def generate_response(self, user_message):
        """Generate a response to the user message"""
        if not self.model_manager or not self.model_manager.is_model_loaded():
            # No model loaded
            self.add_assistant_message(
                "I'm sorry, but no language model is currently loaded. "
                "Please load a model from the Model Selection tab first."
            )
            return
        
        # Get model and tokenizer
        model_id, tokenizer, model = self.model_manager.get_current_model()
        
        if not model or not tokenizer:
            # Model loading issue
            self.add_assistant_message(
                "I'm sorry, but there was an issue with the language model. "
                "Please try reloading it from the Model Selection tab."
            )
            return
        
        try:
            # Update status
            self.status_label.setText("Generating response...")
            
            # Get conversation history for context
            current_conv = self.conversation_manager.get_current_conversation()
            conversation_history = []
            
            if current_conv:
                for msg in current_conv.messages[-6:]:  # Last 6 messages for context
                    if msg.is_user:
                        conversation_history.append(f"User: {msg.text}")
                    else:
                        conversation_history.append(f"Assistant: {msg.text}")
            
            # Create prompt with conversation history
            if conversation_history:
                # Remove the last user message (which we'll add explicitly below)
                if conversation_history[-1].startswith("User:"):
                    conversation_history.pop()
                
                context = "\n".join(conversation_history)
                prompt = f"{context}\n\nUser: {user_message}\n\nAssistant:"
            else:
                prompt = f"User: {user_message}\n\nAssistant:"
            
            # Generate response
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            
            # Move to same device as model
            device = next(model.parameters()).device
            input_ids = input_ids.to(device)
            
            # Set generation parameters
            gen_kwargs = {
                "max_length": min(input_ids.shape[1] + 500, 2048),  # Cap at 2048 tokens
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id
            }
            
            # Generate
            with torch.no_grad():
                output = model.generate(input_ids, **gen_kwargs)
            
            # Decode the output
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract assistant's response
            response = response.split("Assistant:", 1)[-1].strip()
            
            # Check if empty
            if not response:
                response = "I'm sorry, I'm not sure how to respond to that."
            
            # Add assistant message
            self.add_assistant_message(response)
            
            # Update status
            self.status_label.setText("Ready")
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            self.add_assistant_message(
                "I'm sorry, but I encountered an error while generating a response. "
                f"Error: {str(e)}"
            )
            self.status_label.setText("Error generating response")