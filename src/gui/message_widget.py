"""
Message widget module for Advanced Code Generator.
Provides widgets for displaying messages in the conversation tab.
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QMenu, QAction, QFrame, 
                            QSizePolicy, QSpacerItem, QApplication, QTextBrowser)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QMimeData
from PyQt5.QtGui import QIcon, QPixmap, QColor, QPalette, QClipboard

from src.gui.message_formatting import create_formatted_message_widget, format_message_content

# Initialize logger
logger = logging.getLogger(__name__)


class MessageReactionButton(QPushButton):
    """Button for message reactions like thumbs up, copy, etc."""
    
    clicked_with_data = pyqtSignal(str, dict)
    
    def __init__(self, icon_name: str, tooltip: str, action_data: Dict, parent=None):
        """Initialize the reaction button."""
        super().__init__(parent)
        self.action_data = action_data
        
        # Set icon
        self.setIcon(QIcon(f"resources/icons/{icon_name}.png"))
        self.setToolTip(tooltip)
        
        # Style the button
        self.setFlat(True)
        self.setIconSize(QSize(16, 16))
        self.setFixedSize(24, 24)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
                border-radius: 12px;
            }
        """)
        
        # Connect clicked signal
        self.clicked.connect(self.emit_clicked_with_data)
    
    def emit_clicked_with_data(self):
        """Emit clicked signal with action data."""
        self.clicked_with_data.emit(self.action_data.get("action", ""), self.action_data)


class Message:
    """Represents a message in the conversation."""
    
    def __init__(self, content: str, role: str = "user", 
                 timestamp: Optional[datetime] = None,
                 message_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize a message."""
        self.content = content
        self.role = role  # "user", "assistant", "system"
        self.timestamp = timestamp or datetime.now()
        self.message_id = message_id or f"{role}_{int(self.timestamp.timestamp())}"
        self.metadata = metadata or {}
        self.reactions = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "content": self.content,
            "role": self.role,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "metadata": self.metadata,
            "reactions": self.reactions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a message from dictionary."""
        timestamp = datetime.fromisoformat(data.get("timestamp")) if data.get("timestamp") else None
        message = cls(
            content=data.get("content", ""),
            role=data.get("role", "user"),
            timestamp=timestamp,
            message_id=data.get("message_id"),
            metadata=data.get("metadata", {})
        )
        message.reactions = data.get("reactions", [])
        return message
    
    def add_reaction(self, reaction: str) -> None:
        """Add a reaction to the message."""
        if reaction not in self.reactions:
            self.reactions.append(reaction)
    
    def remove_reaction(self, reaction: str) -> None:
        """Remove a reaction from the message."""
        if reaction in self.reactions:
            self.reactions.remove(reaction)
    
    def has_code(self) -> bool:
        """Check if the message contains code blocks."""
        return bool(re.search(r'```.*?```', self.content, re.DOTALL))
    
    def extract_code_blocks(self) -> List[str]:
        """Extract code blocks from the message."""
        code_blocks = []
        matches = re.finditer(r'```(?:\w+)?\s*\n(.*?)\n```', self.content, re.DOTALL)
        for match in matches:
            code_blocks.append(match.group(1))
        return code_blocks


class MessageWidget(QWidget):
    """Widget for displaying a message in the conversation."""
    
    reaction_clicked = pyqtSignal(str, str, dict)  # message_id, action, data
    
    def __init__(self, message: 'Message', 
                on_copy: Optional[Callable] = None,
                on_edit: Optional[Callable] = None,
                on_delete: Optional[Callable] = None,
                parent=None):
        """Initialize the message widget."""
        super().__init__(parent)
        self.message = message
        self.on_copy = on_copy
        self.on_edit = on_edit
        self.on_delete = on_delete
        
        # Setup UI
        self.init_ui()
    
    def init_ui(self):
        """Set up the message widget UI."""
        # Set stylesheet
        self.setStyleSheet(f"""
            QFrame {{
                border-radius: 12px;
                background-color: {self.get_background_color()};
            }}
        """)
        
        # Create main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Message header with user/assistant icon and timestamp
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        
        # Icon and role label
        icon_label = QLabel()
        icon_pixmap = self.get_role_icon()
        icon_label.setPixmap(icon_pixmap)
        header_layout.addWidget(icon_label)
        
        role_label = QLabel(self.get_display_name())
        role_label.setStyleSheet(f"font-weight: bold; color: {self.get_role_color()};")
        header_layout.addWidget(role_label)
        
        # Timestamp
        time_str = self.message.timestamp.strftime("%I:%M %p")
        time_label = QLabel(time_str)
        time_label.setStyleSheet("color: #757575; font-size: 10px;")
        time_label.setObjectName("timestamp_label")
        header_layout.addWidget(time_label)
        
        # Spacer
        header_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        # Add header to main layout
        layout.addLayout(header_layout)
        
        # Message content with markdown support
        content_browser = QTextBrowser()
        content_browser.setOpenExternalLinks(True)
        content_browser.setStyleSheet("""
            QTextBrowser {
                background-color: transparent;
                border: none;
                color: #333333;
                font-size: 14px;
            }
        """)
        
        # Set the HTML content using our formatting function with role
        formatted_content = format_message_content(self.message.content, self.message.role)
        content_browser.setHtml(formatted_content)
        
        # Important: Set object name for identification in update_message method
        content_browser.setObjectName("content_display")
        
        # Add to layout
        layout.addWidget(content_browser)
        
        # Reactions bar
        reactions_layout = QHBoxLayout()
        reactions_layout.setContentsMargins(0, 0, 0, 0)
        reactions_layout.setSpacing(5)
        
        # Copy button
        copy_button = MessageReactionButton(
            "copy", "Copy message", 
            {"action": "copy", "message_id": self.message.message_id},
            self
        )
        copy_button.clicked_with_data.connect(self.handle_reaction)
        reactions_layout.addWidget(copy_button)
        
        # Only show edit button for user messages
        if self.message.role == "user" and self.on_edit:
            edit_button = MessageReactionButton(
                "edit", "Edit message",
                {"action": "edit", "message_id": self.message.message_id},
                self
            )
            edit_button.clicked_with_data.connect(self.handle_reaction)
            reactions_layout.addWidget(edit_button)
        
        # Only show delete button if callback provided
        if self.on_delete:
            delete_button = MessageReactionButton(
                "delete", "Delete message",
                {"action": "delete", "message_id": self.message.message_id},
                self
            )
            delete_button.clicked_with_data.connect(self.handle_reaction)
            reactions_layout.addWidget(delete_button)
        
        # Thumbs up/down for assistant messages
        if self.message.role == "assistant":
            thumbs_up = MessageReactionButton(
                "thumbs_up", "Thumbs up",
                {"action": "thumbs_up", "message_id": self.message.message_id},
                self
            )
            thumbs_up.clicked_with_data.connect(self.handle_reaction)
            reactions_layout.addWidget(thumbs_up)
            
            thumbs_down = MessageReactionButton(
                "thumbs_down", "Thumbs down",
                {"action": "thumbs_down", "message_id": self.message.message_id},
                self
            )
            thumbs_down.clicked_with_data.connect(self.handle_reaction)
            reactions_layout.addWidget(thumbs_down)
        
        # Spacer at the end
        reactions_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        # Add reactions to main layout
        layout.addLayout(reactions_layout)
        
        # Add a line separator at the bottom
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #E0E0E0;")
        layout.addWidget(line)
    
    def get_background_color(self):
        """Get the background color based on message role."""
        if self.message.role == "user":
            return "#F5F5F5"
        elif self.message.role == "assistant":
            return "#FAFAFA"
        elif self.message.role == "system":
            return "#FFFDE7"
        else:
            return "#FFFFFF"
    
    def get_role_color(self):
        """Get the color for the message role."""
        if self.message.role == "user":
            return "#2979FF"  # Blue
        elif self.message.role == "assistant":
            return "#FF5722"  # Orange/Red
        elif self.message.role == "system":
            return "#4CAF50"  # Green
        else:
            return "#9E9E9E"  # Grey

    def get_display_name(self):
        """Get the display name for the role."""
        if self.message.role == "user":
            return "You"
        elif self.message.role == "assistant":
            return "Assistant"
        elif self.message.role == "system":
            return "System"
        else:
            return self.message.role.capitalize()
    
    def get_role_icon(self):
        """Get the icon for the message role."""
        from PyQt5.QtGui import QPixmap, QColor
        
        icon_size = 16
        pixmap = QPixmap(icon_size, icon_size)
        
        # Make sure we create a valid QColor
        color_str = self.get_role_color()
        try:
            color = QColor(color_str)
            if color.isValid():
                pixmap.fill(color)
            else:
                # Fallback to a default color if conversion fails
                pixmap.fill(QColor("#9E9E9E"))
        except Exception as e:
            # Handle any errors in color conversion
            logger.error(f"Error creating color from {color_str}: {str(e)}")
            pixmap.fill(QColor("#9E9E9E"))
            
        return pixmap
    
    def handle_reaction(self, action: str, data: Dict):
        """Handle reaction button clicks."""
        message_id = data.get("message_id", "")
        
        if action == "copy" and self.on_copy:
            self.on_copy(self.message)
            return
        
        if action == "edit" and self.on_edit:
            self.on_edit(self.message)
            return
        
        if action == "delete" and self.on_delete:
            self.on_delete(self.message)
            return
        
        # Emit signal for other reactions
        self.reaction_clicked.emit(message_id, action, data)
    
    def copy_content_to_clipboard(self):
        """Copy message content to clipboard."""
        clipboard = QApplication.clipboard()
        mime_data = QMimeData()
        
        # Set plain text
        mime_data.setText(self.message.content)
        
        # Also set HTML if message has formatting
        if self.message.has_code() or re.search(r'[*_#`]', self.message.content):
            # Use a simple conversion for HTML
            html = self.message.content
            html = html.replace("\n", "<br>")
            html = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
            html = re.sub(r'`(.*?)`', r'<code>\1</code>', html)
            html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', html)
            html = re.sub(r'__(.*?)__', r'<b>\1</b>', html)
            html = re.sub(r'\*(.*?)\*', r'<i>\1</i>', html)
            html = re.sub(r'_(.*?)_', r'<i>\1</i>', html)
            
            mime_data.setHtml(html)
        
        clipboard.setMimeData(mime_data)
    
    def update_message(self, message):
        """Update the message content without rebuilding the entire UI."""
        self.message = message
        
        # Update content in the message content container
        for widget in self.findChildren((QTextBrowser, QLabel)):
            if hasattr(widget, 'setHtml') and widget.objectName() == "content_display":
                # Format the content based on role
                formatted_content = format_message_content(message.content, message.role)
                widget.setHtml(formatted_content)
                logger.debug(f"Updated message content in QTextBrowser: {message.content[:30]}...")
                break
        
        # Update timestamp
        timestamp_labels = [widget for widget in self.findChildren(QLabel) if widget.objectName() == "timestamp_label"]
        if timestamp_labels:
            timestamp_labels[0].setText(self.get_formatted_time(message.timestamp))
    
    def get_formatted_time(self, timestamp):
        """Format timestamp for display in the message."""
        if timestamp:
            return timestamp.strftime("%I:%M %p")
        return ""
