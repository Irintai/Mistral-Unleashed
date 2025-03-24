"""
Message formatting module for Advanced Code Generator.
Provides utilities for formatting messages with markdown and syntax highlighting.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional

from PyQt5.QtWidgets import QTextBrowser, QFrame, QVBoxLayout
from PyQt5.QtCore import Qt, QRegExp
from PyQt5.QtGui import QTextDocument, QSyntaxHighlighter, QTextCharFormat, QFont, QColor, QBrush


# Initialize logger
logger = logging.getLogger(__name__)


class MarkdownHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for Markdown text."""
    
    def __init__(self, document):
        """Initialize with parent document."""
        super().__init__(document)
        self.highlighting_rules = []
        
        # Define highlighting formats
        self.formats = {
            'header': self.create_format('#0066CC', True, size_adjustment=2),
            'header2': self.create_format('#0066CC', True, size_adjustment=1),
            'header3': self.create_format('#0066CC', True),
            'bold': self.create_format('#000000', True),
            'italic': self.create_format('#000000', italic=True),
            'code': self.create_format('#800000', font_family='Consolas'),
            'code_block': self.create_format('#800000', font_family='Consolas', background='#F5F5F5'),
            'blockquote': self.create_format('#808080', italic=True, background='#F9F9F9'),
            'link': self.create_format('#0000FF', underline=True),
            'list': self.create_format('#800000', True),
        }
        
        # Add highlighting rules
        
        # Headers
        self.highlighting_rules.append((QRegExp(r'^# .+$'), 'header'))
        self.highlighting_rules.append((QRegExp(r'^## .+$'), 'header2'))
        self.highlighting_rules.append((QRegExp(r'^### .+$'), 'header3'))
        
        # Bold and italic
        self.highlighting_rules.append((QRegExp(r'\*\*.+?\*\*'), 'bold'))
        self.highlighting_rules.append((QRegExp(r'__.+?__'), 'bold'))
        self.highlighting_rules.append((QRegExp(r'\*.+?\*'), 'italic'))
        self.highlighting_rules.append((QRegExp(r'_.+?_'), 'italic'))
        
        # Inline code
        self.highlighting_rules.append((QRegExp(r'`[^`]+`'), 'code'))
        
        # Links
        self.highlighting_rules.append((QRegExp(r'\[.+?\]\(.+?\)'), 'link'))
        
        # Lists
        self.highlighting_rules.append((QRegExp(r'^\s*[*+-] '), 'list'))
        self.highlighting_rules.append((QRegExp(r'^\s*\d+\. '), 'list'))
        
        # Blockquotes
        self.highlighting_rules.append((QRegExp(r'^\s*> .*$'), 'blockquote'))
        
        # Code blocks
        self.code_block_start = QRegExp(r'```')
        self.code_block_end = QRegExp(r'```')
        
        # State for multiline elements
        self.in_code_block = False
    
    def create_format(self, color, bold=False, italic=False, underline=False, 
                     background=None, font_family=None, size_adjustment=0):
        """Create a text format with the given properties."""
        fmt = QTextCharFormat()
        fmt.setForeground(QBrush(QColor(color)))
        
        if bold:
            fmt.setFontWeight(QFont.Bold)
        
        if italic:
            fmt.setFontItalic(True)
        
        if underline:
            fmt.setFontUnderline(True)
        
        if background:
            fmt.setBackground(QBrush(QColor(background)))
        
        if font_family:
            font = QFont(font_family)
            fmt.setFont(font)
        
        if size_adjustment != 0:
            font = fmt.font()
            size = font.pointSize()
            font.setPointSize(size + size_adjustment)
            fmt.setFont(font)
        
        return fmt
    
    def highlightBlock(self, text):
        """Apply highlighting to a block of text."""
        # Handle code blocks first
        if self.in_code_block:
            # Format as code block
            self.setFormat(0, len(text), self.formats['code_block'])
            
            # Check if this is the end of the code block
            if self.code_block_end.indexIn(text) != -1:
                self.in_code_block = False
            
            return
        
        # Check for start of code block
        if self.code_block_start.indexIn(text) != -1:
            self.in_code_block = True
            self.setFormat(0, len(text), self.formats['code_block'])
            return
        
        # Apply regular rules
        for pattern, format_name in self.highlighting_rules:
            expression = QRegExp(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, self.formats[format_name])
                index = expression.indexIn(text, index + length)


class MessageFormatter:
    """Handles formatting of messages with markdown support."""
    
    def __init__(self):
        """Initialize the message formatter."""
        self.code_block_pattern = re.compile(r'```(?:\w+)?\s*\n(.*?)\n```', re.DOTALL)
        self.inline_code_pattern = re.compile(r'`([^`]+)`')
    
    def format_markdown(self, text: str) -> str:
        """
        Format text with markdown to HTML.
        
        Args:
            text: The markdown text to format
            
        Returns:
            str: HTML formatted text
        """
        # Escape HTML entities to prevent XSS
        text = self._escape_html(text)
        
        # Format headers
        text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        
        # Format bold and italic
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
        text = re.sub(r'_(.+?)_', r'<i>\1</i>', text)
        
        # Format lists
        text = re.sub(r'^\s*[*+-] (.+)$', r'<ul><li>\1</li></ul>', text, flags=re.MULTILINE)
        
        # Format numbered lists
        text = re.sub(r'^\s*\d+\. (.+)$', r'<ol><li>\1</li></ol>', text, flags=re.MULTILINE)
        
        # Format blockquotes
        text = re.sub(r'^\s*> (.+)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)
        
        # Format links
        text = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', text)
        
        # Format code blocks with syntax highlighting
        text = self._format_code_blocks(text)
        
        # Format inline code
        text = self.inline_code_pattern.sub(r'<code>\1</code>', text)
        
        # Convert newlines to <br>
        text = text.replace('\n', '<br>')
        
        return text
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML entities to prevent XSS."""
        replacements = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        }
        
        # Skip escape inside code blocks
        code_blocks = self.code_block_pattern.findall(text)
        for block in code_blocks:
            text = text.replace(block, '#CODE_BLOCK_PLACEHOLDER#')
        
        # Escape HTML entities
        for char, entity in replacements.items():
            text = text.replace(char, entity)
        
        # Restore code blocks
        for block in code_blocks:
            text = text.replace('#CODE_BLOCK_PLACEHOLDER#', block, 1)
        
        return text
    
    def _format_code_blocks(self, text: str) -> str:
        """Format code blocks with syntax highlighting."""
        def replace_code_block(match):
            code = match.group(1)
            lang = match.group(0).split('```')[1].strip()
            
            # Add the language class for potential syntax highlighting
            if lang:
                return f'<pre><code class="language-{lang}">{code}</code></pre>'
            else:
                return f'<pre><code>{code}</code></pre>'
        
        # Replace code blocks with HTML
        text = re.sub(r'```(\w+)?\s*\n(.*?)\n```', replace_code_block, text, flags=re.DOTALL)
        
        return text
    
    def apply_to_text_browser(self, browser: QTextBrowser, text: str) -> None:
        """
        Apply formatted markdown to a QTextBrowser widget.
        
        Args:
            browser: The QTextBrowser widget
            text: The markdown text to format and display
        """
        html = self.format_markdown(text)
        browser.setHtml(html)
        
        # Apply custom highlighting
        highlighter = MarkdownHighlighter(browser.document())
        
        # Enable rich text interaction
        browser.setOpenExternalLinks(True)
        browser.setTextInteractionFlags(Qt.TextBrowserInteraction)


def create_formatted_message_widget(text: str) -> QTextBrowser:
    """
    Create a widget with formatted message content.
    
    Args:
        text: The markdown text to format
        
    Returns:
        QWidget: Formatted content widget
    """
    # Create frame to hold the content
    content_frame = QFrame()
    content_frame.setProperty("name", "content_frame")
    content_frame.setStyleSheet("""
        QFrame {
            background-color: transparent;
            border: none;
        }
    """)
    
    # Create layout
    content_layout = QVBoxLayout(content_frame)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(0)
    
    # Create browser with formatted content
    formatter = MessageFormatter()
    html_content = formatter.format_markdown(text)
    
    # Create browser with formatted content
    browser = QTextBrowser()
    browser.setHtml(html_content)
    browser.setOpenExternalLinks(True)
    browser.setStyleSheet("""
        QTextBrowser {
            background-color: transparent;
            border: none;
            color: #333333;
            font-size: 14px;
        }
    """)
    
    # Set object name for identification in update_message method
    browser.setObjectName("content_display")
    
    # Add to layout
    content_layout.addWidget(browser)
    
    return content_frame


def format_message_content(role: str, content: str, timestamp: str = None) -> str:
    """Format a message content based on its role with timestamp."""
    from src.core.settings import SettingsManager
    
    # Get settings manager instance
    settings_manager = SettingsManager()
    
    # Get styling settings
    font_family = settings_manager.get_value("conversation_font_family", "Arial") 
    font_size = settings_manager.get_value("conversation_font_size", 14)
    user_color = settings_manager.get_value("user_message_color", "#333333")
    assistant_color = settings_manager.get_value("assistant_message_color", "#333333")
    system_color = settings_manager.get_value("system_message_color", "#555555")
    code_font = settings_manager.get_value("code_font_family", "Consolas, monospace")
    code_font_size = settings_manager.get_value("code_font_size", 13)
    code_bg_color = settings_manager.get_value("code_background_color", "#f6f8fa")
    
    # Base styling
    base_style = f"""
    <style>
        .message-content {{
            font-family: {font_family};
            font-size: {font_size}px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .user-message {{
            color: {user_color};
        }}
        .assistant-message {{
            color: {assistant_color};
        }}
        .system-message {{
            color: {system_color};
            font-style: italic;
        }}
        code {{
            font-family: {code_font};
            font-size: {code_font_size}px;
            background-color: {code_bg_color};
            padding: 2px 4px;
            border-radius: 3px;
        }}
        pre {{
            font-family: {code_font};
            font-size: {code_font_size}px;
            background-color: {code_bg_color};
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 10px 0;
        }}
        .timestamp {{
            font-size: 0.8em;
            color: #888;
            margin-top: 5px;
            text-align: right;
        }}
    </style>
    """
    
    # Add timestamp if provided
    timestamp_html = f'<div class="timestamp">{timestamp}</div>' if timestamp else ''
    
    # Format based on role
    if role == "user":
        formatted_content = f"{base_style}<div class='message-content user-message'>{content}</div>{timestamp_html}"
    elif role == "assistant":
        formatted_content = f"{base_style}<div class='message-content assistant-message'>{content}</div>{timestamp_html}"
    else:  # system message or other
        formatted_content = f"{base_style}<div class='message-content system-message'>{content}</div>{timestamp_html}"
    
    return formatted_content


def markdown_to_html(markdown_text: str) -> str:
    """Convert markdown text to HTML with syntax highlighting."""
    from src.core.settings import SettingsManager
    
    # Get settings manager instance
    settings_manager = SettingsManager()
    
    # Get styling settings
    font_family = settings_manager.get_value("conversation_font_family", "Arial") 
    font_size = settings_manager.get_value("conversation_font_size", 14)
    code_font = settings_manager.get_value("code_font_family", "Consolas, monospace")
    code_font_size = settings_manager.get_value("code_font_size", 13)
    code_bg_color = settings_manager.get_value("code_background_color", "#f6f8fa")
    
    if not markdown_text:
        return ""
        
    formatter = MessageFormatter()
    html_content = formatter.format_markdown(markdown_text)
    
    # Ensure proper encoding and HTML structure
    return f"""
    <html>
    <head>
    <style>
        body {{ 
            font-family: {font_family}; 
            font-size: {font_size}px;
            line-height: 1.5;
        }}
        code {{ 
            font-family: {code_font};
            font-size: {code_font_size}px;
            background-color: {code_bg_color}; 
            padding: 2px 4px;
            border-radius: 3px;
        }}
        pre {{ 
            font-family: {code_font};
            font-size: {code_font_size}px;
            background-color: {code_bg_color};
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 10px 0;
        }}
    </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """
