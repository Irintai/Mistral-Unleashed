"""
Code editor module for Advanced Code Generator.
Provides a code editor widget with syntax highlighting and other features.
"""

import os
import sys
import logging
from PyQt5.QtWidgets import QPlainTextEdit, QWidget, QVBoxLayout, QTextEdit
from PyQt5.QtCore import Qt, QRect, QSize
from PyQt5.QtGui import QColor, QPainter, QTextFormat, QFont, QFontMetrics

from src.gui.syntax_highlighter import PythonHighlighter, CppHighlighter, JavaScriptHighlighter, HtmlHighlighter

# Logger for this module
logger = logging.getLogger(__name__)


class LineNumberArea(QWidget):
    """Widget for displaying line numbers in the code editor."""
    
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
    
    def sizeHint(self):
        """Return the size hint for the line number area."""
        return QSize(self.editor.lineNumberAreaWidth(), 0)
    
    def paintEvent(self, event):
        """Handle paint events for the line number area."""
        self.editor.lineNumberAreaPaintEvent(event)


class QCodeEditor(QPlainTextEdit):
    """Enhanced text editor with code editing features."""
    
    def __init__(self, parent=None):
        """Initialize the code editor."""
        super(QCodeEditor, self).__init__(parent)
        
        # Font setup
        self.setFont(QFont("Consolas", 10))
        
        # Line numbers
        self.line_number_area = LineNumberArea(self)
        self.blockCountChanged.connect(self.updateLineNumberAreaWidth)
        self.updateRequest.connect(self.updateLineNumberArea)
        self.cursorPositionChanged.connect(self.highlightCurrentLine)
        self.updateLineNumberAreaWidth(0)
        
        # Syntax highlighter (default to Python)
        self.highlighter = PythonHighlighter(self.document())
        
        # Tab settings
        self.setTabStopWidth(4 * self.fontMetrics().width(' '))
        
        # Auto indent
        self.auto_indent = True
        
        # Set a background color
        self.setStyleSheet("background-color: #F8F8F8; color: #2D2D2D;")
        
        # Setup key handling for common programming features
        self.setup_key_handlers()
        
        # Current language
        self.current_language = "python"
    
    def set_language(self, language):
        """Set the syntax highlighting language."""
        language = language.lower()
        
        if language == self.current_language:
            return
        
        # Remove current highlighter
        if hasattr(self, 'highlighter') and self.highlighter:
            self.highlighter.setDocument(None)
        
        # Apply new highlighter
        if language == "python":
            self.highlighter = PythonHighlighter(self.document())
        elif language == "cpp" or language == "c++" or language == "c":
            self.highlighter = CppHighlighter(self.document())
        elif language == "javascript" or language == "js":
            self.highlighter = JavaScriptHighlighter(self.document())
        elif language == "html":
            self.highlighter = HtmlHighlighter(self.document())
        else:
            # Default to Python
            self.highlighter = PythonHighlighter(self.document())
            language = "python"
        
        self.current_language = language
    
    def auto_detect_language(self):
        """Auto-detect the code language based on content."""
        text = self.toPlainText().lower()
        
        # Python indicators
        if "def " in text or "import " in text or "class " in text:
            self.set_language("python")
            return "python"
        
        # C++ indicators
        elif "#include" in text or "int main" in text:
            self.set_language("cpp")
            return "cpp"
        
        # JavaScript indicators
        elif "function" in text or "var " in text or "const " in text or "let " in text:
            self.set_language("javascript")
            return "javascript"
        
        # HTML indicators
        elif "<html" in text or "<!doctype" in text:
            self.set_language("html")
            return "html"
        
        # Default to Python
        self.set_language("python")
        return "python"
    
    def lineNumberAreaWidth(self):
        """Calculate the width of the line number area."""
        digits = 1
        max_num = max(1, self.blockCount())
        while max_num >= 10:
            max_num //= 10
            digits += 1
        
        space = 3 + self.fontMetrics().width('9') * digits
        return space
    
    def updateLineNumberAreaWidth(self, _):
        """Update the width of the line number area."""
        self.setViewportMargins(self.lineNumberAreaWidth(), 0, 0, 0)
    
    def updateLineNumberArea(self, rect, dy):
        """Update the line number area on scroll."""
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(0, rect.y(), self.line_number_area.width(), rect.height())
        
        if rect.contains(self.viewport().rect()):
            self.updateLineNumberAreaWidth(0)
    
    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.line_number_area.setGeometry(QRect(cr.left(), cr.top(), self.lineNumberAreaWidth(), cr.height()))
    
    def lineNumberAreaPaintEvent(self, event):
        """Paint the line number area."""
        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QColor("#F0F0F0"))
        
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()
        
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(QColor("#A0A0A0"))
                painter.drawText(0, top, self.line_number_area.width(), self.fontMetrics().height(),
                                 Qt.AlignRight, number)
            
            block = block.next()
            top = bottom
            bottom = top + self.blockBoundingRect(block).height()
            block_number += 1
    
    def highlightCurrentLine(self):
        """Highlight the line with the cursor."""
        extra_selections = []
        
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            line_color = QColor("#E8F5E9")
            selection.format.setBackground(line_color)
            selection.format.setProperty(QTextFormat.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            extra_selections.append(selection)
        
        self.setExtraSelections(extra_selections)
    
    def setup_key_handlers(self):
        """Set up key handlers for programming-specific features."""
        self.keyPressEvent = self.custom_key_press_event
    
    def custom_key_press_event(self, event):
        """Custom key press event handler."""
        # Auto-indentation for brackets
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.auto_indent:
                self.handleAutoIndent()
                return
        
        # Auto-completion for brackets and quotes
        if event.key() == Qt.Key_BraceLeft:
            self.insertPlainText("{}")
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.Left)
            self.setTextCursor(cursor)
            return
        
        if event.key() == Qt.Key_BracketLeft:
            self.insertPlainText("[]")
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.Left)
            self.setTextCursor(cursor)
            return
        
        if event.key() == Qt.Key_ParenLeft:
            self.insertPlainText("()")
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.Left)
            self.setTextCursor(cursor)
            return
        
        if event.key() == Qt.Key_QuoteDbl:
            self.insertPlainText("\"\"")
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.Left)
            self.setTextCursor(cursor)
            return
        
        if event.key() == Qt.Key_Apostrophe:
            self.insertPlainText("''")
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.Left)
            self.setTextCursor(cursor)
            return
        
        # Forward to the parent
        super(QCodeEditor, self).keyPressEvent(event)
    
    def handleAutoIndent(self):
        """Handle auto-indentation when Enter is pressed."""
        cursor = self.textCursor()
        text = cursor.block().text()
        
        # Find the indentation level of the current line
        indent = ""
        for char in text:
            if char == ' ' or char == '\t':
                indent += char
            else:
                break
        
        # Check if the line ends with a colon (for Python) or an opening brace
        if self.current_language == "python" and text.rstrip().endswith(':'):
            indent += "    "  # Add additional indentation for Python after colon
        elif text.rstrip().endswith('{') or text.rstrip().endswith('['):
            indent += "    "  # Add additional indentation after braces
        
        # Insert newline and indentation
        cursor.insertText("\n" + indent)
        self.setTextCursor(cursor)
    
    def insertFromMimeData(self, source):
        """Handle paste operations."""
        # Just insert plain text and let the syntax highlighter do its job
        if source.hasText():
            self.insertPlainText(source.text())
        else:
            super(QCodeEditor, self).insertFromMimeData(source)
        
        # After pasting code, try to auto-detect the language
        self.auto_detect_language()
