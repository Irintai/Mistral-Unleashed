"""
Syntax highlighter module for Advanced Code Editor.
Provides syntax highlighting for various programming languages.
"""

import re
from PyQt5.QtGui import QSyntaxHighlighter, QTextCharFormat, QFont, QColor
from PyQt5.QtCore import Qt


class BaseSyntaxHighlighter(QSyntaxHighlighter):
    """Base class for syntax highlighters."""
    
    def __init__(self, document):
        """Initialize with document."""
        super(BaseSyntaxHighlighter, self).__init__(document)
        self.highlighting_rules = []
        self.formats = {}
        
        # Initialize the formats
        self.initialize_formats()
        
        # Initialize the highlighting rules
        self.initialize_rules()
    
    def initialize_formats(self):
        """Initialize the text formats."""
        # Default formats
        self.formats["keyword"] = self.create_format("#0056AA", True)
        self.formats["operator"] = self.create_format("#666666")
        self.formats["brace"] = self.create_format("#666666")
        self.formats["defclass"] = self.create_format("#007788", True)
        self.formats["string"] = self.create_format("#BA2121")
        self.formats["string2"] = self.create_format("#0080D0")
        self.formats["comment"] = self.create_format("#408040", False, True)
        self.formats["self"] = self.create_format("#9A7700", True)
        self.formats["numbers"] = self.create_format("#AA0000")
        self.formats["decorator"] = self.create_format("#AA5500")
    
    def initialize_rules(self):
        """Initialize the highlighting rules. To be overridden by subclasses."""
        pass
    
    def create_format(self, color, bold=False, italic=False):
        """Create a text format with the given attributes."""
        char_format = QTextCharFormat()
        char_format.setForeground(QColor(color))
        
        if bold:
            char_format.setFontWeight(QFont.Bold)
        
        if italic:
            char_format.setFontItalic(True)
        
        return char_format
    
    def highlightBlock(self, text):
        """Highlight the block of text."""
        # Apply the highlighting rules
        for pattern, format_key in self.highlighting_rules:
            expression = re.compile(pattern)
            index = expression.indexIn(text) if hasattr(expression, 'indexIn') else text.find(pattern)
            while index >= 0:
                length = expression.matchedLength() if hasattr(expression, 'matchedLength') else len(pattern)
                self.setFormat(index, length, self.formats[format_key])
                index = expression.indexIn(text, index + length) if hasattr(expression, 'indexIn') else text.find(pattern, index + length)
        
        # Handle multiline comments, strings, etc. in subclasses
        self.handle_multiline(text)
    
    def handle_multiline(self, text):
        """Handle multiline comments, strings, etc. To be overridden by subclasses."""
        pass


class PythonHighlighter(BaseSyntaxHighlighter):
    """Syntax highlighter for Python code."""
    
    def initialize_rules(self):
        """Initialize the highlighting rules for Python."""
        # Python keywords
        keywords = [
            'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 
            'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 
            'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 
            'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'
        ]
        
        # Add keyword rules
        for word in keywords:
            pattern = r'\\b' + word + r'\\b'
            self.highlighting_rules.append((pattern, "keyword"))
        
        # Class/function name
        self.highlighting_rules.append((r'\\bclass\\b\\s*(\\w+)', "defclass"))
        self.highlighting_rules.append((r'\\bdef\\b\\s*(\\w+)', "defclass"))
        
        # Numeric literals
        self.highlighting_rules.append((r'\\b[0-9]+[lL]?\\b', "numbers"))
        self.highlighting_rules.append((r'\\b0[xX][0-9A-Fa-f]+[lL]?\\b', "numbers"))
        self.highlighting_rules.append((r'\\b[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\\b', "numbers"))
        
        # String literals
        self.highlighting_rules.append((r'"[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "string"))
        self.highlighting_rules.append((r"'[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", "string"))
        
        # Triple-quoted strings
        self.highlighting_rules.append((r'""".*?"""', "string2"))
        self.highlighting_rules.append((r"'''.*?'''", "string2"))
        
        # Comments
        self.highlighting_rules.append((r'#[^\n]*', "comment"))
        
        # 'self' keyword
        self.highlighting_rules.append((r'\\bself\\b', "self"))
        
        # Decorators
        self.highlighting_rules.append((r'@\\w+', "decorator"))
        
        # Braces, brackets, parentheses
        self.highlighting_rules.append((r'[\\{\\}\\(\\)\\[\\]]', "brace"))
        
        # Operators
        self.highlighting_rules.append((r'[\\+\\-\\*\\/\\=<>]', "operator"))
    
    def handle_multiline(self, text):
        """Handle multiline strings and comments in Python."""
        # Python's multiline strings are already handled by our regular expressions
        pass


class CppHighlighter(BaseSyntaxHighlighter):
    """Syntax highlighter for C++ code."""
    
    def initialize_rules(self):
        """Initialize the highlighting rules for C++."""
        # C++ keywords
        keywords = [
            'alignas', 'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 
            'bool', 'break', 'case', 'catch', 'char', 'char16_t', 'char32_t', 'class', 
            'compl', 'const', 'constexpr', 'const_cast', 'continue', 'decltype', 
            'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum', 
            'explicit', 'export', 'extern', 'false', 'float', 'for', 'friend', 'goto', 
            'if', 'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 
            'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq', 'private', 
            'protected', 'public', 'register', 'reinterpret_cast', 'return', 'short', 
            'signed', 'sizeof', 'static', 'static_assert', 'static_cast', 'struct', 
            'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try', 
            'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 
            'void', 'volatile', 'wchar_t', 'while', 'xor', 'xor_eq'
        ]
        
        # Add keyword rules
        for word in keywords:
            pattern = r'\\b' + word + r'\\b'
            self.highlighting_rules.append((pattern, "keyword"))
        
        # Class/function name
        self.highlighting_rules.append((r'\\bclass\\b\\s*(\\w+)', "defclass"))
        self.highlighting_rules.append((r'\\b\\w+(?=\\s*\\()', "defclass"))
        
        # Numeric literals
        self.highlighting_rules.append((r'\\b[0-9]+[uUlL]*\\b', "numbers"))
        self.highlighting_rules.append((r'\\b0[xX][0-9A-Fa-f]+[uUlL]*\\b', "numbers"))
        self.highlighting_rules.append((r'\\b[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?[fF]?\\b', "numbers"))
        
        # String literals
        self.highlighting_rules.append((r'"[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "string"))
        self.highlighting_rules.append((r"'.'", "string"))
        
        # Comments
        self.highlighting_rules.append((r'//[^\n]*', "comment"))
        self.highlighting_rules.append((r'/\\*.*?\\*/', "comment"))
        
        # Preprocessor directives
        self.highlighting_rules.append((r'#\\w+', "decorator"))
        
        # Braces, brackets, parentheses
        self.highlighting_rules.append((r'[\\{\\}\\(\\)\\[\\]]', "brace"))
        
        # Operators
        self.highlighting_rules.append((r'[\\+\\-\\*\\/\\=<>]', "operator"))
    
    def handle_multiline(self, text):
        """Handle multiline comments in C++."""
        # Start of a multiline comment
        comment_start = re.search(r'/\\*', text)
        comment_end = re.search(r'\\*/', text)
        
        if comment_start and not comment_end:
            self.setFormat(comment_start.start(), len(text) - comment_start.start(), self.formats["comment"])
        
        # End of a multiline comment
        elif comment_end and not comment_start:
            self.setFormat(0, comment_end.end(), self.formats["comment"])


class JavaScriptHighlighter(BaseSyntaxHighlighter):
    """Syntax highlighter for JavaScript code."""
    
    def initialize_rules(self):
        """Initialize the highlighting rules for JavaScript."""
        # JavaScript keywords
        keywords = [
            'abstract', 'arguments', 'await', 'boolean', 'break', 'byte', 'case', 
            'catch', 'char', 'class', 'const', 'continue', 'debugger', 'default', 
            'delete', 'do', 'double', 'else', 'enum', 'eval', 'export', 'extends', 
            'false', 'final', 'finally', 'float', 'for', 'function', 'goto', 'if', 
            'implements', 'import', 'in', 'instanceof', 'int', 'interface', 'let', 
            'long', 'native', 'new', 'null', 'package', 'private', 'protected', 
            'public', 'return', 'short', 'static', 'super', 'switch', 'synchronized', 
            'this', 'throw', 'throws', 'transient', 'true', 'try', 'typeof', 'var', 
            'void', 'volatile', 'while', 'with', 'yield'
        ]
        
        # Add keyword rules
        for word in keywords:
            pattern = r'\\b' + word + r'\\b'
            self.highlighting_rules.append((pattern, "keyword"))
        
        # Function name
        self.highlighting_rules.append((r'\\bfunction\\b\\s*(\\w+)', "defclass"))
        self.highlighting_rules.append((r'\\b\\w+(?=\\s*\\()', "defclass"))
        
        # Numeric literals
        self.highlighting_rules.append((r'\\b[0-9]+[nN]?\\b', "numbers"))
        self.highlighting_rules.append((r'\\b0[xX][0-9A-Fa-f]+[nN]?\\b', "numbers"))
        self.highlighting_rules.append((r'\\b[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\\b', "numbers"))
        
        # String literals
        self.highlighting_rules.append((r'"[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "string"))
        self.highlighting_rules.append((r"'[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", "string"))
        self.highlighting_rules.append((r'`[^`\\\\]*(?:\\\\.[^`\\\\]*)*`', "string2"))
        
        # Comments
        self.highlighting_rules.append((r'//[^\n]*', "comment"))
        self.highlighting_rules.append((r'/\\*.*?\\*/', "comment"))
        
        # Braces, brackets, parentheses
        self.highlighting_rules.append((r'[\\{\\}\\(\\)\\[\\]]', "brace"))
        
        # Operators
        self.highlighting_rules.append((r'[\\+\\-\\*\\/\\=<>]', "operator"))
    
    def handle_multiline(self, text):
        """Handle multiline comments in JavaScript."""
        # Start of a multiline comment
        comment_start = re.search(r'/\\*', text)
        comment_end = re.search(r'\\*/', text)
        
        if comment_start and not comment_end:
            self.setFormat(comment_start.start(), len(text) - comment_start.start(), self.formats["comment"])
        
        # End of a multiline comment
        elif comment_end and not comment_start:
            self.setFormat(0, comment_end.end(), self.formats["comment"])


class HtmlHighlighter(BaseSyntaxHighlighter):
    """Syntax highlighter for HTML code."""
    
    def initialize_formats(self):
        """Initialize the text formats for HTML."""
        super(HtmlHighlighter, self).initialize_formats()
        # HTML specific formats
        self.formats["tag"] = self.create_format("#008000")
        self.formats["attribute"] = self.create_format("#7D9029")
        self.formats["entity"] = self.create_format("#BA2121")
    
    def initialize_rules(self):
        """Initialize the highlighting rules for HTML."""
        # Tags
        self.highlighting_rules.append((r'<\\s*\\w+.*?>', "tag"))
        self.highlighting_rules.append((r'</\\s*\\w+\\s*>', "tag"))
        
        # Attributes
        self.highlighting_rules.append((r'\\w+(?=\\s*=\\s*["\'])', "attribute"))
        
        # String literals
        self.highlighting_rules.append((r'"[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "string"))
        self.highlighting_rules.append((r"'[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", "string"))
        
        # Comments
        self.highlighting_rules.append((r'<!--.*?-->', "comment"))
        
        # Entities
        self.highlighting_rules.append((r'&[A-Za-z0-9#]+;', "entity"))
    
    def handle_multiline(self, text):
        """Handle multiline comments in HTML."""
        # Start of a multiline comment
        comment_start = re.search(r'<!--', text)
        comment_end = re.search(r'-->', text)
        
        if comment_start and not comment_end:
            self.setFormat(comment_start.start(), len(text) - comment_start.start(), self.formats["comment"])
        
        # End of a multiline comment
        elif comment_end and not comment_start:
            self.setFormat(0, comment_end.end(), self.formats["comment"])
