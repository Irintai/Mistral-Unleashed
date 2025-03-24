"""
Markdown renderer for Advanced Code Generator.
Provides capabilities to render markdown in QTextEdit widgets.
"""

import re
import logging
from PyQt5.QtGui import QTextDocument, QSyntaxHighlighter, QTextCharFormat, QColor, QFont, QTextCursor
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtCore import Qt

# Initialize logger
logger = logging.getLogger(__name__)


class CodeBlockHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for code blocks in markdown"""
    
    def __init__(self, document, language='python'):
        super().__init__(document)
        self.language = language.lower()
        self.highlighting_rules = []
        
        # Define formats for different syntax elements
        self.formats = {
            'keyword': self._create_format("#0000FF", True),
            'string': self._create_format("#008000"),
            'comment': self._create_format("#808080", italic=True),
            'number': self._create_format("#FF8000"),
            'function': self._create_format("#800080", True),
            'operator': self._create_format("#000000", True),
            'class': self._create_format("#0000A0", True),
            'decorator': self._create_format("#808000"),
        }
        
        # Set up highlighting rules based on language
        self._setup_highlighting_rules()
    
    def _create_format(self, color, bold=False, italic=False):
        """Create a text format with specified properties"""
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        if bold:
            fmt.setFontWeight(QFont.Bold)
        if italic:
            fmt.setFontItalic(True)
        return fmt
    
    def _setup_highlighting_rules(self):
        """Set up syntax highlighting rules for the selected language"""
        if self.language == 'python':
            self._setup_python_rules()
        elif self.language in ['javascript', 'typescript']:
            self._setup_javascript_rules()
        elif self.language in ['html', 'xml']:
            self._setup_html_rules()
        elif self.language in ['css']:
            self._setup_css_rules()
        elif self.language in ['c', 'c++', 'cpp', 'csharp', 'c#', 'java']:
            self._setup_c_family_rules()
        # Add more languages as needed
    
    def _setup_python_rules(self):
        """Set up highlighting rules for Python"""
        # Keywords
        keywords = [
            r'\bimport\b', r'\bfrom\b', r'\bdef\b', r'\bclass\b', r'\bif\b', r'\belif\b',
            r'\belse\b', r'\bwhile\b', r'\bfor\b', r'\bin\b', r'\breturn\b', r'\bNone\b',
            r'\bTrue\b', r'\bFalse\b', r'\band\b', r'\bor\b', r'\bnot\b', r'\bwith\b',
            r'\bas\b', r'\btry\b', r'\bexcept\b', r'\bfinally\b', r'\braise\b', r'\bself\b',
            r'\bglobal\b', r'\bnonlocal\b', r'\bassert\b', r'\blambda\b', r'\byield\b'
        ]
        for pattern in keywords:
            rule = (re.compile(pattern), self.formats['keyword'])
            self.highlighting_rules.append(rule)
        
        # Strings
        self.highlighting_rules.append((re.compile(r'\".*?\"'), self.formats['string']))
        self.highlighting_rules.append((re.compile(r'\'.*?\''), self.formats['string']))
        self.highlighting_rules.append((re.compile(r'\"\"\".*?\"\"\"', re.DOTALL), self.formats['string']))
        self.highlighting_rules.append((re.compile(r'\'\'\'.*?\'\'\'', re.DOTALL), self.formats['string']))
        
        # Comments
        self.highlighting_rules.append((re.compile(r'#.*$'), self.formats['comment']))
        
        # Numbers
        self.highlighting_rules.append((re.compile(r'\b\d+\b'), self.formats['number']))
        
        # Functions
        self.highlighting_rules.append((re.compile(r'\bdef\s+(\w+)'), self.formats['function']))
        
        # Classes
        self.highlighting_rules.append((re.compile(r'\bclass\s+(\w+)'), self.formats['class']))
        
        # Decorators
        self.highlighting_rules.append((re.compile(r'@\w+'), self.formats['decorator']))
    
    def _setup_javascript_rules(self):
        """Set up highlighting rules for JavaScript"""
        # Keywords
        keywords = [
            r'\bvar\b', r'\blet\b', r'\bconst\b', r'\bfunction\b', r'\breturn\b',
            r'\bif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b', r'\bdo\b', r'\bswitch\b',
            r'\bcase\b', r'\bbreak\b', r'\bcontinue\b', r'\btry\b', r'\bcatch\b',
            r'\bfinally\b', r'\bthrow\b', r'\bnew\b', r'\bdelete\b', r'\btypeof\b',
            r'\binstance\w+\b', r'\bclass\b', r'\bconstructor\b', r'\bextends\b',
            r'\bimport\b', r'\bexport\b', r'\basync\b', r'\bawait\b', r'\byield\b'
        ]
        for pattern in keywords:
            rule = (re.compile(pattern), self.formats['keyword'])
            self.highlighting_rules.append(rule)
        
        # Strings
        self.highlighting_rules.append((re.compile(r'\".*?\"'), self.formats['string']))
        self.highlighting_rules.append((re.compile(r'\'.*?\''), self.formats['string']))
        self.highlighting_rules.append((re.compile(r'\`.*?\`', re.DOTALL), self.formats['string']))
        
        # Comments
        self.highlighting_rules.append((re.compile(r'//.*$'), self.formats['comment']))
        self.highlighting_rules.append((re.compile(r'/\*.*?\*/', re.DOTALL), self.formats['comment']))
        
        # Numbers
        self.highlighting_rules.append((re.compile(r'\b\d+\b'), self.formats['number']))
        
        # Functions
        self.highlighting_rules.append((re.compile(r'\bfunction\s+(\w+)'), self.formats['function']))
        self.highlighting_rules.append((re.compile(r'(\w+)\s*\('), self.formats['function']))
        
        # Classes
        self.highlighting_rules.append((re.compile(r'\bclass\s+(\w+)'), self.formats['class']))
    
    def _setup_html_rules(self):
        """Set up highlighting rules for HTML"""
        # Tags
        self.highlighting_rules.append((re.compile(r'<[!?]?[a-zA-Z0-9]+'), self.formats['keyword']))
        self.highlighting_rules.append((re.compile(r'</[a-zA-Z0-9]+>'), self.formats['keyword']))
        self.highlighting_rules.append((re.compile(r'[a-zA-Z0-9]+ ?/>'), self.formats['keyword']))
        self.highlighting_rules.append((re.compile(r'>'), self.formats['keyword']))
        
        # Attributes
        self.highlighting_rules.append((re.compile(r'[a-zA-Z-]+='), self.formats['function']))
        
        # Strings
        self.highlighting_rules.append((re.compile(r'\".*?\"'), self.formats['string']))
        self.highlighting_rules.append((re.compile(r'\'.*?\''), self.formats['string']))
        
        # Comments
        self.highlighting_rules.append((re.compile(r'<!--.*?-->', re.DOTALL), self.formats['comment']))
    
    def _setup_css_rules(self):
        """Set up highlighting rules for CSS"""
        # Selectors
        self.highlighting_rules.append((re.compile(r'[.#]?[a-zA-Z0-9_-]+\s*\{'), self.formats['class']))
        
        # Properties
        self.highlighting_rules.append((re.compile(r'[a-zA-Z-]+\s*:'), self.formats['function']))
        
        # Values
        self.highlighting_rules.append((re.compile(r':\s*[^;]+'), self.formats['string']))
        
        # Units
        self.highlighting_rules.append((re.compile(r'\b\d+(%|px|em|rem|vh|vw|pt|pc)\b'), self.formats['number']))
        
        # Colors
        self.highlighting_rules.append((re.compile(r'#[0-9a-fA-F]{3,6}'), self.formats['number']))
        
        # Comments
        self.highlighting_rules.append((re.compile(r'/\*.*?\*/', re.DOTALL), self.formats['comment']))
    
    def _setup_c_family_rules(self):
        """Set up highlighting rules for C-family languages (C, C++, C#, Java)"""
        # Keywords
        keywords = [
            r'\bif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b', r'\bdo\b', r'\bswitch\b',
            r'\bcase\b', r'\bbreak\b', r'\bcontinue\b', r'\breturn\b', r'\bvoid\b', 
            r'\bint\b', r'\bfloat\b', r'\bdouble\b', r'\bchar\b', r'\blong\b', r'\bshort\b',
            r'\bstruct\b', r'\bunion\b', r'\benum\b', r'\bsizeof\b', r'\btypedef\b',
            r'\bstatic\b', r'\bextern\b', r'\bconst\b', r'\bauto\b', r'\bregister\b',
            r'\bvolatile\b', r'\bunsigned\b', r'\bsigned\b', r'\bclass\b', r'\bprivate\b',
            r'\bprotected\b', r'\bpublic\b', r'\bvirtual\b', r'\binline\b', r'\btemplate\b',
            r'\bnamespace\b', r'\busing\b', r'\bnew\b', r'\bdelete\b', r'\btry\b', r'\bcatch\b',
            r'\bthrow\b', r'\bthrows\b', r'\bfinally\b', r'\bimport\b', r'\bpackage\b'
        ]
        for pattern in keywords:
            rule = (re.compile(pattern), self.formats['keyword'])
            self.highlighting_rules.append(rule)
        
        # Strings
        self.highlighting_rules.append((re.compile(r'\".*?\"'), self.formats['string']))
        self.highlighting_rules.append((re.compile(r'\'.\''), self.formats['string']))
        
        # Comments
        self.highlighting_rules.append((re.compile(r'//.*$'), self.formats['comment']))
        self.highlighting_rules.append((re.compile(r'/\*.*?\*/', re.DOTALL), self.formats['comment']))
        
        # Numbers
        self.highlighting_rules.append((re.compile(r'\b\d+\b'), self.formats['number']))
        
        # Functions
        self.highlighting_rules.append((re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*\('), self.formats['function']))
        
        # Classes
        self.highlighting_rules.append((re.compile(r'\bclass\s+(\w+)'), self.formats['class']))
        self.highlighting_rules.append((re.compile(r'\bstruct\s+(\w+)'), self.formats['class']))
    
    def highlightBlock(self, text):
        """Apply highlighting to the given text block"""
        for pattern, format in self.highlighting_rules:
            for match in pattern.finditer(text):
                start = match.start()
                length = match.end() - start
                self.setFormat(start, length, format)


class MarkdownRenderer:
    """Class for rendering markdown text in QTextEdit widgets"""
    
    def __init__(self):
        self.code_highlighters = {}
    
    def render_markdown(self, text_edit, markdown_text):
        """Render markdown text in a QTextEdit widget"""
        if not markdown_text:
            return
        
        # Clear the text edit
        text_edit.clear()
        
        # Convert markdown to HTML
        html = self._markdown_to_html(markdown_text)
        
        # Set the HTML content
        text_edit.setHtml(html)
        
        # Apply syntax highlighting to code blocks
        self._highlight_code_blocks(text_edit)
    
    def _markdown_to_html(self, markdown_text):
        """Convert markdown text to HTML"""
        html = markdown_text
        
        # Replace code blocks with HTML
        html = self._process_code_blocks(html)
        
        # Replace inline code with HTML
        html = self._process_inline_code(html)
        
        # Replace headers with HTML
        html = self._process_headers(html)
        
        # Replace bold text with HTML
        html = self._process_bold(html)
        
        # Replace italic text with HTML
        html = self._process_italic(html)
        
        # Replace lists with HTML
        html = self._process_lists(html)
        
        # Replace links with HTML
        html = self._process_links(html)
        
        # Replace newlines with <br>
        html = html.replace('\n', '<br>')
        
        return html
    
    def _process_code_blocks(self, text):
        """Process code blocks in markdown text"""
        # Find all code blocks delimited by ```
        pattern = r'```(\w*)\n(.*?)```'
        
        def replace_code_block(match):
            language = match.group(1) or 'text'
            code = match.group(2)
            # Store code and language for later syntax highlighting
            block_id = f"code_{len(self.code_highlighters)}"
            self.code_highlighters[block_id] = {
                'language': language,
                'code': code
            }
            # Create HTML for code block
            return f'<pre id="{block_id}" style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; font-family: monospace;">{code}</pre>'
        
        return re.sub(pattern, replace_code_block, text, flags=re.DOTALL)
    
    def _process_inline_code(self, text):
        """Process inline code in markdown text"""
        pattern = r'`(.*?)`'
        return re.sub(pattern, r'<code style="background-color: #f5f5f5; padding: 2px 5px; border-radius: 3px; font-family: monospace;">\1</code>', text)
    
    def _process_headers(self, text):
        """Process headers in markdown text"""
        # H1
        text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
        # H2
        text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
        # H3
        text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        # H4
        text = re.sub(r'^#### (.*?)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
        # H5
        text = re.sub(r'^##### (.*?)$', r'<h5>\1</h5>', text, flags=re.MULTILINE)
        # H6
        text = re.sub(r'^###### (.*?)$', r'<h6>\1</h6>', text, flags=re.MULTILINE)
        return text
    
    def _process_bold(self, text):
        """Process bold text in markdown text"""
        pattern = r'\*\*(.*?)\*\*'
        return re.sub(pattern, r'<b>\1</b>', text)
    
    def _process_italic(self, text):
        """Process italic text in markdown text"""
        pattern = r'\*(.*?)\*'
        return re.sub(pattern, r'<i>\1</i>', text)
    
    def _process_lists(self, text):
        """Process lists in markdown text"""
        # Convert unordered lists
        text = re.sub(r'^\* (.*?)$', r'<ul><li>\1</li></ul>', text, flags=re.MULTILINE)
        text = re.sub(r'<\/ul>\s*<ul>', '', text)  # Merge adjacent ul tags
        
        # Convert ordered lists
        text = re.sub(r'^(\d+)\. (.*?)$', r'<ol><li>\2</li></ol>', text, flags=re.MULTILINE)
        text = re.sub(r'<\/ol>\s*<ol>', '', text)  # Merge adjacent ol tags
        
        return text
    
    def _process_links(self, text):
        """Process links in markdown text"""
        pattern = r'\[(.*?)\]\((.*?)\)'
        return re.sub(pattern, r'<a href="\2">\1</a>', text)
    
    def _highlight_code_blocks(self, text_edit):
        """Apply syntax highlighting to code blocks in the rendered HTML"""
        document = text_edit.document()
        
        for block_id, block_info in self.code_highlighters.items():
            # Find the text block with the code
            cursor = text_edit.textCursor()
            cursor.setPosition(0)
            
            # Try to find the block by its ID
            search_text = f'id="{block_id}"'
            found = text_edit.find(search_text)
            
            if found:
                # Move to the start of the code block content
                cursor = text_edit.textCursor()
                cursor.movePosition(QTextCursor.NextBlock)
                
                # Create a syntax highlighter for this block
                language = block_info['language']
                highlighter = CodeBlockHighlighter(document, language)
                
                # Store the highlighter to prevent it from being garbage collected
                self.code_highlighters[block_id]['highlighter'] = highlighter


class MarkdownTextEdit(QTextEdit):
    """QTextEdit with markdown rendering support"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setAcceptRichText(True)
        self.markdown_renderer = MarkdownRenderer()
    
    def setMarkdownText(self, markdown_text):
        """Set markdown text and render it"""
        self.markdown_renderer.render_markdown(self, markdown_text)
    
    def appendMarkdownText(self, markdown_text):
        """Append markdown text and render it"""
        # Get current content
        current_html = self.toHtml()
        
        # Convert new markdown to HTML
        new_html = self.markdown_renderer._markdown_to_html(markdown_text)
        
        # Remove <html>, <head>, <body> tags from current HTML to get just the content
        content_pattern = r'<body.*?>(.*?)</body>'
        match = re.search(content_pattern, current_html, re.DOTALL)
        current_content = match.group(1) if match else ""
        
        # Combine content
        combined_content = current_content + new_html
        
        # Set the HTML content
        self.setHtml(f"<html><body>{combined_content}</body></html>")
        
        # Apply syntax highlighting to code blocks
        self.markdown_renderer._highlight_code_blocks(self)
