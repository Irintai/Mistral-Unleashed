"""
Enhanced code generation tab for the Advanced Code Generator.
Provides functionality for generating code with various options.
"""

import os
import sys
import logging
import time
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                           QComboBox, QPushButton, QGroupBox, QFormLayout, 
                           QSplitter, QTextEdit, QSpinBox, QDoubleSpinBox,
                           QCheckBox, QToolBar, QAction, QFileDialog, QMessageBox,
                           QTabWidget, QApplication, QSpacerItem, QSizePolicy, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QIcon, QFont, QTextCursor

from PyQt5.Qsci import QsciScintilla, QsciLexerPython, QsciLexerJavaScript, QsciLexerCPP

# Add the src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import constants
from constants import (SUPPORTED_LANGUAGES, COMMENT_SYMBOLS, 
                       DEFAULT_MAX_LENGTH, DEFAULT_TEMPERATURE, 
                       DEFAULT_TOP_P, DEFAULT_REPETITION_PENALTY,
                       FILE_EXTENSIONS)

# StreamingCodeGenerator will be imported from the implementation we'll create

# Logger for this module
logger = logging.getLogger(__name__)


class EnhancedCodeEditor(QsciScintilla):
    """Enhanced code editor with syntax highlighting and advanced features"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set default parameters
        self.current_language = "python"
        
        # Configure editor appearance
        self.setUtf8(True)
        self.setMarginType(0, QsciScintilla.NumberMargin)
        self.setMarginWidth(0, "0000")  # Width for line numbers
        self.setMarginLineNumbers(0, True)
        
        # Configure tabs and indentation
        self.setIndentationsUseTabs(False)
        self.setTabWidth(4)
        self.setAutoIndent(True)
        self.setIndentationGuides(True)
        
        # Enable code folding
        self.setFolding(QsciScintilla.BoxedTreeFoldStyle)
        
        # Set font
        self.setFont(QFont("Consolas" if os.name == "nt" else "Menlo", 10))
        
        # Enable line wrapping
        self.setWrapMode(QsciScintilla.WrapWord)
        
        # Enable brace matching
        self.setBraceMatching(QsciScintilla.SloppyBraceMatch)
        
        # Set syntax highlighting for default language
        self.set_language("python")
    
    def set_language(self, language):
        """Set syntax highlighting based on language"""
        self.current_language = language.lower()
        
        # Create appropriate lexer based on language
        if language == "python":
            lexer = QsciLexerPython()
        elif language in ["javascript", "typescript"]:
            lexer = QsciLexerJavaScript()
        elif language in ["c++", "c#", "java"]:
            lexer = QsciLexerCPP()
        else:
            # If no specific lexer is available, use none
            self.setLexer(None)
            return
        
        # Set lexer font
        lexer.setFont(QFont("Consolas" if os.name == "nt" else "Menlo", 10))
        
        # Apply lexer to the editor
        self.setLexer(lexer)
    
    def set_text(self, text):
        """Set editor text"""
        self.setText(text)
    
    def get_text(self):
        """Get editor text"""
        return self.text()
    
    def set_font_size(self, size):
        """Set font size for editor"""
        current_font = self.font()
        current_font.setPointSize(size)
        self.setFont(current_font)
        
        # Update lexer font if present
        if self.lexer():
            lexer_font = self.lexer().font(0)
            lexer_font.setPointSize(size)
            self.lexer().setFont(lexer_font)
    
    def format_code(self):
        """Format code based on current language"""
        code = self.text()
        
        try:
            if self.current_language == "python":
                # Use black for Python formatting if available
                try:
                    import black
                    mode = black.Mode()
                    formatted_code = black.format_str(code, mode=mode)
                    self.setText(formatted_code)
                    return True
                except ImportError:
                    logger.warning("black package not found, using basic formatting")
                except Exception as e:
                    logger.error(f"Error formatting Python code: {str(e)}")
                    return False
            
            elif self.current_language in ["javascript", "typescript"]:
                # Use js-beautify for JS/TS formatting if available
                try:
                    import jsbeautifier
                    opts = jsbeautifier.default_options()
                    formatted_code = jsbeautifier.beautify(code, opts)
                    self.setText(formatted_code)
                    return True
                except ImportError:
                    logger.warning("jsbeautifier package not found, using basic formatting")
                except Exception as e:
                    logger.error(f"Error formatting JavaScript code: {str(e)}")
                    return False
            
            # Basic indentation formatting as fallback
            self.selectAll()
            self.recolor()
            return True
            
        except Exception as e:
            logger.error(f"Error in format_code: {str(e)}")
            return False
    
    def toggle_comment(self):
        """Toggle comment for selected lines"""
        # Get current selection or current line
        if self.hasSelectedText():
            # Get selection
            line_from, index_from, line_to, index_to = self.getSelection()
        else:
            # Get current line
            line_from = line_to = self.getCursorPosition()[0]
            index_from = index_to = 0
        
        # Get comment symbol for current language
        comment_symbol = COMMENT_SYMBOLS.get(self.current_language, "#")
        
        # Determine if we're commenting or uncommenting
        first_line_text = self.text(line_from)
        is_comment = first_line_text.lstrip().startswith(comment_symbol)
        
        # Start an undo action
        self.beginUndoAction()
        
        for line in range(line_from, line_to + 1):
            line_text = self.text(line)
            
            if is_comment and line_text.lstrip().startswith(comment_symbol):
                # Uncomment: remove first occurrence of comment symbol
                indent = len(line_text) - len(line_text.lstrip())
                if len(line_text) > indent and line_text[indent:indent+len(comment_symbol)] == comment_symbol:
                    new_text = line_text[:indent] + line_text[indent+len(comment_symbol):].lstrip()
                    self.setSelection(line, 0, line, len(line_text))
                    self.replaceSelectedText(new_text)
            elif not is_comment:
                # Comment: add comment symbol to beginning of content
                indent = len(line_text) - len(line_text.lstrip())
                if len(line_text) > indent:  # Skip empty lines
                    new_text = line_text[:indent] + comment_symbol + " " + line_text[indent:]
                    self.setSelection(line, 0, line, len(line_text))
                    self.replaceSelectedText(new_text)
        
        # End the undo action
        self.endUndoAction()


class StreamingGenerationThread(QThread):
    """Thread for handling streaming code generation"""
    
    # Signals
    token_received = pyqtSignal(str)
    generation_started = pyqtSignal()
    generation_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, model, tokenizer, prompt, params, parent=None):
        super().__init__(parent)
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.params = params
        self.is_stopped = False
    
    def run(self):
        """Run the generation process"""
        try:
            self.generation_started.emit()
            
            # Validate inputs
            if self.model is None or self.tokenizer is None:
                raise ValueError("No model is loaded. Please load a model first.")
            
            if not self.prompt:
                raise ValueError("Prompt cannot be empty.")
            
            # Import torch
            import torch
            
            # Encode the prompt
            input_ids = self.tokenizer.encode(self.prompt, return_tensors="pt")
            
            # Move to appropriate device
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            
            # Extract parameters
            max_length = self.params.get("max_length", 500)
            temperature = self.params.get("temperature", 0.7)
            top_p = self.params.get("top_p", 0.9)
            repetition_penalty = self.params.get("repetition_penalty", 1.1)
            stream_interval = self.params.get("stream_interval", 0.05)
            
            # Set up generation parameters
            gen_kwargs = {
                "input_ids": input_ids,
                "max_length": max_length + len(input_ids[0]),
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            # Generate with streaming
            with torch.no_grad():
                # Initial generation
                generated = input_ids.clone()
                past = None
                
                for i in range(max_length):
                    if self.is_stopped:
                        logger.info("Generation stopped by user")
                        break
                    
                    # Forward pass
                    outputs = self.model(generated[:, -1:], past_key_values=past)
                    past = outputs.past_key_values
                    
                    # Get logits for next token
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # Apply temperature and top-p
                    if temperature > 0:
                        next_token_logits = next_token_logits / temperature
                    
                    # Apply repetition penalty
                    for token_idx in set(generated[0].tolist()):
                        next_token_logits[0][token_idx] /= repetition_penalty
                    
                    # Apply top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[0][indices_to_remove] = -float("Inf")
                    
                    # Sample from the distribution
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Break if EOS token
                    if next_token[0, 0].item() == self.tokenizer.eos_token_id:
                        break
                    
                    # Append to generated
                    generated = torch.cat((generated, next_token), dim=1)
                    
                    # Decode and emit token
                    new_tokens = generated[0, -1:].tolist()
                    new_token_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    # Emit token and sleep
                    self.token_received.emit(new_token_text)
                    
                    # Update progress
                    progress = min(100, int((i + 1) / max_length * 100))
                    self.progress_updated.emit(progress)
                    
                    # Small sleep to allow UI updates
                    time.sleep(stream_interval)
            
            # Generation complete
            self.progress_updated.emit(100)
            self.generation_complete.emit()
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            self.error_occurred.emit(f"Generation error: {str(e)}")
            self.generation_complete.emit()
    
    def stop(self):
        """Stop the generation process"""
        self.is_stopped = True


class StreamingCodeGenerator:
    """Handles code generation with streaming output"""
    
    def __init__(self):
        """Initialize the generator"""
        self.thread = None
    
    def generate_code(self, model, tokenizer, prompt, params, callbacks):
        """
        Generate code with streaming output
        
        Args:
            model: The loaded model
            tokenizer: The model tokenizer
            prompt: Prompt text for code generation
            params: Generation parameters
            callbacks: Dict of callback functions for events
        
        Returns:
            bool: True if generation started successfully
        """
        try:
            # Create and start generation thread
            self.thread = StreamingGenerationThread(model, tokenizer, prompt, params)
            
            # Connect signals to callbacks
            if "on_token" in callbacks:
                self.thread.token_received.connect(callbacks["on_token"])
            if "on_start" in callbacks:
                self.thread.generation_started.connect(callbacks["on_start"])
            if "on_complete" in callbacks:
                self.thread.generation_complete.connect(callbacks["on_complete"])
            if "on_error" in callbacks:
                self.thread.error_occurred.connect(callbacks["on_error"])
            if "on_progress" in callbacks:
                self.thread.progress_updated.connect(callbacks["on_progress"])
            
            # Start the thread
            self.thread.start()
            return True
            
        except Exception as e:
            logger.error(f"Error starting code generation: {str(e)}")
            return False
    
    def stop_generation(self):
        """Stop ongoing generation"""
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            return True
        return False
    
    def is_generating(self):
        """Check if generation is in progress"""
        return self.thread is not None and self.thread.isRunning()


class EnhancedCodeGenerationTab(QWidget):
    """Tab for generating code with advanced options"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize components
        self.model_id = None
        self.tokenizer = None
        self.model = None
        self.code_generator = StreamingCodeGenerator()
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components"""
        # Main layout
        main_layout = QVBoxLayout()
        
        # Top toolbar with actions
        self.toolbar = QToolBar()
        
        # New action
        new_action = QAction("New", self)
        new_action.setToolTip("Clear current prompt and generated code")
        new_action.triggered.connect(self.new_generation)
        self.toolbar.addAction(new_action)
        
        # Save action
        save_action = QAction("Save", self)
        save_action.setToolTip("Save generated code to file")
        save_action.triggered.connect(self.save_code)
        self.toolbar.addAction(save_action)
        
        # Copy action
        copy_action = QAction("Copy", self)
        copy_action.setToolTip("Copy generated code to clipboard")
        copy_action.triggered.connect(self.copy_code)
        self.toolbar.addAction(copy_action)
        
        # Format action
        format_action = QAction("Format", self)
        format_action.setToolTip("Format generated code")
        format_action.triggered.connect(self.format_code)
        self.toolbar.addAction(format_action)
        
        # Add toolbar to layout
        main_layout.addWidget(self.toolbar)
        
        # Main splitter (vertical)
        self.main_splitter = QSplitter(Qt.Vertical)
        
        # Prompt section
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout()
        
        # Prompt editor
        self.prompt_editor = QTextEdit()
        self.prompt_editor.setPlaceholderText("Enter your prompt for code generation here...")
        prompt_layout.addWidget(self.prompt_editor)
        
        # Language and parameters form
        form_layout = QHBoxLayout()
        
        # Language selection
        language_layout = QFormLayout()
        self.language_combo = QComboBox()
        self.language_combo.addItems(SUPPORTED_LANGUAGES)
        language_layout.addRow("Language:", self.language_combo)
        
        form_layout.addLayout(language_layout)
        
        # Generation parameters
        params_layout = QFormLayout()
        
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(10, 2000)
        self.max_length_spin.setValue(DEFAULT_MAX_LENGTH)
        params_layout.addRow("Max Length:", self.max_length_spin)
        
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.1, 2.0)
        self.temperature_spin.setValue(DEFAULT_TEMPERATURE)
        self.temperature_spin.setSingleStep(0.1)
        params_layout.addRow("Temperature:", self.temperature_spin)
        
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.1, 1.0)
        self.top_p_spin.setValue(DEFAULT_TOP_P)
        self.top_p_spin.setSingleStep(0.05)
        params_layout.addRow("Top-p:", self.top_p_spin)
        
        form_layout.addLayout(params_layout)
        
        # Generation controls
        buttons_layout = QVBoxLayout()
        
        self.generate_button = QPushButton("Generate Code")
        self.generate_button.clicked.connect(self.generate_code)
        self.generate_button.setEnabled(False)  # Disabled until model is loaded
        buttons_layout.addWidget(self.generate_button)
        
        self.stop_button = QPushButton("Stop Generation")
        self.stop_button.clicked.connect(self.stop_generation)
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)
        
        # Add to history checkbox
        self.add_to_history_check = QCheckBox("Add to History")
        self.add_to_history_check.setChecked(True)
        buttons_layout.addWidget(self.add_to_history_check)
        
        form_layout.addLayout(buttons_layout)
        
        prompt_layout.addLayout(form_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        prompt_layout.addWidget(self.progress_bar)
        
        prompt_group.setLayout(prompt_layout)
        self.main_splitter.addWidget(prompt_group)
        
        # Generated code section
        code_group = QGroupBox("Generated Code")
        code_layout = QVBoxLayout()
        
        # Enhanced code editor
        self.code_editor = EnhancedCodeEditor()
        code_layout.addWidget(self.code_editor)
        
        code_group.setLayout(code_layout)
        self.main_splitter.addWidget(code_group)
        
        # Set initial splitter sizes (40% prompt, 60% code)
        self.main_splitter.setSizes([400, 600])
        
        # Add splitter to main layout
        main_layout.addWidget(self.main_splitter)
        
        # Add status label
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
        # Set the main layout
        self.setLayout(main_layout)
        
        # Connect language change to editor
        self.language_combo.currentTextChanged.connect(self.on_language_changed)
        self.on_language_changed(self.language_combo.currentText())
    
    def set_model(self, model_id, tokenizer, model):
        """Set the model to use for generation"""
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.model = model
        
        # Enable/disable generation button based on model availability
        self.generate_button.setEnabled(model is not None)
        
        if model is None:
            self.status_label.setText("No model loaded")
        else:
            self.status_label.setText(f"Model {model_id} loaded and ready")
    
    def on_language_changed(self, language):
        """Handle language selection change"""
        # Update code editor language
        self.code_editor.set_language(language)
    
    def generate_code(self):
        """Start code generation"""
        # Check if model is loaded
        if not self.model or not self.tokenizer:
            QMessageBox.warning(
                self,
                "No Model",
                "Please load a model first."
            )
            return
        
        # Get prompt
        prompt = self.prompt_editor.toPlainText().strip()
        
        if not prompt:
            QMessageBox.warning(
                self,
                "Empty Prompt",
                "Please enter a prompt for code generation."
            )
            return
        
        # Get parameters
        params = {
            "max_length": self.max_length_spin.value(),
            "temperature": self.temperature_spin.value(),
            "top_p": self.top_p_spin.value(),
            "repetition_penalty": DEFAULT_REPETITION_PENALTY,
            "stream_interval": 0.05,
        }
        
        # Prepare callbacks
        callbacks = {
            "on_token": self.on_token_received,
            "on_start": self.on_generation_started,
            "on_complete": self.on_generation_complete,
            "on_error": self.on_generation_error,
            "on_progress": self.on_progress_updated,
        }
        
        # Clear the code editor
        self.code_editor.clear()
        
        # Start generation
        success = self.code_generator.generate_code(
            self.model, self.tokenizer, prompt, params, callbacks
        )
        
        if not success:
            QMessageBox.critical(
                self,
                "Generation Failed",
                "Failed to start code generation. See log for details."
            )
    
    def stop_generation(self):
        """Stop ongoing code generation"""
        if self.code_generator.is_generating():
            self.code_generator.stop_generation()
            self.status_label.setText("Generation stopped by user")
    
    @pyqtSlot(str)
    def on_token_received(self, token):
        """Handle received token"""
        # Append token to the code editor
        self.code_editor.append(token)
        
        # Move cursor to end
        self.code_editor.setCursorPosition(
            self.code_editor.lines(), 0
        )
        
        # Process application events to update UI
        QApplication.processEvents()
    
    @pyqtSlot()
    def on_generation_started(self):
        """Handle generation start"""
        # Update UI
        self.generate_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Generating code...")
        self.progress_bar.setValue(0)
    
    @pyqtSlot()
    def on_generation_complete(self):
        """Handle generation completion"""
        # Update UI
        self.generate_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Generation complete")
        self.progress_bar.setValue(100)
        
        # Format code if needed
        self.format_code()
        
        # Add to history if checkbox is checked
        if self.add_to_history_check.isChecked():
            self.add_to_history()
    
    @pyqtSlot(str)
    def on_generation_error(self, error_message):
        """Handle generation error"""
        # Update UI
        self.generate_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText(f"Error: {error_message}")
        
        # Show error message
        QMessageBox.critical(
            self,
            "Generation Error",
            error_message
        )
    
    @pyqtSlot(int)
    def on_progress_updated(self, progress):
        """Handle progress update"""
        self.progress_bar.setValue(progress)
    
    def new_generation(self):
        """Clear current prompt and generated code"""
        # Check if generation is in progress
        if self.code_generator.is_generating():
            confirm = QMessageBox.question(
                self,
                "Generation in Progress",
                "A generation is currently in progress. Stop it and start new?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if confirm == QMessageBox.No:
                return
            
            # Stop generation
            self.stop_generation()
        
        # Clear editors
        self.prompt_editor.clear()
        self.code_editor.clear()
        
        # Reset progress
        self.progress_bar.setValue(0)
        
        # Update status
        self.status_label.setText("New generation ready")
    
    def save_code(self):
        """Save generated code to file"""
        # Get current code
        code = self.code_editor.text()
        
        if not code:
            QMessageBox.information(
                self,
                "No Code",
                "There is no code to save."
            )
            return
        
        # Get current language
        language = self.language_combo.currentText().lower()
        
        # Get file extension for current language
        extension = FILE_EXTENSIONS.get(language, ".txt")
        
        # Ask for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Generated Code",
            os.path.expanduser("~/generated_code" + extension),
            f"Code Files (*{extension});;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Save the code
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            self.status_label.setText(f"Code saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving code: {str(e)}")
            QMessageBox.critical(
                self,
                "Save Error",
                f"Error saving code: {str(e)}"
            )
    
    def copy_code(self):
        """Copy generated code to clipboard"""
        # Get current code
        code = self.code_editor.text()
        
        if not code:
            QMessageBox.information(
                self,
                "No Code",
                "There is no code to copy."
            )
            return
        
        # Copy to clipboard
        clipboard = QApplication.clipboard()
        clipboard.setText(code)
        
        self.status_label.setText("Code copied to clipboard")
    
    def format_code(self):
        """Format the generated code"""
        success = self.code_editor.format_code()
        
        if success:
            self.status_label.setText("Code formatted")
        else:
            self.status_label.setText("Code formatting failed")
    
    def add_to_history(self):
        """Add current generation to history"""
        try:
            # Get current state
            prompt = self.prompt_editor.toPlainText().strip()
            code = self.code_editor.text()
            language = self.language_combo.currentText()
            
            if not prompt or not code:
                logger.debug("Not adding to history: empty prompt or code")
                return
            
            # Get parent window to access history manager
            main_window = self.window()
            
            # If history manager exists, add to history
            if hasattr(main_window, 'history_manager'):
                # Create history entry
                entry = {
                    "prompt": prompt,
                    "code": code,
                    "language": language,
                    "model": self.model_id,
                    "timestamp": time.time(),
                    "title": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                    "parameters": {
                        "max_length": self.max_length_spin.value(),
                        "temperature": self.temperature_spin.value(),
                        "top_p": self.top_p_spin.value(),
                    }
                }
                
                # Add to history
                main_window.history_manager.add_entry(entry)
                self.status_label.setText("Added to history")
            else:
                logger.warning("No history manager found")
                
        except Exception as e:
            logger.error(f"Error adding to history: {str(e)}")