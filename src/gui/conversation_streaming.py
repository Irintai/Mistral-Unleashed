"""
Streaming conversation response generation for Advanced Code Generator.
This module provides streaming response capabilities for the conversation tab.
"""

import time
import logging
import torch
from PyQt5.QtCore import QThread, pyqtSignal, QObject, QMutex, QMutexLocker

# Initialize logger
logger = logging.getLogger(__name__)


class StreamingConversationThread(QThread):
    """Thread for handling streaming conversation generation"""
    
    # Signals
    token_received = pyqtSignal(str)
    generation_started = pyqtSignal()
    generation_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, model, tokenizer, prompt, params, callback, parent=None):
        super().__init__(parent)
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.params = params
        self.callback = callback
        self.is_stopped = False
        self.generated_text = ""
        
        # Configure default parameters for generation
        self.default_params = {
            "max_length": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "stream_interval": 0.05,
            "do_sample": True
        }
        
        # Determine device capabilities for optimizations
        self.device = next(self.model.parameters()).device
        self.is_gpu = self.device.type == 'cuda'
        self.use_optimized_generation = self.is_gpu
    
    def run(self):
        """Run the generation process"""
        try:
            self.generation_started.emit()
            
            # Validate inputs
            if self.model is None or self.tokenizer is None:
                raise ValueError("No model is loaded. Please load a model first.")
            
            if not self.prompt:
                raise ValueError("Prompt cannot be empty.")
            
            # Extract parameters with defaults
            max_length = self.params.get("max_length", self.default_params["max_length"])
            temperature = self.params.get("temperature", self.default_params["temperature"])
            top_p = self.params.get("top_p", self.default_params["top_p"])
            repetition_penalty = self.params.get("repetition_penalty", self.default_params["repetition_penalty"])
            stream_interval = self.params.get("stream_interval", self.default_params["stream_interval"])
            do_sample = self.params.get("do_sample", self.default_params["do_sample"])
            
            logger.info(f"Starting generation on device: {self.device}")
            logger.info(f"Model type: {type(self.model).__name__}")
            
            input_ids = self.tokenizer.encode(self.prompt, return_tensors="pt").to(self.device)
            
            # Create attention mask (important for some models)
            attention_mask = torch.ones_like(input_ids).to(self.device)
            
            # For GPU performance, make sure we're using optimal precision
            if self.is_gpu and hasattr(torch, 'float16'):
                # Use half-precision if available
                input_ids = input_ids.to(dtype=torch.long)  # IDs always stay as long
                attention_mask = attention_mask.to(dtype=torch.float16)
            
            # Use the model's generation method directly
            with torch.no_grad():
                try:
                    # Set performance options
                    with torch.amp.autocast(device_type='cuda' if self.is_gpu else 'cpu'):
                        logger.info(f"Generating with params: max_length={max_length}, temp={temperature}, top_p={top_p}")
                        
                        # Different models might need different parameters
                        try:
                            # Generate in a way that makes use of GPU parallelism
                            outputs = self.model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_length=max_length,
                                temperature=temperature,
                                top_p=top_p,
                                repetition_penalty=repetition_penalty,
                                do_sample=do_sample,
                                num_return_sequences=1,
                                pad_token_id=self.tokenizer.eos_token_id
                            )
                        except TypeError as e:
                            # Some models don't accept all parameters - try a more compatible set
                            logger.warning(f"Parameter error in generation, trying with simplified params: {e}")
                            outputs = self.model.generate(
                                input_ids=input_ids,
                                max_length=max_length,
                                do_sample=do_sample,
                                num_return_sequences=1
                            )
                    
                    # Decode the output
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract just the response part (after the prompt)
                    prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    response_text = generated_text[len(prompt_text):].strip()
                    
                    logger.info(f"Generated response of length {len(response_text)}")
                    logger.info(f"Response starts with: {response_text[:30]}...")
                    
                    # Check if response is empty
                    if not response_text:
                        logger.warning("Generated an empty response - trying to extract differently")
                        # Try alternative extraction
                        response_text = generated_text[len(self.prompt):].strip()
                        if not response_text:
                            logger.error("Still got empty response, using full generation")
                            response_text = generated_text
                    
                    # Stream the response one character at a time for UI effect
                    for i, char in enumerate(response_text):
                        if self.is_stopped:
                            logger.info("Generation stopped by user")
                            break
                        
                        # Check if we've been deleted
                        if not self.callback:
                            logger.warning("Callback no longer valid, stopping generation")
                            break
                            
                        try:
                            self.callback(char, False, None, response_text[:i+1])
                        except RuntimeError as e:
                            logger.error(f"Error sending token: {e}")
                            break
                        
                        # Update progress
                        try:
                            progress = min(100, int((i / len(response_text)) * 100))
                            self.callback(None, False, None, response_text[:i+1], progress)
                        except RuntimeError as e:
                            logger.error(f"Error updating progress: {e}")
                            break
                        
                        # Adaptive streaming speed based on GPU capabilities and response length
                        if self.is_gpu and len(response_text) > 100:
                            # GPU can go faster for long responses
                            adjusted_interval = min(stream_interval, 0.005)
                        else: 
                            # Default or slower for shorter responses
                            adjusted_interval = stream_interval
                            
                        time.sleep(adjusted_interval)
                        
                except Exception as e:
                    logger.error(f"Error during model generation: {str(e)}")
                    if self.callback:
                        try:
                            self.callback(None, False, str(e), None)
                        except RuntimeError:
                            pass
            
            # Generation complete - clean up CUDA memory if needed
            if self.is_gpu:
                try:
                    del outputs
                    del input_ids
                    del attention_mask
                    torch.cuda.empty_cache()
                except:
                    pass
            
            # Generation complete
            if self.callback:
                try:
                    self.callback(None, True, None, response_text)
                except RuntimeError as e:
                    logger.error(f"Error during completion: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            if self.callback:
                try:
                    self.callback(None, False, str(e), None)
                except RuntimeError:
                    pass


class StreamingConversationGenerator(QObject):
    """Handles conversation generation with streaming output"""
    
    # Signals
    token_generated = pyqtSignal(str)  # Emitted when a new token is generated
    generation_started = pyqtSignal()  # Emitted when generation starts
    generation_finished = pyqtSignal()  # Emitted when generation is complete
    generation_error = pyqtSignal(str)  # Emitted when an error occurs
    progress_updated = pyqtSignal(int)  # Emitted to update progress (0-100)
    
    def __init__(self, parent=None):
        """Initialize the streaming generator."""
        super().__init__(parent)
        self.model = None
        self.tokenizer = None
        self.prompt = ""
        self.params = {}
        self.mutex = QMutex()  # Mutex to protect thread access
        self.thread = None
    
    def __del__(self):
        """Clean up resources when deleted"""
        self.cleanup()
    
    def setup(self, model=None, tokenizer=None, prompt=None, params=None):
        """Set up the generator with model, tokenizer, and prompt."""
        with QMutexLocker(self.mutex):
            if model is not None:
                self.model = model
            if tokenizer is not None:
                self.tokenizer = tokenizer
            if prompt is not None:
                self.prompt = prompt
            self.params = params or {}
    
    def start_generation(self):
        """Start the generation process."""
        with QMutexLocker(self.mutex):
            # If there's an existing thread running, stop it first
            if self.thread and self.thread.isRunning():
                self.stop()
                self.thread.wait(1000)  # Wait for it to finish
                
            if not self.model or not self.tokenizer or not self.prompt:
                logger.error("Cannot start generation: model, tokenizer, or prompt not set")
                self.generation_error.emit("Model, tokenizer, or prompt not set")
                return False
            
            # Create callback class that can emit signals
            callback = self.StreamingCallback(self)
            
            try:
                # Emit generation started signal
                self.generation_started.emit()
                
                # Create and start generation thread
                self.thread = StreamingConversationThread(
                    self.model, self.tokenizer, self.prompt, self.params, callback
                )
                self.thread.finished.connect(self._on_thread_finished)
                self.thread.start()
                return True
            except Exception as e:
                logger.error(f"Error starting generation: {str(e)}")
                self.generation_error.emit(f"Error starting generation: {str(e)}")
                return False
    
    def _on_thread_finished(self):
        """Handle thread completion"""
        logger.debug("Generation thread finished")
    
    def stop(self):
        """Stop the current generation process."""
        with QMutexLocker(self.mutex):
            if self.thread and self.thread.isRunning():
                logger.info("Stopping generation thread")
                self.thread.is_stopped = True
                # Don't join here to avoid blocking the GUI thread
    
    def is_generating(self):
        """Check if generation is in progress"""
        return self.thread is not None and self.thread.isRunning()
    
    def cleanup(self):
        """Clean up resources when no longer needed."""
        self.stop()
        
        # Wait for thread to finish
        if self.thread and self.thread.isRunning():
            logger.info("Cleaning up generator thread...")
            self.thread.wait(1000)  # Wait up to 1 second
            
            # If still running, terminate it
            if self.thread.isRunning():
                logger.warning("Force terminating generator thread during cleanup")
                self.thread.terminate()
                self.thread.wait()
        
        # Remove references
        self.thread = None
        self.model = None
        self.tokenizer = None
    
    class StreamingCallback:
        """Callback class to handle signals from the generation thread."""
        
        def __init__(self, parent):
            """Initialize with parent StreamingConversationGenerator."""
            self.parent = parent
        
        def __call__(self, token, is_finished, error, response, progress=None):
            if error:
                self.parent.generation_error.emit(str(error))
                return
                
            if is_finished:
                self.parent.generation_finished.emit()
                return
                
            # Emit the token as a signal
            if token:
                self.parent.token_generated.emit(token)
            
            # Emit progress update (approximation)
            if progress is not None:
                self.parent.progress_updated.emit(progress)
