"""
Code streaming module for Advanced Code Generator.
Handles streaming generation of code responses.
"""

import os
import time
import logging
import threading
import torch
from typing import Dict, Any, Optional
from PyQt5.QtCore import QObject, pyqtSignal, QMutex, QMutexLocker

# Logger for this module
logger = logging.getLogger(__name__)


class StreamingGenerationThread(threading.Thread):
    """Thread that handles the actual generation process, feeding tokens back to main thread."""
    
    def __init__(self, model, tokenizer, prompt, params, callback=None):
        """Initialize the generation thread."""
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.params = params
        self.callback = callback
        self.stop_event = threading.Event()
        self.daemon = True  # Thread will exit when main thread exits
    
    def run(self):
        """Run the generation process."""
        try:
            # Encode the prompt
            input_ids = self.tokenizer.encode(self.prompt, return_tensors="pt")
            attention_mask = torch.ones_like(input_ids)
            
            # Move to same device as model
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Get the prompt length to avoid re-generating it
            prompt_length = input_ids.shape[1]
            
            # Extract parameters
            temperature = self.params.get("temperature", 0.7)
            top_p = self.params.get("top_p", 0.9)
            repetition_penalty = self.params.get("repetition_penalty", 1.1)
            max_new_tokens = self.params.get("max_length", 500)
            stream_interval = self.params.get("stream_interval", 0.05)
            
            # Set up generation parameters
            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            # Check if model is compatible with streaming
            supports_streaming = hasattr(self.model, "generate_with_streaming")
            
            if supports_streaming:
                # Model has built-in streaming support
                for output in self.model.generate_with_streaming(**gen_kwargs):
                    if self.stop_event.is_set():
                        logger.info("Generation stopped by user")
                        break
                    
                    # Decode only the new token
                    new_tokens = output[:, prompt_length:].tolist()[0]
                    if new_tokens:
                        token_text = self.tokenizer.decode(new_tokens[-1:], skip_special_tokens=True)
                        if self.callback:
                            self.callback.token_generated.emit(token_text)
                        
                        # Progress update
                        progress = min(100, int((len(new_tokens) / max_new_tokens) * 100))
                        if hasattr(self.callback, "progress_updated"):
                            self.callback.progress_updated.emit(progress)
                        
                        # Small pause to simulate realistic typing
                        time.sleep(stream_interval)
            else:
                # Implement manual streaming for models without built-in support
                generated = input_ids.clone()
                past = None
                
                for i in range(max_new_tokens):
                    if self.stop_event.is_set():
                        logger.info("Generation stopped by user")
                        break
                    
                    with torch.no_grad():
                        if past is None:
                            outputs = self.model(generated)
                        else:
                            outputs = self.model(generated[:, -1:], past_key_values=past)
                        
                        logits = outputs.logits[:, -1, :]
                        past = outputs.past_key_values
                        
                        # Apply temperature
                        logits = logits / max(temperature, 1e-7)
                        
                        # Apply top_p sampling
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                            
                            # Remove tokens with cumulative probability above the threshold
                            sorted_indices_to_remove = cumulative_probs > top_p
                            # Shift the indices to the right to keep also the first token above the threshold
                            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                            sorted_indices_to_remove[:, 0] = 0
                            
                            indices_to_remove = sorted_indices[sorted_indices_to_remove]
                            logits[:, indices_to_remove] = -float('Inf')
                        
                        # Sample from the distribution
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        
                        # Add the new token
                        generated = torch.cat([generated, next_token], dim=-1)
                        
                        # Decode the new token
                        new_token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                        if self.callback:
                            self.callback.token_generated.emit(new_token_text)
                        
                        # Progress update
                        progress = min(100, int((i / max_new_tokens) * 100))
                        if hasattr(self.callback, "progress_updated"):
                            self.callback.progress_updated.emit(progress)
                        
                        # Check if we've hit the end of text token
                        if next_token.item() == self.tokenizer.eos_token_id:
                            break
                        
                        # Small pause to simulate realistic typing
                        time.sleep(stream_interval)
            
            # Signal completion
            if hasattr(self.callback, "generation_finished"):
                self.callback.generation_finished.emit()
                
        except Exception as e:
            logger.error(f"Error in generation thread: {str(e)}")
            if hasattr(self.callback, "generation_error"):
                self.callback.generation_error.emit(str(e))
    
    def stop(self):
        """Stop the generation process."""
        self.stop_event.set()


class StreamingCodeGenerator(QObject):
    """Class to manage streaming generation of code."""
    
    # Signals
    token_generated = pyqtSignal(str)  # Emitted when a new token is generated
    generation_started = pyqtSignal()  # Emitted when generation starts
    generation_finished = pyqtSignal()  # Emitted when generation is complete
    generation_error = pyqtSignal(str)  # Emitted when an error occurs
    progress_updated = pyqtSignal(int)  # Emitted to update progress (0-100)
    
    def __init__(self):
        """Initialize the streaming generator."""
        super().__init__()
        self.thread = None
        self.model = None
        self.tokenizer = None
        self.prompt = None
        self.params = {}
        self.mutex = QMutex()  # Mutex to protect thread access
    
    def setup(self, model, tokenizer, prompt, params=None):
        """Set up the generator with model, tokenizer, and prompt."""
        with QMutexLocker(self.mutex):
            self.model = model
            self.tokenizer = tokenizer
            self.prompt = prompt
            self.params = params or {}
    
    def start_generation(self):
        """Start the generation process."""
        with QMutexLocker(self.mutex):
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
                self.thread = StreamingGenerationThread(
                    self.model, self.tokenizer, self.prompt, self.params, callback
                )
                self.thread.start()
                return True
            except Exception as e:
                logger.error(f"Error starting generation: {str(e)}")
                self.generation_error.emit(f"Error starting generation: {str(e)}")
                return False
    
    def stop(self):
        """Stop the current generation process."""
        with QMutexLocker(self.mutex):
            if self.thread and self.thread.is_alive():
                self.thread.stop()
                # Don't join here to avoid blocking the GUI thread
    
    class StreamingCallback:
        """Callback class to handle signals from the generation thread."""
        
        def __init__(self, parent):
            """Initialize with parent StreamingCodeGenerator."""
            self.parent = parent
        
        def token_generated(self, token):
            """Handle a new token."""
            self.parent.token_generated.emit(token)
        
        def progress_updated(self, progress):
            """Update the progress bar."""
            self.parent.progress_updated.emit(progress)
        
        def generation_finished(self):
            """Handle generation completion."""
            self.parent.generation_finished.emit()
        
        def generation_error(self, error_message):
            """Handle generation error."""
            self.parent.generation_error.emit(error_message)
