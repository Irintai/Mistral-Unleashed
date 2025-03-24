"""
Model manager for handling LLM loading, caching, and information.
"""

import os
import logging
import torch
import gc
from typing import Dict, Tuple, Any, Optional, List

# Initialize logger
logger = logging.getLogger(__name__)

class ModelManager:
    """Manager for language models, handling loading, caching and memory management"""
    
    def __init__(self):
        """Initialize the model manager"""
        # Dictionary to store loaded models: model_id -> (tokenizer, model)
        self.model_cache: Dict[str, Tuple[Any, Any]] = {}
        
        # Currently active model
        self.current_model_id: Optional[str] = None
        self.current_tokenizer = None
        self.current_model = None
        
        # Model load parameters
        self.load_8bit = True  # Enable 8-bit quantization by default
        self.load_4bit = False  # 4-bit quantization (off by default)
        self.device_map = 'auto'  # Use auto device mapping
        
        logger.info("Model manager initialized")
    
    def load_model(self, model_id: str, token: Optional[str] = None, 
                force_reload: bool = False) -> Tuple[Any, Any]:
        """
        Load a model and its tokenizer
        
        Args:
            model_id: HuggingFace model ID
            token: HuggingFace access token
            force_reload: Whether to force reload even if the model is in cache
            
        Returns:
            Tuple of (tokenizer, model)
            
        Raises:
            Exception: If model loading fails
        """
        # Check if model is already loaded and not forcing reload
        if model_id in self.model_cache and not force_reload:
            logger.info(f"Using cached model: {model_id}")
            tokenizer, model = self.model_cache[model_id]
            
            # Update current model
            self.current_model_id = model_id
            self.current_tokenizer = tokenizer
            self.current_model = model
            
            return tokenizer, model
        
        try:
            # Set environment variable to force using PyTorch
            os.environ["USE_TORCH"] = "1"
            os.environ["USE_TF"] = "0"
            
            # Import here to avoid loading transformers at startup
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            logger.info(f"Loading model: {model_id}")
            
            # Check if model is a GPTQ model
            is_gptq = "gptq" in model_id.lower()
            
            # Try to load specialized GPTQ modules if needed
            if is_gptq:
                try:
                    # Try to import auto-gptq
                    try:
                        import auto_gptq
                        logger.info("Using auto_gptq for model loading")
                    except ImportError:
                        logger.warning("auto_gptq not found. Try installing it with: pip install auto-gptq")
                except:
                    logger.warning("Failed to import GPTQ-specific modules")
            
            # Load tokenizer with more specific error handling
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    use_auth_token=token,
                    trust_remote_code=True
                )
            except Exception as e:
                logger.error(f"Error loading tokenizer: {str(e)}")
                raise
            
            # Prepare model loading arguments
            kwargs = {
                'use_auth_token': token,
                'device_map': self.device_map,
                'torch_dtype': torch.float16,
                'trust_remote_code': True,
            }
            
            # Explicitly disable TensorFlow
            kwargs['use_tf'] = False
            
            # Add quantization parameters if needed
            if self.load_8bit:
                kwargs['load_in_8bit'] = True
            elif self.load_4bit:
                kwargs['load_in_4bit'] = True
                kwargs['bnb_4bit_compute_dtype'] = torch.float16
                kwargs['bnb_4bit_quant_type'] = 'nf4'
            
            # Try different loading strategies
            try:
                # First attempt: standard loading
                logger.info("Attempting to load model with standard AutoModelForCausalLM")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"Standard loading failed: {str(e)}")
                
                if is_gptq:
                    # Second attempt: try loading with auto-gptq if available
                    try:
                        logger.info("Attempting to load GPTQ model with specific loader")
                        from auto_gptq import AutoGPTQForCausalLM
                        model = AutoGPTQForCausalLM.from_quantized(
                            model_id,
                            use_triton=False,
                            device_map=self.device_map,
                            trust_remote_code=True,
                            use_auth_token=token
                        )
                    except Exception as gptq_e:
                        logger.error(f"GPTQ-specific loading failed: {str(gptq_e)}")
                        
                        # Third attempt: try suggesting an alternative model
                        if "llama" in model_id.lower():
                            alt_model_id = model_id.replace("GPTQ", "GGUF")
                            logger.warning(f"Consider using {alt_model_id} instead, as GGUF models often have fewer dependencies")
                        
                        # Re-raise the original exception
                        raise e
                else:
                    # Re-raise the original exception
                    raise
            
            # Set model to evaluation mode
            model.eval()
            
            # Store in cache
            self.model_cache[model_id] = (tokenizer, model)
            
            # Update current model
            self.current_model_id = model_id
            self.current_tokenizer = tokenizer
            self.current_model = model
            
            # Report success
            logger.info(f"Model {model_id} loaded successfully")
            
            # Return the model and tokenizer
            return tokenizer, model
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            # Clean up any partial loading
            if model_id in self.model_cache:
                del self.model_cache[model_id]
            
            # Reset current model if it was the one being loaded
            if self.current_model_id == model_id:
                self.current_model_id = None
                self.current_tokenizer = None
                self.current_model = None
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Re-raise the exception with more context
            raise
    
    def unload_model(self, model_id: Optional[str] = None) -> bool:
        """
        Unload a model from memory
        
        Args:
            model_id: Model ID to unload, or None to unload current model
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Determine which model to unload
        if model_id is None:
            model_id = self.current_model_id
        
        if not model_id or model_id not in self.model_cache:
            logger.warning(f"No model to unload: {model_id}")
            return False
        
        try:
            logger.info(f"Unloading model: {model_id}")
            
            # Get the model
            _, model = self.model_cache.pop(model_id)
            
            # If this was the current model, reset current model
            if model_id == self.current_model_id:
                self.current_model_id = None
                self.current_tokenizer = None
                self.current_model = None
            
            # Delete model and run garbage collection
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
            logger.info(f"Model {model_id} unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {str(e)}")
            return False
    
    def clear_cache(self) -> bool:
        """
        Clear the entire model cache
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Clearing model cache")
            
            # Unload all models
            model_ids = list(self.model_cache.keys())
            for model_id in model_ids:
                self.unload_model(model_id)
            
            # Reset current model
            self.current_model_id = None
            self.current_tokenizer = None
            self.current_model = None
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info("Model cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing model cache: {str(e)}")
            return False
    
    def get_current_model(self) -> Tuple[Optional[str], Any, Any]:
        """
        Get the currently active model
        
        Returns:
            Tuple of (model_id, tokenizer, model), or (None, None, None) if no model is loaded
        """
        return self.current_model_id, self.current_tokenizer, self.current_model
    
    def get_model_parameters(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model parameters such as size, device, etc.
        
        Args:
            model_id: Model ID to get parameters for, or None for current model
            
        Returns:
            Dict of model parameters, or empty dict if model not found
        """
        # Determine which model to use
        if model_id is None:
            model_id = self.current_model_id
        
        if not model_id or model_id not in self.model_cache:
            return {}
        
        try:
            _, model = self.model_cache[model_id]
            
            # Get basic parameters
            params = {
                "model_id": model_id,
                "is_quantized": self.load_8bit or self.load_4bit,
                "quantization": "8-bit" if self.load_8bit else "4-bit" if self.load_4bit else "None",
                "device_map": self.device_map,
            }
            
            # Get model specific parameters if available
            if hasattr(model, "config"):
                config = model.config
                
                # Common parameters
                if hasattr(config, "model_type"):
                    params["model_type"] = config.model_type
                
                if hasattr(config, "vocab_size"):
                    params["vocab_size"] = config.vocab_size
                
                if hasattr(config, "hidden_size"):
                    params["hidden_size"] = config.hidden_size
                
                if hasattr(config, "num_hidden_layers"):
                    params["num_layers"] = config.num_hidden_layers
                
                if hasattr(config, "num_attention_heads"):
                    params["num_heads"] = config.num_attention_heads
            
            # Calculate model size (parameters)
            try:
                num_params = sum(p.numel() for p in model.parameters())
                params["num_parameters"] = num_params
                params["num_parameters_millions"] = round(num_params / 1000000, 2)
                params["num_parameters_billions"] = round(num_params / 1000000000, 2)
            except Exception as e:
                logger.warning(f"Could not calculate model size: {str(e)}")
            
            # Get device information
            try:
                if hasattr(model, "device"):
                    params["device"] = str(model.device)
                elif hasattr(model, "hf_device_map"):
                    params["device_map"] = model.hf_device_map
                
                # Check if model is on GPU
                if torch.cuda.is_available():
                    params["is_on_gpu"] = any(p.device.type == "cuda" for p in model.parameters())
                else:
                    params["is_on_gpu"] = False
            except Exception as e:
                logger.warning(f"Could not determine model device: {str(e)}")
            
            # Get memory usage if on GPU
            if params.get("is_on_gpu", False):
                try:
                    memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                    memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
                    params["gpu_memory_allocated_gb"] = round(memory_allocated, 2)
                    params["gpu_memory_reserved_gb"] = round(memory_reserved, 2)
                except Exception as e:
                    logger.warning(f"Could not determine GPU memory usage: {str(e)}")
            
            return params
            
        except Exception as e:
            logger.error(f"Error getting model parameters: {str(e)}")
            return {}
    
    def get_current_model_info(self) -> str:
        """
        Get formatted information about the current model
        
        Returns:
            str: Formatted model information, or message if no model is loaded
        """
        if not self.current_model_id:
            return "No model currently loaded."
        
        try:
            params = self.get_model_parameters()
            
            if not params:
                return f"Model {self.current_model_id} is loaded, but details are not available."
            
            # Format information as HTML for rich display
            info = f"<h3>Model: {params['model_id']}</h3>\n\n"
            
            # Basic information
            info += "<h4>Basic Information:</h4>\n"
            info += "<ul>\n"
            
            if "model_type" in params:
                info += f"<li>Type: {params['model_type']}</li>\n"
            
            if "num_parameters_billions" in params:
                info += f"<li>Size: {params['num_parameters_billions']}B parameters</li>\n"
            elif "num_parameters_millions" in params:
                info += f"<li>Size: {params['num_parameters_millions']}M parameters</li>\n"
            
            info += f"<li>Quantization: {params['quantization']}</li>\n"
            
            info += "</ul>\n\n"
            
            # Architecture details if available
            architecture_params = ["vocab_size", "hidden_size", "num_layers", "num_heads"]
            if any(p in params for p in architecture_params):
                info += "<h4>Architecture:</h4>\n"
                info += "<ul>\n"
                
                for param in architecture_params:
                    if param in params:
                        # Convert param name to readable format
                        param_name = param.replace("_", " ").title()
                        info += f"<li>{param_name}: {params[param]}</li>\n"
                
                info += "</ul>\n\n"
            
            # Device and memory information
            info += "<h4>Resource Usage:</h4>\n"
            info += "<ul>\n"
            
            if "device" in params:
                info += f"<li>Device: {params['device']}</li>\n"
            elif "device_map" in params:
                info += f"<li>Device Map: {params['device_map']}</li>\n"
            
            if "is_on_gpu" in params:
                info += f"<li>Using GPU: {'Yes' if params['is_on_gpu'] else 'No'}</li>\n"
            
            if "gpu_memory_allocated_gb" in params:
                info += f"<li>GPU Memory Allocated: {params['gpu_memory_allocated_gb']} GB</li>\n"
            
            if "gpu_memory_reserved_gb" in params:
                info += f"<li>GPU Memory Reserved: {params['gpu_memory_reserved_gb']} GB</li>\n"
            
            info += "</ul>\n"
            
            return info
            
        except Exception as e:
            logger.error(f"Error formatting model info: {str(e)}")
            return f"Model {self.current_model_id} is loaded, but an error occurred retrieving details."
    
    def set_quantization(self, use_8bit: bool = True, use_4bit: bool = False) -> None:
        """
        Set quantization options for future model loading
        
        Args:
            use_8bit: Whether to use 8-bit quantization
            use_4bit: Whether to use 4-bit quantization (overrides 8-bit if True)
        """
        self.load_8bit = use_8bit
        
        # 4-bit takes precedence over 8-bit
        if use_4bit:
            self.load_8bit = False
            self.load_4bit = True
        else:
            self.load_4bit = False
        
        logger.info(f"Set quantization: 8-bit={self.load_8bit}, 4-bit={self.load_4bit}")
    
    def set_device_map(self, device_map: str) -> None:
        """
        Set device mapping for future model loading
        
        Args:
            device_map: Device mapping strategy ('auto', 'balanced', 'sequential', etc.)
        """
        self.device_map = device_map
        logger.info(f"Set device map: {device_map}")
    
    def is_model_loaded(self, model_id: Optional[str] = None) -> bool:
        """
        Check if a model is loaded
        
        Args:
            model_id: Model ID to check, or None to check if any model is loaded
            
        Returns:
            bool: True if the model is loaded, False otherwise
        """
        if model_id is None:
            return self.current_model_id is not None
        
        return model_id in self.model_cache
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get current memory statistics
        
        Returns:
            Dict with memory statistics in GB
        """
        stats = {}
        
        if torch.cuda.is_available():
            # Get memory stats
            stats["allocated"] = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            stats["reserved"] = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
            stats["max_allocated"] = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            
            # Get per device stats if multiple GPUs
            if torch.cuda.device_count() > 1:
                for i in range(torch.cuda.device_count()):
                    stats[f"device_{i}_allocated"] = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    stats[f"device_{i}_reserved"] = torch.cuda.memory_reserved(i) / (1024 ** 3)
        
        return stats