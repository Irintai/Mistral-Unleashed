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

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError as e:
    logger.error(f"Critical import error: {str(e)}")
    from PyQt5.QtWidgets import QMessageBox
    QMessageBox.critical(
        None,
        "Import Error",
        "Failed to import required libraries. Please install transformers and torch."
    )
# Add at the top of the file, after the imports
HAS_GPTQ_SUPPORT = False
GPTQ_METHOD = None  # Will be set to "auto_gptq" or "transformers" depending on availability

try:
    from transformers import GPTQConfig
    HAS_GPTQ_SUPPORT = True
    GPTQ_METHOD = "transformers"
    logger.info("Using Transformers native GPTQ support")
except ImportError:
    try:
        import auto_gptq
        HAS_GPTQ_SUPPORT = True
        GPTQ_METHOD = "auto_gptq"
        logger.warning("Using AutoGPTQ which is deprecated. Consider installing Transformers with native GPTQ support.")
    except ImportError:
        logger.warning("No GPTQ support available. GPTQ models will require installation of transformers with GPTQ support.")


class ModelManager:
    """Manager for language models, handling loading, caching and memory management"""
    
    def __init__(self):
        """Initialize the model manager"""
        # Dictionary to store loaded models: model_id -> (tokenizer, model)
        self.model_cache: Dict[str, Tuple[Any, Any]] = {}
        
        # Currently active model
        self.active_model_id: Optional[str] = None
        
        # Quantization settings
        self.load_8bit: bool = False
        self.load_4bit: bool = False
        
        # Device settings
        self.device_map: str = "auto"  # "auto", "cuda:0", "cpu", etc.
        
        # GPU optimization settings
        self.use_cuda_graph = False  # CUDA Graph optimization for inference
        self.use_flash_attention = False  # Use flash attention if available
        self.use_better_transformer = True  # Better transformer optimization
        self.use_xformers = False  # Use xformers if available
        self.use_torch_compile = False  # Use torch.compile for further optimization
        
        # Detect GPU capabilities
        self.gpu_info = self._detect_gpu()
        
        # Initialize memory monitoring
        self.memory_usage = 0.0
        self.total_memory = 0.0
        self.update_memory_usage()
        
        logger.info("Model manager initialized")
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU capabilities and return information"""
        gpu_info = {
            "has_gpu": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [],
            "cuda_version": torch.version.cuda,
            "total_memory": 0,
        }
        
        if gpu_info["has_gpu"]:
            for i in range(gpu_info["gpu_count"]):
                device_name = torch.cuda.get_device_name(i)
                gpu_info["gpu_names"].append(device_name)
                
                # Get memory info
                total_mem = torch.cuda.get_device_properties(i).total_memory
                gpu_info["total_memory"] += total_mem
                
                logger.info(f"Found GPU {i}: {device_name} with {total_mem / 1024**3:.2f} GB memory")
                
                # Determine capabilities
                if "RTX" in device_name or "Quadro" in device_name or "Tesla" in device_name:
                    gpu_info["has_tensor_cores"] = True
            
            # Check for available optimization packages
            try:
                import flash_attn
                self.use_flash_attention = True
                logger.info("Flash Attention 2 available and enabled")
            except ImportError:
                self.use_flash_attention = False
                logger.info("Flash Attention 2 not available. Using standard attention mechanisms.")
            
            try:
                import xformers
                self.use_xformers = True
                logger.info("xformers available, enabled for memory-efficient attention")
            except ImportError:
                self.use_xformers = False
                logger.info("xformers not available. Install for memory-efficient attention.")
                
            # Enable torch.compile if we have a capable PyTorch version
            if hasattr(torch, 'compile') and callable(getattr(torch, 'compile')):
                self.use_torch_compile = True
                logger.info("torch.compile available, can be used for optimized inference")
                
        return gpu_info
    
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
            self.active_model_id = model_id
            self.current_tokenizer = tokenizer
            self.current_model = model
            
            return tokenizer, model
        
        try:
            # Set environment variable to force using PyTorch
            os.environ["USE_TORCH"] = "1"
            os.environ["USE_TF"] = "0"
            
            # Import here to avoid loading transformers at startup
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
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
            
            # Add GPU-specific optimizations
            if torch.cuda.is_available():
                # Apply memory-efficient attention mechanisms if available
                if self.use_flash_attention:
                    try:
                        import flash_attn
                        kwargs['attn_implementation'] = 'flash_attention_2'
                        logger.info("Using Flash Attention 2 for efficient memory usage")
                    except ImportError:
                        logger.warning("Flash Attention 2 requested but package not installed")
                
                # For transformers models that support better transformer
                if self.use_better_transformer:
                    try:
                        # Check if BetterTransformer is available in the current transformers version
                        from transformers.utils import is_torch_greater_or_equal_than_1_10
                        if is_torch_greater_or_equal_than_1_10():
                            kwargs['use_bettertransformer'] = True
                            logger.info("Using BetterTransformer for optimized inference")
                    except (ImportError, AttributeError):
                        logger.warning("BetterTransformer not available in this transformers version")
            
            # Add quantization parameters if needed - only for non-GPTQ models
            if not is_gptq:
                quantization_config = None
                if self.load_8bit or self.load_4bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=self.load_8bit,
                        load_in_4bit=self.load_4bit,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                kwargs['quantization_config'] = quantization_config
            
            # Try different loading strategies
            model = None
            try:
                # First attempt: standard loading
                logger.info("Attempting to load model with standard AutoModelForCausalLM")
                # For Llama models, we need to filter out some parameters
                if "llama" in model_id.lower():
                    clean_kwargs = {k: v for k, v in kwargs.items() if k != 'use_tf'}
                    model = AutoModelForCausalLM.from_pretrained(model_id, **clean_kwargs)
                else:
                    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
                
                # Apply post-loading optimizations
                if torch.cuda.is_available():
                    # Apply optimizations if we have a GPU
                    if model.device.type == "cuda":
                        # Check if we need to apply optimizations
                        try:
                            # First check for built-in scaled dot product attention (available in newer PyTorch versions)
                            # If the model already uses this optimization, we don't need to apply others
                            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                                logger.info("Model is using PyTorch's built-in scaled_dot_product_attention")
                                uses_sdpa = True
                            else:
                                uses_sdpa = False
                                
                            # Only apply xformers if not using built-in SDPA
                            if not uses_sdpa and self.use_xformers:
                                import xformers
                                logger.info("Applying xformers memory-efficient attention")
                                model = self.apply_xformers(model)
                        except ImportError:
                            logger.warning("xformers not available, skipping this optimization")
                        
                        # Apply torch.compile for PyTorch 2.0+ when available
                        if self.use_torch_compile and hasattr(torch, 'compile'):
                            logger.info("Applying torch.compile for optimized performance")
                            try:
                                # Use the appropriate backend
                                compile_backend = "inductor"  # Default for PyTorch 2.0+
                                model = torch.compile(model, backend=compile_backend)
                            except Exception as compile_error:
                                logger.warning(f"torch.compile failed: {str(compile_error)}")
            
            except Exception as e:
                logger.warning(f"Standard loading failed: {str(e)}")
                
                if is_gptq:
                    if not HAS_GPTQ_SUPPORT:
                        # No fallback - we require GPTQ support
                        error_msg = ("Cannot load GPTQ model: No GPTQ support available. "
                                    "Please install transformers with GPTQ support or auto_gptq.")
                        logger.error(error_msg)
                        raise ImportError(error_msg)
                    
                    elif GPTQ_METHOD == "transformers":
                        # Use native transformers GPTQ support
                        logger.info("Loading GPTQ model with native Transformers support")
                        # Create a clean set of kwargs for GPTQ - filter out problematic parameters
                        gptq_kwargs = {k: v for k, v in kwargs.items() 
                                    if k not in ['load_in_8bit', 'load_in_4bit', 
                                                'bnb_4bit_compute_dtype', 'bnb_4bit_quant_type',
                                                'use_tf']}  # Remove use_tf parameter
                        
                        gptq_kwargs["quantization_config"] = GPTQConfig(bits=4, use_exllama=True)
                        
                        model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            **gptq_kwargs
                        )
                    else:  # GPTQ_METHOD == "auto_gptq"
                        # Use deprecated AutoGPTQ
                        logger.info("Loading GPTQ model with AutoGPTQ")
                        from auto_gptq import AutoGPTQForCausalLM
                        
                        # Filter out problematic parameters for AutoGPTQ
                        autogptq_kwargs = {k: v for k, v in kwargs.items() 
                                        if k not in ['load_in_8bit', 'load_in_4bit', 
                                                    'bnb_4bit_compute_dtype', 'bnb_4bit_quant_type',
                                                    'use_tf']}
                        
                        model = AutoGPTQForCausalLM.from_quantized(
                            model_id,
                            use_triton=False,
                            **autogptq_kwargs
                        )
                else:
                    try:
                        logger.info("Attempting to load with bitsandbytes 8-bit quantization as fallback")
                        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True, load_in_4bit=False, bnb_4bit_compute_dtype=torch.float16)
                        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
                    except Exception as fallback_e:
                        logger.error(f"Fallback loading failed: {str(fallback_e)}")
                        raise e  # Re-raise original error if fallback also fails

            # Check if model was successfully loaded
            if model is None:
                raise ValueError(f"Failed to load model {model_id} with any available method")
                
            # Set model to evaluation mode for inference
            model.eval()
            
            # Optimize for inference if possible
            model = self.configure_for_inference(model)
            
            # Store in cache
            self.model_cache[model_id] = (tokenizer, model)
            
            # Update current model references
            self.active_model_id = model_id
            self.current_tokenizer = tokenizer
            self.current_model = model
            
            # Report success
            logger.info(f"Model {model_id} loaded successfully")
            
            # Update memory usage statistics
            self.update_memory_usage()
            
            # Return the model and tokenizer
            return tokenizer, model
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            # Clean up any partial loading
            if model_id in self.model_cache:
                del self.model_cache[model_id]
            
            # Reset current model if it was the one being loaded
            if self.active_model_id == model_id:
                self.active_model_id = None
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
            model_id = self.active_model_id
        
        if not model_id or model_id not in self.model_cache:
            logger.warning(f"No model to unload: {model_id}")
            return False
        
        try:
            logger.info(f"Unloading model: {model_id}")
            
            # Get the model
            _, model = self.model_cache.pop(model_id)
            
            # If this was the current model, reset current model
            if model_id == self.active_model_id:
                self.active_model_id = None
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
            self.active_model_id = None
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
        return self.active_model_id, self.current_tokenizer, self.current_model
    
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
            model_id = self.active_model_id
        
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
        if not self.active_model_id:
            return "No model currently loaded."
        
        try:
            params = self.get_model_parameters()
            
            if not params:
                return f"Model {self.active_model_id} is loaded, but details are not available."
            
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
            return f"Model {self.active_model_id} is loaded, but an error occurred retrieving details."
    
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
            return self.active_model_id is not None
        
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
    
    def update_memory_usage(self):
        """Update GPU memory usage statistics"""
        if torch.cuda.is_available():
            # Get memory statistics for all GPUs
            total_memory = 0
            used_memory = 0
            
            for i in range(torch.cuda.device_count()):
                try:
                    # Get memory stats in bytes
                    total_mem = torch.cuda.get_device_properties(i).total_memory
                    reserved_mem = torch.cuda.memory_reserved(i)
                    allocated_mem = torch.cuda.memory_allocated(i)
                    
                    # Convert to more readable formats
                    total_memory += total_mem
                    used_memory += allocated_mem
                    
                    # Log memory usage per device
                    logger.debug(f"GPU {i} Memory: {allocated_mem / 1024**2:.1f}MB allocated, "
                              f"{reserved_mem / 1024**2:.1f}MB reserved, "
                              f"{total_mem / 1024**2:.1f}MB total")
                except Exception as e:
                    logger.error(f"Error getting memory for GPU {i}: {e}")
            
            # Update cached memory values
            self.memory_usage = used_memory
            self.total_memory = total_memory
            
            # Return memory usage percentage
            if total_memory > 0:
                return (used_memory / total_memory) * 100
            return 0
        else:
            return 0
    
    def optimize_memory(self, aggressive=False):
        """Optimize GPU memory usage by clearing unused memory"""
        if not torch.cuda.is_available():
            return
        
        # Standard cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        # More aggressive optimization if requested
        if aggressive:
            # Force a full garbage collection cycle
            gc.collect(generation=2)
            
            # Try to release unused memory back to the GPU
            if hasattr(torch.cuda, 'memory_stats'):
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)
            
            # On CUDA 11.4+, we can use this for even better memory cleanup
            if hasattr(torch.cuda, 'memory_stats') and callable(getattr(torch.cuda, 'memory_stats')):
                torch.cuda.synchronize()
            
            # Log memory usage after optimization
            mem_usage = self.update_memory_usage()
            logger.info(f"Memory optimized. Current usage: {mem_usage:.1f}%")
    
    def configure_for_inference(self, model):
        """Configure model for optimal inference performance"""
        if not model or not torch.cuda.is_available():
            return model
        
        try:
            # Set model to evaluation mode
            model.eval()
            
            # Disable gradient computation for inference
            for param in model.parameters():
                param.requires_grad = False
            
            # Apply CUDA optimizations for inference
            if self.use_cuda_graph and hasattr(torch, 'cuda') and hasattr(torch.cuda, 'make_graphed_callables'):
                try:
                    logger.info("Attempting to use CUDA Graph for inference optimization")
                    # Note: This is an advanced optimization that not all models support
                    # This implementation is simplified and may need model-specific adjustments
                    # torch.cuda.make_graphed_callables() is used in real implementations
                except Exception as e:
                    logger.warning(f"Could not apply CUDA Graph optimization: {e}")
            
            # Return the optimized model
            return model
        except Exception as e:
            logger.error(f"Error configuring model for inference: {e}")
            return model
    
    def apply_xformers(self, model):
        """Apply xformers memory-efficient attention"""
        import xformers
        model = model.to_bettertransformer()
        model.config.attn_implementation = "xformers"
        logger.info("Applied xformers memory-efficient attention")
        return model