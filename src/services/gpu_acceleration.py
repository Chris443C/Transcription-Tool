"""
GPU Acceleration Support for Whisper Transcription
Provides CUDA and OpenCL acceleration with automatic fallback to CPU.
"""

import torch
import platform
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

try:
    import whisper
    from whisper import Whisper
except ImportError:
    whisper = None
    Whisper = None

try:
    import pyopencl as cl
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False
    cl = None

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    """Information about available GPU hardware"""
    device_type: str
    device_name: str
    memory_gb: float
    compute_capability: Optional[str] = None
    opencl_version: Optional[str] = None
    is_available: bool = False

class GPUAccelerator:
    """Manages GPU acceleration for Whisper transcription"""
    
    def __init__(self):
        self.available_devices = self._detect_devices()
        self.preferred_device = self._select_best_device()
        self.current_device = None
        
    def _detect_devices(self) -> List[GPUInfo]:
        """Detect available GPU devices"""
        devices = []
        
        # Check CUDA availability
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                
                devices.append(GPUInfo(
                    device_type="CUDA",
                    device_name=props.name,
                    memory_gb=memory_gb,
                    compute_capability=f"{props.major}.{props.minor}",
                    is_available=True
                ))
                
        # Check OpenCL availability
        if HAS_OPENCL:
            try:
                platforms = cl.get_platforms()
                for platform in platforms:
                    gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
                    for device in gpu_devices:
                        memory_gb = device.global_mem_size / (1024**3)
                        
                        devices.append(GPUInfo(
                            device_type="OpenCL",
                            device_name=device.name.strip(),
                            memory_gb=memory_gb,
                            opencl_version=device.opencl_c_version.strip(),
                            is_available=True
                        ))
            except Exception as e:
                logger.warning(f"OpenCL device detection failed: {e}")
                
        # Add CPU as fallback
        devices.append(GPUInfo(
            device_type="CPU",
            device_name=platform.processor() or "CPU",
            memory_gb=0.0,
            is_available=True
        ))
        
        return devices
    
    def _select_best_device(self) -> Optional[GPUInfo]:
        """Select the best available device for transcription"""
        if not self.available_devices:
            return None
            
        # Prioritize CUDA > OpenCL > CPU
        cuda_devices = [d for d in self.available_devices if d.device_type == "CUDA" and d.memory_gb >= 2.0]
        if cuda_devices:
            return max(cuda_devices, key=lambda d: d.memory_gb)
            
        opencl_devices = [d for d in self.available_devices if d.device_type == "OpenCL" and d.memory_gb >= 2.0]
        if opencl_devices:
            return max(opencl_devices, key=lambda d: d.memory_gb)
            
        # Fallback to CPU
        cpu_devices = [d for d in self.available_devices if d.device_type == "CPU"]
        return cpu_devices[0] if cpu_devices else None
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available devices"""
        return {
            "available_devices": [
                {
                    "type": d.device_type,
                    "name": d.device_name,
                    "memory_gb": d.memory_gb,
                    "compute_capability": d.compute_capability,
                    "opencl_version": d.opencl_version,
                    "available": d.is_available
                }
                for d in self.available_devices
            ],
            "preferred_device": {
                "type": self.preferred_device.device_type,
                "name": self.preferred_device.device_name,
                "memory_gb": self.preferred_device.memory_gb
            } if self.preferred_device else None,
            "cuda_available": torch.cuda.is_available(),
            "opencl_available": HAS_OPENCL
        }
    
    def setup_device(self, force_device: Optional[str] = None) -> str:
        """Setup the device for transcription"""
        if force_device:
            # Try to find requested device
            for device in self.available_devices:
                if device.device_type.lower() == force_device.lower() and device.is_available:
                    self.current_device = device
                    break
                    
        if not self.current_device:
            self.current_device = self.preferred_device
            
        if not self.current_device:
            raise RuntimeError("No suitable device found for transcription")
            
        device_str = self._get_torch_device_string()
        logger.info(f"Using device: {self.current_device.device_name} ({self.current_device.device_type})")
        
        return device_str
    
    def _get_torch_device_string(self) -> str:
        """Get PyTorch device string for current device"""
        if not self.current_device:
            return "cpu"
            
        if self.current_device.device_type == "CUDA":
            return "cuda"
        elif self.current_device.device_type == "OpenCL":
            # PyTorch doesn't natively support OpenCL, fallback to CPU
            logger.warning("OpenCL selected but PyTorch doesn't support it natively, using CPU")
            return "cpu"
        else:
            return "cpu"
    
    def optimize_for_device(self, model_size: str = "medium") -> Dict[str, Any]:
        """Get optimized settings for current device"""
        if not self.current_device:
            return {"batch_size": 1, "fp16": False}
            
        settings = {
            "batch_size": 1,
            "fp16": False,
            "chunk_length": 30
        }
        
        if self.current_device.device_type == "CUDA":
            # Optimize for CUDA
            if self.current_device.memory_gb >= 8.0:
                settings.update({
                    "batch_size": 4 if model_size in ["tiny", "base"] else 2,
                    "fp16": True,
                    "chunk_length": 60
                })
            elif self.current_device.memory_gb >= 4.0:
                settings.update({
                    "batch_size": 2,
                    "fp16": True,
                    "chunk_length": 30
                })
            else:
                settings.update({
                    "batch_size": 1,
                    "fp16": True,
                    "chunk_length": 15
                })
                
        elif self.current_device.device_type == "CPU":
            # Optimize for CPU
            settings.update({
                "batch_size": 1,
                "fp16": False,
                "chunk_length": 30
            })
            
        return settings

class QuantizedModelManager:
    """Manages quantized Whisper models for improved performance"""
    
    SUPPORTED_QUANTIZATIONS = ["int8", "int4", "fp16", "dynamic"]
    MODEL_SIZES = ["tiny", "base", "small", "medium", "large"]
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "whisper_quantized"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._quantization_cache = {}
        
    def is_quantized_available(self, model_name: str, quantization: str) -> bool:
        """Check if quantized model is available"""
        if quantization not in self.SUPPORTED_QUANTIZATIONS:
            return False
            
        model_path = self.cache_dir / f"{model_name}_{quantization}.pt"
        return model_path.exists()
    
    def get_quantized_model_path(self, model_name: str, quantization: str) -> Optional[Path]:
        """Get path to quantized model if available"""
        if self.is_quantized_available(model_name, quantization):
            return self.cache_dir / f"{model_name}_{quantization}.pt"
        return None
    
    async def create_quantized_model(self, model_name: str, quantization: str, device: str = "cpu") -> Optional[Path]:
        """Create and save quantized model"""
        if not whisper or quantization not in self.SUPPORTED_QUANTIZATIONS:
            return None
            
        try:
            logger.info(f"Creating quantized model: {model_name} -> {quantization}")
            
            # Load original model
            model = whisper.load_model(model_name, device=device)
            
            # Apply quantization
            quantized_model = None
            
            if quantization == "int8":
                # Dynamic int8 quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8
                )
            elif quantization == "int4" and hasattr(torch, 'ao'):
                # Int4 quantization (if available)
                try:
                    import torch.ao.quantization as tq
                    quantized_model = tq.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8, reduce_range=True
                    )
                except ImportError:
                    logger.warning("Int4 quantization not available, falling back to int8")
                    quantized_model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
            elif quantization == "fp16":
                # Half precision
                quantized_model = model.half()
            elif quantization == "dynamic":
                # Dynamic quantization with optimal settings
                quantized_model = torch.quantization.quantize_dynamic(
                    model, 
                    {torch.nn.Linear, torch.nn.Conv1d, torch.nn.MultiheadAttention}, 
                    dtype=torch.qint8
                )
            
            if quantized_model is not None:
                # Save quantized model
                output_path = self.cache_dir / f"{model_name}_{quantization}.pt"
                torch.save(quantized_model.state_dict(), output_path)
                
                # Save metadata
                metadata = {
                    'model_name': model_name,
                    'quantization': quantization,
                    'created_at': datetime.now().isoformat(),
                    'device': device,
                    'original_size': sum(p.numel() for p in model.parameters()),
                    'quantized_size': sum(p.numel() for p in quantized_model.parameters())
                }
                
                metadata_path = self.cache_dir / f"{model_name}_{quantization}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Quantized model saved to {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Failed to create quantized model: {e}")
            
        return None
    
    def get_quantization_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about available quantizations for model"""
        info = {
            'model_name': model_name,
            'available_quantizations': [],
            'recommended_quantization': None
        }
        
        for quantization in self.SUPPORTED_QUANTIZATIONS:
            if self.is_quantized_available(model_name, quantization):
                # Load metadata if available
                metadata_path = self.cache_dir / f"{model_name}_{quantization}_metadata.json"
                metadata = {}
                
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                info['available_quantizations'].append({
                    'type': quantization,
                    'file_size': self.get_quantized_model_path(model_name, quantization).stat().st_size,
                    'metadata': metadata
                })
        
        # Determine recommended quantization based on available hardware
        gpu_info = gpu_accelerator.get_device_info()
        if gpu_info.get('cuda_available'):
            info['recommended_quantization'] = 'fp16'
        else:
            info['recommended_quantization'] = 'int8'
            
        return info
    
    def load_model_with_quantization(self, model_name: str, device: str, quantization: Optional[str] = None):
        """Load Whisper model with optimal quantization"""
        if not whisper:
            raise ImportError("Whisper not available")
        
        # Auto-select quantization if not specified
        if not quantization:
            if device == "cuda":
                quantization = "fp16"
            else:
                quantization = "int8"
        
        # Check cache first
        cache_key = f"{model_name}_{quantization}_{device}"
        if cache_key in self._quantization_cache:
            logger.info(f"Using cached quantized model: {cache_key}")
            return self._quantization_cache[cache_key]
        
        model = None
        
        # Try quantized version first if available
        if quantization and self.is_quantized_available(model_name, quantization):
            try:
                quantized_path = self.get_quantized_model_path(model_name, quantization)
                logger.info(f"Loading pre-quantized model: {model_name}_{quantization}")
                
                # Load the base model structure
                base_model = whisper.load_model(model_name, device=device)
                
                # Load quantized weights
                quantized_state = torch.load(quantized_path, map_location=device)
                base_model.load_state_dict(quantized_state)
                
                model = base_model
                
            except Exception as e:
                logger.warning(f"Failed to load pre-quantized model: {e}")
        
        if model is None:
            # Load standard model
            logger.info(f"Loading standard model: {model_name}")
            model = whisper.load_model(model_name, device=device)
            
            # Apply post-loading quantization
            if quantization and quantization != "none":
                try:
                    if quantization == "int8":
                        model = torch.quantization.quantize_dynamic(
                            model, {torch.nn.Linear}, dtype=torch.qint8
                        )
                        logger.info(f"Applied dynamic int8 quantization to {model_name}")
                        
                    elif quantization == "fp16" and device == "cuda":
                        model = model.half()
                        logger.info(f"Applied fp16 quantization to {model_name}")
                        
                    elif quantization == "dynamic":
                        model = torch.quantization.quantize_dynamic(
                            model, 
                            {torch.nn.Linear, torch.nn.Conv1d}, 
                            dtype=torch.qint8
                        )
                        logger.info(f"Applied dynamic quantization to {model_name}")
                        
                except Exception as e:
                    logger.warning(f"Post-loading quantization failed: {e}")
        
        # Cache the model
        self._quantization_cache[cache_key] = model
        
        return model
    
    def get_optimal_quantization(self, model_name: str, device_info: Dict[str, Any]) -> str:
        """Get optimal quantization for given hardware"""
        if device_info.get('cuda_available', False):
            # CUDA device - prefer fp16 for speed
            gpu_memory = device_info.get('preferred_device', {}).get('memory_gb', 0)
            
            if gpu_memory >= 8.0:
                return "fp16"  # Plenty of memory, use fp16
            elif gpu_memory >= 4.0:
                return "int8"   # Limited memory, use int8
            else:
                return "dynamic"  # Very limited memory
        else:
            # CPU device - prefer int8 for size/speed balance
            return "int8"
    
    def estimate_model_memory(self, model_name: str, quantization: str = "none") -> float:
        """Estimate memory usage for model in GB"""
        # Base model parameter counts (approximate)
        param_counts = {
            "tiny": 39_000_000,
            "base": 74_000_000, 
            "small": 244_000_000,
            "medium": 769_000_000,
            "large": 1_550_000_000
        }
        
        params = param_counts.get(model_name, param_counts["medium"])
        
        # Memory multipliers for different quantizations
        multipliers = {
            "none": 4.0,     # fp32 - 4 bytes per param
            "fp16": 2.0,     # fp16 - 2 bytes per param
            "int8": 1.0,     # int8 - 1 byte per param
            "int4": 0.5,     # int4 - 0.5 bytes per param
            "dynamic": 1.5   # mixed precision - average
        }
        
        bytes_per_param = multipliers.get(quantization, 4.0)
        total_bytes = params * bytes_per_param
        
        # Add overhead (approximately 20% for activations and buffers)
        total_bytes *= 1.2
        
        return total_bytes / (1024**3)  # Convert to GB
    
    def cleanup_old_models(self, keep_recent: int = 5):
        """Remove old quantized models to free space"""
        try:
            model_files = list(self.cache_dir.glob("*.pt"))
            model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            for old_file in model_files[keep_recent:]:
                old_file.unlink()
                
                # Remove corresponding metadata
                metadata_file = old_file.with_suffix('.json')
                if metadata_file.exists():
                    metadata_file.unlink()
                    
                logger.info(f"Removed old quantized model: {old_file}")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup old models: {e}")

# Global instance for easy access
gpu_accelerator = GPUAccelerator()
quantized_manager = QuantizedModelManager()

def get_optimal_transcription_config(model_size: str = "medium", force_device: Optional[str] = None) -> Dict[str, Any]:
    """Get optimal configuration for transcription based on available hardware"""
    device_str = gpu_accelerator.setup_device(force_device)
    optimization_settings = gpu_accelerator.optimize_for_device(model_size)
    
    return {
        "device": device_str,
        "model_size": model_size,
        **optimization_settings,
        "gpu_info": gpu_accelerator.get_device_info()
    }

def load_optimized_model(model_name: str, config: Dict[str, Any]):
    """Load Whisper model with optimal configuration"""
    device = config.get("device", "cpu")
    quantization = config.get("quantization")
    
    return quantized_manager.load_model_with_quantization(
        model_name, device, quantization
    )