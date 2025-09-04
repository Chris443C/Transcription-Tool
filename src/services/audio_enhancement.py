"""
AI-Powered Audio Enhancement
Provides intelligent audio processing with noise reduction, speech enhancement, and normalization.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio

try:
    import soundfile as sf
    import librosa
    import scipy.signal
    from scipy.io.wavfile import write as write_wav
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    sf = None
    librosa = None
    scipy = None

try:
    import noisereduce as nr
    HAS_NOISE_REDUCE = True
except ImportError:
    HAS_NOISE_REDUCE = False
    nr = None

try:
    import webrtcvad
    HAS_VAD = True
except ImportError:
    HAS_VAD = False
    webrtcvad = None

logger = logging.getLogger(__name__)

@dataclass
class EnhancementSettings:
    """Audio enhancement configuration"""
    # Noise reduction
    noise_reduction_enabled: bool = True
    noise_reduction_strength: float = 0.8  # 0.0 to 1.0
    stationary_noise_reduction: bool = True
    non_stationary_noise_reduction: bool = True
    
    # Speech enhancement
    speech_enhancement_enabled: bool = True
    voice_activity_detection: bool = True
    speech_boost_db: float = 3.0
    formant_enhancement: bool = True
    
    # Audio normalization
    normalization_enabled: bool = True
    target_lufs: float = -23.0  # EBU R128 standard
    peak_limiting: bool = True
    dynamic_range_compression: float = 0.2  # 0.0 = no compression, 1.0 = heavy
    
    # Frequency processing
    high_pass_filter: bool = True
    high_pass_cutoff: float = 80.0  # Hz
    low_pass_filter: bool = False
    low_pass_cutoff: float = 8000.0  # Hz
    
    # Advanced settings
    sample_rate_optimization: bool = True
    target_sample_rate: int = 16000
    mono_conversion: bool = True
    remove_silence: bool = False
    silence_threshold_db: float = -40.0

class AudioAnalyzer:
    """Analyzes audio characteristics for optimal enhancement"""
    
    def __init__(self):
        self.analysis_cache = {}
        
    def analyze_audio(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Comprehensive audio analysis"""
        if not HAS_AUDIO_LIBS:
            raise ImportError("Audio analysis requires soundfile and librosa")
            
        analysis = {
            'duration': len(audio_data) / sample_rate,
            'sample_rate': sample_rate,
            'channels': 1 if audio_data.ndim == 1 else audio_data.shape[1],
            'peak_amplitude': float(np.max(np.abs(audio_data))),
            'rms_level': float(np.sqrt(np.mean(audio_data**2))),
            'dynamic_range': 0.0,
            'spectral_centroid': 0.0,
            'zero_crossing_rate': 0.0,
            'signal_to_noise_ratio': 0.0,
            'voice_activity_ratio': 0.0,
            'frequency_analysis': {},
            'noise_profile': {}
        }
        
        try:
            # Ensure mono for analysis
            if audio_data.ndim > 1:
                audio_mono = np.mean(audio_data, axis=1)
            else:
                audio_mono = audio_data
                
            # Dynamic range
            analysis['dynamic_range'] = float(
                20 * np.log10(analysis['peak_amplitude'] / (analysis['rms_level'] + 1e-10))
            )
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_mono, sr=sample_rate)
            analysis['spectral_centroid'] = float(np.mean(spectral_centroids))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_mono)
            analysis['zero_crossing_rate'] = float(np.mean(zcr))
            
            # Frequency analysis
            fft = np.fft.fft(audio_mono[:min(len(audio_mono), sample_rate)])  # 1 second window
            freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
            magnitude = np.abs(fft)
            
            # Find dominant frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            # Voice frequency range analysis (80-8000 Hz)
            voice_range_mask = (positive_freqs >= 80) & (positive_freqs <= 8000)
            voice_energy = np.sum(positive_magnitude[voice_range_mask])
            total_energy = np.sum(positive_magnitude)
            
            analysis['frequency_analysis'] = {
                'dominant_frequency': float(positive_freqs[np.argmax(positive_magnitude)]),
                'voice_frequency_ratio': float(voice_energy / (total_energy + 1e-10)),
                'high_frequency_content': float(
                    np.sum(positive_magnitude[positive_freqs > 4000]) / (total_energy + 1e-10)
                ),
                'low_frequency_content': float(
                    np.sum(positive_magnitude[positive_freqs < 200]) / (total_energy + 1e-10)
                )
            }
            
            # Voice Activity Detection if available
            if HAS_VAD:
                try:
                    vad = webrtcvad.Vad()
                    vad.set_mode(3)  # Most aggressive
                    
                    # Convert to 16-bit PCM for WebRTC VAD
                    audio_16bit = (audio_mono * 32767).astype(np.int16)
                    frame_duration = 30  # ms
                    frame_samples = int(sample_rate * frame_duration / 1000)
                    
                    voice_frames = 0
                    total_frames = 0
                    
                    for i in range(0, len(audio_16bit) - frame_samples, frame_samples):
                        frame = audio_16bit[i:i + frame_samples].tobytes()
                        if len(frame) == frame_samples * 2:  # 2 bytes per sample
                            if vad.is_speech(frame, sample_rate):
                                voice_frames += 1
                            total_frames += 1
                            
                    if total_frames > 0:
                        analysis['voice_activity_ratio'] = voice_frames / total_frames
                        
                except Exception as e:
                    logger.warning(f"VAD analysis failed: {e}")
                    
            # Estimate signal-to-noise ratio
            # Simple approach: compare speech segments to non-speech segments
            if analysis['voice_activity_ratio'] > 0.1:
                # Rough SNR estimation
                signal_power = analysis['rms_level']**2
                
                # Estimate noise from quieter segments
                sorted_samples = np.sort(np.abs(audio_mono))
                noise_samples = sorted_samples[:int(len(sorted_samples) * 0.2)]  # Bottom 20%
                noise_power = np.mean(noise_samples**2)
                
                if noise_power > 1e-10:
                    analysis['signal_to_noise_ratio'] = float(
                        10 * np.log10(signal_power / noise_power)
                    )
                    
        except Exception as e:
            logger.error(f"Audio analysis error: {e}")
            
        return analysis

class NoiseReducer:
    """Advanced noise reduction using multiple techniques"""
    
    def __init__(self):
        self.noise_profiles = {}
        
    def reduce_stationary_noise(self, 
                               audio: np.ndarray, 
                               sample_rate: int,
                               strength: float = 0.8) -> np.ndarray:
        """Reduce stationary background noise"""
        
        if not HAS_NOISE_REDUCE:
            logger.warning("noisereduce library not available, using basic noise reduction")
            return self._basic_noise_reduction(audio, strength)
            
        try:
            # Use noisereduce for stationary noise
            reduced_audio = nr.reduce_noise(
                y=audio, 
                sr=sample_rate,
                stationary=True,
                prop_decrease=strength
            )
            
            return reduced_audio
            
        except Exception as e:
            logger.error(f"Noise reduction failed: {e}")
            return self._basic_noise_reduction(audio, strength)
            
    def reduce_non_stationary_noise(self, 
                                  audio: np.ndarray,
                                  sample_rate: int,
                                  strength: float = 0.6) -> np.ndarray:
        """Reduce non-stationary noise (clicks, pops, etc.)"""
        
        if not HAS_NOISE_REDUCE:
            return self._spectral_gating(audio, sample_rate, strength)
            
        try:
            # Non-stationary noise reduction
            reduced_audio = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                stationary=False,
                prop_decrease=strength
            )
            
            return reduced_audio
            
        except Exception as e:
            logger.error(f"Non-stationary noise reduction failed: {e}")
            return self._spectral_gating(audio, sample_rate, strength)
            
    def _basic_noise_reduction(self, audio: np.ndarray, strength: float) -> np.ndarray:
        """Basic noise reduction using spectral subtraction"""
        try:
            # Simple spectral subtraction
            fft = np.fft.fft(audio)
            magnitude = np.abs(fft)
            phase = np.angle(fft)
            
            # Estimate noise floor
            noise_floor = np.percentile(magnitude, 10)  # Bottom 10%
            
            # Spectral subtraction
            enhanced_magnitude = magnitude - strength * noise_floor
            enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
            
            # Reconstruct signal
            enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = np.real(np.fft.ifft(enhanced_fft))
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Basic noise reduction failed: {e}")
            return audio
            
    def _spectral_gating(self, audio: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
        """Spectral gating for noise reduction"""
        try:
            # Short-time Fourier transform
            window_size = int(0.025 * sample_rate)  # 25ms
            hop_size = window_size // 4
            
            # Apply spectral gating
            enhanced_audio = audio.copy()
            
            for i in range(0, len(audio) - window_size, hop_size):
                window = audio[i:i + window_size]
                fft = np.fft.fft(window)
                magnitude = np.abs(fft)
                
                # Apply soft thresholding
                threshold = np.percentile(magnitude, 30) * strength
                mask = magnitude > threshold
                
                enhanced_fft = fft * mask
                enhanced_window = np.real(np.fft.ifft(enhanced_fft))
                
                # Overlap-add
                enhanced_audio[i:i + window_size] += enhanced_window * 0.5
                
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Spectral gating failed: {e}")
            return audio

class SpeechEnhancer:
    """Enhances speech clarity and intelligibility"""
    
    def __init__(self):
        pass
        
    def enhance_speech(self, 
                      audio: np.ndarray,
                      sample_rate: int,
                      boost_db: float = 3.0,
                      formant_enhancement: bool = True) -> np.ndarray:
        """Enhance speech clarity"""
        
        try:
            enhanced_audio = audio.copy()
            
            # Speech frequency boost (300-3400 Hz)
            if boost_db > 0:
                enhanced_audio = self._boost_speech_frequencies(
                    enhanced_audio, sample_rate, boost_db
                )
                
            # Formant enhancement
            if formant_enhancement:
                enhanced_audio = self._enhance_formants(enhanced_audio, sample_rate)
                
            # Dynamic range expansion for speech
            enhanced_audio = self._expand_speech_dynamics(enhanced_audio)
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Speech enhancement failed: {e}")
            return audio
            
    def _boost_speech_frequencies(self, 
                                 audio: np.ndarray,
                                 sample_rate: int,
                                 boost_db: float) -> np.ndarray:
        """Boost speech frequency range"""
        if not HAS_AUDIO_LIBS:
            return audio
            
        try:
            # Design bandpass filter for speech frequencies
            low_freq = 300
            high_freq = 3400
            nyquist = sample_rate / 2
            
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            # Butterworth bandpass filter
            b, a = scipy.signal.butter(4, [low, high], btype='band')
            
            # Filter the audio
            speech_band = scipy.signal.filtfilt(b, a, audio)
            
            # Apply boost
            boost_linear = 10**(boost_db / 20)
            boosted_speech = speech_band * (boost_linear - 1.0)
            
            # Mix back with original
            enhanced_audio = audio + boosted_speech
            
            # Prevent clipping
            max_val = np.max(np.abs(enhanced_audio))
            if max_val > 1.0:
                enhanced_audio = enhanced_audio / max_val
                
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Speech frequency boost failed: {e}")
            return audio
            
    def _enhance_formants(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Enhance speech formants"""
        try:
            # Simple formant enhancement using spectral shaping
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
            
            # Formant frequency ranges (approximate)
            formant_ranges = [(400, 800), (800, 1200), (1200, 2400)]
            
            enhanced_fft = fft.copy()
            
            for low, high in formant_ranges:
                # Find frequency bins
                mask = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
                
                # Slight boost for formants
                enhanced_fft[mask] *= 1.1
                
            enhanced_audio = np.real(np.fft.ifft(enhanced_fft))
            
            # Normalize
            max_val = np.max(np.abs(enhanced_audio))
            if max_val > 1.0:
                enhanced_audio = enhanced_audio / max_val
                
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Formant enhancement failed: {e}")
            return audio
            
    def _expand_speech_dynamics(self, audio: np.ndarray) -> np.ndarray:
        """Expand dynamic range for better speech clarity"""
        try:
            # Simple dynamic range expansion
            # Make loud parts louder and quiet parts relatively quieter
            threshold = 0.1
            ratio = 1.2
            
            expanded_audio = audio.copy()
            
            # Apply expansion above threshold
            above_threshold = np.abs(audio) > threshold
            sign = np.sign(audio)
            
            expanded_audio[above_threshold] = (
                sign[above_threshold] * 
                (threshold + (np.abs(audio[above_threshold]) - threshold) * ratio)
            )
            
            # Normalize
            max_val = np.max(np.abs(expanded_audio))
            if max_val > 1.0:
                expanded_audio = expanded_audio / max_val
                
            return expanded_audio
            
        except Exception as e:
            logger.error(f"Dynamic range expansion failed: {e}")
            return audio

class AudioNormalizer:
    """Advanced audio normalization and dynamics processing"""
    
    def __init__(self):
        pass
        
    def normalize_audio(self,
                       audio: np.ndarray,
                       sample_rate: int,
                       target_lufs: float = -23.0,
                       peak_limiting: bool = True,
                       compression_ratio: float = 0.2) -> np.ndarray:
        """Comprehensive audio normalization"""
        
        try:
            normalized_audio = audio.copy()
            
            # RMS normalization (approximates LUFS)
            normalized_audio = self._rms_normalize(normalized_audio, target_lufs)
            
            # Dynamic range compression
            if compression_ratio > 0:
                normalized_audio = self._compress_dynamics(normalized_audio, compression_ratio)
                
            # Peak limiting
            if peak_limiting:
                normalized_audio = self._peak_limit(normalized_audio)
                
            return normalized_audio
            
        except Exception as e:
            logger.error(f"Audio normalization failed: {e}")
            return audio
            
    def _rms_normalize(self, audio: np.ndarray, target_lufs: float) -> np.ndarray:
        """RMS-based normalization"""
        try:
            # Calculate RMS
            rms = np.sqrt(np.mean(audio**2))
            
            if rms < 1e-10:  # Silence
                return audio
                
            # Convert LUFS target to linear scale (approximation)
            target_rms = 10**(target_lufs / 20)
            
            # Apply gain
            gain = target_rms / rms
            normalized_audio = audio * gain
            
            return normalized_audio
            
        except Exception as e:
            logger.error(f"RMS normalization failed: {e}")
            return audio
            
    def _compress_dynamics(self, audio: np.ndarray, ratio: float) -> np.ndarray:
        """Apply dynamic range compression"""
        try:
            # Simple soft knee compression
            threshold = 0.3
            
            compressed_audio = audio.copy()
            
            # Find samples above threshold
            above_threshold = np.abs(audio) > threshold
            sign = np.sign(audio)
            
            # Apply compression
            excess = np.abs(audio[above_threshold]) - threshold
            compressed_excess = excess * (1 - ratio)
            
            compressed_audio[above_threshold] = (
                sign[above_threshold] * (threshold + compressed_excess)
            )
            
            return compressed_audio
            
        except Exception as e:
            logger.error(f"Dynamic compression failed: {e}")
            return audio
            
    def _peak_limit(self, audio: np.ndarray, limit: float = 0.95) -> np.ndarray:
        """Apply peak limiting"""
        try:
            # Soft limiting
            limited_audio = np.tanh(audio / limit) * limit
            return limited_audio
            
        except Exception as e:
            logger.error(f"Peak limiting failed: {e}")
            return audio

class AudioEnhancementEngine:
    """Main audio enhancement engine"""
    
    def __init__(self):
        self.analyzer = AudioAnalyzer()
        self.noise_reducer = NoiseReducer()
        self.speech_enhancer = SpeechEnhancer()
        self.normalizer = AudioNormalizer()
        
    def enhance_audio_file(self,
                          input_path: str,
                          output_path: str,
                          settings: EnhancementSettings = None) -> bool:
        """Enhance audio file with specified settings"""
        
        if not HAS_AUDIO_LIBS:
            logger.error("Audio enhancement requires soundfile and librosa libraries")
            return False
            
        settings = settings or EnhancementSettings()
        
        try:
            # Load audio
            audio, sample_rate = sf.read(input_path)
            
            # Convert to mono if requested
            if settings.mono_conversion and audio.ndim > 1:
                audio = np.mean(audio, axis=1)
                
            logger.info(f"Processing audio: {len(audio)/sample_rate:.1f}s at {sample_rate}Hz")
            
            # Analyze audio
            analysis = self.analyzer.analyze_audio(audio, sample_rate)
            logger.info(f"Audio analysis: SNR={analysis.get('signal_to_noise_ratio', 0):.1f}dB, "
                       f"Voice activity={analysis.get('voice_activity_ratio', 0):.2f}")
            
            # Apply enhancements
            enhanced_audio = self._apply_enhancements(audio, sample_rate, settings, analysis)
            
            # Save enhanced audio
            sf.write(output_path, enhanced_audio, sample_rate)
            
            logger.info(f"Audio enhancement completed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Audio enhancement failed: {e}")
            return False
            
    def _apply_enhancements(self,
                          audio: np.ndarray,
                          sample_rate: int,
                          settings: EnhancementSettings,
                          analysis: Dict[str, Any]) -> np.ndarray:
        """Apply all enhancement steps"""
        
        enhanced_audio = audio.copy()
        
        # 1. High-pass filter (remove low-frequency noise)
        if settings.high_pass_filter:
            enhanced_audio = self._apply_high_pass_filter(
                enhanced_audio, sample_rate, settings.high_pass_cutoff
            )
            
        # 2. Noise reduction
        if settings.noise_reduction_enabled:
            if settings.stationary_noise_reduction:
                enhanced_audio = self.noise_reducer.reduce_stationary_noise(
                    enhanced_audio, sample_rate, settings.noise_reduction_strength
                )
                
            if settings.non_stationary_noise_reduction:
                enhanced_audio = self.noise_reducer.reduce_non_stationary_noise(
                    enhanced_audio, sample_rate, settings.noise_reduction_strength * 0.8
                )
                
        # 3. Speech enhancement
        if settings.speech_enhancement_enabled:
            enhanced_audio = self.speech_enhancer.enhance_speech(
                enhanced_audio,
                sample_rate,
                settings.speech_boost_db,
                settings.formant_enhancement
            )
            
        # 4. Normalization
        if settings.normalization_enabled:
            enhanced_audio = self.normalizer.normalize_audio(
                enhanced_audio,
                sample_rate,
                settings.target_lufs,
                settings.peak_limiting,
                settings.dynamic_range_compression
            )
            
        # 5. Low-pass filter (if enabled)
        if settings.low_pass_filter:
            enhanced_audio = self._apply_low_pass_filter(
                enhanced_audio, sample_rate, settings.low_pass_cutoff
            )
            
        return enhanced_audio
        
    def _apply_high_pass_filter(self,
                              audio: np.ndarray,
                              sample_rate: int,
                              cutoff: float) -> np.ndarray:
        """Apply high-pass filter"""
        if not HAS_AUDIO_LIBS:
            return audio
            
        try:
            nyquist = sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            
            if normalized_cutoff >= 1.0:
                return audio
                
            # Butterworth high-pass filter
            b, a = scipy.signal.butter(4, normalized_cutoff, btype='high')
            filtered_audio = scipy.signal.filtfilt(b, a, audio)
            
            return filtered_audio
            
        except Exception as e:
            logger.error(f"High-pass filter failed: {e}")
            return audio
            
    def _apply_low_pass_filter(self,
                             audio: np.ndarray,
                             sample_rate: int,
                             cutoff: float) -> np.ndarray:
        """Apply low-pass filter"""
        if not HAS_AUDIO_LIBS:
            return audio
            
        try:
            nyquist = sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            
            if normalized_cutoff >= 1.0:
                return audio
                
            # Butterworth low-pass filter
            b, a = scipy.signal.butter(4, normalized_cutoff, btype='low')
            filtered_audio = scipy.signal.filtfilt(b, a, audio)
            
            return filtered_audio
            
        except Exception as e:
            logger.error(f"Low-pass filter failed: {e}")
            return audio
            
    async def enhance_audio_async(self,
                                input_path: str,
                                output_path: str,
                                settings: EnhancementSettings = None,
                                progress_callback: Optional[callable] = None) -> bool:
        """Asynchronous audio enhancement"""
        
        loop = asyncio.get_event_loop()
        
        def progress_wrapper(stage: str, progress: float):
            if progress_callback:
                asyncio.run_coroutine_threadsafe(
                    progress_callback(stage, progress),
                    loop
                )
                
        # Run enhancement in thread pool
        with ThreadPoolExecutor() as executor:
            future = executor.submit(
                self._enhance_with_progress,
                input_path,
                output_path,
                settings,
                progress_wrapper
            )
            
            result = await asyncio.wrap_future(future)
            
        return result
        
    def _enhance_with_progress(self,
                             input_path: str,
                             output_path: str,
                             settings: EnhancementSettings,
                             progress_callback: callable) -> bool:
        """Enhancement with progress reporting"""
        
        try:
            progress_callback("Loading audio", 0.1)
            
            # Load audio
            audio, sample_rate = sf.read(input_path)
            
            if settings and settings.mono_conversion and audio.ndim > 1:
                audio = np.mean(audio, axis=1)
                
            progress_callback("Analyzing audio", 0.2)
            
            # Analyze
            analysis = self.analyzer.analyze_audio(audio, sample_rate)
            enhanced_audio = audio.copy()
            
            # Apply enhancements with progress reporting
            if settings:
                if settings.high_pass_filter:
                    progress_callback("Applying high-pass filter", 0.3)
                    enhanced_audio = self._apply_high_pass_filter(
                        enhanced_audio, sample_rate, settings.high_pass_cutoff
                    )
                    
                if settings.noise_reduction_enabled:
                    progress_callback("Reducing noise", 0.5)
                    if settings.stationary_noise_reduction:
                        enhanced_audio = self.noise_reducer.reduce_stationary_noise(
                            enhanced_audio, sample_rate, settings.noise_reduction_strength
                        )
                        
                    progress_callback("Reducing non-stationary noise", 0.6)
                    if settings.non_stationary_noise_reduction:
                        enhanced_audio = self.noise_reducer.reduce_non_stationary_noise(
                            enhanced_audio, sample_rate, settings.noise_reduction_strength * 0.8
                        )
                        
                if settings.speech_enhancement_enabled:
                    progress_callback("Enhancing speech", 0.7)
                    enhanced_audio = self.speech_enhancer.enhance_speech(
                        enhanced_audio, sample_rate, settings.speech_boost_db, settings.formant_enhancement
                    )
                    
                if settings.normalization_enabled:
                    progress_callback("Normalizing audio", 0.8)
                    enhanced_audio = self.normalizer.normalize_audio(
                        enhanced_audio, sample_rate, settings.target_lufs,
                        settings.peak_limiting, settings.dynamic_range_compression
                    )
                    
                if settings.low_pass_filter:
                    progress_callback("Applying low-pass filter", 0.9)
                    enhanced_audio = self._apply_low_pass_filter(
                        enhanced_audio, sample_rate, settings.low_pass_cutoff
                    )
                    
            progress_callback("Saving enhanced audio", 0.95)
            
            # Save
            sf.write(output_path, enhanced_audio, sample_rate)
            
            progress_callback("Complete", 1.0)
            return True
            
        except Exception as e:
            logger.error(f"Audio enhancement failed: {e}")
            return False

# Global instance
audio_enhancement_engine = AudioEnhancementEngine()

# Preset configurations
PRESET_CONFIGURATIONS = {
    "podcast": EnhancementSettings(
        noise_reduction_strength=0.7,
        speech_boost_db=2.5,
        target_lufs=-18.0,
        dynamic_range_compression=0.3
    ),
    "lecture": EnhancementSettings(
        noise_reduction_strength=0.8,
        speech_boost_db=3.5,
        target_lufs=-20.0,
        formant_enhancement=True
    ),
    "interview": EnhancementSettings(
        noise_reduction_strength=0.6,
        speech_boost_db=2.0,
        voice_activity_detection=True,
        dynamic_range_compression=0.4
    ),
    "meeting": EnhancementSettings(
        noise_reduction_strength=0.9,
        speech_boost_db=4.0,
        target_lufs=-16.0,
        high_pass_cutoff=100.0
    ),
    "music": EnhancementSettings(
        noise_reduction_enabled=False,
        speech_enhancement_enabled=False,
        normalization_enabled=True,
        target_lufs=-14.0,
        dynamic_range_compression=0.1
    )
}