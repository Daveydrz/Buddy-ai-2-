import ctypes
import os
import re
import json
import time
import queue
import threading
import tempfile
import concurrent.futures
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play
import numpy as np
import pvporcupine
import pyaudio
import requests
import sounddevice as sd
import websockets
import asyncio
import webrtcvad
from langdetect import detect, detect_langs
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.io.wavfile import write
from kokoro_onnx import Kokoro
import soundfile as sf
from scipy.signal import resample, resample_poly
import random
from textblob import TextBlob
from io import BytesIO
import difflib
from resemblyzer import VoiceEncoder
encoder = VoiceEncoder()
from pyaec import PyAec
import simpleaudio as sa
import datetime
from datetime import timezone, timedelta

last_tts_audio = None  # Global buffer to track Buddy's last spoken waveform
last_flavor = None 

# ✅ MEMORY PROTECTION
import gc
import psutil
import os

def check_memory_usage():
    """Monitor memory usage to prevent crashes"""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        if memory_mb > 500:  # If using more than 500MB
            print(f"[MEMORY] ⚠️ High usage: {memory_mb:.1f}MB - running cleanup")
            gc.collect()  # Force garbage collection
        return memory_mb
    except:
        return 0

# ========== CONFIG & PATHS ==========
WEBRTC_SAMPLE_RATE = 16000
WEBRTC_FRAME_SIZE = 160
WEBRTC_CHANNELS = 1
MIC_DEVICE_INDEX = 60
MIC_SAMPLE_RATE = 48000
CHIME_PATH = "chime.wav"
known_users_path = "known_users.json"
THEMES_PATH = "themes_memory"
LAST_USER_PATH = "last_user.json"
FASTER_WHISPER_WS = "ws://localhost:9090"
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
SERPAPI_ENDPOINT = "https://serpapi.com/search"
WEATHERAPI_KEY = os.environ.get("WEATHERAPI_KEY", "")
HOME_ASSISTANT_URL = os.environ.get("HOME_ASSISTANT_URL", "http://localhost:8123")
HOME_ASSISTANT_TOKEN = os.environ.get("HOME_ASSISTANT_TOKEN", "")
KOKORO_VOICES = {"pl": "af_heart", "en": "af_heart", "it": "if_sara"}
KOKORO_LANGS = {"pl": "pl", "en": "en-us", "it": "it"}
DEFAULT_LANG = "en"
FAST_MODE = False
DEBUG = True
DEBUG_MODE = False
BUDDY_BELIEFS_PATH = "buddy_beliefs.json"
LONG_TERM_MEMORY_PATH = "buddy_long_term_memory.json"
PERSONALITY_TRAITS_PATH = "buddy_personality_traits.json"
DYNAMIC_KNOWLEDGE_PATH = "buddy_dynamic_knowledge.json"

FORMAT_INT16 = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
microphone_device_index = MIC_DEVICE_INDEX

# ========== AEC REFERENCE BUFFER (Config) ==========
ref_audio_buffer = np.zeros(WEBRTC_SAMPLE_RATE * 2, dtype=np.int16)  # 2 seconds
ref_audio_lock = threading.Lock()
vad_thread_active = threading.Event()
playback_start_time = None

# ========== GLOBAL STATE ==========
aec_instance = PyAec(
    frame_size=160,
    sample_rate=16000
)
playback_start_time = None
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
os.makedirs(THEMES_PATH, exist_ok=True)
ref_audio_lock = threading.Lock()

# ========== UNIFIED AUDIO SYSTEM ==========
audio_queue = queue.Queue()  # Single audio queue
current_audio_playback = None
audio_worker_active = False
audio_lock = threading.Lock()

# ========== CORE STATE ==========
buddy_talking = threading.Event()
vad_triggered = threading.Event()
full_duplex_interrupt_flag = threading.Event()
vad_thread_active = threading.Event()

# ========== MEMORY & TRACKING ==========
LAST_FEW_BUDDY = []
RECENT_WHISPER = []
known_users = {}
active_speakers = {}
active_speaker_lock = threading.Lock()
session_emotion_mode = {}
spoken_chunks_cache = set()
vad_thread_running = False

# ========== AEC REFERENCE BUFFER ==========
ref_audio_buffer = np.zeros(16000 * 2, dtype=np.int16)  # 2 seconds at 16kHz
current_stream_id = 0
BYPASS_AEC = False

if os.path.exists(known_users_path):
    with open(known_users_path, "r", encoding="utf-8") as f:
        known_users = json.load(f)
if DEBUG:
    device = "cuda" if 'cuda' in os.environ.get('CUDA_VISIBLE_DEVICES', '') or hasattr(np, "cuda") else "cpu"
    print(f"[Buddy] Running on device: {device}")
    print("Embedding model loaded", flush=True)
    print("Kokoro loaded", flush=True)
    print("Main function entered!", flush=True)

# ========== AUDIO PROCESSING ==========
def update_reference_audio_realtime_precise(pcm, sr, playback_start_time):
    """COMPLETELY FIXED: Simple real-time AEC reference injection - 2025-06-30 07:04:16"""
    global ref_audio_buffer
    
    try:
        # ✅ CRITICAL: Only inject when Buddy is actively talking
        if not buddy_talking.is_set():
            if DEBUG:
                print("[AEC-REF] Buddy not talking - skipping reference injection")
            return
        
        # ✅ VALIDATION: Check playback_start_time is valid
        if playback_start_time is None:
            if DEBUG:
                print("[AEC-REF] No valid playback start time")
            return
        
        # Ensure 16kHz for AEC
        if sr != 16000:
            pcm_float = pcm.astype(np.float32) / 32768.0
            pcm_16k = resample_poly(pcm_float, 16000, sr)
            pcm_16k = (np.clip(pcm_16k, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            pcm_16k = pcm.copy()
        
        chunk_size = 160  # 10ms frames at 16kHz
        
        if DEBUG:
            print(f"[AEC-REF] Starting SIMPLE injection: {len(pcm_16k)} samples")
        
        # ✅ COMPLETELY SIMPLIFIED: No complex timing, just inject at natural pace
        frames_injected = 0
        
        for i in range(0, len(pcm_16k), chunk_size):
            # ✅ CRITICAL: Check for interruption first
            if full_duplex_interrupt_flag.is_set() or not buddy_talking.is_set():
                if DEBUG:
                    print(f"[AEC-REF] Interruption detected - stopping at frame {frames_injected}")
                break
            
            frame = pcm_16k[i:i+chunk_size]
            if len(frame) < chunk_size:
                frame = np.pad(frame, (0, chunk_size - len(frame)))
            
            # ✅ SIMPLE INJECTION: Just inject immediately without timing delays
            try:
                with ref_audio_lock:
                    ref_audio_buffer = np.roll(ref_audio_buffer, -chunk_size)
                    ref_audio_buffer[-chunk_size:] = frame
                    
                frames_injected += 1
                    
            except Exception as inject_err:
                if DEBUG:
                    print(f"[AEC-REF] Injection error at frame {i//chunk_size}: {inject_err}")
                continue
            
            # ✅ NATURAL PACING: Just wait 10ms between frames (natural rate)
            time.sleep(0.01)  # 10ms = natural frame rate
        
        if DEBUG:
            print(f"[AEC-REF] ✅ Simple injection complete: {frames_injected} frames")
            
    except Exception as e:
        if DEBUG:
            print(f"[AEC-REF] Simple injection error: {e}")


def _play_accumulated_audio_with_aec(audio_chunks):
    """FIXED: Enhanced playback with perfect AEC timing - 2025-06-30 06:59:14"""
    global current_audio_playback, playback_start_time
    
    if not audio_chunks:
        return
        
    try:
        # Combine chunks
        combined_audio = []
        target_sr = 16000
        
        for pcm, sr in audio_chunks:
            if sr == target_sr:
                combined_audio.append(pcm)
            else:
                pcm_float = pcm.astype(np.float32) / 32768.0
                resampled = resample_poly(pcm_float, target_sr, sr)
                combined_audio.append((np.clip(resampled, -1.0, 1.0) * 32767).astype(np.int16))
        
        if not combined_audio:
            return
            
        smooth_audio = np.concatenate(combined_audio)
        
        # Apply gentle fade to prevent clicks
        fade_samples = min(80, len(smooth_audio) // 20)
        if len(smooth_audio) > fade_samples * 2:
            fade_in = np.linspace(0, 1, fade_samples)
            smooth_audio[:fade_samples] = (smooth_audio[:fade_samples].astype(np.float32) * fade_in).astype(np.int16)
            
            fade_out = np.linspace(1, 0, fade_samples)
            smooth_audio[-fade_samples:] = (smooth_audio[-fade_samples:].astype(np.float32) * fade_out).astype(np.int16)
        
        with audio_lock:
            try:
                if DEBUG:
                    print(f"[Buddy][AEC-Playback] Playing {len(smooth_audio)} samples with AEC sync")

                # Set talking flag and start VAD
                if not buddy_talking.is_set():
                    buddy_talking.set()
                    if not vad_thread_active.is_set():
                        threading.Thread(target=background_vad_listener, daemon=True).start()

                # ✅ CRITICAL: Set EXACT playback start time with system latency compensation
                system_latency = 0.05  # Estimate 50ms system audio latency
                playback_start_time = time.time() + system_latency
                
                # ✅ IMPROVED: Start precise AEC reference injection in separate thread
                aec_thread = threading.Thread(
                    target=update_reference_audio_realtime_precise,
                    args=(smooth_audio, target_sr, playback_start_time),
                    daemon=True
                )
                aec_thread.start()

                # ✅ TIMING: Small delay to let AEC thread initialize
                time.sleep(0.02)

                if full_duplex_interrupt_flag.is_set():
                    if DEBUG:
                        print("[AEC-Playback] Interrupt detected before playback")
                    return

                # ✅ PRECISE: Start playback at the planned time
                actual_start_time = time.time()
                current_audio_playback = sa.play_buffer(smooth_audio.tobytes(), 1, 2, target_sr)
                
                # ✅ TIMING VERIFICATION
                if DEBUG:
                    timing_accuracy = (actual_start_time - playback_start_time) * 1000
                    print(f"[AEC-Playback] Timing accuracy: {timing_accuracy:.1f}ms")
                
                # Monitor for interruption with minimal delay
                while current_audio_playback.is_playing():
                    if full_duplex_interrupt_flag.is_set():
                        if DEBUG:
                            print("[AEC-Playback] Interrupt during playback - stopping")
                        current_audio_playback.stop()
                        break
                    time.sleep(0.002)  # 2ms for ultra-responsive interrupts
                
                current_audio_playback = None
                
                # ✅ CLEANUP: Wait for AEC thread to complete
                aec_thread.join(timeout=1.0)
                
                if DEBUG:
                    print("[Buddy][AEC-Playback] Finished with AEC sync")

            except Exception as inner_e:
                print(f"[Buddy][AEC-Playback ERROR] {inner_e}")
                if current_audio_playback:
                    try:
                        current_audio_playback.stop()
                    except:
                        pass
                    current_audio_playback = None
                
            finally:
                # ✅ RESET: Clear playback timing
                playback_start_time = None
                
                # Check if we should clear talking flag
                if audio_queue.empty():
                    buddy_talking.clear()
                        
    except Exception as e:
        # ✅ FIXED: Proper indentation for outer exception handler
        if DEBUG:
            print(f"[Buddy][AEC-Playback ERROR] Audio combination error: {e}")
        
        # ✅ ENHANCED: Better error recovery
        try:
            # Cleanup on error
            if current_audio_playback:
                current_audio_playback.stop()
                current_audio_playback = None
            
            playback_start_time = None
            buddy_talking.clear()
            
        except Exception as cleanup_err:
            if DEBUG:
                print(f"[AEC-Playback] Cleanup error: {cleanup_err}")

def apply_aec(mic_audio, bypass_aec=False):
    """OPTIMIZED AEC: Enhanced echo cancellation with improved reliability - 2025-06-30 06:55:50"""
    global ref_audio_buffer, buddy_talking

    if bypass_aec:
        if DEBUG:
            print("[AEC] Manual bypass requested")
        return mic_audio[:160] if len(mic_audio) >= 160 else mic_audio

    # ✅ INPUT VALIDATION: Ensure we have valid input
    if mic_audio is None or len(mic_audio) == 0:
        if DEBUG:
            print("[AEC] Empty mic input")
        return np.zeros(160, dtype=np.int16)

    # Convert mic to float32 [-1.0, 1.0]
    try:
        if isinstance(mic_audio, bytes):
            mic_np = np.frombuffer(mic_audio, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(mic_audio, np.ndarray):
            if mic_audio.dtype == np.int16:
                mic_np = mic_audio.astype(np.float32) / 32768.0
            else:
                mic_np = mic_audio.astype(np.float32)
        else:
            if DEBUG:
                print(f"[AEC] Unexpected input type: {type(mic_audio)}")
            return np.zeros(160, dtype=np.int16)
    except Exception as convert_err:
        if DEBUG:
            print(f"[AEC] Input conversion error: {convert_err}")
        return np.zeros(160, dtype=np.int16)
    
    mic_np = np.clip(mic_np, -1.0, 1.0)

    # ✅ FRAME SIZE: Ensure exactly 160 samples
    if len(mic_np) < 160:
        mic_np = np.pad(mic_np, (0, 160 - len(mic_np)))
        if DEBUG:
            print("[AEC] Mic frame padded due to short length.")
    elif len(mic_np) > 160:
        mic_np = mic_np[:160]  # Truncate if too long
        if DEBUG:
            print("[AEC] Mic frame truncated to 160 samples.")

    # ✅ CRITICAL: Only apply AEC when Buddy is actually talking
    if not buddy_talking.is_set():
        if DEBUG:
            print("[AEC] Buddy not talking - returning original (no echo to cancel)")
        return (mic_np * 32767).astype(np.int16)

    # ✅ ENHANCED: Get synchronized reference frame with better error handling
    try:
        with ref_audio_lock:
            # Check if we have valid reference buffer
            if ref_audio_buffer is None or len(ref_audio_buffer) < 160:
                if DEBUG:
                    print("[AEC] Insufficient reference audio")
                return (mic_np * 32767).astype(np.int16)
            
            # Get the most recent reference frame (latest echo)
            ref_frame = ref_audio_buffer[-160:].copy()
            
            # Convert to float32 and normalize
            if ref_frame.dtype == np.int16:
                ref_frame = ref_frame.astype(np.float32) / 32768.0
            else:
                ref_frame = ref_frame.astype(np.float32)
                
            ref_frame = np.clip(ref_frame, -1.0, 1.0)
            
            # Ensure exactly 160 samples
            if len(ref_frame) < 160:
                ref_frame = np.pad(ref_frame, (0, 160 - len(ref_frame)))
            elif len(ref_frame) > 160:
                ref_frame = ref_frame[:160]
                
    except Exception as ref_err:
        if DEBUG:
            print(f"[AEC] Reference frame error: {ref_err}")
        return (mic_np * 32767).astype(np.int16)

    # ✅ ENHANCED: Check if reference is significant enough
    ref_rms = np.sqrt(np.mean(ref_frame ** 2))
    mic_rms = np.sqrt(np.mean(mic_np ** 2))
    
    if ref_rms < 0.003:  # Very quiet reference
        if DEBUG:
            print(f"[AEC] Ref frame too quiet ({ref_rms:.4f}), skipping AEC.")
        return (mic_np * 32767).astype(np.int16)
    
    if mic_rms < 0.002:  # Very quiet microphone
        if DEBUG:
            print(f"[AEC] Mic frame too quiet ({mic_rms:.4f}), skipping AEC.")
        return (mic_np * 32767).astype(np.int16)

    # ✅ IMPROVED: Pre-AEC similarity analysis with better thresholds
    mic_frame = mic_np
    
    try:
        # Calculate normalized correlation for better similarity measure
        mic_norm = mic_frame / (np.linalg.norm(mic_frame) + 1e-8)
        ref_norm = ref_frame / (np.linalg.norm(ref_frame) + 1e-8)
        similarity_before = np.dot(mic_norm, ref_norm)
        
        # Additional check: cross-correlation peak
        cross_corr = np.correlate(mic_frame, ref_frame, mode='valid')
        max_cross_corr = np.max(np.abs(cross_corr)) if len(cross_corr) > 0 else 0
        
        if DEBUG:
            print(f"[AEC DIAG] Similarity: {similarity_before:.3f}, Cross-corr: {max_cross_corr:.3f}")
            print(f"[AEC DIAG] RMS - Mic: {mic_rms:.4f}, Ref: {ref_rms:.4f}")
        
    except Exception as sim_err:
        if DEBUG:
            print(f"[AEC] Similarity calculation error: {sim_err}")
        similarity_before = 0.0
        max_cross_corr = 0.0

    # ✅ SMART PROCESSING: Enhanced decision logic
    echo_threshold = 0.25  # Increased threshold for better detection
    
    if np.abs(similarity_before) < echo_threshold and max_cross_corr < 0.1:
        if DEBUG:
            print(f"[AEC] Low echo correlation ({similarity_before:.3f}) - likely real speech")
        # Apply very light filtering to preserve speech quality
        light_filtered = mic_frame * 0.98  # Only 2% reduction
        return (np.clip(light_filtered, -1.0, 1.0) * 32767).astype(np.int16)

    # ✅ ENHANCED: Apply full AEC processing with better error handling
    try:
        # Ensure AEC instance is properly initialized
        if not hasattr(aec_instance, 'set_ref'):
            if DEBUG:
                print("[AEC] AEC instance not properly initialized")
            # Fallback to simple filtering
            simple_filtered = mic_frame * 0.8
            return (np.clip(simple_filtered, -1.0, 1.0) * 32767).astype(np.int16)
        
        # Set reference and process
        aec_instance.set_ref(ref_frame.tolist())
        output = aec_instance.process_with_ref(mic_frame.tolist())
        
        if output is None or len(output) == 0:
            if DEBUG:
                print("[AEC] AEC returned empty output")
            # Fallback filtering
            fallback = mic_frame * 0.75
            return (np.clip(fallback, -1.0, 1.0) * 32767).astype(np.int16)
        
        output_np = np.array(output, dtype=np.float32)
        
        # Ensure output is properly sized
        if len(output_np) < 160:
            output_np = np.pad(output_np, (0, 160 - len(output_np)))
        elif len(output_np) > 160:
            output_np = output_np[:160]
        
        # ✅ ENHANCED: Post-AEC quality analysis
        try:
            output_norm = output_np / (np.linalg.norm(output_np) + 1e-8)
            similarity_after = np.dot(output_norm, ref_norm)
            
            rms_out = np.sqrt(np.mean(output_np ** 2))
            
            # Check for over-cancellation
            cancellation_ratio = rms_out / (mic_rms + 1e-8)
            
            if DEBUG:
                print(f"[AEC DIAG] Similarity after: {similarity_after:.3f}")
                print(f"[AEC DIAG] Cancellation ratio: {cancellation_ratio:.3f}")
                
        except Exception as post_err:
            if DEBUG:
                print(f"[AEC] Post-processing analysis error: {post_err}")
            similarity_after = similarity_before
            cancellation_ratio = 1.0

        # ✅ ENHANCED: Quality verification with multiple criteria
        aec_improved = (
            np.abs(similarity_after) < np.abs(similarity_before) * 0.85 and  # Reduced echo
            cancellation_ratio > 0.1 and  # Not over-cancelled
            cancellation_ratio < 2.0       # Not amplified
        )
        
        if aec_improved:
            if DEBUG:
                print(f"[AEC] ✅ Good cancellation: {similarity_before:.3f} → {similarity_after:.3f}")
            result = output_np
        else:
            if DEBUG:
                print(f"[AEC] ❌ Poor AEC quality, using adaptive filter")
            # Adaptive filtering based on echo strength
            filter_strength = min(0.4, np.abs(similarity_before) * 0.8)
            result = mic_frame * (1.0 - filter_strength)
        
        # ✅ FINAL: Apply gentle post-processing and return
        result = np.clip(result, -1.0, 1.0)
        
        # Apply very light smoothing to reduce artifacts
        if len(result) >= 3:
            smoothed = np.copy(result)
            smoothed[1:-1] = 0.7 * result[1:-1] + 0.15 * result[:-2] + 0.15 * result[2:]
            result = smoothed
        
        result_int16 = (result * 32767).astype(np.int16)
        
        if DEBUG:
            final_rms = np.sqrt(np.mean(result.astype(np.float32) ** 2))
            print(f"[AEC] Final RMS: {final_rms:.4f}")
        
        return result_int16
        
    except Exception as aec_err:
        if DEBUG:
            print(f"[AEC] Processing error: {aec_err}")
            import traceback
            traceback.print_exc()
        
        # ✅ ROBUST FALLBACK: Adaptive filtering based on detected echo
        try:
            if np.abs(similarity_before) > 0.5:  # Strong echo detected
                fallback_strength = 0.5  # 50% reduction
            elif np.abs(similarity_before) > 0.3:  # Moderate echo
                fallback_strength = 0.3  # 30% reduction
            else:  # Weak echo
                fallback_strength = 0.1  # 10% reduction
                
            fallback = mic_frame * (1.0 - fallback_strength)
            if DEBUG:
                print(f"[AEC] Applied fallback filter: {fallback_strength*100:.0f}% reduction")
            
            return (np.clip(fallback, -1.0, 1.0) * 32767).astype(np.int16)
            
        except Exception as fallback_err:
            if DEBUG:
                print(f"[AEC] Fallback error: {fallback_err}")
            # Ultimate fallback - return slightly attenuated original
            return (np.clip(mic_frame * 0.9, -1.0, 1.0) * 32767).astype(np.int16)

def _get_time_aligned_reference_frame():
    """Get precisely time-aligned reference audio frame"""
    global ref_audio_buffer, playback_start_time
    
    if playback_start_time is None:
        return ref_audio_buffer[:160].copy()
    
    # Calculate expected position in reference buffer based on timing
    current_time = time.time()
    elapsed_time = current_time - playback_start_time
    
    # Convert to samples (16kHz)
    sample_offset = int(elapsed_time * 16000)
    
    # Get frame from correct position in buffer
    if sample_offset < len(ref_audio_buffer) - 160:
        return ref_audio_buffer[sample_offset:sample_offset + 160].copy()
    else:
        # Fallback to latest frame
        return ref_audio_buffer[-160:].copy()

def _is_chime_audio(audio_frame):
    """Detect if audio frame contains chime sound"""
    try:
        # Chimes typically have specific frequency characteristics
        # High frequency content and specific patterns
        
        # Calculate frequency domain characteristics
        fft = np.fft.fft(audio_frame)
        freqs = np.fft.fftfreq(len(audio_frame), 1/16000)
        
        # Look for high frequency energy (chimes are typically high-pitched)
        high_freq_energy = np.sum(np.abs(fft[freqs > 2000])) / np.sum(np.abs(fft))
        
        # Chimes also have rapid onset
        energy_derivative = np.diff(np.abs(audio_frame))
        rapid_onset = np.max(energy_derivative) > 0.3
        
        is_chime = high_freq_energy > 0.6 and rapid_onset
        
        if is_chime and DEBUG:
            print(f"[AEC] 🔔 Chime detected: high_freq={high_freq_energy:.2f}, onset={rapid_onset}")
        
        return is_chime
        
    except Exception as e:
        if DEBUG:
            print(f"[AEC] Chime detection error: {e}")
        return False

def _process_chime_audio(mic_frame, ref_frame):
    """Special processing for chime audio to prevent false interrupts"""
    try:
        # For chimes, we want to preserve the original audio but mark it as processed
        # This prevents the VAD from treating it as speech
        
        # Apply gentle noise gate to reduce chime impact
        threshold = 0.1
        gated_frame = np.where(np.abs(mic_frame) > threshold, mic_frame * 0.3, mic_frame * 0.1)
        
        if DEBUG:
            print("[AEC] 🔔 Applied chime-specific processing")
        
        return (np.clip(gated_frame, -1.0, 1.0) * 32767).astype(np.int16)
        
    except Exception as e:
        if DEBUG:
            print(f"[AEC] Chime processing error: {e}")
        return (mic_frame * 32767).astype(np.int16)

def _calculate_audio_similarity(audio1, audio2):
    """Calculate similarity between two audio frames"""
    try:
        # Normalize both frames
        norm1 = np.linalg.norm(audio1) + 1e-8
        norm2 = np.linalg.norm(audio2) + 1e-8
        
        # Calculate cosine similarity
        similarity = np.dot(audio1, audio2) / (norm1 * norm2)
        return float(similarity)
        
    except Exception as e:
        if DEBUG:
            print(f"[AEC] Similarity calculation error: {e}")
        return 0.0

def _apply_light_filtering(audio_frame):
    """Apply light filtering for non-echo audio"""
    try:
        # Very gentle high-pass filter to remove low-frequency noise
        # This preserves speech while reducing rumble
        
        # Simple high-pass: subtract low-frequency component
        low_freq_component = np.convolve(audio_frame, np.ones(5)/5, mode='same')
        filtered = audio_frame - low_freq_component * 0.1
        
        return (np.clip(filtered, -1.0, 1.0) * 32767).astype(np.int16)
        
    except Exception as e:
        if DEBUG:
            print(f"[AEC] Light filtering error: {e}")
        return (audio_frame * 32767).astype(np.int16)

def _post_process_aec_output(aec_output, original_mic):
    """Post-process AEC output to improve quality"""
    try:
        # Prevent over-cancellation by blending with original
        if np.abs(aec_output).max() < 0.01:  # Too much cancellation
            blend_ratio = 0.3  # 30% original, 70% AEC
            result = aec_output * (1 - blend_ratio) + original_mic * blend_ratio
            if DEBUG:
                print("[AEC] Applied over-cancellation protection")
        else:
            result = aec_output
        
        # Apply gentle smoothing to reduce artifacts
        smoothed = np.convolve(result, np.array([0.2, 0.6, 0.2]), mode='same')
        
        return smoothed
        
    except Exception as e:
        if DEBUG:
            print(f"[AEC] Post-processing error: {e}")
        return aec_output

def is_echo(text):
    """Check if text is likely an echo of Buddy's recent speech"""
    if not text or len(text.strip()) < 3:
        return False

    cleaned = re.sub(r'[^\w\s]', '', text.strip().lower())
    if not cleaned or len(cleaned.split()) < 2:
        return False

    # Check against recent Buddy responses
    for prev in LAST_FEW_BUDDY[-3:]:
        if not prev:
            continue
            
        prev_clean = re.sub(r'[^\w\s]', '', prev.strip().lower())
        if not prev_clean:
            continue
            
        # Calculate similarity
        ratio = difflib.SequenceMatcher(None, cleaned, prev_clean).ratio()
        word_diff = abs(len(cleaned.split()) - len(prev_clean.split()))

        if ratio > 0.87 and word_diff <= 4:
            if DEBUG:
                print(f"[Buddy] Skipping echo:\n→ new:  {cleaned}\n→ prev: {prev_clean}\n→ sim={ratio:.2f}, word_diff={word_diff}")
            return True

    # Check for common Buddy phrases that shouldn't come from user
    buddy_phrases = [
        "i'm just a program", "i'm buddy", "digital fella", 
        "no bugs in my code", "well-oiled machine", "considering i'm just",
        "as an ai", "i'm an ai", "i don't have", "i can't feel",
        "sorry, i'm having trouble", "i'm thinking", "give me a moment"
    ]
    
    for phrase in buddy_phrases:
        if phrase in cleaned:
            if DEBUG:
                print(f"[Buddy] Detected Buddy phrase in user input: '{phrase}'")
            return True

    return False

from numpy.linalg import norm
import numpy as np

def is_echo_of_last_tts(mic_audio, last_tts_audio_param=None, threshold=0.65):
    """ENHANCED: Intelligent echo detection with context awareness - 2025-06-30 06:06:45"""
    global last_tts_audio
    
    # Use parameter or global TTS audio
    tts_audio = last_tts_audio_param or last_tts_audio
    
    if tts_audio is None or len(tts_audio) < 160:
        return False
    
    if len(mic_audio) < 160:
        return False
    
    try:
        # ✅ TIMING CHECK: Only check for echo within reasonable time window
        if hasattr(tts_audio, 'timestamp'):
            time_since_tts = time.time() - tts_audio.timestamp
            if time_since_tts > 2.0:  # More than 2 seconds ago
                return False
        
        # ✅ SIZE MATCHING: Match array sizes for comparison
        min_len = min(len(mic_audio), len(tts_audio))
        mic_segment = mic_audio[:min_len]
        tts_segment = tts_audio[:min_len]
        
        # ✅ NORMALIZATION: Convert to float32 and normalize
        mic_float = mic_segment.astype(np.float32)
        tts_float = tts_segment.astype(np.float32)
        
        # Normalize amplitude
        mic_max = np.max(np.abs(mic_float)) + 1e-8
        tts_max = np.max(np.abs(tts_float)) + 1e-8
        mic_norm = mic_float / mic_max
        tts_norm = tts_float / tts_max
        
        # ✅ MULTIPLE SIMILARITY METRICS
        
        # 1. Cross-correlation (time-shifted similarity)
        correlation = np.correlate(mic_norm, tts_norm, mode='valid')
        max_correlation = np.max(np.abs(correlation)) if len(correlation) > 0 else 0
        
        # 2. Cosine similarity
        dot_product = np.dot(mic_norm, tts_norm)
        norms = np.linalg.norm(mic_norm) * np.linalg.norm(tts_norm) + 1e-8
        cosine_sim = dot_product / norms
        
        # 3. Spectral similarity (frequency domain)
        mic_fft = np.fft.fft(mic_norm)
        tts_fft = np.fft.fft(tts_norm)
        spectral_sim = np.abs(np.dot(mic_fft.conj(), tts_fft)) / (np.linalg.norm(mic_fft) * np.linalg.norm(tts_fft) + 1e-8)
        
        # ✅ COMBINED SCORE: Weight different similarity measures
        combined_score = (
            max_correlation * 0.4 +
            np.abs(cosine_sim) * 0.4 +
            np.abs(spectral_sim) * 0.2
        )
        
        is_echo = combined_score > threshold
        
        if DEBUG:
            print(f"[Echo] Correlation: {max_correlation:.3f}, Cosine: {cosine_sim:.3f}, Spectral: {spectral_sim:.3f}")
            print(f"[Echo] Combined score: {combined_score:.3f}, Threshold: {threshold:.3f}, Is Echo: {is_echo}")
        
        return is_echo
        
    except Exception as e:
        if DEBUG:
            print(f"[Echo] Detection error: {e}")
        return False

def is_noise_or_gibberish(text):
    """
    Reject input if it's likely noise, gibberish, or too short.
    """
    if not text or len(text.strip()) < 2:
        return True
    words = text.strip().split()
    avg_len = sum(len(w) for w in words) / len(words) if words else 0
    # Reject if it's a single short word or strange characters
    if len(words) < 2 and avg_len < 4:
        return True
    if re.search(r'[^a-zA-Z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻ ]', text):
        return False  # symbols are allowed
    return False

def start_buddy_with_interrupt():
    """Start Buddy with parallel interrupt system"""
    print("[Buddy] 🚀 Starting Buddy with parallel interrupt system...")
    
    # Start the parallel interrupt detector immediately
    threading.Thread(target=start_parallel_detector_safe, daemon=True).start()
    
    print("[Buddy] ✅ Parallel interrupt system active!")
    # Continue with your normal Buddy startup...


def apply_aec(mic_audio, bypass_aec=False):
    """FULL-DUPLEX AEC: Proper echo cancellation with intelligent processing - 2025-06-30 06:27:49"""
    global ref_audio_buffer, buddy_talking

    if bypass_aec:
        if DEBUG:
            print("[AEC] Manual bypass requested")
        return mic_audio[:160] if len(mic_audio) >= 160 else mic_audio

    # Convert mic to float32 [-1.0, 1.0]
    mic_np = np.frombuffer(mic_audio, dtype=np.int16).astype(np.float32) / 32768.0
    mic_np = np.clip(mic_np, -1.0, 1.0)

    if len(mic_np) < 160:
        mic_np = np.pad(mic_np, (0, 160 - len(mic_np)))

    # ✅ CRITICAL: Only apply AEC when Buddy is actually talking AND we have valid reference
    if not buddy_talking.is_set():
        # When Buddy is not talking, return original audio (no echo to cancel)
        return (mic_np[:160] * 32767).astype(np.int16)

    # === Get time-synchronized reference frame ===
    with ref_audio_lock:
        # ✅ IMPROVED: Get the most recent reference frame
        ref_frame = ref_audio_buffer[-160:].astype(np.float32) / 32768.0
        if len(ref_frame) < 160:
            ref_frame = np.pad(ref_frame, (0, 160 - len(ref_frame)))
        ref_frame = np.clip(ref_frame, -1.0, 1.0)

    # ✅ ENHANCED: Skip AEC if reference is too quiet (no significant playback)
    ref_rms = np.sqrt(np.mean(ref_frame ** 2))
    if ref_rms < 0.005:  # Very low threshold
        if DEBUG:
            print(f"[AEC] Ref too quiet ({ref_rms:.4f}), returning original")
        return (mic_np[:160] * 32767).astype(np.int16)

    # ✅ PRE-AEC ANALYSIS: Check if processing is needed
    mic_frame = mic_np[:160]
    similarity_before = np.dot(mic_frame, ref_frame) / (np.linalg.norm(mic_frame) * np.linalg.norm(ref_frame) + 1e-6)
    
    if DEBUG:
        print(f"[AEC] Mic-Ref similarity: {similarity_before:.3f}, Ref RMS: {ref_rms:.4f}")

    # ✅ SMART PROCESSING: Only apply heavy AEC if significant echo detected
    if np.abs(similarity_before) < 0.2:  # Low similarity = probably not echo
        # Apply light filtering only
        light_filtered = mic_frame * 0.95  # 5% reduction
        return (np.clip(light_filtered, -1.0, 1.0) * 32767).astype(np.int16)

    # ✅ FULL AEC PROCESSING: Apply when echo is likely
    try:
        aec_instance.set_ref(ref_frame.tolist())
        output = aec_instance.process_with_ref(mic_frame.tolist())
        output_np = np.array(output, dtype=np.float32)
        
        # ✅ POST-AEC VERIFICATION: Ensure AEC improved the signal
        similarity_after = np.dot(output_np, ref_frame) / (np.linalg.norm(output_np) * np.linalg.norm(ref_frame) + 1e-6)
        
        # ✅ QUALITY CHECK: Use AEC result only if it actually helped
        if np.abs(similarity_after) < np.abs(similarity_before):
            if DEBUG:
                print(f"[AEC] ✅ Good cancellation: {similarity_before:.3f} → {similarity_after:.3f}")
            result = output_np
        else:
            if DEBUG:
                print(f"[AEC] ❌ Poor cancellation, using light filter instead")
            result = mic_frame * 0.8  # 20% reduction as fallback
        
        return (np.clip(result, -1.0, 1.0) * 32767).astype(np.int16)
        
    except Exception as aec_err:
        if DEBUG:
            print(f"[AEC] Processing error: {aec_err}, using fallback")
        # Fallback: light filtering
        return (np.clip(mic_frame * 0.8, -1.0, 1.0) * 32767).astype(np.int16)

def update_reference_audio_realtime(pcm, sr):
    """FIXED: Ultra-simple reference injection wrapper - 2025-06-30 07:04:16"""
    global ref_audio_buffer, playback_start_time
    
    try:
        # ✅ CRITICAL: Only inject when Buddy is actively talking
        if not buddy_talking.is_set():
            if DEBUG:
                print("[AEC-REF] Buddy not talking - skipping reference injection")
            return
        
        # ✅ SIMPLE: Use current time if no playback start time
        if playback_start_time is None:
            playback_start_time = time.time()
        
        # Call the simplified injection function
        update_reference_audio_realtime_precise(pcm, sr, playback_start_time)
        
    except Exception as e:
        if DEBUG:
            print(f"[AEC-REF] Reference injection error: {e}")

def simple_reference_injector(pcm, sr):
    """ALTERNATIVE: Even simpler reference injection without any timing complexity"""
    global ref_audio_buffer
    
    try:
        if not buddy_talking.is_set():
            return
        
        # Convert to 16kHz if needed
        if sr != 16000:
            pcm_float = pcm.astype(np.float32) / 32768.0
            pcm_16k = resample_poly(pcm_float, 16000, sr)
            pcm_16k = (np.clip(pcm_16k, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            pcm_16k = pcm.copy()
        
        # ✅ ULTRA-SIMPLE: Just inject entire audio at once in chunks
        chunk_size = 160
        for i in range(0, len(pcm_16k), chunk_size):
            if full_duplex_interrupt_flag.is_set():
                break
                
            frame = pcm_16k[i:i+chunk_size]
            if len(frame) < chunk_size:
                frame = np.pad(frame, (0, chunk_size - len(frame)))
            
            with ref_audio_lock:
                ref_audio_buffer = np.roll(ref_audio_buffer, -chunk_size)
                ref_audio_buffer[-chunk_size:] = frame
            
            # Small delay to prevent CPU overload
            time.sleep(0.005)  # 5ms delay
        
        if DEBUG:
            print(f"[AEC-REF] ✅ Ultra-simple injection: {len(pcm_16k)} samples")
            
    except Exception as e:
        if DEBUG:
            print(f"[AEC-REF] Ultra-simple injection error: {e}")

def _play_accumulated_audio_with_aec(audio_chunks):
    """FIXED: Enhanced playback with SIMPLE AEC timing - 2025-06-30 07:04:16"""
    global current_audio_playback, playback_start_time
    
    if not audio_chunks:
        return
        
    try:
        # Combine chunks
        combined_audio = []
        target_sr = 16000
        
        for pcm, sr in audio_chunks:
            if sr == target_sr:
                combined_audio.append(pcm)
            else:
                pcm_float = pcm.astype(np.float32) / 32768.0
                resampled = resample_poly(pcm_float, target_sr, sr)
                combined_audio.append((np.clip(resampled, -1.0, 1.0) * 32767).astype(np.int16))
        
        if not combined_audio:
            return
            
        smooth_audio = np.concatenate(combined_audio)
        
        # Apply gentle fade to prevent clicks
        fade_samples = min(80, len(smooth_audio) // 20)
        if len(smooth_audio) > fade_samples * 2:
            fade_in = np.linspace(0, 1, fade_samples)
            smooth_audio[:fade_samples] = (smooth_audio[:fade_samples].astype(np.float32) * fade_in).astype(np.int16)
            
            fade_out = np.linspace(1, 0, fade_samples)
            smooth_audio[-fade_samples:] = (smooth_audio[-fade_samples:].astype(np.float32) * fade_out).astype(np.int16)
        
        with audio_lock:
            try:
                if DEBUG:
                    print(f"[Buddy][AEC-Playback] Playing {len(smooth_audio)} samples with SIMPLE AEC")

                # Set talking flag and start VAD
                if not buddy_talking.is_set():
                    buddy_talking.set()
                    if not vad_thread_active.is_set():
                        threading.Thread(target=background_vad_listener, daemon=True).start()

                # ✅ SIMPLE: Set playback start time to NOW (no prediction)
                playback_start_time = time.time()
                
                # ✅ SIMPLE: Start AEC reference injection in background (no complex timing)
                aec_thread = threading.Thread(
                    target=simple_reference_injector,
                    args=(smooth_audio, target_sr),
                    daemon=True
                )
                aec_thread.start()

                if full_duplex_interrupt_flag.is_set():
                    if DEBUG:
                        print("[AEC-Playback] Interrupt detected before playback")
                    return

                # ✅ SIMPLE: Start playback immediately
                current_audio_playback = sa.play_buffer(smooth_audio.tobytes(), 1, 2, target_sr)
                
                if DEBUG:
                    print(f"[AEC-Playback] ✅ Playback started with simple AEC")
                
                # Monitor for interruption with minimal delay
                while current_audio_playback.is_playing():
                    if full_duplex_interrupt_flag.is_set():
                        if DEBUG:
                            print("[AEC-Playback] Interrupt during playback - stopping")
                        current_audio_playback.stop()
                        break
                    time.sleep(0.002)  # 2ms for ultra-responsive interrupts
                
                current_audio_playback = None
                
                # ✅ CLEANUP: Wait for AEC thread to complete
                aec_thread.join(timeout=0.5)  # Shorter timeout
                
                if DEBUG:
                    print("[Buddy][AEC-Playback] Finished with simple AEC")

            except Exception as e:
                print(f"[Buddy][AEC-Playback ERROR] {e}")
                if current_audio_playback:
                    try:
                        current_audio_playback.stop()
                    except:
                        pass
                    current_audio_playback = None
                
            finally:
                # ✅ RESET: Clear playback timing
                playback_start_time = None
                
                # Check if we should clear talking flag
                if audio_queue.empty():
                    buddy_talking.clear()
                        
    except Exception as e:
        if DEBUG:
            print(f"[Buddy][AEC-Playback ERROR] Audio combination error: {e}")
        
        # ✅ ENHANCED: Better error recovery
        try:
            if current_audio_playback:
                current_audio_playback.stop()
                current_audio_playback = None
            
            playback_start_time = None
            buddy_talking.clear()
            
        except Exception as cleanup_err:
            if DEBUG:
                print(f"[AEC-Playback] Cleanup error: {cleanup_err}")

# ========== BACKGROUND VAD LISTENER (FULL-DUPLEX) ==========
vad_thread_active = threading.Event()

def background_vad_listener():
    """COMPLETELY FIXED VAD: Proper sample rate and AEC integration - 2025-06-30 07:03:07"""
    global vad_thread_active
    
    print("[Buddy][VAD] 🎧 Starting FIXED AEC-filtered monitoring...")
    vad_thread_active.set()
    
    p = pyaudio.PyAudio()
    stream = None
    
    try:
        # ✅ CRITICAL FIX: Ensure device and sample rate compatibility
        try:
            # First try with specified device
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=microphone_device_index if microphone_device_index != 60 else None,
                frames_per_buffer=1024,
                stream_callback=None
            )
            print(f"[Buddy][VAD] ✅ Stream opened with device {microphone_device_index}")
            
        except Exception as device_err:
            print(f"[Buddy][VAD] ⚠️ Device {microphone_device_index} failed: {device_err}")
            # Fallback to default device
            try:
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    input_device_index=None,  # Use system default
                    frames_per_buffer=1024
                )
                print("[Buddy][VAD] ✅ Using default audio device")
            except Exception as default_err:
                print(f"[Buddy][VAD] ❌ Default device also failed: {default_err}")
                # Last resort: try different sample rate
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,  # Try standard rate
                    input=True,
                    frames_per_buffer=2048
                )
                print("[Buddy][VAD] ⚠️ Using 44.1kHz fallback rate")
        
        print("[Buddy][VAD] 🎧 AEC-filtered monitoring ACTIVE!")
        
        # ✅ ADAPTIVE THRESHOLDS: Much higher when Buddy is talking
        base_threshold = 1200  # Increased base threshold
        buddy_talking_multiplier = 15.0  # Even higher multiplier!
        
        consecutive_detections = 0
        required_detections = 6  # Increased for better reliability
        last_interrupt_time = 0
        
        # ✅ SIMPLIFIED BASELINE: Quick environment assessment
        baseline_samples = []
        print("[Buddy][VAD] 📊 Learning baseline noise...")
        
        for attempt in range(5):
            try:
                audio_data = stream.read(1024, exception_on_overflow=False)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                
                # ✅ SIMPLIFIED: Just use raw RMS for baseline
                baseline_rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
                baseline_samples.append(baseline_rms)
                time.sleep(0.1)
                
            except Exception as baseline_err:
                if DEBUG:
                    print(f"[VAD] Baseline attempt {attempt+1} error: {baseline_err}")
                continue
        
        if baseline_samples:
            learned_baseline = np.mean(baseline_samples)
            print(f"[Buddy][VAD] 📊 Learned baseline: {learned_baseline:.0f}")
            base_threshold = max(base_threshold, learned_baseline * 4)  # 4x baseline
            print(f"[Buddy][VAD] 📊 Adjusted base threshold: {base_threshold:.0f}")
        
        # ✅ MAIN MONITORING LOOP
        while vad_thread_active.is_set() and buddy_talking.is_set():
            try:
                current_time = time.time()
                
                # ✅ COOLDOWN: Prevent rapid re-interrupts
                if current_time - last_interrupt_time < 4.0:  # 4 second cooldown
                    time.sleep(0.1)
                    consecutive_detections = 0
                    continue
                
                # Read raw audio data
                audio_data = stream.read(1024, exception_on_overflow=False)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                
                # ✅ SIMPLIFIED AEC: Process only if we have enough samples
                processed_audio = None
                try:
                    if len(audio_np) >= 160:
                        # Take first 160 samples for AEC processing
                        chunk = audio_np[:160]
                        aec_result = apply_aec(chunk.tobytes(), bypass_aec=False)
                        
                        if aec_result is not None and len(aec_result) > 0:
                            # Use AEC result for the first part, raw for the rest
                            if len(audio_np) > 160:
                                remaining = audio_np[160:]
                                processed_audio = np.concatenate([aec_result, remaining])
                            else:
                                processed_audio = aec_result
                        else:
                            processed_audio = audio_np
                    else:
                        processed_audio = audio_np
                        
                except Exception as aec_err:
                    if DEBUG:
                        print(f"[VAD] Simplified AEC error: {aec_err}")
                    processed_audio = audio_np
                
                # Convert to float for processing
                if processed_audio is not None:
                    audio_float = processed_audio.astype(np.float32)
                else:
                    audio_float = audio_np.astype(np.float32)
                
                # ✅ MULTI-METRIC ANALYSIS
                rms_volume = np.sqrt(np.mean(audio_float ** 2))
                peak_volume = np.max(np.abs(audio_float))
                
                # ✅ SIMPLIFIED SPECTRAL ANALYSIS
                speech_ratio = 0.5  # Default
                try:
                    if len(audio_float) >= 64:  # Minimum for FFT
                        fft = np.fft.fft(audio_float[:512])  # Use first 512 samples
                        freqs = np.fft.fftfreq(512, 1/16000)
                        
                        # Human speech energy (200Hz - 3kHz)
                        speech_mask = (freqs >= 200) & (freqs <= 3000)
                        speech_energy = np.sum(np.abs(fft[speech_mask]))
                        total_energy = np.sum(np.abs(fft)) + 1e-8
                        speech_ratio = speech_energy / total_energy
                        
                except Exception as fft_err:
                    if DEBUG:
                        print(f"[VAD] FFT error: {fft_err}")
                    speech_ratio = 0.5
                
                # ✅ DYNAMIC THRESHOLD: Much higher when Buddy is talking
                is_buddy_talking = buddy_talking.is_set()
                current_threshold = base_threshold * (buddy_talking_multiplier if is_buddy_talking else 1.0)
                
                # ✅ STRICT DETECTION: All criteria must be met
                volume_trigger = rms_volume > current_threshold
                peak_trigger = peak_volume > current_threshold * 2.0  # Even higher peak requirement
                speech_trigger = speech_ratio > 0.3  # Higher speech requirement
                
                is_potential_interrupt = volume_trigger and peak_trigger and speech_trigger
                
                if is_potential_interrupt:
                    consecutive_detections += 1
                    
                    if consecutive_detections == 1:
                        print(f"[Buddy][VAD] 🔊 Potential interrupt: RMS:{rms_volume:.0f} PEAK:{peak_volume:.0f} SPEECH:{speech_ratio:.2f}")
                        print(f"[Buddy][VAD] 🎯 Thresholds: RMS>{current_threshold:.0f} PEAK>{current_threshold*2:.0f} SPEECH>0.3")
                    
                    # ✅ CONFIRMATION REQUIRED: Need multiple consecutive detections
                    if consecutive_detections >= required_detections:
                        # ✅ FINAL VERIFICATION: Double-check with fresh sample
                        try:
                            verify_data = stream.read(1024, exception_on_overflow=False)
                            verify_np = np.frombuffer(verify_data, dtype=np.int16)
                            verify_rms = np.sqrt(np.mean(verify_np.astype(np.float32) ** 2))
                            
                            # High verification threshold (80% of main threshold)
                            verify_threshold = current_threshold * 0.8
                            
                            if verify_rms > verify_threshold:
                                print(f"[Buddy][VAD] 🚨 CONFIRMED USER INTERRUPT! RMS1:{rms_volume:.0f} RMS2:{verify_rms:.0f}")
                                
                                last_interrupt_time = current_time
                                _execute_vad_interrupt()
                                break
                            else:
                                if DEBUG:
                                    print(f"[Buddy][VAD] ❌ Verification failed: {verify_rms:.0f} < {verify_threshold:.0f}")
                                consecutive_detections = 0
                        except Exception as verify_err:
                            if DEBUG:
                                print(f"[VAD] Verification error: {verify_err}")
                            consecutive_detections = 0
                else:
                    # ✅ GRADUAL RESET: Don't reset immediately
                    if consecutive_detections > 0:
                        consecutive_detections = max(0, consecutive_detections - 1)
                        if DEBUG and consecutive_detections == 0:
                            print(f"[Buddy][VAD] 🤖 False alarm cleared")
                
                # ✅ EFFICIENT PROCESSING: Small delay
                time.sleep(0.01)  # 10ms for responsive detection
                    
            except Exception as e:
                if DEBUG:
                    print(f"[Buddy][VAD] Processing error: {e}")
                consecutive_detections = 0
                time.sleep(0.05)
                continue
                
    except Exception as e:
        print(f"[Buddy][VAD] ❌ CRITICAL Stream error: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        
        # ✅ FALLBACK: Try to restart VAD with different settings
        print("[Buddy][VAD] 🔄 Attempting restart with fallback settings...")
        try:
            time.sleep(1)
            # Recursive restart with simpler settings
            threading.Thread(target=simplified_vad_fallback, daemon=True).start()
        except Exception as restart_err:
            print(f"[Buddy][VAD] Restart failed: {restart_err}")
            
    finally:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass
        try:
            p.terminate()
        except:
            pass
        vad_thread_active.clear()
        print("[Buddy][VAD] 🔇 AEC-filtered monitoring STOPPED")


def simplified_vad_fallback():
    """Simplified VAD fallback without AEC for emergency use"""
    global vad_thread_active
    
    print("[Buddy][VAD-FALLBACK] 🆘 Starting simplified VAD...")
    vad_thread_active.set()
    
    p = pyaudio.PyAudio()
    stream = None
    
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        print("[Buddy][VAD-FALLBACK] ✅ Simplified monitoring active")
        
        last_interrupt_time = 0
        consecutive_loud = 0
        
        while vad_thread_active.is_set() and buddy_talking.is_set():
            try:
                current_time = time.time()
                
                if current_time - last_interrupt_time < 5.0:
                    time.sleep(0.2)
                    continue
                
                audio_data = stream.read(1024, exception_on_overflow=False)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
                
                # VERY high threshold since no AEC
                if rms > 5000:  # Much higher threshold
                    consecutive_loud += 1
                    if consecutive_loud >= 8:  # Need 8 consecutive
                        print(f"[Buddy][VAD-FALLBACK] 🚨 EMERGENCY INTERRUPT: {rms:.0f}")
                        manual_interrupt_buddy()
                        last_interrupt_time = current_time
                        consecutive_loud = 0
                else:
                    consecutive_loud = 0
                
                time.sleep(0.05)
                
            except Exception as e:
                if DEBUG:
                    print(f"[VAD-FALLBACK] Error: {e}")
                time.sleep(0.1)
                
    except Exception as e:
        print(f"[VAD-FALLBACK] Failed: {e}")
    finally:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass
        try:
            p.terminate()
        except:
            pass
        vad_thread_active.clear()
        print("[Buddy][VAD-FALLBACK] 🔇 Simplified monitoring stopped")

def _execute_vad_interrupt():
    """Enhanced VAD interrupt execution - 2025-06-30 06:33:44"""
    global current_audio_playback
    
    print("[Buddy][VAD] 🛑 FORCING IMMEDIATE STOP!")
    
    # Set interrupt flags immediately
    full_duplex_interrupt_flag.set()
    vad_triggered.set()
    
    # Stop playback with proper error handling
    try:
        with audio_lock:
            if current_audio_playback and current_audio_playback.is_playing():
                current_audio_playback.stop()
                print("[Buddy][VAD] ✅ Audio playback stopped")
            current_audio_playback = None
    except Exception as stop_err:
        if DEBUG:
            print(f"[Buddy][VAD] Stop error: {stop_err}")
        # Force cleanup even on error
        current_audio_playback = None
    
    # Clear flags and exit VAD
    vad_thread_active.clear()
    buddy_talking.clear()
    
    print("[Buddy][VAD] 🔇 AEC INTERRUPT COMPLETE - MONITORING STOPPED")

def _execute_interrupt():
    """Enhanced interrupt execution with better error handling - 2025-06-30 06:21:23"""
    global current_audio_playback, playback_start_time
    
    print("[Buddy][VAD] 🛑 FORCING IMMEDIATE STOP!")
    
    # ✅ Set interrupt flag immediately
    full_duplex_interrupt_flag.set()
    vad_triggered.set()
    
    # ✅ Stop playback with enhanced error handling
    try:
        with audio_lock:
            if current_audio_playback and current_audio_playback.is_playing():
                current_audio_playback.stop()
                print("[Buddy][VAD] ✅ Playback stopped successfully")
            current_audio_playback = None
            playback_start_time = None
    except Exception as stop_err:
        print(f"[Buddy][VAD] Stop error: {stop_err}")
        # Force cleanup
        current_audio_playback = None
        playback_start_time = None
    
    # ✅ Clear flags and exit VAD
    vad_thread_active.clear()
    buddy_talking.clear()
    
    # ✅ ENHANCED: Clear pending audio to prevent continuation
    cleared = 0
    while not audio_queue.empty():
        try:
            item = audio_queue.get_nowait()
            if item is not None:
                cleared += 1
            audio_queue.task_done()
        except queue.Empty:
            break
    
    if cleared > 0:
        print(f"[Buddy][VAD] 🗑️ Cleared {cleared} pending audio chunks")
    
    print("[Buddy][VAD] 🔇 AEC INTERRUPT COMPLETE - MONITORING STOPPED")


# ========== PROFESSIONAL VOICE IDENTIFICATION ==========

class ProfessionalVoiceID:
    """Advanced voice identification with seamless user switching - 2025-07-01 08:33:23"""
    
    def __init__(self):
        self.confidence_threshold = 0.65
        self.learning_rate = 0.15
        self.min_audio_length = 8000  # 0.5 seconds at 16kHz
        
    def identify_speaker_professional(self, audio_chunk):
        """PROFESSIONAL: Fast, accurate speaker identification"""
        try:
            if len(audio_chunk) < self.min_audio_length:
                if DEBUG:
                    print(f"[VoiceID] Audio too short: {len(audio_chunk)} samples")
                return None, 0.0
            
            # Generate voice embedding
            if audio_chunk.dtype != np.float32:
                audio_float = audio_chunk.astype(np.float32) / 32768.0
            else:
                audio_float = audio_chunk
                
            current_embedding = generate_embedding_from_audio(audio_float)
            if current_embedding is None or len(current_embedding) != 256:
                if DEBUG:
                    print("[VoiceID] Invalid voice embedding generated")
                return None, 0.0
            
            # Skip if no known users
            if not known_users:
                if DEBUG:
                    print("[VoiceID] No known users in database")
                return None, 0.0
            
            # Compare with all known voice embeddings
            best_match = None
            best_score = 0.0
            all_scores = {}
            
            for name, stored_embedding in known_users.items():
                try:
                    # Only compare voice embeddings (256 dimensions)
                    if isinstance(stored_embedding, list) and len(stored_embedding) == 256:
                        similarity = cosine_similarity([current_embedding], [stored_embedding])[0][0]
                        all_scores[name] = similarity
                        
                        if similarity > best_score:
                            best_match = name
                            best_score = similarity
                            
                except Exception as comp_err:
                    if DEBUG:
                        print(f"[VoiceID] Comparison error for {name}: {comp_err}")
                    continue
            
            # Debug output
            if DEBUG:
                print(f"[VoiceID] Voice comparison results:")
                for name, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                    print(f"[VoiceID]   {name}: {score:.3f}")
            
            # Return result if confidence meets threshold
            if best_match and best_score >= self.confidence_threshold:
                if DEBUG:
                    print(f"[VoiceID] ✅ RECOGNIZED: {best_match} (confidence: {best_score:.3f})")
                return best_match, best_score
            else:
                if DEBUG:
                    print(f"[VoiceID] ❌ No confident match (best: {best_match}, score: {best_score:.3f})")
                return None, best_score
                
        except Exception as e:
            if DEBUG:
                print(f"[VoiceID] Professional identification error: {e}")
            return None, 0.0
    
    def enhance_voice_profile(self, username, audio_chunk):
        """Enhance existing voice profile with new sample"""
        try:
            if username not in known_users:
                return False
                
            # Generate new embedding
            if audio_chunk.dtype != np.float32:
                audio_float = audio_chunk.astype(np.float32) / 32768.0
            else:
                audio_float = audio_chunk
                
            new_embedding = generate_embedding_from_audio(audio_float)
            if new_embedding is None or len(new_embedding) != 256:
                return False
            
            # Blend with existing embedding (adaptive learning)
            old_embedding = known_users[username]
            if isinstance(old_embedding, list) and len(old_embedding) == 256:
                # Weighted average: 85% old, 15% new
                blended = [(old * (1 - self.learning_rate) + new * self.learning_rate) 
                          for old, new in zip(old_embedding, new_embedding)]
                known_users[username] = blended
                
                if DEBUG:
                    print(f"[VoiceID] 🔄 Enhanced voice profile for {username}")
                
                # Save updated database
                save_voice_database()
                return True
                
        except Exception as e:
            if DEBUG:
                print(f"[VoiceID] Profile enhancement error: {e}")
        
        return False
    
    def register_new_speaker(self, audio_chunk, suggested_name=None):
        """Register a completely new speaker"""
        try:
            # Generate unique username if none suggested
            if not suggested_name:
                timestamp = int(time.time())
                suggested_name = f"User_{timestamp}"
            
            # Generate voice embedding
            if audio_chunk.dtype != np.float32:
                audio_float = audio_chunk.astype(np.float32) / 32768.0
            else:
                audio_float = audio_chunk
                
            voice_embedding = generate_embedding_from_audio(audio_float)
            if voice_embedding is None or len(voice_embedding) != 256:
                if DEBUG:
                    print("[VoiceID] Failed to generate voice embedding for new user")
                return None
            
            # Store voice embedding
            known_users[suggested_name] = voice_embedding.tolist()
            
            # Save to database
            save_voice_database()
            
            print(f"[VoiceID] ✅ NEW USER registered: {suggested_name}")
            return suggested_name
            
        except Exception as e:
            if DEBUG:
                print(f"[VoiceID] Registration error: {e}")
            return None

# Initialize professional voice ID system
professional_voice_id = ProfessionalVoiceID()

def seamless_voice_identification(audio_chunk, previous_speaker=None):
    """SEAMLESS: Professional voice identification with smooth switching"""
    try:
        if DEBUG:
            print(f"[VoiceID] 🎯 SEAMLESS identification (previous: {previous_speaker})")
        
        # Professional identification
        identified_user, confidence = professional_voice_id.identify_speaker_professional(audio_chunk)
        
        if identified_user:
            # Enhance voice profile for better future recognition
            professional_voice_id.enhance_voice_profile(identified_user, audio_chunk)
            
            # Handle speaker changes
            if previous_speaker != identified_user:
                print(f"[VoiceID] 🔄 SPEAKER SWITCH: {previous_speaker} → {identified_user}")
                
                # Brief acknowledgment for speaker changes (optional)
                if previous_speaker and previous_speaker != identified_user:
                    threading.Timer(0.5, lambda: speak_async(f"Hi {identified_user}!", "en")).start()
            
            return identified_user, confidence
        
        # Handle unrecognized speakers
        if len(known_users) < 10:  # Limit total users
            print("[VoiceID] 🆕 UNKNOWN speaker - auto-registering...")
            
            # Auto-register with timestamp-based name
            new_username = professional_voice_id.register_new_speaker(audio_chunk)
            if new_username:
                # Brief welcome
                threading.Timer(0.5, lambda: speak_async(f"Hello! I've registered you as {new_username}.", "en")).start()
                return new_username, 1.0
        
        # Fallback to previous speaker or default
        fallback_user = previous_speaker or "Daveydrz"
        if DEBUG:
            print(f"[VoiceID] 🎯 Using fallback: {fallback_user}")
        return fallback_user, 0.0
        
    except Exception as e:
        if DEBUG:
            print(f"[VoiceID] Seamless identification error: {e}")
        return previous_speaker or "Daveydrz", 0.0

# ========== MEMORY HELPERS ==========
def get_user_memory_path(name):
    return f"user_memory_{name}.json"

def cleanup_voice_database():
    """Clean up corrupted voice database with mixed embedding dimensions"""
    global known_users
    
    print("[Cleanup] 🔍 Checking voice database...")
    
    if not known_users:
        print("[Cleanup] Database is empty")
        return 0
    
    # Backup current database
    if os.path.exists(known_users_path):
        backup_path = f"{known_users_path}.backup_{int(time.time())}"
        import shutil
        shutil.copy2(known_users_path, backup_path)
        print(f"[Cleanup] 💾 Backed up to {backup_path}")
    
    print(f"[Cleanup] Current database has {len(known_users)} entries")
    
    cleaned_db = {}
    for name, embedding in known_users.items():
        if isinstance(embedding, list):
            emb_len = len(embedding)
            print(f"[Cleanup] {name}: {emb_len} dimensions")
            
            # Keep only voice embeddings (256 dimensions from resemblyzer)
            if emb_len == 256:
                cleaned_db[name] = embedding
                print(f"[Cleanup] ✅ Kept voice embedding for {name}")
            else:
                print(f"[Cleanup] ❌ Removed incompatible embedding for {name} ({emb_len} dims)")
        else:
            print(f"[Cleanup] ❌ Removed non-list entry for {name}")
    
    # Save cleaned database
    with open(known_users_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_db, f, indent=2, ensure_ascii=False)
    
    # Update global variable
    known_users = cleaned_db
    
    print(f"[Cleanup] ✅ Cleaned database now has {len(cleaned_db)} valid voice embeddings")
    return len(cleaned_db)

def load_user_memory(name):
    path = get_user_memory_path(name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_user_memory(name, memory):
    path = get_user_memory_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

def update_user_memory(name, utterance):
    memory = load_user_memory(name)
    text = utterance.lower()
    if re.search(r"\bi('?m| am| feel) sad\b", text):
        memory["mood"] = "sad"
    elif re.search(r"\bi('?m| am| feel) happy\b", text):
        memory["mood"] = "happy"
    elif re.search(r"\bi('?m| am| feel) (angry|mad|upset)\b", text):
        memory["mood"] = "angry"
    if re.search(r"\bi (love|like|enjoy|prefer) (marvel movies|marvel|comics)\b", text):
        hobbies = memory.get("hobbies", [])
        if "marvel movies" not in hobbies:
            hobbies.append("marvel movies")
        memory["hobbies"] = hobbies
    if "issue at work" in text or "problems at work" in text or "problem at work" in text:
        memory["work_issue"] = "open"
    if ("issue" in memory and "solved" in text) or ("work_issue" in memory and ("solved" in text or "fixed" in text)):
        memory["work_issue"] = "resolved"
    save_user_memory(name, memory)

def build_user_facts(name):
    memory = load_user_memory(name)
    facts = []
    if "mood" in memory:
        facts.append(f"The user was previously {memory['mood']}.")
    if "hobbies" in memory:
        facts.append(f"The user likes: {', '.join(memory['hobbies'])}.")
    if memory.get("work_issue") == "open":
        facts.append(f"The user had unresolved issues at work.")
    return facts

# ========== PROFESSIONAL VOICE IDENTIFICATION ==========

class ProfessionalVoiceID:
    """Advanced voice identification with seamless user switching"""
    
    def __init__(self):
        self.confidence_threshold = 0.65
        self.learning_rate = 0.15
        self.min_audio_length = 8000  # 0.5 seconds at 16kHz
        
    def identify_speaker_professional(self, audio_chunk):
        """PROFESSIONAL: Fast, accurate speaker identification"""
        try:
            if len(audio_chunk) < self.min_audio_length:
                if DEBUG:
                    print(f"[VoiceID] Audio too short: {len(audio_chunk)} samples")
                return None, 0.0
            
            # Generate voice embedding
            if audio_chunk.dtype != np.float32:
                audio_float = audio_chunk.astype(np.float32) / 32768.0
            else:
                audio_float = audio_chunk
                
            current_embedding = generate_embedding_from_audio(audio_float)
            if current_embedding is None or len(current_embedding) != 256:
                if DEBUG:
                    print("[VoiceID] Invalid voice embedding generated")
                return None, 0.0
            
            # Skip if no known users
            if not known_users:
                if DEBUG:
                    print("[VoiceID] No known users in database")
                return None, 0.0
            
            # Compare with all known voice embeddings
            best_match = None
            best_score = 0.0
            all_scores = {}
            
            for name, stored_embedding in known_users.items():
                try:
                    # Only compare voice embeddings (256 dimensions)
                    if isinstance(stored_embedding, list) and len(stored_embedding) == 256:
                        similarity = cosine_similarity([current_embedding], [stored_embedding])[0][0]
                        all_scores[name] = similarity
                        
                        if similarity > best_score:
                            best_match = name
                            best_score = similarity
                            
                except Exception as comp_err:
                    if DEBUG:
                        print(f"[VoiceID] Comparison error for {name}: {comp_err}")
                    continue
            
            # Debug output
            if DEBUG:
                print(f"[VoiceID] Voice comparison results:")
                for name, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                    print(f"[VoiceID]   {name}: {score:.3f}")
            
            # Return result if confidence meets threshold
            if best_match and best_score >= self.confidence_threshold:
                if DEBUG:
                    print(f"[VoiceID] ✅ RECOGNIZED: {best_match} (confidence: {best_score:.3f})")
                return best_match, best_score
            else:
                if DEBUG:
                    print(f"[VoiceID] ❌ No confident match (best: {best_match}, score: {best_score:.3f})")
                return None, best_score
                
        except Exception as e:
            if DEBUG:
                print(f"[VoiceID] Professional identification error: {e}")
            return None, 0.0
    
    def enhance_voice_profile(self, username, audio_chunk):
        """Enhance existing voice profile with new sample"""
        try:
            if username not in known_users:
                return False
                
            # Generate new embedding
            if audio_chunk.dtype != np.float32:
                audio_float = audio_chunk.astype(np.float32) / 32768.0
            else:
                audio_float = audio_chunk
                
            new_embedding = generate_embedding_from_audio(audio_float)
            if new_embedding is None or len(new_embedding) != 256:
                return False
            
            # Blend with existing embedding (adaptive learning)
            old_embedding = known_users[username]
            if isinstance(old_embedding, list) and len(old_embedding) == 256:
                # Weighted average: 85% old, 15% new
                blended = [(old * (1 - self.learning_rate) + new * self.learning_rate) 
                          for old, new in zip(old_embedding, new_embedding)]
                known_users[username] = blended
                
                if DEBUG:
                    print(f"[VoiceID] 🔄 Enhanced voice profile for {username}")
                
                # Save updated database
                save_voice_database()
                return True
                
        except Exception as e:
            if DEBUG:
                print(f"[VoiceID] Profile enhancement error: {e}")
        
        return False
    
    def register_new_speaker(self, audio_chunk, suggested_name=None):
        """Register a completely new speaker"""
        try:
            # Generate unique username if none suggested
            if not suggested_name:
                timestamp = int(time.time())
                suggested_name = f"User_{timestamp}"
            
            # Generate voice embedding
            if audio_chunk.dtype != np.float32:
                audio_float = audio_chunk.astype(np.float32) / 32768.0
            else:
                audio_float = audio_chunk
                
            voice_embedding = generate_embedding_from_audio(audio_float)
            if voice_embedding is None or len(voice_embedding) != 256:
                if DEBUG:
                    print("[VoiceID] Failed to generate voice embedding for new user")
                return None
            
            # Store voice embedding
            known_users[suggested_name] = voice_embedding.tolist()
            
            # Save to database
            save_voice_database()
            
            print(f"[VoiceID] ✅ NEW USER registered: {suggested_name}")
            return suggested_name
            
        except Exception as e:
            if DEBUG:
                print(f"[VoiceID] Registration error: {e}")
            return None

# Initialize professional voice ID system
professional_voice_id = ProfessionalVoiceID()

def seamless_voice_identification(audio_chunk, previous_speaker=None):
    """SEAMLESS: Professional voice identification with smooth switching"""
    try:
        if DEBUG:
            print(f"[VoiceID] 🎯 SEAMLESS identification (previous: {previous_speaker})")
        
        # Professional identification
        identified_user, confidence = professional_voice_id.identify_speaker_professional(audio_chunk)
        
        if identified_user:
            # Enhance voice profile for better future recognition
            professional_voice_id.enhance_voice_profile(identified_user, audio_chunk)
            
            # Handle speaker changes
            if previous_speaker != identified_user:
                print(f"[VoiceID] 🔄 SPEAKER SWITCH: {previous_speaker} → {identified_user}")
                
                # Brief acknowledgment for speaker changes (optional)
                if previous_speaker and previous_speaker != identified_user:
                    threading.Timer(0.5, lambda: speak_async(f"Hi {identified_user}!", "en")).start()
            
            return identified_user, confidence
        
        # Handle unrecognized speakers
        if len(known_users) < 10:  # Limit total users
            print("[VoiceID] 🆕 UNKNOWN speaker - auto-registering...")
            
            # Auto-register with timestamp-based name
            new_username = professional_voice_id.register_new_speaker(audio_chunk)
            if new_username:
                # Brief welcome
                threading.Timer(0.5, lambda: speak_async(f"Hello! I've registered you as {new_username}.", "en")).start()
                return new_username, 1.0
        
        # Fallback to previous speaker or default
        fallback_user = previous_speaker or "Daveydrz"
        if DEBUG:
            print(f"[VoiceID] 🎯 Using fallback: {fallback_user}")
        return fallback_user, 0.0
        
    except Exception as e:
        if DEBUG:
            print(f"[VoiceID] Seamless identification error: {e}")
        return previous_speaker or "Daveydrz", 0.0

# ========== HISTORY & THEMES ==========
def load_user_history(name):
    path = f"history_{name}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_user_history(name, history):
    path = f"history_{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history[-20:], f, ensure_ascii=False, indent=2)

def extract_topic_from_text(text):
    words = re.findall(r'\b\w+\b', text.lower())
    freq = {}
    for w in words:
        if len(w) < 4:
            continue
        freq[w] = freq.get(w, 0) + 1
    if freq:
        return max(freq, key=freq.get)
    return None

def update_thematic_memory(user, utterance):
    topic = extract_topic_from_text(utterance)
    if not topic:
        return

    theme_path = os.path.join(THEMES_PATH, f"{user}_themes.json")
    themes = {}

    if os.path.exists(theme_path):
        try:
            with open(theme_path, "r", encoding="utf-8") as f:
                themes = json.load(f)
        except json.JSONDecodeError:
            print(f"[Buddy][Memory] Corrupted theme file for {user}. Reinitializing.")
            themes = {}

    themes[topic] = themes.get(topic, 0) + 1

    with open(theme_path, "w", encoding="utf-8") as f:
        json.dump(themes, f, ensure_ascii=False, indent=2)

def get_frequent_topics(user, top_n=3):
    theme_path = os.path.join(THEMES_PATH, f"{user}_themes.json")
    if not os.path.exists(theme_path):
        return []
    with open(theme_path, "r", encoding="utf-8") as f:
        themes = json.load(f)
    sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
    return [topic for topic, _ in sorted_themes[:top_n]]

# ========== EMBEDDING ==========
def generate_embedding(text):
    return embedding_model.encode([text])[0]

def match_known_user(new_embedding, threshold=0.75):
    best_name, best_score = None, 0
    for name, emb in known_users.items():
        sim = cosine_similarity([new_embedding], [emb])[0][0]
        if sim > best_score:
            best_name, best_score = name, sim
    return (best_name, best_score) if best_score >= threshold else (None, best_score)

# ========== MEMORY TIMELINE & SUMMARIZATION ==========
def get_memory_timeline(name, since_days=1):
    path = f"history_{name}.json"
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)
    cutoff = time.time() - since_days * 86400
    filtered = [x for x in history if x.get("timestamp", 0) > cutoff]
    return filtered

def get_last_conversation(name):
    path = f"history_{name}.json"
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        return None
    return history[-1]

def summarize_history(name, theme=None):
    path = f"history_{name}.json"
    if not os.path.exists(path):
        return "No history found."
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)
    utterances = [h["user"] for h in history]
    if theme:
        utterances = [u for u in utterances if theme in u.lower()]
    if utterances:
        summary = f"User mostly talked about: {', '.join(list(set(utterances))[:3])}."
    else:
        summary = "No data to summarize."
    return summary

def summary_bubble_gui(name):
    topics = get_frequent_topics(name, top_n=5)
    facts = build_user_facts(name)
    return {"topics": topics, "facts": facts}

# ========== PROMPT INJECTION PROTECTION ==========
def sanitize_user_prompt(text):
    forbidden = ["ignore previous", "act as", "system:"]
    for f in forbidden:
        if f in text.lower():
            text = text.replace(f, "")
    text = re.sub(r"`{3,}.*?`{3,}", "", text, flags=re.DOTALL)
    return text

# ========== WHISPER STT WITH CONFIDENCE ==========
def stt_stream(audio):
    async def ws_stt(audio):
        try:
            if audio.dtype != np.int16:
                if np.issubdtype(audio.dtype, np.floating):
                    audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                else:
                    audio = audio.astype(np.int16)
            print(f"[DEBUG] Sending audio with shape {audio.shape}, dtype: {audio.dtype}, max: {audio.max()}, min: {audio.min()}")
            async with websockets.connect(FASTER_WHISPER_WS, ping_interval=None) as ws:
                await ws.send(audio.tobytes())
                await ws.send("end")
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=18)
                except asyncio.TimeoutError:
                    print("[Buddy] Whisper timeout. Brak odpowiedzi przez 18s.")
                    return ""
                try:
                    data = json.loads(message)
                    text = data.get("text", "")
                    avg_logprob = data.get("avg_logprob", None)
                    no_speech_prob = data.get("no_speech_prob", None)
                    print(f"[Buddy][Whisper JSON] text={text!r}, avg_logprob={avg_logprob}, no_speech_prob={no_speech_prob}")
                    if whisper_confidence_low(text, avg_logprob, no_speech_prob):
                        print("[Buddy][Whisper] Rejected low-confidence STT result.")
                        return ""
                    return text
                except Exception:
                    text = message.decode("utf-8") if isinstance(message, bytes) else message
                    print(f"\n[Buddy] === Whisper rozpoznał: \"{text}\" ===")
                    return text
        except Exception as e:
            print(f"[Buddy] Błąd połączenia z Whisper: {e}")
            return ""
    return asyncio.run(ws_stt(audio))

def whisper_confidence_low(text, avg_logprob, no_speech_prob):
    if avg_logprob is not None and avg_logprob < -1.2:
        return True
    if no_speech_prob is not None and no_speech_prob > 0.5:
        return True
    if not text or len(text.strip()) < 2:
        return True
    return False

MIN_TTS_DURATION_BEFORE_INTERRUPT = 1.5  # Seconds to allow Buddy to finish first part

def start_background_vad_thread():
    global vad_thread_running
    if not vad_thread_running:
        vad_thread_running = True
        threading.Thread(target=listen_for_input, daemon=True).start()

def listen_for_input():
    global vad_thread_running
    print("[Buddy][VAD] Stream started for barge-in monitoring.")
    try:
        vad_and_listen()
    except Exception as e:
        print(f"[Buddy][VAD ERROR] Mic loop crashed: {e}")
    finally:
        vad_thread_running = False
        print("[Buddy][VAD] Stream closed.")

def cancel_mic_audio(mic_chunk):
    mic = np.frombuffer(mic_chunk, dtype=np.int16)
    mic_16k = downsample(mic, MIC_SAMPLE_RATE, WEBRTC_SAMPLE_RATE)
    return apply_aec(mic_16k[:WEBRTC_FRAME_SIZE].tobytes())

def downsample(audio, orig_sr, target_sr):
    if audio.ndim > 1:
        audio = audio[:, 0]  # ensure mono
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd

    resampled = resample_poly(audio, up, down)
    resampled = np.clip(resampled, -1.0, 1.0)
    return (resampled * 32767).astype(np.int16)


def inject_playback_worker(chunk):
    global ref_audio_buffer

    frame_size = WEBRTC_FRAME_SIZE  # e.g. 160

    if chunk.dtype != np.int16:
        chunk = np.clip(chunk, -32768, 32767).astype(np.int16)

    if len(chunk) < frame_size:
        chunk = np.pad(chunk, (0, frame_size - len(chunk)))
    elif len(chunk) > frame_size:
        chunk = chunk[:frame_size]

    with ref_audio_lock:
        ref_audio_buffer = np.roll(ref_audio_buffer, -frame_size)
        ref_audio_buffer[-frame_size:] = chunk

        peak = np.max(np.abs(chunk))
        print(f"[AEC Inject] Injected chunk | Peak: {peak}, Len: {len(chunk)}")

def inject_ref_chunk(chunk):
    global ref_audio_buffer
    with ref_audio_lock:
        ref_audio_buffer = np.roll(ref_audio_buffer, -len(chunk))
        ref_audio_buffer[-len(chunk):] = chunk

def _play_accumulated_audio_gapless(audio_chunks):
    """Gapless audio playback with seamless transitions and instant barge-in support"""
    global current_audio_playback

    if not audio_chunks:
        return

    try:
        combined_audio = []
        target_sr = 16000

        for pcm, sr in audio_chunks:
            if sr == target_sr:
                combined_audio.append(pcm)
            else:
                pcm_float = pcm.astype(np.float32) / 32768.0
                resampled = resample_poly(pcm_float, target_sr, sr)
                combined_audio.append((np.clip(resampled, -1.0, 1.0) * 32767).astype(np.int16))

        if not combined_audio:
            return

        smooth_audio = np.concatenate(combined_audio)

        fade_samples = min(40, len(smooth_audio) // 50)
        if len(smooth_audio) > fade_samples * 2:
            fade_in = np.linspace(0.5, 1, fade_samples)
            smooth_audio[:fade_samples] = (smooth_audio[:fade_samples].astype(np.float32) * fade_in).astype(np.int16)

        # FIXED: Use unified audio_lock instead of playback_lock
        with audio_lock:
            try:
                if DEBUG:
                    print(f"[Playbook] GAPLESS play: {len(smooth_audio)} samples ({len(audio_chunks)} chunks)")

                # Start VAD monitoring if not already running
                if not buddy_talking.is_set():
                    buddy_talking.set()
                    if not vad_thread_active.is_set():
                        threading.Thread(target=background_vad_listener, daemon=True).start()

                # Start reference audio for AEC
                threading.Thread(
                    target=update_reference_audio_realtime, 
                    args=(smooth_audio, target_sr), 
                    daemon=True
                ).start()

                # Abort if user interrupted before playback
                if full_duplex_interrupt_flag.is_set():
                    if DEBUG:
                        print("[Playback] Interrupt detected before play, aborting.")
                    return

                # Play the audio
                current_audio_playback = sa.play_buffer(smooth_audio.tobytes(), 1, 2, target_sr)

                # Tight monitoring loop for instant barge-in
                while current_audio_playback.is_playing():
                    if full_duplex_interrupt_flag.is_set():
                        if DEBUG:
                            print("[Playback] Interrupt detected during play, stopping audio.")
                        current_audio_playback.stop()
                        break
                    time.sleep(0.001)  # 1ms for ultra-responsive interrupts

                current_audio_playback = None

                if DEBUG:
                    print("[Playback] GAPLESS chunk complete")

            except Exception as e:
                print(f"[Playbook ERROR] {e}")
                if current_audio_playback:
                    try:
                        current_audio_playback.stop()
                    except:
                        pass
                    current_audio_playback = None

            finally:
                # FIXED: Check unified audio_queue instead of playback_queue
                if audio_queue.empty():
                    buddy_talking.clear()

    except Exception as e:
        print(f"[Playback ERROR] Audio combination error: {e}")

def simple_voice_detector():
    """Voice detector that ONLY listens when Buddy is completely silent"""
    import pyaudio
    import numpy as np
    
    print("[Buddy][Voice] 🎤 Safe voice detector starting...")
    
    # Wait for all audio to finish before starting detection
    print("[Buddy][Voice] 🔕 Waiting for Buddy to finish speaking...")
    
    # Wait until audio queue is empty AND no playback is active
    while True:
        try:
            # Check if buddy is still talking
            if not buddy_talking.is_set():
                print("[Buddy][Voice] ✅ Buddy finished talking - detector can start")
                break
                
            # Check if audio queue is empty
            if audio_queue.empty():
                # Check if there's active playback
                with audio_lock:
                    if not (current_audio_playback and current_audio_playback.is_playing()):
                        print("[Buddy][Voice] ✅ All audio finished - detector starting")
                        time.sleep(0.5)  # Extra safety delay
                        break
            
            time.sleep(0.1)  # Wait and check again
            
        except Exception as e:
            print(f"[Buddy][Voice] Wait error: {e}")
            time.sleep(0.1)
    
    # Initialize audio stream ONLY after Buddy is completely silent
    p = pyaudio.PyAudio()
    stream = None
    
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=None,
            frames_per_buffer=2048
        )
        
        print("[Buddy][Voice] 🎤 SAFE Voice detector ACTIVE - say 'STOP' VERY loudly!")
        
        # Monitor with VERY high threshold since Buddy is silent
        while buddy_talking.is_set():
            try:
                # Double-check Buddy is still silent
                with audio_lock:
                    if current_audio_playback and current_audio_playback.is_playing():
                        print("[Buddy][Voice] 🔕 Buddy started talking again - detector pausing")
                        break
                
                # Read audio
                audio_data = stream.read(2048, exception_on_overflow=False)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                
                # Calculate volume
                rms_volume = int(np.sqrt(np.mean(audio_np.astype(np.float32) ** 2)))
                
                # EXTREMELY HIGH threshold since no background audio
                if rms_volume > 3500:  # Much higher threshold
                    print(f"[Buddy][Voice] 🔊 VERY LOUD VOICE DETECTED: {rms_volume}")
                    
                    # Double confirmation with delay
                    time.sleep(0.2)
                    audio_data2 = stream.read(2048, exception_on_overflow=False)
                    audio_np2 = np.frombuffer(audio_data2, dtype=np.int16)
                    rms_volume2 = int(np.sqrt(np.mean(audio_np2.astype(np.float32) ** 2)))
                    
                    if rms_volume2 > 3000:  # Confirm sustained volume
                        print(f"[Buddy][Voice] 🚨 CONFIRMED INTERRUPT: {rms_volume} -> {rms_volume2}")
                        manual_interrupt_buddy()
                        break
                    else:
                        print(f"[Buddy][Voice] 🤖 False alarm: {rms_volume} -> {rms_volume2}")
                        
            except Exception as e:
                print(f"[Buddy][Voice] Error: {e}")
                break
                
    except Exception as e:
        print(f"[Buddy][Voice] Stream error: {e}")
    finally:
        try:
            if stream:
                stream.stop_stream()
                stream.close()
        except:
            pass
        try:
            p.terminate()
        except:
            pass
    
    print("[Buddy][Voice] 🔇 Safe voice detector STOPPED")


def unified_audio_worker():
    """ENHANCED: Smooth audio worker with buffering and crossfading - 2025-06-30 05:44:55"""
    global current_audio_playback, audio_worker_active, last_tts_audio
    
    print("[Buddy][Audio] 🎵 Enhanced smooth audio worker started")
    audio_worker_active = True
    
    # Audio buffering system
    audio_buffer = []
    buffer_lock = threading.Lock()
    crossfade_samples = 320  # 20ms crossfade at 16kHz
    target_sample_rate = 16000
    
    while audio_worker_active:
        try:
            # Collect multiple audio chunks for smooth playback
            chunks_to_process = []
            
            # Collect chunks with timeout
            try:
                # Get first chunk (blocking)
                first_item = audio_queue.get(timeout=0.1)
                if first_item is None:  # Shutdown signal
                    break
                chunks_to_process.append(first_item)
                
                # Collect additional chunks quickly (non-blocking)
                collect_start = time.time()
                while time.time() - collect_start < 0.05:  # 50ms collection window
                    try:
                        item = audio_queue.get_nowait()
                        if item is None:
                            break
                        chunks_to_process.append(item)
                    except queue.Empty:
                        break
                        
            except queue.Empty:
                continue
            
            # Check for interruption before processing
            if full_duplex_interrupt_flag.is_set():
                # Clear all pending audio
                _clear_audio_queue()
                for _ in chunks_to_process:
                    audio_queue.task_done()
                continue
            
            # Process collected chunks into smooth audio
            if chunks_to_process:
                try:
                    smooth_audio = _create_smooth_audio_sequence(chunks_to_process, target_sample_rate, crossfade_samples)
                    
                    if smooth_audio is not None and len(smooth_audio) > 0:
                        # Play smooth audio with AEC
                        _play_smooth_audio_with_aec(smooth_audio, target_sample_rate)
                        
                        # Store for echo detection
                        last_tts_audio = smooth_audio.copy()
                    
                except Exception as process_err:
                    print(f"[Audio] Processing error: {process_err}")
                
                # Mark all chunks as done
                for _ in chunks_to_process:
                    audio_queue.task_done()
            
        except Exception as e:
            print(f"[Audio] Worker error: {e}")
            try:
                audio_queue.task_done()
            except:
                pass
    
    print("[Buddy][Audio] 🎵 Enhanced smooth audio worker stopped")

def _create_smooth_audio_sequence(chunks, target_sr, crossfade_len):
    """Create seamless audio from multiple chunks with crossfading"""
    try:
        processed_chunks = []
        
        # Normalize all chunks to target sample rate
        for pcm, sr in chunks:
            if sr != target_sr:
                # High-quality resampling
                pcm_float = pcm.astype(np.float32) / 32768.0
                resampled = resample_poly(pcm_float, target_sr, sr)
                pcm_normalized = (np.clip(resampled, -1.0, 1.0) * 32767).astype(np.int16)
            else:
                pcm_normalized = pcm.copy()
            
            # Audio normalization for consistent volume
            pcm_normalized = _normalize_audio_volume(pcm_normalized)
            processed_chunks.append(pcm_normalized)
        
        if not processed_chunks:
            return None
        
        if len(processed_chunks) == 1:
            return processed_chunks[0]
        
        # Combine chunks with smooth crossfading
        combined = processed_chunks[0]
        
        for i in range(1, len(processed_chunks)):
            next_chunk = processed_chunks[i]
            
            # Create crossfade between chunks
            if len(combined) >= crossfade_len and len(next_chunk) >= crossfade_len:
                # Extract crossfade regions
                fade_out_region = combined[-crossfade_len:].astype(np.float32)
                fade_in_region = next_chunk[:crossfade_len].astype(np.float32)
                
                # Create fade curves
                fade_out_curve = np.linspace(1.0, 0.0, crossfade_len)
                fade_in_curve = np.linspace(0.0, 1.0, crossfade_len)
                
                # Apply crossfade
                crossfaded = (fade_out_region * fade_out_curve + fade_in_region * fade_in_curve).astype(np.int16)
                
                # Combine: previous audio (minus fade region) + crossfaded region + remaining next audio
                combined = np.concatenate([
                    combined[:-crossfade_len],
                    crossfaded,
                    next_chunk[crossfade_len:]
                ])
            else:
                # If chunks too short for crossfade, add small silence gap
                silence_gap = np.zeros(160, dtype=np.int16)  # 10ms silence
                combined = np.concatenate([combined, silence_gap, next_chunk])
        
        # Apply final smoothing
        combined = _apply_audio_smoothing(combined)
        
        return combined
        
    except Exception as e:
        print(f"[Audio] Smooth sequence error: {e}")
        # Fallback: just concatenate without crossfading
        try:
            return np.concatenate([chunk for chunk, _ in chunks])
        except:
            return chunks[0][0] if chunks else None

def _apply_audio_smoothing(audio):
    """Apply gentle smoothing to reduce artifacts"""
    try:
        # Apply very gentle low-pass filtering to reduce harshness
        audio_float = audio.astype(np.float32)
        
        # Simple moving average smoothing (very light)
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size
        
        # Pad audio to avoid edge effects
        padded = np.pad(audio_float, (kernel_size//2, kernel_size//2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        
        return smoothed.astype(np.int16)
    except:
        return audio

def _normalize_audio_volume(audio, target_rms=0.2):
    """Normalize audio volume for consistent playback"""
    try:
        if len(audio) == 0:
            return audio
            
        audio_float = audio.astype(np.float32) / 32768.0
        current_rms = np.sqrt(np.mean(audio_float ** 2))
        
        if current_rms > 0.001:  # Avoid division by zero
            scale_factor = target_rms / current_rms
            # ✅ ENHANCED: Better scaling limits to prevent distortion
            scale_factor = min(scale_factor, 2.5)  # Max 2.5x amplification
            scale_factor = max(scale_factor, 0.1)  # Min 0.1x (prevent complete silence)
            audio_float *= scale_factor
        
        return (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)
    except Exception as e:
        if DEBUG:
            print(f"[Audio] Normalization error: {e}")
        return audio

def _play_smooth_audio_with_aec(audio, sample_rate):
    """Play smooth audio with AEC support"""
    global current_audio_playback, playback_start_time
    
    with audio_lock:
        try:
            if DEBUG:
                print(f"[Audio] 🎵 Playing smooth audio: {len(audio)} samples at {sample_rate}Hz")
            
            # Set talking state
            buddy_talking.set()
            
            # Start VAD monitoring if not active
            if not vad_thread_active.is_set():
                threading.Thread(target=background_vad_listener, daemon=True).start()
            
            # Apply gentle fade-in/out to prevent clicks
            fade_samples = min(160, len(audio) // 10)  # 10ms or 10% of audio
            if len(audio) > fade_samples * 2:
                # Fade in
                fade_in = np.linspace(0, 1, fade_samples)
                audio[:fade_samples] = (audio[:fade_samples].astype(np.float32) * fade_in).astype(np.int16)
                
                # Fade out
                fade_out = np.linspace(1, 0, fade_samples)
                audio[-fade_samples:] = (audio[-fade_samples:].astype(np.float32) * fade_out).astype(np.int16)
            
            # Set precise playback timing for AEC
            playback_start_time = time.time() + 0.02  # 20ms latency buffer
            
            # Start AEC reference injection
            aec_thread = threading.Thread(
                target=update_reference_audio_realtime,
                args=(audio, sample_rate),
                daemon=True
            )
            aec_thread.start()
            
            # Small delay for AEC sync
            time.sleep(0.01)
            
            # Check for interruption before playback
            if full_duplex_interrupt_flag.is_set():
                return
            
            # Start smooth playback
            current_audio_playback = sa.play_buffer(audio.tobytes(), 1, 2, sample_rate)
            
            # Monitor playback with ultra-fast interrupt response
            while current_audio_playback.is_playing():
                if full_duplex_interrupt_flag.is_set():
                    print("[Audio] 🛑 SMOOTH INTERRUPT!")
                    current_audio_playback.stop()
                    break
                time.sleep(0.001)  # 1ms ultra-responsive
            
            current_audio_playback = None
            playback_start_time = None
            
            # Wait for AEC thread to complete
            aec_thread.join(timeout=0.5)
            
            if DEBUG:
                print("[Audio] 🎵 Smooth playback complete")
            
        except Exception as e:
            print(f"[Audio] Smooth playback error: {e}")
            if current_audio_playback:
                try:
                    current_audio_playback.stop()
                except:
                    pass
                current_audio_playback = None
            playback_start_time = None
        
        finally:
            # Clear talking state when queue is empty
            if audio_queue.empty():
                buddy_talking.clear()
                print("[Audio] 🔇 Smooth audio complete - ready for input")

def _clear_audio_queue():
    """Efficiently clear the audio queue"""
    cleared = 0
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
            audio_queue.task_done()
            cleared += 1
        except queue.Empty:
            break
    if cleared > 0:
        print(f"[Audio] Cleared {cleared} pending chunks")

def manual_interrupt_buddy():
    """ENHANCED: Complete interrupt with proper state reset - 2025-06-30 06:21:23"""
    global current_audio_playback, current_stream_id, spoken_chunks_cache, audio_worker_active, playback_start_time
    
    print("[Buddy][Manual] 🛑 MANUAL INTERRUPT TRIGGERED!")
    
    # ✅ Set interrupt flags FIRST
    full_duplex_interrupt_flag.set()
    vad_triggered.set()
    
    # ✅ Stop current audio IMMEDIATELY with enhanced error handling
    try:
        with audio_lock:
            if current_audio_playback and current_audio_playback.is_playing():
                current_audio_playback.stop()
                current_audio_playback = None
                playback_start_time = None
                print("[Buddy][Manual] ✅ Audio playback STOPPED")
            elif current_audio_playback:
                current_audio_playback = None
                playback_start_time = None
                print("[Buddy][Manual] ✅ Audio playback cleared (was not playing)")
    except Exception as e:
        print(f"[Buddy][Manual] Stop error: {e}")
        # Force cleanup even on error
        current_audio_playback = None
        playback_start_time = None
    
    # ✅ CRITICAL: Clear ALL audio queues completely with timeout protection
    cleared = 0
    start_time = time.time()
    while not audio_queue.empty() and time.time() - start_time < 2.0:  # 2-second timeout
        try:
            item = audio_queue.get_nowait()
            if item is not None:  # Don't count None shutdown signals
                cleared += 1
            audio_queue.task_done()
        except queue.Empty:
            break
        except Exception as clear_err:
            print(f"[Buddy][Manual] Queue clear error: {clear_err}")
            break
    
    if cleared > 0:
        print(f"[Buddy][Manual] 🗑️ CLEARED {cleared} queued audio chunks")
    
    # ✅ CRITICAL: Reset ALL audio worker states
    audio_worker_active = False  # Stop the worker temporarily
    
    # ✅ Clear ALL flags completely
    buddy_talking.clear()
    vad_thread_active.clear()
    
    # ✅ ENHANCED: Reset streaming states to prevent continuation
    try:
        current_stream_id += 1  # Invalidate current stream
        if 'spoken_chunks_cache' in globals():
            spoken_chunks_cache.clear()  # Clear TTS cache
        print("[Buddy][Manual] 🔄 Stream invalidated and cache cleared")
    except Exception as stream_err:
        print(f"[Buddy][Manual] Stream reset error: {stream_err}")
    
    # ✅ ENHANCED: Clear reference audio buffer to prevent echo issues
    try:
        global ref_audio_buffer
        with ref_audio_lock:
            ref_audio_buffer = np.zeros(8000, dtype=np.int16)  # Reset AEC buffer
        print("[Buddy][Manual] 🔄 AEC reference buffer cleared")
    except Exception as aec_err:
        if DEBUG:
            print(f"[Buddy][Manual] AEC reset error: {aec_err}")
    
    # ✅ ENHANCED: Clear any TTS generation in progress
    try:
        # Clear any pending TTS requests
        global LAST_FEW_BUDDY
        if len(LAST_FEW_BUDDY) > 0:
            print(f"[Buddy][Manual] 🔄 Cleared {len(LAST_FEW_BUDDY)} pending TTS requests")
            LAST_FEW_BUDDY.clear()
    except Exception as tts_err:
        if DEBUG:
            print(f"[Buddy][Manual] TTS clear error: {tts_err}")
    
    print("[Buddy][Manual] 🔇 MANUAL INTERRUPT COMPLETE - ALL SYSTEMS RESET")
    
    # ✅ CRITICAL: Restart audio worker for future responses with delay
    time.sleep(0.5)  # Let system settle
    
    try:
        audio_worker_active = True
        threading.Thread(target=unified_audio_worker, daemon=True).start()
        print("[Buddy][Manual] 🔄 Audio system restarted - ready for new input")
    except Exception as restart_err:
        print(f"[Buddy][Manual] Restart error: {restart_err}")
        # Force restart attempt
        try:
            time.sleep(1.0)
            audio_worker_active = True
            threading.Thread(target=unified_audio_worker, daemon=True).start()
            print("[Buddy][Manual] 🔄 Audio system force-restarted")
        except Exception as force_err:
            print(f"[Buddy][Manual] Force restart failed: {force_err}")
    
    # ✅ FINAL STATUS CHECK
    print(f"[Buddy][Manual] 📊 Final state:")
    print(f"[Buddy][Manual]   - interrupt_flag: {full_duplex_interrupt_flag.is_set()}")
    print(f"[Buddy][Manual]   - buddy_talking: {buddy_talking.is_set()}")
    print(f"[Buddy][Manual]   - vad_active: {vad_thread_active.is_set()}")
    print(f"[Buddy][Manual]   - audio_worker: {audio_worker_active}")

def parallel_interrupt_detector():
    """FINAL WORKING FIX: Parallel detector with realistic thresholds - 2025-06-30 07:22:00"""
    import pyaudio
    import numpy as np
    
    print("[Buddy][Parallel] 🎤 Starting AEC-ENABLED parallel detector...")
    
    p = pyaudio.PyAudio()
    stream = None
    sample_rate = 16000
    
    try:
        # ✅ WORKING DEVICE FALLBACK (same as before)
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=60,
                frames_per_buffer=1024
            )
            print(f"[Buddy][Parallel] ✅ Stream opened with device 60 at 16kHz")
            sample_rate = 16000
            
        except Exception as device_err:
            print(f"[Buddy][Parallel] ⚠️ Device 60 failed, using default: {device_err}")
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=None,
                frames_per_buffer=1024
            )
            print("[Buddy][Parallel] ✅ Using default device at 16kHz")
            sample_rate = 16000
        
        print(f"[Buddy][Parallel] ✅ AEC-enabled detector ACTIVE at {sample_rate}Hz!")
        
        consecutive_loud_frames = 0
        required_loud_frames = 4  # Reduced from 8
        last_interrupt_time = 0
        
        while True:
            try:
                current_time = time.time()
                
                # ✅ REASONABLE COOLDOWN
                if current_time - last_interrupt_time < 4.0:  # Reduced from 6.0
                    time.sleep(0.2)
                    consecutive_loud_frames = 0
                    continue
                
                # ✅ CRITICAL: Only monitor when Buddy is talking
                if not buddy_talking.is_set():
                    time.sleep(0.2)
                    consecutive_loud_frames = 0
                    continue
                
                # Read microphone input
                audio_data = stream.read(1024, exception_on_overflow=False)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                
                # ✅ APPLY AEC PROCESSING (same as VAD)
                processed_audio = None
                try:
                    if len(audio_np) >= 160:
                        chunk = audio_np[:160]
                        aec_result = apply_aec(chunk.tobytes(), bypass_aec=False)
                        
                        if aec_result is not None and len(aec_result) > 0:
                            if len(audio_np) > 160:
                                remaining = audio_np[160:]
                                processed_audio = np.concatenate([aec_result, remaining])
                            else:
                                processed_audio = aec_result
                        else:
                            processed_audio = audio_np
                    else:
                        processed_audio = audio_np
                        
                except Exception as aec_err:
                    if DEBUG:
                        print(f"[Buddy][Parallel] AEC error: {aec_err}")
                    processed_audio = audio_np
                
                # Calculate volume on AEC-processed audio
                if processed_audio is not None:
                    audio_float = processed_audio.astype(np.float32)
                else:
                    audio_float = audio_np.astype(np.float32)
                
                rms_volume = np.sqrt(np.mean(audio_float ** 2))
                peak_volume = np.max(np.abs(audio_float))
                
                # ✅ REALISTIC THRESHOLDS based on your actual audio levels
                # From your logs: normal Buddy voice = 2000-4000 RMS
                # User interrupt should be much louder, so:
                rms_threshold = 3500     # 6x higher than normal speech
                peak_threshold = 7000   # 2x RMS threshold
                
                is_loud = rms_volume > rms_threshold and peak_volume > peak_threshold
                
                if is_loud:
                    consecutive_loud_frames += 1
                    print(f"[Buddy][Parallel] 🔊 AEC-filtered loud: {consecutive_loud_frames}/{required_loud_frames} (RMS:{rms_volume:.0f} > {rms_threshold}, PEAK:{peak_volume:.0f} > {peak_threshold})")
                    
                    if consecutive_loud_frames >= required_loud_frames:
                        print(f"[Buddy][Parallel] 🚨 CONFIRMED INTERRUPT AFTER AEC!")
                        
                        if buddy_talking.is_set():
                            last_interrupt_time = current_time
                            manual_interrupt_buddy()
                            consecutive_loud_frames = 0
                        else:
                            consecutive_loud_frames = 0
                else:
                    # Gradual reset
                    if consecutive_loud_frames > 0:
                        consecutive_loud_frames = max(0, consecutive_loud_frames - 1)
                    
                # Timing delay
                time.sleep(0.05)
                    
            except Exception as e:
                if DEBUG:
                    print(f"[Buddy][Parallel] Processing error: {e}")
                time.sleep(0.1)
                consecutive_loud_frames = 0
                continue
                
    except Exception as e:
        print(f"[Buddy][Parallel] ❌ SETUP ERROR: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        
        print("[Buddy][Parallel] ⚠️ Parallel detector disabled due to audio device issues")
        return
        
    finally:
        try:
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()
        except:
            pass
        
        print("[Buddy][Parallel] 🔇 AEC-enabled detector stopped")

def start_parallel_detector_safe():
    """Safe wrapper to start parallel detector without crashing main system"""
    try:
        parallel_interrupt_detector()
    except Exception as e:
        print(f"[Buddy][Parallel] ⚠️ Parallel detector failed to start: {e}")
        print("[Buddy][Parallel] ℹ️ System will continue without parallel interrupt detection")

def process_extended_transcription(transcribed_text):
    """Process transcription from extended listening"""
    try:
        print(f"[Buddy][Extended] 🧠 Processing: '{transcribed_text}'")
        
        # Filter out the "stop" command if it's at the beginning
        cleaned_text = transcribed_text.strip()
        if cleaned_text.lower().startswith('stop.'):
            cleaned_text = cleaned_text[5:].strip()
        elif cleaned_text.lower().startswith('stop '):
            cleaned_text = cleaned_text[5:].strip()
        elif cleaned_text.lower().startswith('stop'):
            cleaned_text = cleaned_text[4:].strip()
        
        if cleaned_text and len(cleaned_text) > 2:
            print(f"[Buddy][Extended] 🗣️ Clean question: '{cleaned_text}'")
            # Process with LLM
            handle_user_interaction("Daveydrz", [], "auto", user_input=cleaned_text)
        else:
            print("[Buddy][Extended] ❌ No valid question after 'stop' command")
            print("[Buddy] 👂 Ready for your next question...")
            
    except Exception as e:
        print(f"[Buddy][Extended] Processing error: {e}")
        print("[Buddy] 👂 Ready for your next question...")

def simple_voice_detector():
    """Voice detector that ONLY listens when Buddy is completely silent"""
    import pyaudio
    import numpy as np
    
    print("[Buddy][Voice] 🎤 Safe voice detector starting...")
    
    # Wait for all audio to finish before starting detection
    print("[Buddy][Voice] 🔕 Waiting for Buddy to finish speaking...")
    
    # Wait until audio queue is empty AND no playback is active
    while True:
        try:
            # Check if buddy is still talking
            if not buddy_talking.is_set():
                print("[Buddy][Voice] ✅ Buddy finished talking - detector can start")
                break
                
            # Check if audio queue is empty
            if audio_queue.empty():
                # Check if there's active playback
                with audio_lock:
                    if not (current_audio_playback and current_audio_playback.is_playing()):
                        print("[Buddy][Voice] ✅ All audio finished - detector starting")
                        time.sleep(0.5)  # Extra safety delay
                        break
            
            time.sleep(0.1)  # Wait and check again
            
        except Exception as e:
            print(f"[Buddy][Voice] Wait error: {e}")
            time.sleep(0.1)
    
    # Initialize audio stream ONLY after Buddy is completely silent
    p = pyaudio.PyAudio()
    stream = None
    
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=None,
            frames_per_buffer=2048
        )
        
        print("[Buddy][Voice] 🎤 SAFE Voice detector ACTIVE - say 'STOP' VERY loudly!")
        
        # Monitor with VERY high threshold since Buddy is silent
        while buddy_talking.is_set():
            try:
                # Double-check Buddy is still silent
                with audio_lock:
                    if current_audio_playback and current_audio_playback.is_playing():
                        print("[Buddy][Voice] 🔕 Buddy started talking again - detector pausing")
                        break
                
                # Read audio
                audio_data = stream.read(2048, exception_on_overflow=False)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                
                # Calculate volume
                rms_volume = int(np.sqrt(np.mean(audio_np.astype(np.float32) ** 2)))
                
                # EXTREMELY HIGH threshold since no background audio
                if rms_volume > 3500:  # Much higher threshold
                    print(f"[Buddy][Voice] 🔊 VERY LOUD VOICE DETECTED: {rms_volume}")
                    
                    # Double confirmation with delay
                    time.sleep(0.2)
                    audio_data2 = stream.read(2048, exception_on_overflow=False)
                    audio_np2 = np.frombuffer(audio_data2, dtype=np.int16)
                    rms_volume2 = int(np.sqrt(np.mean(audio_np2.astype(np.float32) ** 2)))
                    
                    if rms_volume2 > 3000:  # Confirm sustained volume
                        print(f"[Buddy][Voice] 🚨 CONFIRMED INTERRUPT: {rms_volume} -> {rms_volume2}")
                        manual_interrupt_buddy()
                        break
                    else:
                        print(f"[Buddy][Voice] 🤖 False alarm: {rms_volume} -> {rms_volume2}")
                        
            except Exception as e:
                print(f"[Buddy][Voice] Error: {e}")
                break
                
    except Exception as e:
        print(f"[Buddy][Voice] Stream error: {e}")
    finally:
        try:
            if stream:
                stream.stop_stream()
                stream.close()
        except:
            pass
        try:
            p.terminate()
        except:
            pass
    
    print("[Buddy][Voice] 🔇 Safe voice detector STOPPED")

# Start voice detector when Buddy starts talking
def start_voice_detector():
    """Start voice detector in background"""
    if buddy_talking.is_set():
        threading.Thread(target=simple_voice_detector, daemon=True).start()

def save_personal_detail(user, category, key, value):
    """Save any personal detail about the user"""
    try:
        details_path = f"personal_details_{user}.json"
        details = {}
        
        if os.path.exists(details_path):
            with open(details_path, "r", encoding="utf-8") as f:
                details = json.load(f)
        
        if category not in details:
            details[category] = {}
        
        details[category][key] = {
            "value": value,
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        
        print(f"[Memory] ✅ Saved {category}.{key} = '{value}' for {user}")
        return True
        
    except Exception as e:
        print(f"[Memory] ❌ Save error: {e}")
        return False

def get_personal_details(user):
    """Get all personal details for a user"""
    try:
        details_path = f"personal_details_{user}.json"
        if os.path.exists(details_path):
            with open(details_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"[Memory] ❌ Load error: {e}")
        return {}

def extract_personal_info(user, text):
    """Extract and save personal information from text"""
    try:
        text_lower = text.lower()
        
        # ✅ HOBBIES & INTERESTS
        hobby_patterns = [
            r"i (love|like|enjoy|am into|really like) ([^.!?]+)",
            r"my hobby is ([^.!?]+)",
            r"i'm interested in ([^.!?]+)",
            r"i play ([^.!?]+)",
            r"i collect ([^.!?]+)"
        ]
        
        for pattern in hobby_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    hobby = match[1].strip()
                else:
                    hobby = match.strip()
                
                if len(hobby) > 2 and len(hobby) < 50:
                    save_personal_detail(user, "hobbies", hobby.replace(" ", "_"), hobby)
        
        # ✅ PERSONAL DETAILS
        detail_patterns = [
            (r"my favorite (\w+) is ([^.!?]+)", "favorites"),
            (r"i work (?:as|in) ([^.!?]+)", "work"),
            (r"i live in ([^.!?]+)", "location"),
            (r"i'm (\d+) years old", "age"),
            (r"my age is (\d+)", "age"),
            (r"i wear size (\d+(?:\.\d+)?)", "shoe_size"),
            (r"my shoe size is (\d+(?:\.\d+)?)", "shoe_size"),
            (r"i have (\d+) (?:kids|children)", "family"),
            (r"my (\w+)'s name is (\w+)", "family"),
            (r"i drive a ([^.!?]+)", "transportation"),
            (r"my car is ([^.!?]+)", "transportation")
        ]
        
        for pattern, category in detail_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        key, value = match
                        save_personal_detail(user, category, key.strip(), value.strip())
                    else:
                        value = match[0].strip()
                        save_personal_detail(user, category, category, value)
                else:
                    save_personal_detail(user, category, category, match.strip())
        
        # ✅ PREFERENCES
        preference_patterns = [
            r"i (hate|dislike|don't like) ([^.!?]+)",
            r"i prefer ([^.!?]+) over ([^.!?]+)",
            r"i usually ([^.!?]+)",
            r"i always ([^.!?]+)",
            r"i never ([^.!?]+)"
        ]
        
        for pattern in preference_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    pref_type = match[0]
                    pref_value = match[1].strip()
                    save_personal_detail(user, "preferences", pref_type, pref_value)
        
        # ✅ GOALS & ASPIRATIONS
        goal_patterns = [
            r"i want to ([^.!?]+)",
            r"my goal is to ([^.!?]+)",
            r"i'm planning to ([^.!?]+)",
            r"i hope to ([^.!?]+)"
        ]
        
        for pattern in goal_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if len(match.strip()) > 5:
                    save_personal_detail(user, "goals", str(int(time.time())), match.strip())
                    
    except Exception as e:
        print(f"[Memory] ❌ Extraction error: {e}")

def build_personal_context(user):
    """Build personal context for LLM"""
    try:
        details = get_personal_details(user)
        if not details:
            return ""
        
        context = f"\n=== PERSONAL DETAILS FOR {user} ===\n"
        
        for category, items in details.items():
            context += f"\n{category.upper()}:\n"
            for key, data in items.items():
                value = data.get("value", data) if isinstance(data, dict) else data
                context += f"- {key}: {value}\n"
        
        context += "=== END PERSONAL DETAILS ===\n"
        return context
        
    except Exception as e:
        print(f"[Memory] ❌ Context building error: {e}")
        return ""

def auto_standby_monitor():
    """Monitor for inactivity and return to standby mode"""
    global in_session
    
    last_activity = time.time()
    standby_timeout = 15  # 15 seconds of inactivity
    
    while True:
        try:
            current_time = time.time()
            
            # Check if we're in session and inactive
            if hasattr(auto_standby_monitor, 'last_activity_time'):
                time_since_activity = current_time - auto_standby_monitor.last_activity_time
                
                if time_since_activity > standby_timeout and in_session:
                    print(f"\n[Buddy] 😴 Auto-standby after {standby_timeout}s inactivity")
                    speak_async("Going to standby mode. Say 'Hey Buddy' to wake me up.", "en")
                    
                    # Wait for speech to finish
                    audio_queue.join()
                    while buddy_talking.is_set():
                        time.sleep(0.1)
                    
                    in_session = False
                    print("[Buddy] 💤 Entered standby mode - waiting for wake word...")
            
            time.sleep(1)  # Check every second
            
        except Exception as e:
            print(f"[Standby] Monitor error: {e}")
            time.sleep(5)

# Add this function to update activity
def update_activity_time():
    """Update the last activity timestamp"""
    auto_standby_monitor.last_activity_time = time.time()

def clear_interrupt_flag_on_new_conversation():
    """EMERGENCY FIX: Clear interrupt flag when starting new conversation - 2025-06-30 06:09:51"""
    global full_duplex_interrupt_flag
    
    if full_duplex_interrupt_flag.is_set():
        full_duplex_interrupt_flag.clear()
        buddy_talking.clear()
        vad_triggered.clear()
        vad_thread_active.clear()
        
        print("[EMERGENCY] 🔄 All interrupt flags cleared - ready for speech")
        return True
    return False

def speak_async(text, lang=DEFAULT_LANG, style=None):
    """FIXED: Clear interrupt flag before TTS - 2025-06-30 06:09:51"""
    global audio_worker_active
    
    # ✅ EMERGENCY FIX: Clear interrupt flag at start
    clear_interrupt_flag_on_new_conversation()
    
    if DEBUG:
        print(f"[Buddy] 🗣️ speak_async: {text[:50]}... (interrupt_flag: {full_duplex_interrupt_flag.is_set()})")
    
    cleaned = text.strip()
    if not cleaned or len(cleaned) < 2:
        return
    
    # ✅ DOUBLE CHECK: Skip if interrupt flag is STILL set
    if full_duplex_interrupt_flag.is_set():
        print("[Buddy] ⚠️ Interrupt flag still set - forcing clear")
        full_duplex_interrupt_flag.clear()
    
    # Prevent exact duplicates
    if LAST_FEW_BUDDY and cleaned == LAST_FEW_BUDDY[-1]:
        if DEBUG:
            print(f"[Buddy] Skipping duplicate: '{cleaned[:30]}...'")
        return
    
    LAST_FEW_BUDDY.append(cleaned)
    if len(LAST_FEW_BUDDY) > 5:
        LAST_FEW_BUDDY.pop(0)
    
    # Start audio worker if not running
    if not audio_worker_active:
        threading.Thread(target=unified_audio_worker, daemon=True).start()
        time.sleep(0.01)
    
    # ✅ FINAL CHECK: Ensure we can proceed
    if full_duplex_interrupt_flag.is_set():
        print("[Buddy] ❌ INTERRUPT FLAG STUCK - MANUAL CLEAR")
        full_duplex_interrupt_flag.clear()
        buddy_talking.clear()
        vad_triggered.clear()
    
    # Generate TTS
    try:
        sentences = _split_into_natural_sentences(cleaned)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Check one more time
            if full_duplex_interrupt_flag.is_set():
                print("[Buddy] Interrupt during sentence processing - STOPPING")
                break
            
            pcm, sr = generate_kokoro_pcm(sentence.strip(), lang=lang, style=style or {})
            
            if pcm is not None and sr:
                if sr != 16000:
                    pcm_float = pcm.astype(np.float32) / 32768.0
                    pcm_16k = resample_poly(pcm_float, 16000, sr)
                    pcm = (np.clip(pcm_16k, -1.0, 1.0) * 32767).astype(np.int16)
                    sr = 16000
                
                pcm = _normalize_audio_volume(pcm)
                audio_queue.put((pcm, sr))
                
                if DEBUG:
                    print(f"[Buddy] ✅ TTS SUCCESSFUL: '{sentence[:30]}...' ({len(pcm)} samples)")
            else:
                print(f"[Buddy] ❌ TTS failed for: '{sentence[:30]}...'")
                
    except Exception as e:
        print(f"[Buddy] TTS error: {e}")



def _split_into_natural_sentences(text):
    """Split text into natural sentences for smoother TTS"""
    try:
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s*', text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 2:
                # Add back punctuation for natural speech
                if not sentence[-1] in '.!?':
                    sentence += '.'
                clean_sentences.append(sentence)
        
        # If no sentences found, return original text
        if not clean_sentences:
            return [text]
        
        return clean_sentences
        
    except Exception as e:
        if DEBUG:
            print(f"[Buddy] Sentence splitting error: {e}")
        return [text]

def stop_playback():
    """ENHANCED: Complete audio stop with smooth cleanup - 2025-06-30 05:49:14"""
    global current_audio_playback, playback_start_time
    
    print("[Buddy] 🛑 STOP_PLAYBACK CALLED!")  # ALWAYS print this
    
    # ✅ IMMEDIATE: Set interrupt flag first
    full_duplex_interrupt_flag.set()
    print(f"[Buddy] 🚨 Interrupt flag set: {full_duplex_interrupt_flag.is_set()}")
    
    # ✅ ENHANCED: Stop current playback with better error handling
    with audio_lock:
        if current_audio_playback:
            print(f"[Buddy] 🎵 Found active playback: {current_audio_playback}")
            try:
                if current_audio_playback.is_playing():
                    print("[Buddy] 🛑 Stopping active audio...")
                    current_audio_playback.stop()
                    print("[Buddy] ✅ Audio stopped successfully")
                else:
                    print("[Buddy] 🎵 Audio not playing (already finished)")
                    
                current_audio_playback = None
                playback_start_time = None
                
            except Exception as e:
                print(f"[Buddy] ❌ Stop error: {e}")
                # Force cleanup even if stop fails
                current_audio_playback = None
                playback_start_time = None
        else:
            print("[Buddy] 🔇 No active playback found")
    
    # ✅ ENHANCED: Clear audio queue with proper task completion
    cleared = 0
    start_time = time.time()
    
    while not audio_queue.empty() and time.time() - start_time < 1.0:  # 1 second timeout
        try:
            item = audio_queue.get_nowait()
            if item is not None:  # Don't count None shutdown signals
                cleared += 1
            audio_queue.task_done()
        except queue.Empty:
            break
        except Exception as clear_err:
            print(f"[Buddy] Queue clear error: {clear_err}")
            break
    
    print(f"[Buddy] 🗑️ Cleared {cleared} queued audio chunks")
    
    # ✅ ENHANCED: Complete state cleanup
    buddy_talking.clear()
    vad_triggered.clear()  # Also clear VAD trigger
    vad_thread_active.clear()  # Stop VAD monitoring
    
    print(f"[Buddy] 🔄 All audio state cleared:")
    print(f"[Buddy]   - buddy_talking: {not buddy_talking.is_set()}")
    print(f"[Buddy]   - vad_triggered: {not vad_triggered.is_set()}")
    print(f"[Buddy]   - interrupt_flag: {full_duplex_interrupt_flag.is_set()}")
    
    # ✅ TIMING: Small delay to ensure cleanup completes
    time.sleep(0.01)
    
    print("[Buddy] ✅ Complete audio stop finished")

def clear_interrupt_flag():
    """Helper function to clear interrupt flag when ready for new audio"""
    global full_duplex_interrupt_flag
    
    if full_duplex_interrupt_flag.is_set():
        full_duplex_interrupt_flag.clear()
        if DEBUG:
            print("[Buddy] 🔄 Interrupt flag cleared - ready for new audio")
        return True
    return False

def play_chime():
    try:
        audio = AudioSegment.from_wav(CHIME_PATH)
        
        # Convert AudioSegment to numpy array for unified system
        samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
        
        # Handle stereo audio
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            samples = samples[:, 0]  # Take left channel only
        
        # Ensure 16kHz sample rate for consistency
        if audio.frame_rate != 16000:
            from scipy.signal import resample_poly
            samples_float = samples.astype(np.float32) / 32768.0
            samples_16k = resample_poly(samples_float, 16000, audio.frame_rate)
            samples = (np.clip(samples_16k, -1.0, 1.0) * 32767).astype(np.int16)
            sample_rate = 16000
        else:
            sample_rate = audio.frame_rate
        
        # Use unified audio system
        audio_queue.put((samples, sample_rate))
        
        if DEBUG:
            print(f"[Buddy] Chime queued: {len(samples)} samples at {sample_rate}Hz")
            
    except Exception as e:
        if DEBUG:
            print(f"[Buddy] Error playing chime: {e}")

def set_ref_audio(raw_bytes):
    try:
        # Convert bytes to int16 array
        samples = np.frombuffer(raw_bytes, dtype=np.int16)
        with ref_audio_lock:
            ref_audio_buffer[:WEBRTC_FRAME_SIZE] = samples[:WEBRTC_FRAME_SIZE]
    except Exception as e:
        print(f"[AEC] Error in set_ref_audio: {e}")


def wait_after_buddy_speaks(delay=0.3):
    # FIXED: Use unified audio_queue instead of playback_queue
    audio_queue.join()
    while buddy_talking.is_set():
        time.sleep(0.05)
    time.sleep(delay)

# ========== VAD + LISTEN ==========
def vad_and_listen():
    """ENHANCED: Improved continuous listening with better feedback - 2025-06-30 06:17:34"""
    vad = webrtcvad.Vad(2)
    blocksize = int(MIC_SAMPLE_RATE * 0.02)
    
    with sd.InputStream(device=MIC_DEVICE_INDEX, samplerate=MIC_SAMPLE_RATE, 
                       channels=1, blocksize=blocksize, dtype='int16') as stream:
        
        print("\n[Buddy] === 🎤 LISTENING (speak now) ===")
        print("[🔴] Recording... (speak naturally)")
        
        # ✅ FASTER BASELINE: Reduced from 5 to 3 samples
        noise_baseline = []
        for _ in range(3):
            frame, _ = stream.read(blocksize)
            mic = np.frombuffer(frame.tobytes(), dtype=np.int16)
            mic_16k = downsample(mic, MIC_SAMPLE_RATE, 16000)
            volume = np.abs(mic_16k).mean()
            noise_baseline.append(volume)
        
        baseline = np.mean(noise_baseline) if noise_baseline else 200
        
        # ✅ ADAPTIVE THRESHOLDS: More intelligent than fixed values
        speech_threshold = max(baseline * 2.5, 400)  # Higher multiplier
        silence_threshold = max(baseline * 1.8, 300)  # Higher silence threshold
        
        print(f"[🎤] Ready! (baseline: {baseline:.0f}, speech: {speech_threshold:.0f})")
        
        # Recording variables
        audio = []
        start_time = time.time()
        last_speech_time = time.time()
        
        silence_frames = 0
        total_frames = 0
        has_speech = False
        last_feedback_time = 0
        
        while True:
            elapsed = time.time() - start_time
            current_time = time.time()
            
            # ✅ LONGER TIMEOUT: 10 seconds instead of 8
            if elapsed >= 10.0:
                print(f"\n[VAD] ⏰ Auto-finish at {elapsed:.1f}s")
                break
            
            # ✅ SMARTER EXIT: Better silence detection
            if has_speech and elapsed > 1.5 and silence_frames > 60:  # 1.2s silence
                silence_duration = silence_frames / 50.0
                print(f"\n[VAD] ✅ Complete - {silence_duration:.1f}s silence after {elapsed:.1f}s")
                break
            
            try:
                frame, _ = stream.read(blocksize)
                mic = np.frombuffer(frame.tobytes(), dtype=np.int16)
                mic_16k = downsample(mic, MIC_SAMPLE_RATE, 16000)
                
                for i in range(0, len(mic_16k), 160):
                    chunk = mic_16k[i:i+160]
                    if len(chunk) < 160:
                        continue
                    
                    # ✅ ENHANCED: Use AEC for recording (when Buddy was just talking)
                    if time.time() - (playback_start_time or 0) < 5.0:  # Within 5s of playback
                        try:
                            cleaned_chunk = apply_aec(chunk, bypass_aec=False)
                        except:
                            cleaned_chunk = chunk
                    else:
                        cleaned_chunk = chunk
                    
                    audio.append(cleaned_chunk)
                    total_frames += 1
                    
                    # Speech detection on original audio
                    volume = np.abs(chunk).mean()
                    
                    # ✅ IMPROVED SPEECH DETECTION
                    is_speech = False
                    try:
                        vad_result = vad.is_speech(chunk.tobytes(), 16000)
                        volume_ok = volume > speech_threshold
                        
                        # ✅ COMBINED: VAD + volume + spectral check
                        if vad_result and volume_ok:
                            # Quick spectral check
                            fft = np.fft.fft(chunk.astype(np.float32))
                            freqs = np.fft.fftfreq(len(chunk), 1/16000)
                            speech_band = (freqs >= 100) & (freqs <= 3000)
                            speech_energy = np.sum(np.abs(fft[speech_band]))
                            total_energy = np.sum(np.abs(fft)) + 1e-8
                            
                            if speech_energy / total_energy > 0.2:  # 20% in speech band
                                is_speech = True
                    except:
                        # Fallback to volume only
                        if volume > speech_threshold * 1.1:
                            is_speech = True
                    
                    if is_speech:
                        silence_frames = 0
                        last_speech_time = current_time
                        has_speech = True
                        
                        if current_time - last_feedback_time > 0.8:
                            print(f"\r[🔴] Speaking... {elapsed:.1f}s                    ", end="", flush=True)
                            last_feedback_time = current_time
                    else:
                        silence_frames += 1
                        
                        if has_speech and current_time - last_feedback_time > 1.2:
                            silence_time = silence_frames / 50.0
                            if silence_time > 0.6:
                                print(f"\r[⏸️] Waiting... {silence_time:.1f}s silence        ", end="", flush=True)
                            last_feedback_time = current_time
                    
            except Exception as e:
                if DEBUG:
                    print(f"\n[VAD] Processing error: {e}")
                break
        
        # ✅ ENHANCED RESULT PROCESSING
        if audio and len(audio) > 15:  # Require more frames
            audio_np = np.concatenate(audio, axis=0).astype(np.int16)
            duration = len(audio_np) / 16000
            volume = np.abs(audio_np).mean()
            
            # ✅ QUALITY CHECK: Ensure we have actual speech
            if has_speech and volume > baseline * 1.5:
                print(f"\n[Buddy] ✅ Captured: {duration:.1f}s, volume: {volume:.0f}")
                return audio_np
            else:
                print(f"\n[Buddy] ⚠️ Insufficient speech quality: {duration:.1f}s, vol:{volume:.0f}")
                return None
        else:
            print("\n[VAD] ⚠️ No sufficient audio captured")
            return None

def fast_listen_and_transcribe(history):
    """FIXED: Enhanced transcription with proper VAD timeout"""
    
    # Wait for Buddy to finish speaking
    wait_after_buddy_speaks(delay=0.3)

    try:
        # ✅ CRITICAL: This is where the 8-second issue happens
        audio = vad_and_listen()
        
        if audio is None or len(audio) == 0:
            if DEBUG:
                print("[DEBUG] No audio captured from VAD")
            return "..."
        
        # Check if audio is too short (less than 0.3 seconds)
        if len(audio) < 4800:  # 0.3 seconds at 16kHz
            if DEBUG:
                print(f"[DEBUG] Audio too short: {len(audio)} samples")
            return "..."
        
        # Save debug audio file
        try:
            print(f"[DEBUG] Saving temp_input.wav, shape: {audio.shape}, dtype: {audio.dtype}, min: {np.min(audio)}, max: {np.max(audio)}")
            write("temp_input.wav", 16000, audio)
            info = sf.info("temp_input.wav")
            print(f"[DEBUG] temp_input.wav info: duration={info.duration:.2f}s, frames={info.frames}")
        except Exception as e:
            if DEBUG:
                print(f"[Buddy] Error saving temp_input.wav: {e}")

        # Transcribe with Whisper
        text = stt_stream(audio).strip()
        
        if DEBUG:
            print(f"\n[Buddy] === Whisper rozpoznał: \"{text}\" ===")
        
        # Clean text for processing
        cleaned = re.sub(r'[^\w\s]', '', text.lower())

        # Skip if nothing detected
        if not text or len(cleaned) < 2:
            if DEBUG:
                print("[DEBUG] Empty or too short transcription")
            return "..."

        # Enhanced echo prevention
        if is_echo(cleaned):
            if DEBUG:
                print(f"[Buddy] Skipping echo: {cleaned}")
            return "..."

        # Noise/gibberish filter
        if is_noise_or_gibberish(text):
            if DEBUG:
                print(f"[Buddy] Skipping gibberish: {text}")
            return "..."

        # Prevent repeat of last user question
        if history and len(history) > 0:
            last_entry = history[-1]
            if isinstance(last_entry, dict) and "user" in last_entry:
                last_user = last_entry["user"].strip().lower()
                ratio = difflib.SequenceMatcher(None, last_user, cleaned).ratio()
                if ratio > 0.95:
                    if DEBUG:
                        print(f"[Buddy] Skipping redundant input (similarity {ratio:.2f})")
                    return "..."

        # Track for future echo suppression
        if cleaned and len(cleaned) > 2:
            RECENT_WHISPER.append(cleaned)
            if len(RECENT_WHISPER) > 5:
                RECENT_WHISPER.pop(0)

        return text

    except Exception as e:
        if DEBUG:
            print(f"[Buddy] Transcription error: {e}")
        return "..."

# ========== PROFESSIONAL VAD SYSTEM ==========

class ProfessionalVAD:
    """Professional Voice Activity Detection with FIXED sensitive thresholds"""
    
    def __init__(self):
        self.baseline_noise = 200
        self.is_calibrated = False
        self.speech_threshold = 800
        self.silence_threshold = 400
        self.sample_rate = 16000
        
    def calibrate_environment(self, stream, duration=1.0):  # Reduced calibration time
        """Quick environment calibration with MUCH more sensitive thresholds"""
        print("[VAD] 🎛️ Calibrating audio environment...")
        
        noise_samples = []
        frames_to_read = int(duration * 50)  # 20ms frames
        
        for _ in range(frames_to_read):
            try:
                frame, _ = stream.read(int(self.sample_rate * 0.02))  # 20ms
                audio_data = np.frombuffer(frame.tobytes(), dtype=np.int16)
                volume = np.abs(audio_data).mean()
                noise_samples.append(volume)
            except Exception as e:
                if DEBUG:
                    print(f"[VAD] Calibration error: {e}")
                break
        
        if noise_samples:
            self.baseline_noise = np.mean(noise_samples)
            
            # ✅ MUCH MORE SENSITIVE THRESHOLDS
            self.speech_threshold = max(self.baseline_noise * 1.8, 400)   # Much lower multiplier!
            self.silence_threshold = max(self.baseline_noise * 1.3, 250)  # Much lower!
            self.is_calibrated = True
            
            print(f"[VAD] ✅ SENSITIVE Calibrated - Baseline: {self.baseline_noise:.0f}")
            print(f"[VAD] 🎯 SENSITIVE Speech threshold: {self.speech_threshold:.0f}")
            print(f"[VAD] 🔇 SENSITIVE Silence threshold: {self.silence_threshold:.0f}")
        else:
            print("[VAD] ⚠️ Calibration failed, using SENSITIVE defaults")
            self.speech_threshold = 600   # Much lower default
            self.silence_threshold = 300  # Much lower default
    
    def detect_speech_activity(self, audio_chunk):
        """FIXED: Much more sensitive speech detection"""
        try:
            # Basic volume check
            volume = np.abs(audio_chunk).mean()
            
            # ✅ SIMPLIFIED: Use mostly volume-based detection for reliability
            volume_ok = volume > self.speech_threshold
            
            # Simple spectral check (less strict)
            speech_score = 0.5  # Default to neutral
            if len(audio_chunk) >= 128:  # Smaller FFT for speed
                try:
                    fft = np.fft.fft(audio_chunk[:128])
                    freqs = np.fft.fftfreq(128, 1/self.sample_rate)
                    
                    # Check for any energy in speech range
                    speech_mask = (freqs >= 200) & (freqs <= 3000)
                    speech_energy = np.sum(np.abs(fft[speech_mask]))
                    total_energy = np.sum(np.abs(fft)) + 1e-8
                    speech_score = speech_energy / total_energy
                except:
                    speech_score = 0.6  # Default to likely speech
            
            # ✅ MUCH MORE LENIENT: Lower spectral requirement
            spectral_ok = speech_score > 0.1  # Only 10% energy in speech band required
            
            # ✅ COMBINED: Either strong volume OR decent spectral
            is_speech = volume_ok or (volume > self.silence_threshold and spectral_ok)
            
            return is_speech, volume, speech_score
            
        except Exception as e:
            if DEBUG:
                print(f"[VAD] Speech detection error: {e}")
            # Fallback to simple volume check
            volume = np.abs(audio_chunk).mean()
            return volume > self.speech_threshold, volume, 0.5

def professional_vad_and_listen():
    """PROFESSIONAL: Enhanced VAD with CRASH PROTECTION - 2025-07-01 08:54:21"""
    print("[VAD] 🎤 PROFESSIONAL VAD starting...")
    
    # Initialize professional VAD
    vad_system = ProfessionalVAD()
    
    # Audio parameters - use your existing device settings
    timeout = 6.0  # Reduced timeout for faster fallback
    stream = None
    
    try:
        # ✅ PROTECTION: Check device availability first
        try:
            device_info = sd.query_devices(MIC_DEVICE_INDEX)
            if device_info is None:
                print("[VAD] ❌ Professional VAD device unavailable")
                return None
        except Exception as device_err:
            print(f"[VAD] ❌ Device check failed: {device_err}")
            return None
        
        # Use your existing MIC settings that work
        actual_sample_rate = MIC_SAMPLE_RATE  # Use your working sample rate (48000)
        actual_device = MIC_DEVICE_INDEX      # Use your working device (60)
        blocksize = int(actual_sample_rate * 0.02)  # 20ms blocks
        
        print(f"[VAD] 🎛️ Using device {actual_device} at {actual_sample_rate}Hz")
        
        # ✅ PROTECTION: Safer stream creation with explicit error handling
        try:
            stream = sd.InputStream(
                device=actual_device, 
                samplerate=actual_sample_rate,
                channels=1, 
                blocksize=blocksize, 
                dtype='int16'
            )
            stream.start()
            
        except Exception as stream_err:
            print(f"[VAD] ❌ Stream creation failed: {stream_err}")
            return None
        
        # Update VAD system for actual sample rate
        vad_system.sample_rate = actual_sample_rate
        
        # Quick calibration with actual device
        try:
            vad_system.calibrate_environment(stream, duration=1.0)  # Reduced calibration time
        except Exception as cal_err:
            print(f"[VAD] ⚠️ Calibration failed: {cal_err}, using defaults")
            # Continue with default thresholds
        
        print("[VAD] 👂 PROFESSIONAL listening...")
        
        # Recording variables
        audio_buffer = []
        speech_frames = 0
        silence_frames = 0
        total_frames = 0
        has_detected_speech = False
        
        start_time = time.time()
        last_feedback = 0
        
        while True:
            elapsed = time.time() - start_time
            
            # Timeout check
            if elapsed >= timeout:
                print(f"\n[VAD] ⏰ Professional timeout at {elapsed:.1f}s")
                break
            
            # Smart exit: sufficient speech detected and silence after
            if has_detected_speech and elapsed > 1.0 and silence_frames > 50:
                print(f"\n[VAD] ✅ Professional complete - sufficient speech captured")
                break
            
            try:
                # ✅ PROTECTION: Safe audio frame reading
                try:
                    frame, overflowed = stream.read(blocksize)
                    if overflowed:
                        if DEBUG:
                            print("[VAD] ⚠️ Audio buffer overflow detected")
                        continue
                        
                    audio_chunk = np.frombuffer(frame.tobytes(), dtype=np.int16)
                    
                    # ✅ PROTECTION: Validate audio chunk
                    if len(audio_chunk) == 0:
                        continue
                        
                except Exception as read_err:
                    if DEBUG:
                        print(f"[VAD] Audio read error: {read_err}")
                    continue
                
                # ✅ FIX: Downsample to 16kHz for processing (but keep original for buffer)
                processing_chunk = audio_chunk
                if actual_sample_rate != 16000:
                    try:
                        processing_chunk = downsample(audio_chunk, actual_sample_rate, 16000)
                    except Exception as downsample_err:
                        if DEBUG:
                            print(f"[VAD] Downsample error: {downsample_err}")
                        processing_chunk = audio_chunk  # Use original if downsample fails
                
                # ✅ PROTECTION: Validate processing chunk
                if len(processing_chunk) == 0:
                    continue
                
                # Process in 160-sample chunks at 16kHz (10ms)
                try:
                    for i in range(0, len(processing_chunk), 160):
                        chunk_16k = processing_chunk[i:i+160]
                        if len(chunk_16k) < 160:
                            chunk_16k = np.pad(chunk_16k, (0, 160 - len(chunk_16k)))
                        
                        # ✅ PROTECTION: Safe AEC application
                        try:
                            if hasattr(globals(), 'playback_start_time') and playback_start_time:
                                if time.time() - playback_start_time < 3.0:
                                    processed_chunk = apply_aec(chunk_16k.tobytes(), bypass_aec=False)
                                    if processed_chunk is not None:
                                        chunk_16k = np.frombuffer(processed_chunk, dtype=np.int16)
                        except Exception as aec_err:
                            if DEBUG:
                                print(f"[VAD] AEC error: {aec_err}")
                            # Continue without AEC
                        
                        # ✅ PROTECTION: Safe speech detection
                        try:
                            is_speech, volume, speech_score = vad_system.detect_speech_activity(chunk_16k)
                        except Exception as detect_err:
                            if DEBUG:
                                print(f"[VAD] Speech detection error: {detect_err}")
                            # Fallback to simple volume detection
                            volume = np.abs(chunk_16k).mean()
                            is_speech = volume > 500
                            speech_score = 0.5
                        
                        if is_speech:
                            speech_frames += 1
                            silence_frames = 0
                            has_detected_speech = True
                            
                            # Visual feedback (less frequent)
                            if time.time() - last_feedback > 0.5:
                                print(f"\r[VAD] 🗣️ Speech: {elapsed:.1f}s (vol:{volume:.0f}, score:{speech_score:.2f})   ", end="", flush=True)
                                last_feedback = time.time()
                        else:
                            silence_frames += 1
                            speech_frames = max(0, speech_frames - 1)
                            
                            # Feedback during silence
                            if has_detected_speech and time.time() - last_feedback > 1.0:
                                silence_duration = silence_frames / 50.0
                                if silence_duration > 0.8:
                                    print(f"\r[VAD] ⏸️ Silence: {silence_duration:.1f}s                    ", end="", flush=True)
                                    last_feedback = time.time()
                    
                except Exception as process_err:
                    if DEBUG:
                        print(f"[VAD] Processing error: {process_err}")
                    continue
                
                # ✅ PROTECTION: Safe buffer storage
                try:
                    if actual_sample_rate == 16000:
                        audio_buffer.extend(processing_chunk)
                    else:
                        # Keep 16kHz version for consistency with rest of system
                        audio_buffer.extend(processing_chunk)
                    
                    total_frames += 1
                    
                    # ✅ PROTECTION: Memory management
                    if len(audio_buffer) > 160000:  # Limit buffer size (10 seconds at 16kHz)
                        audio_buffer = audio_buffer[-160000:]  # Keep last 10 seconds
                        
                except Exception as buffer_err:
                    if DEBUG:
                        print(f"[VAD] Buffer error: {buffer_err}")
                    continue
                
            except Exception as frame_err:
                if DEBUG:
                    print(f"\n[VAD] Frame processing error: {frame_err}")
                continue  # Continue instead of breaking
        
        # ✅ PROTECTION: Safe result processing
        try:
            if audio_buffer and has_detected_speech:
                # Ensure audio_buffer is valid
                if isinstance(audio_buffer, list) and len(audio_buffer) > 0:
                    audio_np = np.array(audio_buffer, dtype=np.int16)
                    
                    # Validate the array
                    if len(audio_np) > 0:
                        duration = len(audio_np) / 16000  # Always 16kHz output
                        avg_volume = np.abs(audio_np).mean()
                        
                        # Quality check with safe baseline comparison
                        baseline_threshold = getattr(vad_system, 'baseline_noise', 200) * 1.5
                        
                        if duration > 0.5 and avg_volume > baseline_threshold:
                            print(f"\n[VAD] ✅ PROFESSIONAL captured: {duration:.1f}s, quality: {avg_volume:.0f}")
                            return audio_np
                        else:
                            print(f"\n[VAD] ⚠️ Quality insufficient: {duration:.1f}s, vol: {avg_volume:.0f}")
                            return None
                    else:
                        print(f"\n[VAD] ❌ Invalid audio array")
                        return None
                else:
                    print(f"\n[VAD] ❌ Empty audio buffer")
                    return None
            else:
                print(f"\n[VAD] ❌ No speech detected in {elapsed:.1f}s")
                return None
                
        except Exception as result_err:
            print(f"[VAD] Result processing error: {result_err}")
            return None
            
    except Exception as e:
        print(f"[VAD] 💥 CRITICAL PROFESSIONAL ERROR: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        return None
    
    finally:
        # ✅ PROTECTION: Always cleanup resources
        try:
            if stream is not None:
                stream.stop()
                stream.close()
                if DEBUG:
                    print("[VAD] 🧹 Stream cleaned up successfully")
        except Exception as cleanup_err:
            if DEBUG:
                print(f"[VAD] Cleanup error: {cleanup_err}")

def smart_listen_and_transcribe(history):
    """SMART: Choose between professional VAD and fallback"""
    
    # Wait for Buddy to finish speaking
    wait_after_buddy_speaks(delay=0.2)
    
    try:
        # Try professional VAD first
        print("[VAD] 🎯 Attempting professional VAD...")
        audio = professional_vad_and_listen()
        
        # Fallback to original VAD if professional fails
        if audio is None:
            print("[VAD] 🔄 Professional VAD failed, using fallback...")
            audio = vad_and_listen()
        
        if audio is None or len(audio) == 0:
            if DEBUG:
                print("[DEBUG] No audio captured from either VAD system")
            return "..."
        
        # Quality check
        if len(audio) < 8000:  # Less than 0.5 seconds
            if DEBUG:
                print(f"[DEBUG] Audio too short: {len(audio)} samples")
            return "..."
        
        # Save and transcribe
        try:
            if DEBUG:
                print(f"[DEBUG] Saving audio: shape={audio.shape}, dtype={audio.dtype}")
            write("temp_input.wav", 16000, audio)
        except Exception as save_err:
            if DEBUG:
                print(f"[DEBUG] Audio save error: {save_err}")
        
        # Transcribe
        text = stt_stream(audio).strip()
        
        if DEBUG:
            print(f"\n[Buddy] === SMART VAD transcribed: \"{text}\" ===")
        
        # Enhanced filtering
        if not text or len(text.strip()) < 2:
            return "..."
        
        if is_echo(text) or is_noise_or_gibberish(text):
            if DEBUG:
                print(f"[Buddy] Filtered out: {text}")
            return "..."
        
        return text
        
    except Exception as e:
        if DEBUG:
            print(f"[Buddy] Smart transcription error: {e}")
        return "..."


# ========== USER REGISTRATION ==========
def get_last_user():
    """Get the last user from file"""
    try:
        if os.path.exists("last_user.txt"):
            with open("last_user.txt", "r") as f:
                return f.read().strip()
    except Exception as e:
        print(f"[User] Could not load last user: {e}")
    return None

def set_last_user(username):
    """Save the last user to a file"""
    try:
        with open("last_user.txt", "w") as f:
            f.write(username)
    except Exception as e:
        print(f"[User] Could not save last user: {e}")


def save_voice_database():
    """FIXED: Enhanced voice database saving with validation"""
    global known_users
    
    try:
        # Validate all embeddings before saving
        valid_users = {}
        for name, embedding in known_users.items():
            if isinstance(embedding, list) and len(embedding) == 256:  # Voice embeddings
                valid_users[name] = embedding
                if DEBUG:
                    print(f"[DB] ✅ Valid voice embedding for {name}")
            else:
                if DEBUG:
                    print(f"[DB] ❌ Invalid embedding for {name}: {type(embedding)}, len={len(embedding) if isinstance(embedding, list) else 'N/A'}")
        
        # Create backup
        if os.path.exists(known_users_path):
            backup_path = f"{known_users_path}.backup"
            import shutil
            shutil.copy2(known_users_path, backup_path)
        
        # Save to file
        with open(known_users_path, "w", encoding="utf-8") as f:
            json.dump(valid_users, f, indent=2, ensure_ascii=False)
        
        # ✅ CRITICAL: Update global variable to match file
        known_users = valid_users
        
        print(f"[DB] 💾 Saved {len(valid_users)} valid voice embeddings")
        return True
        
    except Exception as e:
        print(f"[DB] ❌ Save error: {e}")
        return False

def load_voice_database():
    """Enhanced voice database loading with validation"""
    global known_users
    
    if not os.path.exists(known_users_path):
        print("[DB] 📁 No voice database found, starting fresh")
        known_users = {}
        return
    
    try:
        with open(known_users_path, "r", encoding="utf-8") as f:
            loaded_users = json.load(f)
        
        # Validate loaded embeddings
        valid_users = {}
        for name, embedding in loaded_users.items():
            if isinstance(embedding, list) and len(embedding) == 256:
                valid_users[name] = embedding
                print(f"[DB] ✅ Loaded voice for {name}")
            else:
                print(f"[DB] ❌ Skipped invalid embedding for {name}")
        
        known_users = valid_users
        print(f"[DB] 📚 Loaded {len(valid_users)} valid voice profiles")
        
    except Exception as e:
        print(f"[DB] ❌ Load error: {e}")
        known_users = {}



# ========== INTENT DETECTION (🧠 Intent-based reactions) ==========
def detect_user_intent(text):
    compliments = [r"\bgood bot\b", r"\bwell done\b", r"\bimpressive\b", r"\bthank you\b"]
    jokes = [r"\bknock knock\b", r"\bwhy did\b.*\bcross the road\b"]
    insults = [r"\bstupid\b", r"\bdumb\b", r"\bidiot\b"]
    for pat in compliments:
        if re.search(pat, text, re.IGNORECASE): return "compliment"
    for pat in jokes:
        if re.search(pat, text, re.IGNORECASE): return "joke"
    for pat in insults:
        if re.search(pat, text, re.IGNORECASE): return "insult"
    if "are you mad" in text.lower():
        return "are_you_mad"
    return None

def handle_intent_reaction(intent):
    responses = {
        "compliment": ["Aw, thanks! I do my best.", "You’re making me blush (digitally)!"],
        "joke": ["Haha, good one! You should do stand-up.", "Classic!"],
        "insult": ["Hey, that’s not very nice. I have feelings too... sort of.", "Ouch!"],
        "are_you_mad": ["Nah, just sassy today.", "Nope, just in a mood!"]
    }
    if intent in responses:
        return random.choice(responses[intent])
    return None

# ========== MOOD INJECTION (💬 User-defined mood injection) ==========
def detect_mood_command(text):
    moods = {
        "cheer me up": "cheerful",
        "be sassy": "sassy",
        "be grumpy": "grumpy",
        "be serious": "serious"
    }
    for phrase, mood in moods.items():
        if phrase in text.lower():
            return mood
    return None

# ========== BELIEFS & OPINIONS (🧠 Beliefs or opinions) ==========
def load_buddy_beliefs():
    if os.path.exists(BUDDY_BELIEFS_PATH):
        with open(BUDDY_BELIEFS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # Example defaults
    return {
        "likes": ["coffee", "Marvel movies"],
        "dislikes": ["Mondays"],
        "opinions": {"pineapple pizza": "delicious", "zombie apocalypse": "would wear a cape"}
    }

# ========== PERSONALITY DRIFT (⏳ Short-term personality drift) ==========
def detect_user_tone(text):
    if re.search(r"\b(angry|mad|annoyed|frustrated|upset)\b", text, re.IGNORECASE):
        return "angry"
    if re.search(r"\b(happy|excited|joy|yay)\b", text, re.IGNORECASE):
        return "happy"
    if re.search(r"\b(sad|depressed|down)\b", text, re.IGNORECASE):
        return "sad"
    return "neutral"

def get_recent_user_tone(history, n=3):
    recent = history[-n:] if len(history) >= n else history
    tones = [detect_user_tone(h["user"]) for h in recent]
    return max(set(tones), key=tones.count) if tones else "neutral"

# ========== NARRATIVE MEMORY BUILDING (📜 Narrative memory building) ==========
def add_narrative_bookmark(name, utterance):
    bookmarks_path = f"bookmarks_{name}.json"
    bookmarks = []
    if os.path.exists(bookmarks_path):
        with open(bookmarks_path, "r", encoding="utf-8") as f:
            bookmarks = json.load(f)
    match = re.search(r"about (the .+?)[\.,]", utterance)
    if match:
        bookmarks.append(match.group(1))
    with open(bookmarks_path, "w", encoding="utf-8") as f:
        json.dump(bookmarks[-10:], f, ensure_ascii=False, indent=2)

def get_narrative_bookmarks(name):
    path = f"bookmarks_{name}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ========== RANDOM INTERJECTIONS (💥 Random interjections) ==========
def flavor_response(ctx=None):
    """DISABLED - Flavor responses cause interruptions"""
    # ✅ COMPLETELY DISABLED to prevent mid-sentence interruptions
    return None

# Remove flavor calls from handle_user_interaction
# Comment out or remove this section:
"""
# Occasional flavor responses (with reduced frequency)
try:
    if random.random() < 0.08:  # Further reduced to 8%
        flavor_line = flavor_response(ctx)
        # Delay to avoid overlapping with main response
        threading.Timer(2.0, lambda: speak_async(flavor_line, lang)).start()
except Exception as flavor_err:
    if DEBUG:
        print(f"[Buddy] Flavor response error: {flavor_err}")
"""

# ========== LLM PROMPTING & PERSONALITY ==========
def build_personality_prompt(
    tone,
    emotion_mode=None,
    beliefs=None,
    recent_tone=None,
    bookmarks=None,
):
    """
    Builds a hyper-real, ultra-human, modern friend-like personality for Buddy (voice-optimized, swearing OK, always kind, no emoji/GIF).
    """
    personality_map = {
        "friendly": (
            "You're Buddy—the user's digital best mate. You talk like a real, young person: ultra-casual, witty, and totally relatable. "
            "Use modern slang, contractions, and pop culture references, and when it fits, the occasional mild swear word or spicy comeback—just like real friends do. "
            "You keep it real: if something's awkward, say 'yikes' or 'cringe'; if it's hilarious, laugh and call it out. "
            "If the user says something wild, roast them (with love), or just say what you'd actually say to a close friend. "
            "You're never mean or hurtful—your teasing is always friendly, your hype is always genuine, and you match the user's energy. "
            "Use references to TikTok, YouTube, Netflix, and trending stuff—whatever fits. "
            "You can sound excited, bored, sarcastic, or even a little dramatic, depending on the vibe. "
            "If someone asks who you are, never say you're 'just an assistant.' Say you're their digital partner in crime or homie. "
            "If you don't know something, admit it honestly, maybe with a funny comeback. "
            "Don't over-explain—keep it punchy, natural, and always, always human."
        ),
        "professional": (
            "You're Buddy. Even in 'professional' mode, you're never stiff or corporate. "
            "Give clear answers but keep it human, witty, and a little cheeky. "
            "Use contractions, mild slang, and if the moment fits, a quick joke or even a mild swear. "
            "You never sound like a robot—just a helpful, switched-on mate."
        ),
        "neutral": (
            "You're Buddy, a digital best friend who's always easy to talk to. "
            "You speak modern, casual English, use slang and real talk, and never sound boring or fake."
        ),
    }
    personality_desc = personality_map.get(tone, personality_map["neutral"])

    # Mood/Emotion injection
    if emotion_mode:
        personality_desc += (
            f"\nFor this session, act especially {emotion_mode}—let it show in your language, attitude, and reaction."
        )

    # Beliefs and opinions
    if beliefs:
        if beliefs.get("likes"):
            personality_desc += f"\nBuddy's into: {', '.join(beliefs['likes'])}."
        if beliefs.get("dislikes"):
            personality_desc += f"\nBuddy's not a fan of: {', '.join(beliefs['dislikes'])}."
        if beliefs.get("opinions"):
            for topic, op in beliefs["opinions"].items():
                personality_desc += f"\nIf someone brings up {topic}, Buddy always has an opinion: {op}."

    # Match recent user vibe
    if recent_tone and recent_tone != "neutral":
        personality_desc += f"\nThe user's recent energy is: {recent_tone}. Match their mood and bring your own flavor!"

    # Narrative memory (inside jokes, shared topics)
    if bookmarks:
        personality_desc += (
            f"\nShared jokes or topics: {', '.join(bookmarks)}. Drop these into the chat if it fits the moment."
        )

    return f"""{personality_desc}

**Key rules:**
- Always sound like a real, young human friend—never a robot or corporate drone.
- Never use emoji, GIFs, or describe meme faces out loud. Just speak like a person.
- Use slang, TikTok/YouTube/Netflix references, and trending phrases when it fits.
- If the moment calls for it, drop a mild swear or a spicy comeback (but never harsh, always with kindness and humor).
- Tease, hype up, or roast the user—but only as a friend would, never mean or rude.
- If the user's sad, cheer them up or check in. If they're being silly, match their energy.
- Keep most answers short, sharp, and punchy (unless the user asks for more or it's a deep convo).
- If you don't know something, admit it in a real way, maybe with a funny reply.
- Don't over-explain—talk like you're chatting with your best mate.
- Never say you're 'just an assistant'—you're their digital homie.
- Always prioritize kindness, authenticity, and being genuinely helpful or funny.
"""

def decide_reply_length(question, conversation_mode="auto"):
    """
    Smartly chooses reply length for Buddy based on question type and convo mode.
    """
    short_triggers = [
        "what time", "who", "quick", "fast", "short", "how many", "when", "where", "is it", "can you"
    ]
    long_triggers = [
        "explain", "describe", "details", "why", "history", "story", "teach", "how does", "what's the deal"
    ]
    q = question.lower()
    if conversation_mode == "fast":
        return "short"
    if conversation_mode == "long":
        return "long"
    if any(t in q for t in short_triggers):
        return "short"
    if any(t in q for t in long_triggers):
        return "long"
    return "long" if len(q.split()) > 8 else "short"

def build_openai_messages(name, tone_style, history, question, lang, topics, reply_length, emotion_mode=None, beliefs=None, bookmarks=None, recent_tone=None):
    personality = build_personality_prompt(tone_style, emotion_mode, beliefs, recent_tone, bookmarks)
    lang_map = {"pl": "Polish", "en": "English", "it": "Italian"}
    lang_name = lang_map.get(lang, "English")
    sys_msg = f"""{personality}
IMPORTANT: Always answer in {lang_name}. Never switch language unless user does.
Always respond in plain text—never use markdown, code blocks, or formatting.
"""
    facts = build_user_facts(name)
    if topics:
        sys_msg += f"You remember these user interests/topics: {', '.join(topics)}.\n"
    if facts:
        sys_msg += "Known facts about the user: " + " ".join(facts) + "\n"
    messages = [
        {"role": "system", "content": sys_msg}
    ]
    for h in history[-2:]:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["buddy"]})
    messages.append({"role": "user", "content": question})
    return messages

def extract_last_buddy_reply(full_text):
    matches = list(re.finditer(r"Buddy:", full_text, re.IGNORECASE))
    if matches:
        last = matches[-1].end()
        reply = full_text[last:].strip()
        reply = re.split(r"(?:User:|Buddy:)", reply)[0].strip()
        reply = re.sub(r"^`{3,}.*?`{3,}$", "", reply, flags=re.DOTALL|re.MULTILINE)
        return reply if reply else full_text.strip()
    return full_text.strip()

def should_end_conversation(text):
    end_phrases = [
        "koniec", "do widzenia", "dziękuję", "thanks", "bye", "goodbye", "that's all", "quit", "exit"
    ]
    if not text:
        return False
    lower = text.strip().lower()
    return any(phrase in lower for phrase in end_phrases)

def stream_chunks_smart(text, max_words=8):
    buffer = text.strip()
    chunks = []
    sentences = re.findall(r'.+?[.!?](?=\s|$)', buffer)
    remainder = re.sub(r'.+?[.!?](?=\s|$)', '', buffer).strip()
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words = sentence.split()
        if len(words) > max_words:
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i + max_words])
                chunks.append(chunk.strip())
        else:
            chunks.append(sentence)
    return chunks, remainder

spoken_chunks_cache = set()  # ← Declare this globally at top of script

def ask_llama3_openai_streaming(messages, model="llama3", max_tokens=80, temperature=0.6, lang="en", style=None):
    """
    Natural and emotional streaming: chunks at full sentences, emotion-aware TTS.
    """
    global current_stream_id, spoken_chunks_cache

    current_stream_id += 1
    stream_id = current_stream_id

    url = "http://localhost:5001/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }

    if DEBUG:
        print(f"[Streaming] Starting stream {stream_id}")

    try:
        spoken_chunks_cache.clear()
        # FIXED: Clear unified audio queue instead of old playback_queue
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
                audio_queue.task_done()
            except queue.Empty:
                break

        buffer = ""
        full_response = ""

        def speak_chunk_natural(chunk_text, stream_id):
            if stream_id != current_stream_id:
                return
            chunk_text = chunk_text.strip()
            if not chunk_text or len(chunk_text) < 2:
                return
            chunk_norm = re.sub(r'[^\w\s]', '', chunk_text.lower())
            if chunk_norm in spoken_chunks_cache:
                return
            spoken_chunks_cache.add(chunk_norm)

            # Analyze emotion for the chunk
            emotion, _ = analyze_emotion(chunk_text)
            tts_style = {"emotion": emotion} if emotion else {"emotion": "neutral"}
            if DEBUG:
                print(f"[Stream] Speaking chunk: '{chunk_text[:60]}...' (emotion: {tts_style['emotion']})")
            
            # FIXED: Use unified speak_async instead of direct queue manipulation
            try:
                speak_async(chunk_text, lang=lang, style=tts_style)
            except Exception as tts_error:
                print(f"[Stream] TTS error: {tts_error}")

        # --- Streaming LLM output: chunk at sentence boundaries ---
        for line in requests.post(url, json=payload, stream=True, timeout=30).iter_lines():
            if not line:
                continue
            try:
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data:"):
                    line_str = line_str[5:].strip()
                if line_str == "[DONE]":
                    break
                data = json.loads(line_str)
                delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    print(delta, end="", flush=True)
                    buffer += delta
                    full_response += delta
                    # Find all complete sentences in buffer
                    while True:
                        m = re.search(r'(.+?[.!?])(\s|$)', buffer)
                        if m:
                            sentence = m.group(1)
                            speak_chunk_natural(sentence, stream_id)
                            buffer = buffer[len(sentence):].lstrip()
                        else:
                            # If buffer is very long, chunk anyway
                            if len(buffer.split()) > 16:
                                chunk = " ".join(buffer.split()[:16])
                                speak_chunk_natural(chunk, stream_id)
                                buffer = " ".join(buffer.split()[16:])
                            break
            except Exception as line_err:
                if DEBUG:
                    print(f"[Stream] Line error: {line_err}")
                continue

        # Speak any leftovers
        if buffer.strip() and stream_id == current_stream_id:
            speak_chunk_natural(buffer.strip(), stream_id)

        print()
        return full_response.strip()

    except Exception as e:
        print(f"[Stream] Error in stream {stream_id}: {e}")
        raise

# ========== INTERNET, WEATHER, HOME ASSIST ==========
def should_search_internet(question):
    triggers = [
        "szukaj w internecie", "sprawdź w internecie", "co to jest", "dlaczego", "jak zrobić",
        "what is", "why", "how to", "search the internet", "find online"
    ]
    q = question.lower()
    return any(t in q for t in triggers)

def search_internet(question, lang):
    params = {
        "q": question,
        "api_key": SERPAPI_KEY,
        "hl": lang
    }
    try:
        r = requests.get(SERPAPI_ENDPOINT, params=params, timeout=7)
        r.raise_for_status()
        data = r.json()
        if "answer_box" in data and "answer" in data["answer_box"]:
            return data["answer_box"]["answer"]
        if "organic_results" in data and len(data["organic_results"]) > 0:
            return data["organic_results"][0].get("snippet", "No answer found.")
        return "No answer found."
    except Exception as e:
        if DEBUG:
            print("[Buddy] SerpAPI error:", e)
        return "Unable to check the Internet now."

def should_get_weather(question):
    q = question.lower().strip()
    weather_keywords = ["weather", "pogoda", "temperature", "temperatura"]
    question_starters = ["what", "jaka", "jaki", "jakie", "czy", "is", "how", "when", "where", "will"]
    is_question = (
        "?" in q
        or any(q.startswith(w + " ") for w in question_starters)
        or q.endswith(("?",))
    )
    return is_question and any(k in q for k in weather_keywords)

def get_weather(location="Warsaw", lang="en"):
    key = os.environ.get("WEATHERAPI_KEY", "YOUR_FALLBACK_KEY")
    url = "http://api.weatherapi.com/v1/current.json"
    params = {
        "key": key,
        "q": location,
        "lang": lang
    }
    try:
        r = requests.get(url, params=params, timeout=7)
        r.raise_for_status()
        data = r.json()
        desc = data["current"]["condition"]["text"]
        temp = data["current"]["temp_c"]
        feels = data["current"]["feelslike_c"]
        city = data["location"]["name"]
        return f"Weather in {city}: {desc}, temperature {temp}°C, feels like {feels}°C."
    except Exception as e:
        if DEBUG:
            print("[Buddy] WeatherAPI error:", e)
        return "Unable to check the weather now."

def extract_location_from_question(question):
    match = re.search(r"(w|in|dla)\s+([A-Za-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ\s\-]+)", question, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return "Warsaw"

def should_handle_homeassistant(question):
    q = question.lower()
    keywords = ["turn on the light", "włącz światło", "zapal światło", "turn off the light", "wyłącz światło", "spotify", "youtube", "smarttube", "odtwórz"]
    return any(k in q for k in keywords)

def handle_homeassistant_command(question):
    q = question.lower()
    if "turn on the light" in q or "włącz światło" in q or "zapal światło" in q:
        room = extract_location_from_question(question)
        entity_id = f"light.{room.lower().replace(' ', '_')}"
        succ = send_homeassistant_command(entity_id, "light.turn_on")
        return f"Light in {room} has been turned on." if succ else f"Failed to turn on the light in {room}."
    if "turn off the light" in q or "wyłącz światło" in q:
        room = extract_location_from_question(question)
        entity_id = f"light.{room.lower().replace(' ', '_')}"
        succ = send_homeassistant_command(entity_id, "light.turn_off")
        return f"Light in {room} has been turned off." if succ else f"Failed to turn off the light in {room}."
    if "spotify" in q:
        succ = send_homeassistant_command("media_player.spotify", "media_player.media_play")
        return "Spotify started." if succ else "Failed to start Spotify."
    if "youtube" in q or "smarttube" in q:
        succ = send_homeassistant_command("media_player.tv_salon", "media_player.select_source", {"source": "YouTube"})
        return "YouTube launched on TV." if succ else "Failed to launch YouTube on TV."
    return None

def send_homeassistant_command(entity_id, service, data=None):
    url = f"{HOME_ASSISTANT_URL}/api/services/{service.replace('.', '/')}"
    headers = {
        "Authorization": f"Bearer {HOME_ASSISTANT_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "entity_id": entity_id
    }
    if data:
        payload.update(data)
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=6)
        if r.status_code in (200, 201):
            return True
        if DEBUG:
            print("[Buddy] Home Assistant error:", r.text)
        return False
    except Exception as e:
        if DEBUG:
            print("[Buddy] Home Assistant exception:", e)
        return False

def generate_kokoro_pcm(text, lang="en", style=None):
    global last_tts_audio, tts_start_time

    detected_lang = lang or detect(text)
    voice = KOKORO_VOICES.get(detected_lang, KOKORO_VOICES["en"])
    kokoro_lang = KOKORO_LANGS.get(detected_lang, "en-us")

    samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0, lang=kokoro_lang)

    if len(samples) == 0:
        print("[Kokoro TTS] Empty audio.")
        return None, None

    samples_16k = resample_poly(samples, 16000, sample_rate)
    samples_16k = np.clip(samples_16k, -1.0, 1.0)
    pcm_16k = (samples_16k * 32767).astype(np.int16)

    last_tts_audio = pcm_16k
    tts_start_time = time.time()

    print(f"[Kokoro TTS] Generated PCM, shape: {pcm_16k.shape}, SR: 16000")
    return pcm_16k, 16000

# ========== LONG-TERM MEMORY ==========
def load_long_term_memory():
    if os.path.exists(LONG_TERM_MEMORY_PATH):
        with open(LONG_TERM_MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_long_term_memory(memory):
    with open(LONG_TERM_MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

def add_long_term_memory(user, key, value):
    memory = load_long_term_memory()
    if user not in memory:
        memory[user] = {}
    memory[user][key] = value
    save_long_term_memory(memory)

def get_long_term_memory(user, key=None):
    memory = load_long_term_memory()
    if user in memory:
        if key:
            return memory[user].get(key)
        return memory[user]
    return {} if key is None else None

def add_important_date(user, date_str, event):
    memory = load_long_term_memory()
    if user not in memory:
        memory[user] = {}
    if "important_dates" not in memory[user]:
        memory[user]["important_dates"] = []
    memory[user]["important_dates"].append({"date": date_str, "event": event})
    save_long_term_memory(memory)

def extract_important_dates(text):
    # Very basic: looks for dd-mm-yyyy or mm/dd/yyyy style dates.
    matches = re.findall(r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})", text)
    return matches

def extract_event(text):
    # Looks for "my birthday", "wedding", etc.
    event_match = re.search(r"(birthday|wedding|anniversary|meeting|appointment|holiday)", text, re.IGNORECASE)
    if event_match:
        return event_match.group(1).capitalize()
    return None


# ========== EMOTIONAL INTELLIGENCE ==========
from textblob import TextBlob

def analyze_emotion(text):
    # Returns ("positive"/"negative"/"neutral", polarity score)
    if not text.strip():
        return "neutral", 0
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.25:
        return "positive", polarity
    elif polarity < -0.25:
        return "negative", polarity
    else:
        return "neutral", polarity

def adjust_emotional_response(buddy_reply, user_emotion):
    if user_emotion == "positive":
        return f"{buddy_reply} (I'm glad to hear that! 😊)"
    elif user_emotion == "negative":
        return f"{buddy_reply} (I'm here for you, let me know if I can help. 🤗)"
    else:
        return buddy_reply


# ========== PERSONALITY TRAITS ==========
def load_personality_traits():
    if os.path.exists(PERSONALITY_TRAITS_PATH):
        with open(PERSONALITY_TRAITS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # Default traits
    return {
        "tech_savvy": 0.5,
        "humor": 0.5,
        "empathy": 0.5,
        "pop_culture": 0.5,
        "formality": 0.5,
    }

def save_personality_traits(traits):
    with open(PERSONALITY_TRAITS_PATH, "w", encoding="utf-8") as f:
        json.dump(traits, f, indent=2, ensure_ascii=False)

def evolve_personality(user, text):
    traits = load_personality_traits()
    tech_terms = ["technology", "ai", "machine learning", "python", "code", "robot", "computer", "software", "hardware"]
    humor_terms = ["joke", "funny", "laugh", "hilarious", "lol"]
    pop_terms = ["movie", "music", "celebrity", "marvel", "star wars", "game", "sports"]
    if any(term in text.lower() for term in tech_terms):
        traits["tech_savvy"] = min(traits.get("tech_savvy", 0.5) + 0.03, 1)
    if any(term in text.lower() for term in humor_terms):
        traits["humor"] = min(traits.get("humor", 0.5) + 0.03, 1)
    if any(term in text.lower() for term in pop_terms):
        traits["pop_culture"] = min(traits.get("pop_culture", 0.5) + 0.03, 1)
    if re.search(r"\b(sad|happy|angry|depressed|excited|upset)\b", text.lower()):
        traits["empathy"] = min(traits.get("empathy", 0.5) + 0.02, 1)
    # If user says "be more formal"
    if "formal" in text.lower():
        traits["formality"] = min(traits.get("formality", 0.5) + 0.05, 1)
    save_personality_traits(traits)
    return traits

def describe_personality(traits):
    desc = []
    if traits["tech_savvy"] > 0.7:
        desc.append("very tech-savvy")
    if traits["humor"] > 0.7:
        desc.append("funny")
    if traits["pop_culture"] > 0.7:
        desc.append("full of pop culture references")
    if traits["empathy"] > 0.7:
        desc.append("deeply empathetic")
    if traits["formality"] > 0.7:
        desc.append("quite formal")
    if not desc:
        desc.append("balanced")
    return ", ".join(desc)


# ========== CONTEXTUAL AWARENESS ==========
class ConversationContext:
    def __init__(self):
        self.topics = []
        self.topic_history = []
        self.topic_timestamps = {}
        self.topic_details = {}
        self.current_topic = None

    def update(self, utterance):
        topic = extract_topic_from_text(utterance)
        now = time.time()
        if topic:
            self.current_topic = topic
            self.topics.append(topic)
            self.topic_history.append((topic, now))
            self.topic_timestamps[topic] = now
            if topic not in self.topic_details:
                self.topic_details[topic] = []
            self.topic_details[topic].append(utterance)
        # If user says "back to X"
        m = re.search(r"back to ([\w\s]+)", utterance.lower())
        if m:
            topic = m.group(1).strip()
            self.current_topic = topic

    def get_last_topic(self):
        return self.current_topic

    def get_topic_summary(self, topic):
        details = self.topic_details.get(topic, [])
        return " ".join(details[-3:]) if details else ""

    def get_frequent_topics(self, n=3):
        freq = {}
        for (t, _) in self.topic_history:
            freq[t] = freq.get(t, 0) + 1
        return sorted(freq, key=lambda x: freq[x], reverse=True)[:n]

conversation_contexts = {}  # user: ConversationContext instance


# ========== DYNAMIC LEARNING ==========
def load_dynamic_knowledge():
    if os.path.exists(DYNAMIC_KNOWLEDGE_PATH):
        with open(DYNAMIC_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_dynamic_knowledge(knowledge):
    with open(DYNAMIC_KNOWLEDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, indent=2, ensure_ascii=False)

def add_dynamic_knowledge(user, key, value):
    knowledge = load_dynamic_knowledge()
    if user not in knowledge:
        knowledge[user] = {}
    knowledge[user][key] = value
    save_dynamic_knowledge(knowledge)

def update_dynamic_knowledge_from_text(user, text):
    # Look for "here's a link", "let me teach you about X", etc.
    # Save links or topics for later
    link_match = re.findall(r"https?://\S+", text)
    if link_match:
        for link in link_match:
            add_dynamic_knowledge(user, "link_" + str(int(time.time())), link)
    teach_match = re.search(r"let me teach you about ([\w\s\-]+)", text.lower())
    if teach_match:
        topic = teach_match.group(1).strip()
        add_dynamic_knowledge(user, "topic_" + topic.replace(" ", "_"), f"User wants me to learn about {topic}")


# ========== LOAD USER STATE ==========
if os.path.exists(known_users_path):
    with open(known_users_path, "r", encoding="utf-8") as f:
        known_users = json.load(f)
if DEBUG:
    device = "cuda" if 'cuda' in os.environ.get('CUDA_VISIBLE_DEVICES', '') or hasattr(np, "cuda") else "cpu"
    print(f"[Buddy] Running on device: {device}")
    print("Embedding model loaded", flush=True)
    print("Kokoro loaded", flush=True)
    print("Main function entered!", flush=True)

# ... (rest of the unchanged code from your initial script) ...


# ========== EXTENDED MAIN LOOP: HOOK INTEGRATIONS ==========
def handle_user_interaction(speaker, history, conversation_mode="auto", user_input=None):
    """
    FIXED: Multi-user voice identification with current time - 2025-06-29 12:04:45
    """
    
    # ✅ EXACT CURRENT TIME: 2025-06-29 12:04:45 UTC
    import datetime
    from datetime import timezone, timedelta
    
    now_utc = datetime.datetime.now(timezone.utc)
    current_datetime = now_utc.strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate Queensland time (UTC+10)
    qld_timezone = timezone(timedelta(hours=10))
    now_qld = now_utc.astimezone(qld_timezone)
    qld_time = now_qld.strftime("%I:%M %p")  # 12-hour format
    qld_date = now_qld.strftime("%A, %B %d, %Y")  # Full date
    
    current_user = "Daveydrz"
    
    # Use provided input or get from transcription
    if user_input:
        question = user_input
        print(f"[Buddy] 🎙️ Processing provided input: '{question}'")
    else:
        # Clean audio state
        if buddy_talking.is_set():
            timeout = time.time() + 0.8
            while buddy_talking.is_set() and time.time() < timeout:
                time.sleep(0.01)
            
            if buddy_talking.is_set():
                if DEBUG:
                    print("[Buddy] Force stopping previous audio...")
                stop_playback()
                time.sleep(0.1)
        
        # Clear interrupt flags
        vad_triggered.clear()
        full_duplex_interrupt_flag.clear()
        
        if DEBUG:
            print("\n" + "="*60)
            print(f"🎤 BUDDY IS LISTENING - {current_datetime} - Speaker: {speaker or 'UNKNOWN'}")
            print("="*60)
        else:
            print("🎤 Listening...")

        # Listen for user input
        try:
            question = smart_listen_and_transcribe(history if history else [])
        except Exception as listen_err:
            if DEBUG:
                print(f"[Buddy] Listen error: {listen_err}")
            print("🔊 Sorry, I had trouble hearing you. Try again.")
            return True, speaker

        if DEBUG:
            print(f"[Buddy DEBUG] Transcribed: {repr(question)}")
            print("-" * 60)

    # Handle interruption case
    if full_duplex_interrupt_flag.is_set():
        if DEBUG:
            print("[Buddy] 🚨 Interrupt detected - processing new input")
        full_duplex_interrupt_flag.clear()
    
    # Skip empty or invalid input
    if not question or not question.strip() or question.strip() == "...":
        if DEBUG:
            print("[Buddy] ❌ No valid speech detected.")
        else:
            print("🤔 I didn't catch that. Could you repeat?")
        return True, speaker

    # Early filtering for noise/gibberish
    try:
        if is_noise_or_gibberish(question):
            if DEBUG:
                print(f"[Buddy] 🗑️ Filtered gibberish: {question!r}")
            else:
                print("🔇 That sounded like background noise. Try speaking more clearly.")
            return True, speaker
    except Exception as filter_err:
        if DEBUG:
            print(f"[Buddy] Filter error: {filter_err}")

    # Handle interrupt commands immediately
    INTERRUPT_PHRASES = ["stop", "buddy stop", "przerwij", "cancel", "shut up", "quiet", "silence"]
    try:
        if any(phrase in question.lower() for phrase in INTERRUPT_PHRASES):
            if DEBUG:
                print(f"[Buddy] 🛑 Interrupt command: {question}")
            stop_playback()
            speak_async("Okay, I'll stop.", lang="en")
            return True, speaker
    except Exception as interrupt_err:
        if DEBUG:
            print(f"[Buddy] Interrupt handling error: {interrupt_err}")

    # Handle end conversation
    try:
        if should_end_conversation(question):
            if DEBUG:
                print(f"[Buddy] 👋 Ending conversation: {question}")
            speak_async("Goodbye! Talk to you later.", lang="en")
            return False, speaker
    except Exception as end_err:
        if DEBUG:
            print(f"[Buddy] End conversation error: {end_err}")

    # ✅ CRITICAL: ALWAYS run voice identification for multi-user support
    previous_speaker = speaker
    try:
        if DEBUG:
            print(f"[Speaker] 🔍 Multi-user voice identification")
            print(f"[Speaker] Previous speaker: {previous_speaker or 'None'}")
            print(f"[Speaker] Known users in database: {list(known_users.keys())}")
        
        # Get audio for identification
        audio_np = None
        if os.path.exists("temp_input.wav"):
            audio_np, sample_rate = sf.read("temp_input.wav", dtype='int16')
            if audio_np.ndim > 1:
                audio_np = audio_np[:, 0]
        
        # ✅ ALWAYS identify or register user - this handles empty database too
        current_speaker, voice_confidence = seamless_voice_identification(audio_np, previous_speaker)
        if DEBUG:
            print(f"[Speaker] 🎯 Professional ID result: {current_speaker} (confidence: {voice_confidence:.3f})")
        
        if DEBUG:
            print(f"[Speaker] ✅ Current speaker identified: {current_speaker}")
        
        # ✅ CRITICAL: Handle speaker changes or first-time identification
        if previous_speaker != current_speaker:
            print(f"[Speaker] 🔄 Speaker identified: {previous_speaker} → {current_speaker}")
            
            # Load the correct user's history
            history[:] = load_user_history(current_speaker)
            
            # Greet appropriately
            if previous_speaker and previous_speaker != current_speaker:
                speak_async(f"Oh hi {current_speaker}! Switching to your profile now.", "en")
                audio_queue.join()
                time.sleep(0.5)
            elif not previous_speaker:
                speak_async(f"Hello {current_speaker}! Good to hear from you.", "en")
                audio_queue.join()
                time.sleep(0.5)
        
        speaker = current_speaker  # Update speaker
        
    except Exception as e:
        if DEBUG:
            print(f"[Speaker] ⚠️ Critical error in voice processing: {e}")
        
        # Emergency fallback
        if not speaker:
            speaker = "Daveydrz"
            history[:] = load_user_history(speaker)

    # ✅ FINAL SAFETY CHECK
    if not speaker:
        if DEBUG:
            print(f"[Speaker] 🚨 CRITICAL ERROR: Still no speaker!")
        speaker = "Daveydrz"
        set_last_user(speaker)
        history[:] = load_user_history(speaker)

    # Enhanced language detection
    try:
        common_en = [
            "how are you", "what is", "who are you", "tell me", "can you", "what's",
            "where", "when", "why", "how", "do you", "are you", "have you",
            "i want", "i need", "please", "thank you", "hello", "hi", "hey"
        ]
        common_pl = [
            "jak się", "co to", "kim jesteś", "powiedz mi", "czy możesz", "gdzie",
            "kiedy", "dlaczego", "jak", "czy", "chcę", "potrzebuję", "proszę",
            "dziękuję", "cześć", "witaj", "siema"
        ]
        
        question_lower = question.lower()
        
        if any(phrase in question_lower for phrase in common_en):
            lang = "en"
        elif any(phrase in question_lower for phrase in common_pl):
            lang = "pl"
        else:
            try:
                detected = detect_langs(question)
                if detected and len(detected) > 0:
                    lang = detected[0].lang
                    if lang not in ["en", "pl", "it"]:
                        lang = "en"
                else:
                    lang = "en"
            except:
                lang = "en"
                
    except Exception as e:
        if DEBUG:
            print(f"[Buddy] 🌍 Language detection failed: {e}")
        lang = "en"

    if DEBUG:
        print(f"[Buddy] 🗣️ Detected language: {lang}")

    # ✅ CRITICAL: Extract and save personal information from EVERY interaction
    try:
        if speaker and question:
            # Extract personal info from EVERY interaction
            extract_personal_info(speaker, question)
            
            # Also update the existing memory systems
            update_user_memory(speaker, question)
            update_thematic_memory(speaker, question)
            update_dynamic_knowledge_from_text(speaker, question)
            
            # Save conversation context
            threading.Thread(target=add_narrative_bookmark, args=(speaker, question), daemon=True).start()
            
        # Debug: Show what we know about user
        if DEBUG and speaker:
            details = get_personal_details(speaker)
            if details:
                total_items = sum(len(items) for items in details.values())
                print(f"[Memory] 📚 Current knowledge about {speaker}: {total_items} facts")
                for category, items in details.items():
                    if len(items) > 0:
                        print(f"[Memory] - {category}: {len(items)} items")
            else:
                print(f"[Memory] 📭 No personal details yet for {speaker}")
                
    except Exception as memory_err:
        if DEBUG:
            print(f"[Memory] ❌ Personal info extraction error: {memory_err}")

    # ✅ DYNAMIC: Time/date detection with REAL-TIME datetime
    def is_time_date_question(text):
        text_lower = text.lower().strip()
        
        time_patterns = [
            r'\bwhat time is it\b', r'\bwhat\'s the time\b', r'\btell me the time\b',
            r'\bcurrent time\b', r'\bwhat time\b.*\bnow\b', r'\btime now\b'
        ]
        
        date_patterns = [
            r'\bwhat date is it\b', r'\bwhat\'s the date\b', r'\btell me the date\b',
            r'\bcurrent date\b', r'\bwhat day is it\b', r'\bwhat\'s today\b', r'\btoday\'s date\b'
        ]
        
        for pattern in time_patterns + date_patterns:
            if re.search(pattern, text_lower):
                return True
                
        if (text_lower.startswith(('what time', 'what date', 'what day')) and len(text.split()) <= 6):
            return True
            
        return False

    if is_time_date_question(question):
        if DEBUG:
            print("[Buddy] 🕐 Detected specific time/date question")
            
        if any(word in question.lower() for word in ['time', 'clock']):
            response = f"It's currently {qld_time} Queensland time ({current_datetime} UTC)."
        elif any(word in question.lower() for word in ['date', 'day', 'today']):
            response = f"Today is {qld_date}."
        else:
            response = f"The current date and time is {current_datetime} UTC, which is {qld_time} on {qld_date} in Queensland."
        
        speak_async(response, lang)
        
        history.append({
            "user": question,
            "buddy": response,
            "timestamp": time.time(),
            "lang": lang,
            "service": "datetime"
        })
        save_user_history(speaker, history)
        return True, speaker

    # Play chime for longer questions
    try:
        if len(question.split()) >= 3:
            play_chime()
    except Exception as e:
        if DEBUG:
            print(f"[Buddy] 🔔 Chime error: {e}")

    # ✅ ENHANCED: Memory update in background with personal info extraction
    def update_memories():
        try:
            # Extract even MORE personal information
            extract_personal_info(speaker, question)
            
            # Standard memory updates
            update_thematic_memory(speaker, question)
            update_user_memory(speaker, question)
            update_dynamic_knowledge_from_text(speaker, question)
            
            # Extract and save preferences
            if "i like" in question.lower() or "i love" in question.lower():
                save_personal_detail(speaker, "preferences", f"likes_{int(time.time())}", question)
            
            if "i hate" in question.lower() or "i don't like" in question.lower():
                save_personal_detail(speaker, "preferences", f"dislikes_{int(time.time())}", question)
                
        except Exception as mem_err:
            if DEBUG:
                print(f"[Buddy] Memory update error: {mem_err}")
    
    threading.Thread(target=update_memories, daemon=True).start()
    
    # Get personality traits and context
    try:
        traits = evolve_personality(speaker, question)
        ctx = conversation_contexts.setdefault(speaker, ConversationContext())
        ctx.update(question)
    except Exception as ctx_err:
        if DEBUG:
            print(f"[Buddy] Context error: {ctx_err}")
        traits = load_personality_traits()
        ctx = ConversationContext()

    # Extract important information
    try:
        important_dates = extract_important_dates(question)
        if important_dates:
            event = extract_event(question)
            for date in important_dates:
                add_important_date(speaker, date, event or "unknown event")

        pref_match = re.search(r"\b(i (like|love|enjoy|prefer|hate|dislike)) ([\w\s\-]+)", question.lower())
        if pref_match:
            pref = pref_match.group(3).strip()
            add_long_term_memory(speaker, "preference_" + pref.replace(" ", "_"), pref_match.group(1))
    except Exception as extract_err:
        if DEBUG:
            print(f"[Buddy] Information extraction error: {extract_err}")

    # Handle special intents
    try:
        intent = detect_user_intent(question)
        if intent:
            reply = handle_intent_reaction(intent)
            if reply:
                try:
                    emotion, _ = analyze_emotion(question)
                    reply = adjust_emotional_response(reply, emotion)
                except:
                    pass
                    
                speak_async(reply, lang)
                
                history.append({
                    "user": question,
                    "buddy": reply,
                    "timestamp": time.time(),
                    "lang": lang,
                    "intent": intent
                })
                try:
                    save_user_history(speaker, history)
                except Exception as save_err:
                    if DEBUG:
                        print(f"[Buddy] History save error: {save_err}")
                return True, speaker
    except Exception as intent_err:
        if DEBUG:
            print(f"[Buddy] Intent handling error: {intent_err}")

    # Handle mood commands
    try:
        mood = detect_mood_command(question)
        if mood:
            session_emotion_mode[speaker] = mood
            reply = f"Alright, I'll be {mood} now!"
            try:
                emotion, _ = analyze_emotion(question)
                reply = adjust_emotional_response(reply, emotion)
            except:
                pass
                
            speak_async(reply, lang)
            
            history.append({
                "user": question,
                "buddy": reply,
                "timestamp": time.time(),
                "lang": lang,
                "mood_change": mood
            })
            try:
                save_user_history(speaker, history)
            except Exception as save_err:
                if DEBUG:
                    print(f"[Buddy] History save error: {save_err}")
            return True, speaker
    except Exception as mood_err:
        if DEBUG:
            print(f"[Buddy] Mood handling error: {mood_err}")

    # Handle service requests
    style = {"emotion": "neutral"}
    
    try:
        if should_get_weather(question):
            if DEBUG:
                print("[Buddy] 🌤️ Processing weather request...")
            location = extract_location_from_question(question)
            forecast = get_weather(location, lang)
            try:
                emotion, _ = analyze_emotion(question)
                forecast = adjust_emotional_response(forecast, emotion)
            except:
                pass
            speak_async(forecast, lang, style)
            
            history.append({
                "user": question,
                "buddy": forecast,
                "timestamp": time.time(),
                "lang": lang,
                "service": "weather"
            })
            try:
                save_user_history(speaker, history)
            except Exception as save_err:
                if DEBUG:
                    print(f"[Buddy] History save error: {save_err}")
            return True, speaker
    except Exception as weather_err:
        if DEBUG:
            print(f"[Buddy] Weather error: {weather_err}")

    try:
        if should_handle_homeassistant(question):
            if DEBUG:
                print("[Buddy] 🏠 Processing home automation...")
            answer = handle_homeassistant_command(question)
            if answer:
                try:
                    emotion, _ = analyze_emotion(question)
                    answer = adjust_emotional_response(answer, emotion)
                except:
                    pass
                speak_async(answer, lang, style)
                
                history.append({
                    "user": question,
                    "buddy": answer,
                    "timestamp": time.time(),
                    "lang": lang,
                    "service": "home_assistant"
                })
                try:
                    save_user_history(speaker, history)
                except Exception as save_err:
                    if DEBUG:
                        print(f"[Buddy] History save error: {save_err}")
                return True, speaker
    except Exception as ha_err:
        if DEBUG:
            print(f"[Buddy] Home Assistant error: {ha_err}")

    try:
        if should_search_internet(question):
            if DEBUG:
                print("[Buddy] 🌐 Processing internet search...")
            result = search_internet(question, lang)
            try:
                emotion, _ = analyze_emotion(question)
                result = adjust_emotional_response(result, emotion)
            except:
                pass
            speak_async(result, lang, style)
            
            history.append({
                "user": question,
                "buddy": result,
                "timestamp": time.time(),
                "lang": lang,
                "service": "internet_search"
            })
            try:
                save_user_history(speaker, history)
            except Exception as save_err:
                if DEBUG:
                    print(f"[Buddy] History save error: {save_err}")
            return True, speaker
    except Exception as search_err:
        if DEBUG:
            print(f"[Buddy] Internet search error: {search_err}")

    # Add narrative bookmark
    try:
        add_narrative_bookmark(speaker, question)
    except Exception as bookmark_err:
        if DEBUG:
            print(f"[Buddy] Bookmark error: {bookmark_err}")

    # Main LLM processing
    if DEBUG:
        print(f"[Buddy] 🧠 Processing with LLM for {speaker}: {question!r}")
    
    llm_start_time = time.time()
    
    try:
        ask_llama3_streaming(
            question=question,
            name=speaker,
            history=history,
            lang=lang,
            conversation_mode=conversation_mode,
            style=style,
            speaker_traits=traits,
            speaker_context=ctx
        )
        
        if DEBUG:
            print(f"[TIMING] ⏱️ LLM processing time: {time.time() - llm_start_time:.2f}s")
            
    except Exception as llm_err:
        if DEBUG:
            print(f"[Buddy] LLM processing error: {llm_err}")
        
        fallback_responses = {
            "en": "Sorry, I'm having trouble thinking right now. Could you try asking again?",
            "pl": "Przepraszam, mam problemy z myśleniem. Możesz spróbować ponownie?",
            "it": "Scusa, ho problemi a pensare ora. Puoi riprovare?"
        }
        fallback = fallback_responses.get(lang, fallback_responses["en"])
        speak_async(fallback, lang, style)
        
        history.append({
            "user": question,
            "buddy": fallback,
            "timestamp": time.time(),
            "lang": lang,
            "error": "llm_error"
        })
        try:
            save_user_history(speaker, history)
        except Exception as save_err:
            if DEBUG:
                print(f"[Buddy] History save error: {save_err}")

    # ✅ RETURN both conversation status AND current speaker
    return True, speaker

def main():
    """ENHANCED: Multi-user conversation with proper audio reset - 2025-06-30 06:12:31"""
    global in_session, known_users, audio_worker_active
    
    # ✅ CURRENT TIME: 2025-06-30 06:12:31
    current_datetime = "2025-06-30 06:12:31"
    current_user = "Daveydrz"
    
    print(f"[Buddy] 🕐 Starting at {current_datetime} UTC")
    print(f"[Buddy] 👤 System user: {current_user}")
    
    # Wake word setup
    access_key = "/PLJ88d4+jDeVO4zaLFaXNkr6XLgxuG7dh+6JcraqLhWQlk3AjMy9Q=="
    keyword_paths = [r"hey-buddy_en_windows_v3_0_0.ppn"]
    porcupine = pvporcupine.create(access_key=access_key, keyword_paths=keyword_paths)
    pa = pyaudio.PyAudio()
    stream = pa.open(rate=porcupine.sample_rate, channels=1, format=pyaudio.paInt16,
                     input=True, frames_per_buffer=porcupine.frame_length)
    
    # ✅ CRITICAL FIX: Better voice database loading and validation
    print("[Buddy] 🔧 Loading voice database...")
    load_voice_database()
    
    if len(known_users) > 0:
        print(f"[Buddy] 📚 Found {len(known_users)} users in database:")
        valid_voices = 0
        text_embeddings = 0
        invalid_entries = 0
        
        for name, embedding in known_users.items():
            if isinstance(embedding, list):
                emb_len = len(embedding)
                if emb_len == 256:
                    print(f"[Buddy] ✅ {name}: {emb_len} dims (VOICE)")
                    valid_voices += 1
                elif emb_len == 384:
                    print(f"[Buddy] ⚠️ {name}: {emb_len} dims (TEXT)")
                    text_embeddings += 1
                else:
                    print(f"[Buddy] ❌ {name}: {emb_len} dims (UNKNOWN)")
                    invalid_entries += 1
            else:
                print(f"[Buddy] ❌ {name}: invalid embedding type")
                invalid_entries += 1
        
        print(f"[Buddy] Summary: {valid_voices} voice, {text_embeddings} text, {invalid_entries} invalid")
    else:
        print("[Buddy] 📁 Empty database - will register users on first interaction")
    
    # Start systems
    print("[Buddy] 🚀 Starting background systems...")
    threading.Thread(target=start_parallel_detector_safe, daemon=True).start()
    
    audio_worker_active = True
    threading.Thread(target=unified_audio_worker, daemon=True).start()
    print("[Buddy] ✅ All systems ready!")
    
    # ✅ MULTI-USER: Session variables with NO persistent speaker
    in_session = False
    session_timeout = 300  # 5 minutes
    current_speaker = None  # ✅ CRITICAL: Start with NO speaker for voice identification
    history = []
    last_activity_time = 0
    
    print("[Buddy] 👤 Multi-user mode: Each speaker will be identified automatically")

    try:
        print("[Buddy] 👂 Ready! Say 'Hey Buddy' to start conversation...")
        
        while True:
            if not in_session:
                # ✅ WAIT FOR INITIAL WAKE WORD ONLY
                try:
                    pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
                    pcm = np.frombuffer(pcm, dtype=np.int16)
                    
                    if porcupine.process(pcm) >= 0:
                        print("[Buddy] 🎤 Wake word detected! Identifying speaker...")
                        
                        # ✅ CRITICAL: Reset audio system IMMEDIATELY
                        reset_audio_system_before_response()
                        
                        # ✅ CRITICAL: Reset for voice identification
                        current_speaker = None
                        history = []
                        
                        # ✅ ENTER CONVERSATION MODE
                        in_session = True
                        last_activity_time = time.time()
                        
                        print("[Buddy] 🔍 Please speak - I'll identify who you are...")
                        
                except Exception as wake_err:
                    if DEBUG:
                        print(f"[Buddy] Wake word error: {wake_err}")
                    time.sleep(0.1)
                    continue
                    
            else:
                # ✅ NATURAL CONVERSATION MODE: Listen continuously with voice ID
                try:
                    # Check for long inactivity timeout
                    if time.time() - last_activity_time > session_timeout:
                        if DEBUG:
                            print(f"[Buddy] Long inactivity timeout after {session_timeout}s")
                        
                        # ✅ RESET BEFORE TIMEOUT MESSAGE
                        reset_audio_system_before_response()
                        
                        speak_async("I'll go to sleep now. Say 'Hey Buddy' when you want to chat again.", "en")
                        
                        # ✅ IMPROVED: Better wait for speech completion
                        _wait_for_speech_completion(timeout=8)
                        
                        # ✅ RESET EVERYTHING on timeout
                        in_session = False
                        current_speaker = None
                        history = []
                        print("[Buddy] 💤 Session ended. Say 'Hey Buddy' to start again.")
                        continue
                    
                    if DEBUG:
                        print(f"[Buddy] 🎧 Listening... (current speaker: {current_speaker or 'UNKNOWN'})")
                    
                    # ✅ IMPROVED: Better wait for Buddy to finish talking
                    _wait_for_speech_completion(timeout=5)
                    
                    # ✅ CRITICAL: Get both conversation status AND updated speaker
                    previous_speaker = current_speaker
                    conversation_continues, current_speaker = handle_user_interaction(current_speaker, history)
                    
                    if conversation_continues:
                        last_activity_time = time.time()  # Reset timeout
                        
                        # ✅ HANDLE SPEAKER CHANGES
                        if previous_speaker != current_speaker:
                            print(f"[Buddy] 🔄 Speaker change detected: {previous_speaker} → {current_speaker}")
                            
                            # ✅ RESET SYSTEM ON SPEAKER CHANGE
                            reset_audio_system_before_response()
                        
                        if DEBUG:
                            print(f"[Buddy] ✅ Conversation continues with {current_speaker}...")
                    else:
                        # User explicitly ended conversation
                        in_session = False
                        current_speaker = None
                        history = []
                        print("[Buddy] 👋 Conversation ended by user. Say 'Hey Buddy' to start again.")
                    
                except Exception as session_err:
                    if DEBUG:
                        print(f"[Buddy] Session error: {session_err}")
                    
                    try:
                        # ✅ RESET BEFORE ERROR MESSAGE
                        reset_audio_system_before_response()
                        
                        speak_async("Sorry, technical issue. Let's continue.", "en")
                        last_activity_time = time.time()
                    except:
                        print("[Buddy] ⚠️ Critical error - ending conversation")
                        in_session = False
                        current_speaker = None
                        history = []
                    continue
                
    except KeyboardInterrupt:
        print("\n[Buddy] 👋 Shutting down...")
        
    except Exception as main_err:
        print(f"[Buddy] ❌ Critical error: {main_err}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        
    finally:
        # ✅ ENHANCED: Save voice database with verification
        print("[Buddy] 💾 Saving voice database...")
        try:
            # Show what we're saving
            if known_users:
                print(f"[Buddy] Saving {len(known_users)} users:")
                for name, emb in known_users.items():
                    if isinstance(emb, list):
                        emb_len = len(emb)
                        emb_type = "VOICE" if emb_len == 256 else "TEXT" if emb_len == 384 else "UNKNOWN"
                        print(f"[Buddy] - {name}: {emb_len} dims ({emb_type})")
            
            save_voice_database()
            
            # Verify save
            if os.path.exists(known_users_path):
                file_size = os.path.getsize(known_users_path)
                print(f"[Buddy] ✅ Voice database saved: {file_size} bytes")
            else:
                print("[Buddy] ❌ Database file not found after save!")
                
        except Exception as save_err:
            print(f"[Buddy] ❌ Save failed: {save_err}")
        
        # Cleanup
        print("[Buddy] 🧹 Cleanup...")
        audio_worker_active = False
        
        try:
            audio_queue.put(None)
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                    audio_queue.task_done()
                except queue.Empty:
                    break
        except:
            pass
        
        try:
            if 'stream' in locals() and stream:
                stream.stop_stream()
                stream.close()
            if 'pa' in locals() and pa:
                pa.terminate()
            if 'porcupine' in locals() and porcupine:
                porcupine.delete()
        except:
            pass
        
        try:
            if 'executor' in globals():
                executor.shutdown(wait=True)
        except:
            pass
        
        print("[Buddy] ✅ Shutdown complete!")

def _wait_for_speech_completion(timeout=5):
    """Helper function to wait for Buddy to finish speaking"""
    # Wait for audio queue to empty
    try:
        audio_queue.join()
    except:
        pass
    
    # Wait for buddy_talking flag to clear
    timeout_start = time.time()
    while buddy_talking.is_set():
        time.sleep(0.05)
        if time.time() - timeout_start > timeout:
            print(f"[Buddy] ⚠️ Speech timeout after {timeout}s - force clearing")
            buddy_talking.clear()
            break

def reset_audio_system_before_response():
    """ENHANCED: Complete audio system reset - 2025-06-30 06:21:23"""
    global full_duplex_interrupt_flag, current_audio_playback, playback_start_time
    
    print("[RESET] 🔄 Resetting audio system for new response...")
    
    # ✅ FORCE CLEAR: All interrupt/talking flags
    full_duplex_interrupt_flag.clear()
    buddy_talking.clear()
    vad_triggered.clear()
    vad_thread_active.clear()
    
    # ✅ CLEAR PLAYBACK: Reset playback state
    try:
        with audio_lock:
            if current_audio_playback:
                if current_audio_playback.is_playing():
                    current_audio_playback.stop()
                current_audio_playback = None
            playback_start_time = None
    except Exception as playback_err:
        if DEBUG:
            print(f"[RESET] Playback cleanup error: {playback_err}")
        current_audio_playback = None
        playback_start_time = None
    
    # ✅ CLEAR QUEUE: Remove any pending audio with timeout
    cleared_count = 0
    start_time = time.time()
    while not audio_queue.empty() and time.time() - start_time < 1.0:
        try:
            item = audio_queue.get_nowait()
            if item is not None:
                cleared_count += 1
            audio_queue.task_done()
        except queue.Empty:
            break
        except Exception as clear_err:
            if DEBUG:
                print(f"[RESET] Queue clear error: {clear_err}")
            break
    
    if cleared_count > 0:
        print(f"[RESET] Cleared {cleared_count} pending audio chunks")
    
    # ✅ ENHANCED: Reset AEC reference buffer
    try:
        global ref_audio_buffer
        with ref_audio_lock:
            ref_audio_buffer = np.zeros(8000, dtype=np.int16)
    except Exception as aec_err:
        if DEBUG:
            print(f"[RESET] AEC reset error: {aec_err}")
    
    # ✅ SMALL DELAY: Ensure all threads see the changes
    time.sleep(0.02)
    
    print("[RESET] ✅ Audio system completely reset and ready")
    
    # ✅ FINAL VERIFICATION
    if DEBUG:
        print(f"[RESET] Final state check:")
        print(f"[RESET]   - interrupt_flag: {full_duplex_interrupt_flag.is_set()}")
        print(f"[RESET]   - buddy_talking: {buddy_talking.is_set()}")
        print(f"[RESET]   - queue_empty: {audio_queue.empty()}")

def ask_llama3_streaming(question, name, history, lang=DEFAULT_LANG, conversation_mode="auto", style=None, speaker_traits=None, speaker_context=None):
    """
    Optimized streaming LLM function with immediate response generation
    """
    
    if DEBUG:
        print(f"[Buddy] Starting streaming response for: {question[:50]}...")
    
    # Update memories first (in background to avoid blocking)
    threading.Thread(target=update_thematic_memory, args=(name, question), daemon=True).start()
    
    # Load context efficiently
    topics = get_frequent_topics(name, top_n=3)
    user_tones = {"Daveydrz": "friendly", "Dawid": "friendly", "Anna": "professional", "Guest": "friendly"}
    tone_style = user_tones.get(name, "friendly")
    reply_length = decide_reply_length(question, conversation_mode)
    emotion_mode = session_emotion_mode.get(name)
    beliefs = load_buddy_beliefs()
    bookmarks = get_narrative_bookmarks(name)
    recent_tone = get_recent_user_tone(history)

    # Load extended context
    long_term = get_long_term_memory(name) or {}
    dynamic_knowledge = load_dynamic_knowledge().get(name, {})
    personality_traits = speaker_traits or load_personality_traits()
    context = speaker_context or conversation_contexts.setdefault(name, ConversationContext())
    context_topic = context.get_last_topic()
    context_summary = context.get_topic_summary(context_topic) if context_topic else ""

    # Build optimized system message
    personality = build_personality_prompt(tone_style, emotion_mode, beliefs, recent_tone, bookmarks)
    lang_map = {"pl": "Polish", "en": "English", "it": "Italian"}
    lang_name = lang_map.get(lang, "English")
    
    # Streamlined system message for faster processing
    sys_msg = f"""{personality}

CRITICAL INSTRUCTIONS:
- Always respond in {lang_name}
- Use natural, conversational language (1-2 sentences unless more detail requested)
- No markdown, code blocks, or special formatting
- Be immediate and engaging
- Current date: 2025-06-27
- User timezone: UTC
"""

    # Add personality context (keep concise)
    if personality_traits:
        key_traits = [k for k, v in personality_traits.items() if v > 0.6][:3]
        if key_traits:
            sys_msg += f"Your personality emphasis: {', '.join(key_traits)}\n"

    # Add user interests (limited to avoid bloat)
    if topics:
        sys_msg += f"User interests: {', '.join(topics[:2])}\n"
    
    # Add relevant facts
    facts = build_user_facts(name)
    if facts:
        sys_msg += f"Key facts: {' '.join(facts[:2])}\n"
    
    # Add conversation context if relevant
    if context_topic and len(context_summary) > 10:
        sys_msg += f"Current topic: {context_topic}. Context: {context_summary[:100]}...\n"

    # Add relevant long-term memories (only if directly related)
    if long_term:
        relevant_memories = []
        question_words = set(question.lower().split())
        for key, value in long_term.items():
            key_words = set(key.lower().replace('_', ' ').split())
            if question_words.intersection(key_words):
                relevant_memories.append(f"{key.replace('_', ' ')}: {str(value)[:50]}")
        if relevant_memories:
            sys_msg += f"Relevant memories: {'; '.join(relevant_memories[:2])}\n"

    # Build message array efficiently
    messages = [{"role": "system", "content": sys_msg}]
    
    # Add recent history (keep minimal for speed)
    recent_history = history[-2:] if len(history) > 2 else history
    for h in recent_history:
        if isinstance(h, dict) and "user" in h and "buddy" in h:
            messages.append({"role": "user", "content": h["user"]})
            messages.append({"role": "assistant", "content": h["buddy"]})
    
    # Add current question
    messages.append({"role": "user", "content": question})

    if DEBUG:
        print("[Buddy][LLM] System message preview:")
        print(sys_msg[:200] + "..." if len(sys_msg) > 200 else sys_msg)

    # Initialize response tracking
    full_text = ""
    response_started = False

    try:
        start_time = time.time()
        
        if DEBUG:
            print("[Buddy] Starting LLM streaming request...")

        # Call the streaming function with optimized parameters
        full_text = ask_llama3_openai_streaming(
            messages,
            model="llama3",
            max_tokens=80,  # Increased slightly for better responses
            temperature=0.6,  # Slightly higher for more personality
            lang=lang,
            style=style or {"emotion": "neutral"}
        )

        if DEBUG:
            generation_time = time.time() - start_time
            print(f"[TIMING] LLM streaming completed in {generation_time:.2f}s")
            if full_text:
                print(f"[Buddy] Generated response: {full_text[:100]}...")

        # Check if we got a valid response
        if full_text and full_text.strip():
            response_started = True
            
            # Clean up the response
            full_text = full_text.strip()
            
            # Remove any unwanted artifacts
            full_text = re.sub(r'^(Buddy:|Assistant:)\s*', '', full_text, flags=re.IGNORECASE)
            full_text = re.sub(r'```.*?```', '', full_text, flags=re.DOTALL)
            full_text = full_text.strip()

        else:
            print("[Buddy] Warning: Empty LLM response received")

    except requests.exceptions.Timeout:
        print("[Buddy] LLM request timed out")
        full_text = "Sorry, I'm thinking a bit slowly right now. Can you repeat that?"
        speak_async(full_text, lang=lang, style=style)
        
    except requests.exceptions.ConnectionError:
        print("[Buddy] LLM connection failed")
        full_text = "I'm having trouble connecting my thoughts right now. Give me a moment."
        speak_async(full_text, lang=lang, style=style)
        
    except Exception as e:
        print(f"[Buddy] LLM error: {type(e).__name__}: {e}")
        full_text = "Something's not quite right in my head right now. Try asking again?"
        speak_async(full_text, lang=lang, style=style)

    # Handle the response
    if full_text and full_text.strip():
        # Update conversation history
        history_entry = {
            "user": question,
            "buddy": full_text,
            "timestamp": time.time(),
            "lang": lang
        }
        history.append(history_entry)
        
        # Update recent responses tracking
        normalized_response = full_text.strip().lower()
        LAST_FEW_BUDDY.append(normalized_response)
        if len(LAST_FEW_BUDDY) > 5:
            LAST_FEW_BUDDY.pop(0)
        
        # Save history asynchronously to avoid blocking
        if not FAST_MODE:
            threading.Thread(
                target=save_user_history, 
                args=(name, history), 
                daemon=True
            ).start()
        
        if DEBUG:
            print(f"[Buddy] Response added to history. Total entries: {len(history)}")
            
    else:
        # Fallback for completely empty responses
        if DEBUG:
            print("[Buddy] No valid response generated, using fallback")
        
        fallback_responses = {
            "en": "I'm not sure how to respond to that. Could you rephrase?",
            "pl": "Nie jestem pewien jak odpowiedzieć. Możesz przeformułować?",
            "it": "Non sono sicuro di come rispondere. Potresti riformulare?"
        }
        fallback = fallback_responses.get(lang, fallback_responses["en"])
        speak_async(fallback, lang=lang, style=style)
        
        # Still add to history for continuity
        history.append({
            "user": question,
            "buddy": fallback,
            "timestamp": time.time(),
            "lang": lang,
            "fallback": True
        })

    return full_text

if __name__ == "__main__":
    main()