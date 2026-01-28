"""
Production-Ready Thai-Isan Speaker Recording System
Professional interface for recording native Thai and Isan speakers
"""

import os
import json
import logging
import threading
import queue
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import librosa
import librosa.display
from scipy import signal
from scipy.signal import spectrogram
import wave
import tempfile
import json

# Quality assessment
from sklearn.metrics import mean_squared_error
from scipy.stats import kurtosis, skew

logger = logging.getLogger(__name__)

@dataclass
class RecordingSession:
    """Recording session information"""
    session_id: str
    speaker_id: str
    language: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_recordings: int = 0
    successful_recordings: int = 0
    average_quality_score: float = 0.0
    notes: str = ""
    equipment_used: str = ""
    recording_conditions: str = ""

@dataclass
class RecordingPrompt:
    """Text prompt for recording"""
    prompt_id: str
    text: str
    language: str
    difficulty: str  # 'easy', 'medium', 'hard'
    category: str  # 'greeting', 'daily', 'formal', 'cultural'
    phonetic_transcription: str = ""
    expected_duration: float = 0.0
    cultural_notes: str = ""
    tone_pattern: str = ""

@dataclass
class AudioQualityMetrics:
    """Audio quality assessment metrics"""
    signal_to_noise_ratio: float
    total_harmonic_distortion: float
    background_noise_level: float
    clipping_detected: bool
    dynamic_range: float
    spectral_centroid: float
    zero_crossing_rate: float
    spectral_rolloff: float
    mfcc_deltas: np.ndarray
    quality_score: float

@dataclass
class RecordingInterfaceConfig:
    """Configuration for recording interface"""
    
    # Audio settings
    sample_rate: int = 48000
    bit_depth: int = 24
    channels: int = 1
    chunk_size: int = 1024
    
    # Quality thresholds
    min_snr_db: float = 30.0
    max_background_noise: float = -40.0
    min_clarity_score: float = 0.8
    min_duration: float = 1.0
    max_duration: float = 15.0
    
    # User interface
    theme: str = "DarkBlue"
    font_size: int = 12
    show_waveform: bool = True
    show_spectrum: bool = True
    real_time_monitoring: bool = True
    
    # Storage settings
    base_data_dir: str = "./recording_data"
    backup_enabled: bool = True
    compression_format: str = "flac"
    
    # Language settings
    supported_languages: List[str] = None
    
    # Recording session settings
    max_session_duration: int = 120  # minutes
    break_interval: int = 30  # minutes
    max_consecutive_failures: int = 5
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["th", "tts"]

class ProfessionalAudioRecorder:
    """Professional audio recorder with real-time monitoring and quality assessment"""
    
    def __init__(self, config: RecordingInterfaceConfig):
        self.config = config
        self.is_recording = False
        self.is_monitoring = False
        self.audio_queue = queue.Queue()
        self.recording_data = []
        self.monitoring_data = []
        self.recording_thread = None
        self.monitoring_thread = None
        self.current_session = None
        self.current_prompt = None
        
        # Setup directories
        self._setup_directories()
        
        # Initialize audio devices
        self._initialize_audio_devices()
        
        # Quality assessment
        self.quality_threshold = config.min_clarity_score
        
    def _setup_directories(self):
        """Setup directory structure"""
        dirs = [
            'sessions',
            'audio/thai',
            'audio/isan',
            'quality_reports',
            'backup',
            'temp'
        ]
        
        for dir_name in dirs:
            Path(self.config.base_data_dir, dir_name).mkdir(parents=True, exist_ok=True)
    
    def _initialize_audio_devices(self):
        """Initialize and test audio devices"""
        try:
            # List available devices
            self.devices = sd.query_devices()
            logger.info(f"Found {len(self.devices)} audio devices")
            
            # Find default input device
            self.input_device = sd.default.device[0]
            self.output_device = sd.default.device[1]
            
            logger.info(f"Input device: {self.input_device}")
            logger.info(f"Output device: {self.output_device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio devices: {e}")
            raise
    
    def start_recording_session(self, speaker_id: str, language: str) -> str:
        """Start a new recording session"""
        session_id = f"{speaker_id}_{language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = RecordingSession(
            session_id=session_id,
            speaker_id=speaker_id,
            language=language,
            start_time=datetime.now()
        )
        
        logger.info(f"Started recording session: {session_id}")
        return session_id
    
    def record_audio(self, duration: float, prompt: RecordingPrompt) -> Optional[np.ndarray]:
        """Record high-quality audio with real-time monitoring"""
        try:
            self.current_prompt = prompt
            logger.info(f"Recording prompt: {prompt.text} (duration: {duration}s)")
            
            # Start monitoring
            if self.config.real_time_monitoring:
                self.start_monitoring()
            
            # Record audio
            self.is_recording = True
            self.recording_data = []
            
            # Record with callback for real-time processing
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype='float32',
                device=self.input_device,
                callback=self._recording_callback
            ):
                time.sleep(duration)
            
            self.is_recording = False
            
            # Stop monitoring
            if self.config.real_time_monitoring:
                self.stop_monitoring()
            
            # Convert to numpy array
            if self.recording_data:
                audio = np.concatenate(self.recording_data, axis=0)
                
                # Quality assessment
                quality_metrics = self.assess_audio_quality(audio)
                
                if quality_metrics.quality_score >= self.quality_threshold:
                    logger.info(f"Recording completed. Quality score: {quality_metrics.quality_score:.2f}")
                    return audio
                else:
                    logger.warning(f"Recording quality too low: {quality_metrics.quality_score:.2f}")
                    return None
            else:
                logger.error("No audio data recorded")
                return None
                
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            return None
    
    def _recording_callback(self, indata, frames, time_info, status):
        """Callback for audio recording"""
        if status:
            logger.warning(f"Recording status: {status}")
        
        if self.is_recording:
            self.recording_data.append(indata.copy())
    
    def start_monitoring(self):
        """Start real-time audio monitoring"""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.start()
        logger.info("Started audio monitoring")
    
    def stop_monitoring(self):
        """Stop real-time audio monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Stopped audio monitoring")
    
    def _monitoring_loop(self):
        """Real-time monitoring loop"""
        while self.is_monitoring:
            try:
                # Record small chunk for monitoring
                monitoring_chunk = sd.rec(
                    int(0.5 * self.config.sample_rate),
                    samplerate=self.config.sample_rate,
                    channels=self.config.channels,
                    dtype='float32'
                )
                sd.wait()
                
                # Analyze the chunk
                if len(monitoring_chunk) > 0:
                    self._analyze_monitoring_chunk(monitoring_chunk.flatten())
                    
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                break
            
            time.sleep(0.1)
    
    def _analyze_monitoring_chunk(self, audio_chunk: np.ndarray):
        """Analyze monitoring audio chunk"""
        # Calculate basic metrics
        rms = np.sqrt(np.mean(audio_chunk**2))
        peak = np.max(np.abs(audio_chunk))
        
        # Check for clipping
        if peak > 0.95:
            logger.warning("Clipping detected! Please adjust input volume.")
        
        # Check for silence
        if rms < 0.01:
            logger.warning("Very low input level detected.")
    
    def assess_audio_quality(self, audio: np.ndarray) -> AudioQualityMetrics:
        """Comprehensive audio quality assessment"""
        try:
            # Basic metrics
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))
            
            # Signal-to-noise ratio (simplified)
            signal_power = np.mean(audio**2)
            noise_power = np.mean(audio[audio < 0.1 * np.max(audio)]**2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            # Total Harmonic Distortion (simplified)
            thd = self._calculate_thd(audio)
            
            # Background noise level
            background_noise = self._estimate_background_noise(audio)
            
            # Clipping detection
            clipping_detected = peak > 0.95
            
            # Dynamic range
            dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
            
            # Spectral features
            spectral_centroid = self._calculate_spectral_centroid(audio)
            spectral_rolloff = self._calculate_spectral_rolloff(audio)
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=self.config.sample_rate, n_mfcc=13)
            mfcc_deltas = np.mean(librosa.feature.delta(mfcc), axis=1)
            
            # Overall quality score (simplified)
            quality_score = self._calculate_quality_score(snr, thd, clipping_detected, dynamic_range)
            
            return AudioQualityMetrics(
                signal_to_noise_ratio=snr,
                total_harmonic_distortion=thd,
                background_noise_level=background_noise,
                clipping_detected=clipping_detected,
                dynamic_range=dynamic_range,
                spectral_centroid=spectral_centroid,
                zero_crossing_rate=zero_crossing_rate,
                spectral_rolloff=spectral_rolloff,
                mfcc_deltas=mfcc_deltas,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            # Return default metrics
            return AudioQualityMetrics(
                signal_to_noise_ratio=20.0,
                total_harmonic_distortion=0.1,
                background_noise_level=-30.0,
                clipping_detected=False,
                dynamic_range=40.0,
                spectral_centroid=1000.0,
                zero_crossing_rate=0.1,
                spectral_rolloff=5000.0,
                mfcc_deltas=np.zeros(13),
                quality_score=0.5
            )
    
    def _calculate_thd(self, audio: np.ndarray) -> float:
        """Calculate Total Harmonic Distortion (simplified)"""
        try:
            # FFT
            fft = np.fft.fft(audio)
            magnitude = np.abs(fft)
            
            # Find fundamental frequency
            freqs = np.fft.fftfreq(len(audio), 1/self.config.sample_rate)
            fundamental_idx = np.argmax(magnitude[:len(magnitude)//2])
            fundamental_freq = abs(freqs[fundamental_idx])
            
            # Calculate THD (simplified)
            if fundamental_freq > 0:
                harmonics = []
                for i in range(2, 6):  # First 4 harmonics
                    harmonic_freq = fundamental_freq * i
                    harmonic_idx = int(harmonic_freq * len(audio) / self.config.sample_rate)
                    if harmonic_idx < len(magnitude):
                        harmonics.append(magnitude[harmonic_idx])
                
                if harmonics:
                    thd = np.sqrt(np.sum(np.array(harmonics)**2)) / magnitude[fundamental_idx]
                    return min(thd, 1.0)  # Cap at 1.0
            
            return 0.0
            
        except Exception:
            return 0.1
    
    def _estimate_background_noise(self, audio: np.ndarray) -> float:
        """Estimate background noise level"""
        try:
            # Simple noise estimation using quiet parts
            quiet_threshold = 0.1 * np.max(audio)
            quiet_parts = audio[audio < quiet_threshold]
            
            if len(quiet_parts) > 0:
                noise_power = np.mean(quiet_parts**2)
                return 10 * np.log10(noise_power + 1e-10)
            else:
                return -40.0  # Default low noise level
                
        except Exception:
            return -30.0
    
    def _calculate_spectral_centroid(self, audio: np.ndarray) -> float:
        """Calculate spectral centroid"""
        try:
            centroid = librosa.feature.spectral_centroid(y=audio, sr=self.config.sample_rate)[0]
            return np.mean(centroid)
        except Exception:
            return 1000.0
    
    def _calculate_spectral_rolloff(self, audio: np.ndarray) -> float:
        """Calculate spectral rolloff"""
        try:
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.config.sample_rate)[0]
            return np.mean(rolloff)
        except Exception:
            return 5000.0
    
    def _calculate_quality_score(self, snr: float, thd: float, clipping: bool, dynamic_range: float) -> float:
        """Calculate overall quality score"""
        # SNR score (0-1)
        snr_score = min(max((snr - 10) / 30, 0), 1)
        
        # THD score (0-1)
        thd_score = max(1 - thd * 10, 0)
        
        # Clipping penalty
        clipping_penalty = 0.5 if clipping else 0.0
        
        # Dynamic range score
        dr_score = min(dynamic_range / 60, 1)
        
        # Combined score
        quality_score = (snr_score + thd_score + dr_score) / 3 - clipping_penalty
        return max(quality_score, 0)
    
    def save_recording(self, audio: np.ndarray, session_id: str, prompt_id: str) -> str:
        """Save recording with metadata"""
        try:
            recording_id = f"{session_id}_{prompt_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save audio file
            audio_path = Path(self.config.base_data_dir) / "audio" / self.current_session.language / f"{recording_id}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            
            sf.write(audio_path, audio, self.config.sample_rate)
            
            # Update session
            if self.current_session:
                self.current_session.total_recordings += 1
                self.current_session.successful_recordings += 1
            
            logger.info(f"Saved recording: {recording_id}")
            return str(recording_id)
            
        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
            return ""
    
    def create_gui_interface(self):
        """Create professional GUI interface for recording"""
        layout = [
            [sg.Text("Thai-Isan Speaker Recording System", font=("Arial", 16))],
            [sg.HorizontalSeparator()],
            
            # Speaker info
            [sg.Text("Speaker ID:", size=(15, 1)), sg.Input(key="-SPEAKER_ID-", size=(20, 1))],
            [sg.Text("Language:", size=(15, 1)), sg.Combo(["th", "tts"], default_value="th", key="-LANGUAGE-")],
            [sg.Text("Session ID:", size=(15, 1)), sg.Text("Not started", key="-SESSION_ID-", size=(20, 1))],
            [sg.HorizontalSeparator()],
            
            # Current prompt
            [sg.Text("Current Prompt:", font=("Arial", 12))],
            [sg.Multiline("", key="-PROMPT_TEXT-", size=(60, 3), disabled=True)],
            [sg.Text("Phonetic:", size=(10, 1)), sg.Text("", key="-PHONETIC-", size=(50, 1))],
            [sg.HorizontalSeparator()],
            
            # Recording controls
            [sg.Button("Start Session", key="-START_SESSION-", size=(15, 2)),
             sg.Button("Record", key="-RECORD-", size=(15, 2), disabled=True),
             sg.Button("Play", key="-PLAY-", size=(15, 2), disabled=True),
             sg.Button("Save", key="-SAVE-", size=(15, 2), disabled=True),
             sg.Button("Next Prompt", key="-NEXT_PROMPT-", size=(15, 2))],
            
            # Quality indicators
            [sg.Text("Quality Score:", size=(15, 1)), sg.Text("N/A", key="-QUALITY_SCORE-", size=(10, 1))],
            [sg.Text("SNR (dB):", size=(15, 1)), sg.Text("N/A", key="-SNR-", size=(10, 1))],
            [sg.Text("Clipping:", size=(15, 1)), sg.Text("N/A", key="-CLIPPING-", size=(10, 1))],
            
            # Progress
            [sg.Text("Progress:", size=(15, 1)), sg.Text("0/0", key="-PROGRESS-", size=(10, 1))],
            [sg.HorizontalSeparator()],
            
            # Status
            [sg.Text("Status:", size=(10, 1)), sg.Text("Ready", key="-STATUS-", size=(50, 1))],
            
            # Exit
            [sg.Button("Exit", key="-EXIT-", size=(15, 2))]
        ]
        
        return layout
    
    def run_gui_interface(self):
        """Run the GUI interface"""
        layout = self.create_gui_interface()
        
        window = sg.Window("Thai-Isan Speaker Recording System", layout, 
                          font=("Arial", self.config.font_size), 
                          element_justification='center')
        
        # Initialize prompts
        self._initialize_prompts()
        self.current_prompt_index = 0
        self.current_audio = None
        self.current_session_id = None
        
        while True:
            event, values = window.read(timeout=100)
            
            if event == sg.WINDOW_CLOSED or event == "-EXIT-":
                break
            
            elif event == "-START_SESSION-":
                self._handle_start_session(window, values)
            
            elif event == "-RECORD-":
                self._handle_record(window)
            
            elif event == "-PLAY-":
                self._handle_play(window)
            
            elif event == "-SAVE-":
                self._handle_save(window)
            
            elif event == "-NEXT_PROMPT-":
                self._handle_next_prompt(window)
        
        window.close()
    
    def _initialize_prompts(self):
        """Initialize recording prompts"""
        # This would be loaded from a file or database
        self.thai_prompts = [
            RecordingPrompt("th_001", "สวัสดีครับ", "th", "easy", "greeting", "sa-wat-dee-krap"),
            RecordingPrompt("th_002", "สบายดีไหมครับ", "th", "easy", "greeting", "sa-bai-dee-mai-krap"),
            RecordingPrompt("th_003", "ขอบคุณมากครับ", "th", "easy", "formal", "kop-kun-mak-krap"),
        ]
        
        self.isan_prompts = [
            RecordingPrompt("tts_001", "สบายดีบ่", "tts", "easy", "greeting", "sa-bai-dee-bor"),
            RecordingPrompt("tts_002", "กินข้าวหรือยัง", "tts", "easy", "greeting", "gin-kaao-reu-yang"),
            RecordingPrompt("tts_003", "ขอบคุณหลายๆ", "tts", "easy", "formal", "kop-kun-lai-lai"),
        ]
        
        self.all_prompts = self.thai_prompts + self.isan_prompts
    
    def _handle_start_session(self, window, values):
        """Handle start session button"""
        speaker_id = values["-SPEAKER_ID-"]
        language = values["-LANGUAGE-"]
        
        if not speaker_id:
            sg.popup_error("Please enter a speaker ID")
            return
        
        # Start session
        self.current_session_id = self.start_recording_session(speaker_id, language)
        window["-SESSION_ID-"].update(self.current_session_id)
        window["-STATUS-"].update("Session started")
        
        # Enable recording
        window["-RECORD-"].update(disabled=False)
        
        # Load first prompt
        self._load_current_prompt(window)
    
    def _load_current_prompt(self, window):
        """Load current prompt into interface"""
        if self.current_prompt_index < len(self.all_prompts):
            prompt = self.all_prompts[self.current_prompt_index]
            window["-PROMPT_TEXT-"].update(prompt.text)
            window["-PHONETIC-"].update(prompt.phonetic_transcription)
            
            # Update progress
            window["-PROGRESS-"].update(f"{self.current_prompt_index + 1}/{len(self.all_prompts)}")
    
    def _handle_record(self, window):
        """Handle record button"""
        if not self.current_prompt:
            sg.popup_error("No prompt loaded")
            return
        
        window["-STATUS-"].update("Recording...")
        window.refresh()
        
        # Record audio
        duration = min(self.current_prompt.expected_duration + 2, 8) if self.current_prompt.expected_duration > 0 else 5
        audio = self.record_audio(duration, self.current_prompt)
        
        if audio is not None:
            self.current_audio = audio
            
            # Assess quality
            quality_metrics = self.assess_audio_quality(audio)
            
            # Update quality indicators
            window["-QUALITY_SCORE-"].update(f"{quality_metrics.quality_score:.2f}")
            window["-SNR-"].update(f"{quality_metrics.signal_to_noise_ratio:.1f}")
            window["-CLIPPING-"].update("Yes" if quality_metrics.clipping_detected else "No")
            
            # Enable play and save
            window["-PLAY-"].update(disabled=False)
            window["-SAVE-"].update(disabled=False)
            
            window["-STATUS-"].update("Recording completed")
        else:
            sg.popup_error("Recording failed or quality too low")
            window["-STATUS-"].update("Recording failed")
    
    def _handle_play(self, window):
        """Handle play button"""
        if self.current_audio is not None:
            window["-STATUS-"].update("Playing...")
            sd.play(self.current_audio, self.config.sample_rate)
            sd.wait()
            window["-STATUS-"].update("Ready")
    
    def _handle_save(self, window):
        """Handle save button"""
        if self.current_audio is not None and self.current_session_id:
            recording_id = self.save_recording(self.current_audio, self.current_session_id, 
                                              self.all_prompts[self.current_prompt_index].prompt_id)
            
            if recording_id:
                window["-STATUS-"].update(f"Saved: {recording_id}")
                
                # Move to next prompt
                self.current_prompt_index += 1
                if self.current_prompt_index < len(self.all_prompts):
                    self._load_current_prompt(window)
                    # Disable play and save until next recording
                    window["-PLAY-"].update(disabled=True)
                    window["-SAVE-"].update(disabled=True)
                else:
                    sg.popup_info("All prompts completed!")
                    window["-STATUS-"].update("Session completed")
            else:
                sg.popup_error("Failed to save recording")
    
    def _handle_next_prompt(self, window):
        """Handle next prompt button"""
        self.current_prompt_index += 1
        if self.current_prompt_index < len(self.all_prompts):
            self._load_current_prompt(window)
            # Reset audio
            self.current_audio = None
            window["-PLAY-"].update(disabled=True)
            window["-SAVE-"].update(disabled=True)
        else:
            sg.popup_info("All prompts completed!")

def main():
    """Main function to demonstrate the recording system"""
    # Initialize configuration
    config = RecordingInterfaceConfig()
    
    # Create recorder
    recorder = ProfessionalAudioRecorder(config)
    
    # Run GUI interface
    print("Starting Thai-Isan Speaker Recording System...")
    print("Make sure you have a microphone connected and configured.")
    print("The GUI interface will open shortly.")
    
    recorder.run_gui_interface()
    
    print("Recording session completed.")

if __name__ == "__main__":
    main()