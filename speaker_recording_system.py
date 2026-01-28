"""
Thai-Isan Speaker Recording System
Professional speaker recording interface for native Thai and Isan speakers
"""

import os
import json
import logging
import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import PySimpleGUI as sg
import threading
import queue
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
import wave
import tempfile
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range

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
    languages: List[str] = None
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["th", "tts"]

class AudioRecorder:
    """Professional audio recorder with real-time monitoring"""
    
    def __init__(self, config: RecordingInterfaceConfig):
        self.config = config
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        self.recording_data = []
        
        # Initialize audio
        self.sample_rate = config.sample_rate
        self.channels = config.channels
        
        # Find available audio devices
        self.devices = self.get_audio_devices()
        self.current_device = self.find_best_device()
    
    def get_audio_devices(self) -> List[Dict]:
        """Get list of available audio devices"""
        devices = []
        device_count = sd.query_hostapis()[0]['deviceCount']
        
        for i in range(device_count):
            device_info = sd.query_devices(i)
            if device_info['max_input_channels'] > 0:
                devices.append({
                    'id': i,
                    'name': device_info['name'],
                    'channels': device_info['max_input_channels'],
                    'sample_rate': device_info['default_samplerate']
                })
        
        return devices
    
    def find_best_device(self) -> int:
        """Find the best available audio device"""
        if not self.devices:
            return 0
        
        # Prefer devices with higher channel count and sample rate
        best_device = self.devices[0]
        for device in self.devices:
            if (device['channels'] > best_device['channels'] or
                (device['channels'] == best_device['channels'] and 
                 device['sample_rate'] > best_device['sample_rate'])):
                best_device = device
        
        return best_device['id']
    
    def start_recording(self):
        """Start audio recording"""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.recording_data = []
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()
    
    def record_audio(self):
        """Record audio in separate thread"""
        def callback(indata, frames, time_info, status):
            if self.is_recording:
                self.recording_data.append(indata.copy())
                if self.config.real_time_monitoring:
                    self.audio_queue.put(indata.copy())
        
        # Start recording
        with sd.InputStream(
            device=self.current_device,
            channels=self.channels,
            samplerate=self.sample_rate,
            callback=callback,
            blocksize=self.config.chunk_size
        ):
            while self.is_recording:
                time.sleep(0.1)
    
    def stop_recording(self) -> np.ndarray:
        """Stop recording and return audio data"""
        self.is_recording = False
        
        if self.recording_thread:
            self.recording_thread.join()
        
        # Combine all recorded chunks
        if self.recording_data:
            audio = np.concatenate(self.recording_data, axis=0)
            return audio.squeeze()
        else:
            return np.array([])
    
    def save_recording(self, audio: np.ndarray, filepath: str):
        """Save recording to file"""
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Save as WAV
        sf.write(filepath, audio, self.sample_rate, subtype='PCM_24')
        
        # Also save as FLAC if enabled
        if self.config.compression_format == "flac":
            flac_path = filepath.replace('.wav', '.flac')
            sf.write(flac_path, audio, self.sample_rate, format='FLAC')

class RecordingQualityAnalyzer:
    """Analyze recording quality in real-time"""
    
    def __init__(self, config: RecordingInterfaceConfig):
        self.config = config
        self.sample_rate = config.sample_rate
    
    def analyze_audio(self, audio: np.ndarray) -> Dict[str, float]:
        """Comprehensive audio quality analysis"""
        results = {}
        
        # Basic metrics
        results['duration'] = len(audio) / self.sample_rate
        results['amplitude'] = float(np.max(np.abs(audio)))
        results['rms'] = float(np.sqrt(np.mean(audio**2)))
        
        # Signal-to-noise ratio
        results['snr_db'] = self.calculate_snr(audio)
        
        # Background noise level
        results['background_noise_db'] = self.estimate_background_noise(audio)
        
        # Clarity score (0-1)
        results['clarity_score'] = self.calculate_clarity_score(audio)
        
        # Frequency analysis
        results['spectral_centroid'] = self.calculate_spectral_centroid(audio)
        results['spectral_rolloff'] = self.calculate_spectral_rolloff(audio)
        
        # Voice activity detection
        results['voice_activity_ratio'] = self.calculate_voice_activity(audio)
        
        return results
    
    def calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        # Simple SNR calculation
        signal_power = np.mean(audio**2)
        
        # Estimate noise in quiet segments
        noise_segments = self.get_quiet_segments(audio)
        if len(noise_segments) > 0:
            noise_power = np.mean([np.mean(seg**2) for seg in noise_segments])
        else:
            noise_power = signal_power * 0.01  # Assume 1% noise
        
        if noise_power > 0:
            return 10 * np.log10(signal_power / noise_power)
        return 60.0  # High SNR
    
    def estimate_background_noise(self, audio: np.ndarray) -> float:
        """Estimate background noise level"""
        quiet_segments = self.get_quiet_segments(audio)
        if len(quiet_segments) > 0:
            noise_level = np.mean([np.sqrt(np.mean(seg**2)) for seg in quiet_segments])
            return 20 * np.log10(noise_level + 1e-10)
        return -50.0  # Very low noise
    
    def get_quiet_segments(self, audio: np.ndarray, threshold: float = 0.1) -> List[np.ndarray]:
        """Extract quiet segments from audio"""
        # Simple energy-based segmentation
        frame_size = int(0.025 * self.sample_rate)  # 25ms frames
        hop_size = int(0.01 * self.sample_rate)     # 10ms hop
        
        segments = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            energy = np.mean(frame**2)
            if energy < threshold:
                segments.append(frame)
        
        return segments
    
    def calculate_clarity_score(self, audio: np.ndarray) -> float:
        """Calculate clarity score (0-1)"""
        # Based on SNR and spectral characteristics
        snr = self.calculate_snr(audio)
        
        # Convert SNR to clarity score
        if snr > 40:
            clarity = 1.0
        elif snr > 20:
            clarity = 0.8 + 0.2 * (snr - 20) / 20
        elif snr > 10:
            clarity = 0.5 + 0.3 * (snr - 10) / 10
        else:
            clarity = 0.1 + 0.4 * snr / 10
        
        return max(0.0, min(1.0, clarity))
    
    def calculate_spectral_centroid(self, audio: np.ndarray) -> float:
        """Calculate spectral centroid"""
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        frequency = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        
        # Only use positive frequencies
        positive_freqs = frequency[:len(frequency)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        
        if np.sum(positive_magnitude) > 0:
            return np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)
        return 0.0
    
    def calculate_spectral_rolloff(self, audio: np.ndarray, rolloff_point: float = 0.85) -> float:
        """Calculate spectral rolloff point"""
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        frequency = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        
        # Only use positive frequencies
        positive_freqs = frequency[:len(frequency)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        
        # Calculate cumulative magnitude
        cumulative_magnitude = np.cumsum(positive_magnitude)
        total_magnitude = np.sum(positive_magnitude)
        
        # Find rolloff point
        rolloff_threshold = rolloff_point * total_magnitude
        rolloff_idx = np.where(cumulative_magnitude >= rolloff_threshold)[0]
        
        if len(rolloff_idx) > 0:
            return positive_freqs[rolloff_idx[0]]
        return 0.0
    
    def calculate_voice_activity(self, audio: np.ndarray) -> float:
        """Calculate voice activity ratio"""
        # Simple energy-based voice activity detection
        frame_size = int(0.025 * self.sample_rate)
        hop_size = int(0.01 * self.sample_rate)
        
        voice_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            energy = np.mean(frame**2)
            
            if energy > 0.01:  # Threshold for voice activity
                voice_frames += 1
            total_frames += 1
        
        return voice_frames / total_frames if total_frames > 0 else 0.0

class ThaiIsanRecordingGUI:
    """Professional GUI for Thai-Isan speaker recording"""
    
    def __init__(self, config: RecordingInterfaceConfig):
        self.config = config
        self.recorder = AudioRecorder(config)
        self.quality_analyzer = RecordingQualityAnalyzer(config)
        
        # Current session
        self.current_session = None
        self.current_speaker = None
        self.current_prompt = None
        self.recording_history = []
        
        # Setup GUI
        self.setup_gui()
        self.setup_prompts()
    
    def setup_gui(self):
        """Setup the graphical user interface"""
        sg.theme(self.config.theme)
        
        # Main layout
        layout = [
            # Header
            [sg.Text("Thai-Isan Speaker Recording System", font=("Arial", 16, "bold"))],
            [sg.HorizontalSeparator()],
            
            # Speaker info
            [sg.Frame("Speaker Information", [
                [sg.Text("Speaker ID:"), sg.Input(key="-SPEAKER_ID-", size=(20, 1))],
                [sg.Text("Language:"), sg.Combo(["th", "tts"], default_value="th", key="-LANGUAGE-")],
                [sg.Text("Gender:"), sg.Combo(["male", "female"], default_value="male", key="-GENDER-")],
                [sg.Text("Age:"), sg.Combo(["young", "adult", "senior"], default_value="adult", key="-AGE-")],
                [sg.Text("Region:"), sg.Combo(["central", "northeastern", "northern", "southern"], 
                                            default_value="central", key="-REGION-")],
                [sg.Button("Register Speaker", key="-REGISTER-"), 
                 sg.Button("Load Speaker", key="-LOAD_SPEAKER-")]
            ])],
            
            # Recording controls
            [sg.Frame("Recording Controls", [
                [sg.Text("Status: Ready", key="-STATUS-", size=(30, 1))],
                [sg.Button("Start Recording", key="-START-", size=(15, 2)),
                 sg.Button("Stop Recording", key="-STOP-", size=(15, 2), disabled=True)],
                [sg.Text("Duration: 0.0s", key="-DURATION-", size=(15, 1))],
                [sg.ProgressBar(100, orientation='h', size=(30, 10), key="-PROGRESS-")]
            ])],
            
            # Quality monitoring
            [sg.Frame("Quality Monitoring", [
                [sg.Text("SNR: -- dB", key="-SNR-", size=(15, 1)),
                 sg.Text("Clarity: --", key="-CLARITY-", size=(15, 1))],
                [sg.Text("Noise: -- dB", key="-NOISE-", size=(15, 1)),
                 sg.Text("Activity: --%", key="-ACTIVITY-", size=(15, 1))],
                [sg.Canvas(key="-WAVEFORM-", size=(400, 100))]
            ])],
            
            # Text prompts
            [sg.Frame("Recording Prompts", [
                [sg.Text("Current Prompt:"), sg.Text("", key="-PROMPT_TEXT-", size=(50, 2))],
                [sg.Button("Next Prompt", key="-NEXT_PROMPT-"),
                 sg.Button("Previous Prompt", key="-PREV_PROMPT-")]
            ])],
            
            # Recording history
            [sg.Frame("Recent Recordings", [
                [sg.Listbox(values=[], key="-RECORDING_LIST-", size=(60, 6))],
                [sg.Button("Play", key="-PLAY_RECORDING-"),
                 sg.Button("Delete", key="-DELETE_RECORDING-")]
            ])],
            
            # Bottom buttons
            [sg.Button("Save Session", key="-SAVE_SESSION-"),
             sg.Button("Export Data", key="-EXPORT_DATA-"),
             sg.Button("Exit", key="-EXIT-")]
        ]
        
        # Create window
        self.window = sg.Window("Thai-Isan Recording System", layout, 
                              finalize=True, size=(800, 800))
        
        # Setup event handlers
        self.setup_event_handlers()
    
    def setup_event_handlers(self):
        """Setup event handlers for GUI"""
        self.event_handlers = {
            '-REGISTER-': self.register_speaker,
            '-LOAD_SPEAKER-': self.load_speaker,
            '-START-': self.start_recording,
            '-STOP-': self.stop_recording,
            '-NEXT_PROMPT-': self.next_prompt,
            '-PREV_PROMPT-': self.previous_prompt,
            '-PLAY_RECORDING-': self.play_recording,
            '-DELETE_RECORDING-': self.delete_recording,
            '-SAVE_SESSION-': self.save_session,
            '-EXPORT_DATA-': self.export_data,
            '-EXIT-': self.exit_application
        }
    
    def setup_prompts(self):
        """Setup recording prompts for Thai and Isan"""
        self.prompts = {
            'th': [
                RecordingPrompt("th_001", "สวัสดีครับ", "th", "easy", "greeting"),
                RecordingPrompt("th_002", "คุณสบายดีไหมครับ", "th", "easy", "greeting"),
                RecordingPrompt("th_003", "ผมมาจากกรุงเทพฯ", "th", "medium", "daily"),
                RecordingPrompt("th_004", "วันนี้อากาศดีมาก", "th", "easy", "daily"),
                RecordingPrompt("th_005", "ขอบคุณมากครับ", "th", "easy", "formal"),
            ],
            'tts': [
                RecordingPrompt("tts_001", "สบายดีบ่", "tts", "easy", "greeting"),
                RecordingPrompt("tts_002", "เฮาไปกินข้าวกันเด้อ", "tts", "easy", "daily"),
                RecordingPrompt("tts_003", "บ่เข้าใจที่คุณพูด", "tts", "medium", "daily"),
                RecordingPrompt("tts_004", "เฮามาจากอีสาน", "tts", "medium", "daily"),
                RecordingPrompt("tts_005", "ขอบคุณหลายๆ", "tts", "easy", "formal"),
            ]
        }
        
        self.current_prompt_index = {'th': 0, 'tts': 0}
    
    def register_speaker(self):
        """Register a new speaker"""
        speaker_info = {
            'speaker_id': self.window['-SPEAKER_ID-'].get(),
            'language': self.window['-LANGUAGE-'].get(),
            'gender': self.window['-GENDER-'].get(),
            'age_group': self.window['-AGE-'].get(),
            'region': self.window['-REGION-'].get(),
            'registration_time': datetime.now().isoformat()
        }
        
        if not speaker_info['speaker_id']:
            sg.popup_error("Please enter a speaker ID")
            return
        
        # Create speaker directory
        speaker_dir = Path(self.config.base_data_dir) / "speakers" / speaker_info['speaker_id']
        speaker_dir.mkdir(parents=True, exist_ok=True)
        
        # Save speaker info
        with open(speaker_dir / "info.json", 'w', encoding='utf-8') as f:
            json.dump(speaker_info, f, ensure_ascii=False, indent=2)
        
        self.current_speaker = speaker_info
        sg.popup(f"Speaker {speaker_info['speaker_id']} registered successfully!")
        
        # Update UI
        self.update_prompt_display()
    
    def load_speaker(self):
        """Load existing speaker"""
        speaker_id = self.window['-SPEAKER_ID-'].get()
        if not speaker_id:
            sg.popup_error("Please enter a speaker ID")
            return
        
        speaker_file = Path(self.config.base_data_dir) / "speakers" / speaker_id / "info.json"
        if not speaker_file.exists():
            sg.popup_error(f"Speaker {speaker_id} not found")
            return
        
        with open(speaker_file, 'r', encoding='utf-8') as f:
            self.current_speaker = json.load(f)
        
        # Update UI
        self.window['-LANGUAGE-'].update(self.current_speaker['language'])
        self.window['-GENDER-'].update(self.current_speaker['gender'])
        self.window['-AGE-'].update(self.current_speaker['age_group'])
        self.window['-REGION-'].update(self.current_speaker['region'])
        
        self.update_prompt_display()
        sg.popup(f"Speaker {speaker_id} loaded successfully!")
    
    def start_recording(self):
        """Start recording audio"""
        if not self.current_speaker:
            sg.popup_error("Please register or load a speaker first")
            return
        
        self.recorder.start_recording()
        self.window['-START-'].update(disabled=True)
        self.window['-STOP-'].update(disabled=False)
        self.window['-STATUS-'].update("Status: Recording...")
        
        # Start recording timer
        self.recording_start_time = time.time()
        self.update_recording_timer()
    
    def stop_recording(self):
        """Stop recording and process audio"""
        self.recorder.stop_recording()
        self.window['-START-'].update(disabled=False)
        self.window['-STOP-'].update(disabled=True)
        self.window['-STATUS-'].update("Status: Processing...")
        
        # Get recorded audio
        audio = self.recorder.stop_recording()
        
        if len(audio) == 0:
            sg.popup_error("No audio recorded")
            return
        
        # Analyze quality
        quality_metrics = self.quality_analyzer.analyze_audio(audio)
        self.update_quality_display(quality_metrics)
        
        # Save recording
        self.save_recording(audio, quality_metrics)
        
        self.window['-STATUS-'].update("Status: Ready")
    
    def update_recording_timer(self):
        """Update recording timer display"""
        if self.recorder.is_recording:
            duration = time.time() - self.recording_start_time
            self.window['-DURATION-'].update(f"Duration: {duration:.1f}s")
            
            # Update progress bar
            progress = min(100, int((duration / 15.0) * 100))  # 15 second max
            self.window['-PROGRESS-'].update(progress)
            
            # Schedule next update
            threading.Timer(0.1, self.update_recording_timer).start()
    
    def update_quality_display(self, metrics: Dict[str, float]):
        """Update quality monitoring display"""
        self.window['-SNR-'].update(f"SNR: {metrics['snr_db']:.1f} dB")
        self.window['-CLARITY-'].update(f"Clarity: {metrics['clarity_score']:.2f}")
        self.window['-NOISE-'].update(f"Noise: {metrics['background_noise_db']:.1f} dB")
        self.window['-ACTIVITY-'].update(f"Activity: {metrics['voice_activity_ratio']*100:.1f}%")
    
    def update_prompt_display(self):
        """Update prompt text display"""
        if not self.current_speaker:
            return
        
        language = self.current_speaker['language']
        prompt_index = self.current_prompt_index[language]
        prompts = self.prompts[language]
        
        if prompt_index < len(prompts):
            prompt = prompts[prompt_index]
            self.window['-PROMPT_TEXT-'].update(prompt.text)
            self.current_prompt = prompt
    
    def next_prompt(self):
        """Move to next prompt"""
        if not self.current_speaker:
            return
        
        language = self.current_speaker['language']
        prompts = self.prompts[language]
        
        self.current_prompt_index[language] = (self.current_prompt_index[language] + 1) % len(prompts)
        self.update_prompt_display()
    
    def previous_prompt(self):
        """Move to previous prompt"""
        if not self.current_speaker:
            return
        
        language = self.current_speaker['language']
        prompts = self.prompts[language]
        
        self.current_prompt_index[language] = (self.current_prompt_index[language] - 1) % len(prompts)
        self.update_prompt_display()
    
    def save_recording(self, audio: np.ndarray, quality_metrics: Dict[str, float]):
        """Save recording with metadata"""
        if not self.current_speaker or not self.current_prompt:
            return
        
        speaker_id = self.current_speaker['speaker_id']
        language = self.current_speaker['language']
        prompt_id = self.current_prompt.prompt_id
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{speaker_id}_{prompt_id}_{timestamp}"
        
        # Save audio
        audio_dir = Path(self.config.base_data_dir) / "recordings" / language
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        audio_path = audio_dir / f"{filename}.wav"
        self.recorder.save_recording(audio, str(audio_path))
        
        # Save metadata
        metadata = {
            'filename': filename,
            'speaker_id': speaker_id,
            'prompt_id': prompt_id,
            'text': self.current_prompt.text,
            'language': language,
            'timestamp': timestamp,
            'quality_metrics': quality_metrics,
            'duration': len(audio) / self.config.sample_rate
        }
        
        metadata_path = audio_dir / f"{filename}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Update recording history
        self.recording_history.append({
            'filename': filename,
            'text': self.current_prompt.text,
            'quality': quality_metrics['clarity_score'],
            'timestamp': timestamp
        })
        
        # Update UI
        self.update_recording_list()
        
        sg.popup(f"Recording saved: {filename}")
    
    def update_recording_list(self):
        """Update recording list display"""
        items = [f"{rec['filename']} - {rec['text'][:30]}... - Quality: {rec['quality']:.2f}" 
                for rec in self.recording_history]
        self.window['-RECORDING_LIST-'].update(items)
    
    def play_recording(self):
        """Play selected recording"""
        selection = self.window['-RECORDING_LIST-'].get()
        if not selection:
            return
        
        selected_item = selection[0]
        filename = selected_item.split(' - ')[0]
        
        # Find audio file
        if self.current_speaker:
            language = self.current_speaker['language']
            audio_path = Path(self.config.base_data_dir) / "recordings" / language / f"{filename}.wav"
            
            if audio_path.exists():
                # Play audio
                audio, sr = librosa.load(audio_path)
                sd.play(audio, sr)
                sd.wait()
            else:
                sg.popup_error(f"Audio file not found: {audio_path}")
    
    def delete_recording(self):
        """Delete selected recording"""
        selection = self.window['-RECORDING_LIST-'].get()
        if not selection:
            return
        
        if sg.popup_yes_no("Are you sure you want to delete this recording?") == 'Yes':
            selected_item = selection[0]
            filename = selected_item.split(' - ')[0]
            
            # Remove from history
            self.recording_history = [rec for rec in self.recording_history 
                                      if rec['filename'] != filename]
            
            # Delete files (optional)
            if self.current_speaker:
                language = self.current_speaker['language']
                audio_path = Path(self.config.base_data_dir) / "recordings" / language / f"{filename}.wav"
                metadata_path = audio_path.with_suffix('.json')
                
                try:
                    if audio_path.exists():
                        audio_path.unlink()
                    if metadata_path.exists():
                        metadata_path.unlink()
                except Exception as e:
                    sg.popup_error(f"Error deleting files: {e}")
            
            self.update_recording_list()
    
    def save_session(self):
        """Save current recording session"""
        if not self.current_speaker:
            return
        
        session_data = {
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'speaker_id': self.current_speaker['speaker_id'],
            'recordings': self.recording_history,
            'total_recordings': len(self.recording_history),
            'timestamp': datetime.now().isoformat()
        }
        
        session_path = Path(self.config.base_data_dir) / "sessions" / f"{session_data['session_id']}.json"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(session_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        sg.popup(f"Session saved: {session_data['session_id']}")
    
    def export_data(self):
        """Export recording data"""
        # This would implement data export functionality
        sg.popup("Data export functionality would be implemented here")
    
    def exit_application(self):
        """Exit the application"""
        if self.recorder.is_recording:
            self.recorder.stop_recording()
        
        self.window.close()
    
    def run(self):
        """Main GUI event loop"""
        while True:
            event, values = self.window.read(timeout=100)
            
            if event == sg.WINDOW_CLOSED or event == '-EXIT-':
                break
            
            # Handle events
            if event in self.event_handlers:
                self.event_handlers[event]()
            
            # Handle audio queue for real-time monitoring
            if self.config.real_time_monitoring:
                try:
                    while True:
                        audio_chunk = self.recorder.audio_queue.get_nowait()
                        # Process audio chunk for real-time display
                        # This would update waveform display, etc.
                except queue.Empty:
                    pass
        
        self.exit_application()

def main():
    """Main function"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = RecordingInterfaceConfig(
        base_data_dir="./recording_data",
        show_waveform=True,
        real_time_monitoring=True
    )
    
    # Create and run GUI
    app = ThaiIsanRecordingGUI(config)
    app.run()

if __name__ == "__main__":
    main()