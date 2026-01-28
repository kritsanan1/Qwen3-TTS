"""
Test utilities and mock objects for Thai-Isan TTS testing
"""

import os
import tempfile
import numpy as np
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock
from datetime import datetime


class MockAudioData:
    """Mock audio data generator for testing"""
    
    @staticmethod
    def generate_sine_wave(duration, sample_rate=48000, frequency=440):
        """Generate a sine wave for testing"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        return np.sin(2 * np.pi * frequency * t)
    
    @staticmethod
    def create_mock_audio_file(filepath, duration=2.0, sample_rate=48000):
        """Create a mock audio file"""
        audio_data = MockAudioData.generate_sine_wave(duration, sample_rate)
        # Mock implementation - in real tests, use actual audio libraries
        return audio_data


class MockSpeaker:
    """Mock speaker data for testing"""
    
    @staticmethod
    def create_mock_speaker(speaker_id="test_speaker_001", language="th"):
        """Create a mock speaker profile"""
        return {
            "speaker_id": speaker_id,
            "language": language,
            "gender": "female",
            "age_group": "adult",
            "region": "central" if language == "th" else "northeastern",
            "dialect": "central_thai" if language == "th" else "central_isan",
            "native_language": language,
            "years_speaking": 25,
            "recording_quality": "studio",
            "accent_rating": 4.8,
            "clarity_rating": 4.9
        }


class MockConfig:
    """Mock configuration objects"""
    
    @staticmethod
    def create_mock_data_collection_config():
        """Create mock data collection configuration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            return {
                "target_hours_thai": 1,  # Reduced for testing
                "target_hours_isan": 1,
                "min_speakers_per_language": 2,
                "max_recordings_per_speaker": 5,
                "target_sample_rate": 48000,
                "target_bit_depth": 24,
                "min_duration_seconds": 1.0,
                "max_duration_seconds": 5.0,
                "target_snr_db": 30.0,
                "min_quality_score": 0.7,
                "max_background_noise": -40.0,
                "min_clarity_rating": 3.5,
                "base_data_dir": temp_dir,
                "backup_enabled": False,
                "compression_format": "flac",
                "num_workers": 2,
                "batch_size": 4,
                "validation_split": 0.2,
                "test_split": 0.2
            }
    
    @staticmethod
    def create_mock_quality_metrics():
        """Create mock quality metrics"""
        return {
            "mel_cepstral_distortion": 5.2,
            "fundamental_frequency_correlation": 0.85,
            "spectral_convergence": 0.78,
            "signal_to_noise_ratio": 32.5,
            "total_harmonic_distortion": 0.02,
            "pesq_score": 3.2,
            "stoi_score": 0.88,
            "tone_accuracy": 0.94,
            "phoneme_accuracy": 0.93,
            "phoneme_error_rate": 0.07,
            "pace_naturalness": 0.82,
            "pitch_naturalness": 0.79,
            "rhythm_accuracy": 0.85,
            "regional_authenticity": 0.88,
            "cultural_appropriateness": 0.91,
            "native_speaker_similarity": 0.87
        }


class TestDataManager:
    """Manage test data and fixtures"""
    
    def __init__(self, base_temp_dir=None):
        self.base_temp_dir = base_temp_dir or tempfile.mkdtemp()
        self.test_data_dir = Path(self.base_temp_dir) / "test_data"
        self.test_data_dir.mkdir(exist_ok=True)
    
    def create_test_audio_files(self, num_files=5):
        """Create test audio files"""
        audio_files = []
        for i in range(num_files):
            filepath = self.test_data_dir / f"test_audio_{i:03d}.wav"
            # Create mock audio file
            MockAudioData.create_mock_audio_file(filepath)
            audio_files.append(filepath)
        return audio_files
    
    def create_test_speaker_profiles(self, num_speakers=3):
        """Create test speaker profiles"""
        speakers = []
        for i in range(num_speakers):
            language = "th" if i % 2 == 0 else "tts"
            speaker = MockSpeaker.create_mock_speaker(f"test_speaker_{i:03d}", language)
            speakers.append(speaker)
        return speakers
    
    def create_test_transcriptions(self, num_transcriptions=10):
        """Create test transcriptions"""
        thai_texts = [
            "สวัสดีครับ นี่คือการทดสอบระบบ",
            "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย",
            "การศึกษาเป็นสิ่งสำคัญสำหรับการพัฒนาประเทศ",
            "ภาษาไทยมีลักษณะเฉพาะที่น่าสนใจ",
            "วัฒนธรรมไทยมีความหลากหลายและงดงาม"
        ]
        
        isan_texts = [
            "สวัสดีครับ นี่คือการทดสอบระบบภาษาอีสาน",
            "ภาคอีสานของประเทศไทยมีวัฒนธรรมที่หลากหลาย",
            "ภาษาอีสานมีลักษณะเฉพาะที่น่าสนใจ",
            "อาหารอีสานมีรสชาติที่จัดจ้าน",
            "ประเพณีของภาคอีสานมีความงดงาม"
        ]
        
        transcriptions = []
        for i in range(num_transcriptions):
            language = "th" if i % 2 == 0 else "tts"
            texts = thai_texts if language == "th" else isan_texts
            text = texts[i % len(texts)]
            
            transcriptions.append({
                "text_id": f"text_{i:03d}",
                "text": text,
                "language": language,
                "duration": np.random.uniform(2.0, 5.0),
                "difficulty": np.random.choice(["easy", "medium", "hard"])
            })
        
        return transcriptions


class MockModel:
    """Mock model for testing"""
    
    def __init__(self):
        self.is_loaded = False
        self.device = "cpu"
    
    def load(self):
        """Mock model loading"""
        self.is_loaded = True
        return True
    
    def synthesize(self, text, language="th", speaker_id=None):
        """Mock speech synthesis"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Generate mock audio data
        duration = len(text) * 0.1  # Rough estimate
        audio_data = MockAudioData.generate_sine_wave(duration)
        
        return {
            "audio": audio_data,
            "sample_rate": 48000,
            "duration": duration,
            "language": language,
            "speaker_id": speaker_id
        }
    
    def unload(self):
        """Mock model unloading"""
        self.is_loaded = False


def setup_test_environment():
    """Set up test environment"""
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    return {
        "temp_dir": temp_dir,
        "test_data_manager": TestDataManager(temp_dir)
    }


def cleanup_test_environment(test_env):
    """Clean up test environment"""
    import shutil
    if "temp_dir" in test_env and os.path.exists(test_env["temp_dir"]):
        shutil.rmtree(test_env["temp_dir"])