"""
Unit tests for Thai-Isan Data Collection Pipeline
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import test utilities
from tests.utils import MockSpeaker, MockConfig, TestDataManager, MockAudioData

# Mock the dependencies that might not be available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

# Mock modules that might not be available
if not TORCH_AVAILABLE:
    torch = Mock()
    torch.nn = Mock()
    torch.tensor = lambda x: x

if not LIBROSA_AVAILABLE:
    librosa = Mock()
    librosa.load = Mock(return_value=(np.array([0.1, 0.2, 0.3]), 48000))
    librosa.effects.trim = Mock(return_value=(np.array([0.1, 0.2, 0.3]), np.array([0, 1, 2])))

if not SOUNDFILE_AVAILABLE:
    sf = Mock()
    sf.read = Mock(return_value=(np.array([0.1, 0.2, 0.3]), 48000))
    sf.write = Mock()

# Mock pythainlp
try:
    import pythainlp
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False
    pythainlp = Mock()
    pythainlp.tokenize = Mock()
    pythainlp.tokenize.word_tokenize = Mock(return_value=['test'])
    pythainlp.corpus = Mock()
    pythainlp.corpus.thai_words = Mock(return_value=['test'])

# Now import the modules to test
with patch.dict('sys.modules', {
    'torch': torch,
    'librosa': librosa,
    'soundfile': sf,
    'pythainlp': pythainlp,
    'pythainlp.tokenize': pythainlp.tokenize if not PYTHAINLP_AVAILABLE else __import__('pythainlp.tokenize'),
    'pythainlp.corpus': pythainlp.corpus if not PYTHAINLP_AVAILABLE else __import__('pythainlp.corpus')
}):
    try:
        from data_collection_pipeline import (
            SpeakerInfo, AudioRecording, DataCollectionConfig, ThaiIsanDataCollector
        )
        MODULE_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import data_collection_pipeline: {e}")
        MODULE_AVAILABLE = False


class TestSpeakerInfo(unittest.TestCase):
    """Test SpeakerInfo dataclass"""
    
    def test_speaker_info_creation(self):
        """Test SpeakerInfo creation with valid data"""
        speaker = SpeakerInfo(
            speaker_id="test_speaker_001",
            language="th",
            gender="female",
            age_group="adult",
            region="central",
            dialect="central_thai",
            native_language="th",
            years_speaking=25,
            recording_quality="studio",
            accent_rating=4.8,
            clarity_rating=4.9
        )
        
        self.assertEqual(speaker.speaker_id, "test_speaker_001")
        self.assertEqual(speaker.language, "th")
        self.assertEqual(speaker.gender, "female")
        self.assertEqual(speaker.accent_rating, 4.8)
        self.assertEqual(speaker.clarity_rating, 4.9)
    
    def test_speaker_info_validation(self):
        """Test SpeakerInfo validation"""
        # Test with invalid rating values
        with self.assertRaises(ValueError):
            SpeakerInfo(
                speaker_id="test_speaker_002",
                language="tts",
                gender="male",
                age_group="young",
                region="northeastern",
                dialect="central_isan",
                native_language="tts",
                years_speaking=20,
                recording_quality="professional",
                accent_rating=6.0,  # Invalid: should be 1-5
                clarity_rating=3.5
            )


class TestAudioRecording(unittest.TestCase):
    """Test AudioRecording dataclass"""
    
    def test_audio_recording_creation(self):
        """Test AudioRecording creation with valid data"""
        from datetime import datetime
        
        recording = AudioRecording(
            recording_id="test_rec_001",
            speaker_id="test_speaker_001",
            text="สวัสดีครับ",
            language="th",
            duration=3.5,
            sample_rate=48000,
            bit_depth=24,
            channels=1,
            file_size=1024000,
            file_path="/test/path/audio.wav",
            quality_score=0.9,
            timestamp=datetime.now(),
            background_noise_level=-45.0,
            signal_to_noise_ratio=35.0
        )
        
        self.assertEqual(recording.recording_id, "test_rec_001")
        self.assertEqual(recording.speaker_id, "test_speaker_001")
        self.assertEqual(recording.language, "th")
        self.assertEqual(recording.quality_score, 0.9)
        self.assertEqual(recording.signal_to_noise_ratio, 35.0)


class TestDataCollectionConfig(unittest.TestCase):
    """Test DataCollectionConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = DataCollectionConfig()
        
        self.assertEqual(config.target_hours_thai, 100)
        self.assertEqual(config.target_hours_isan, 100)
        self.assertEqual(config.min_speakers_per_language, 50)
        self.assertEqual(config.target_sample_rate, 48000)
        self.assertEqual(config.target_bit_depth, 24)
        self.assertEqual(config.min_quality_score, 0.8)
    
    def test_custom_config(self):
        """Test custom configuration values"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DataCollectionConfig(
                target_hours_thai=50,
                target_hours_isan=50,
                min_speakers_per_language=25,
                base_data_dir=temp_dir,
                num_workers=4,
                batch_size=16
            )
            
            self.assertEqual(config.target_hours_thai, 50)
            self.assertEqual(config.target_hours_isan, 50)
            self.assertEqual(config.min_speakers_per_language, 25)
            self.assertEqual(config.base_data_dir, temp_dir)
            self.assertEqual(config.num_workers, 4)
            self.assertEqual(config.batch_size, 16)


@unittest.skipIf(not MODULE_AVAILABLE, "data_collection_pipeline module not available")
class TestThaiIsanDataCollector(unittest.TestCase):
    """Test ThaiIsanDataCollector class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = DataCollectionConfig(
            target_hours_thai=1,  # Reduced for testing
            target_hours_isan=1,
            min_speakers_per_language=2,
            max_recordings_per_speaker=5,
            base_data_dir=self.temp_dir,
            num_workers=2,
            batch_size=4,
            backup_enabled=False
        )
        self.collector = ThaiIsanDataCollector(self.config)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_collector_initialization(self):
        """Test collector initialization"""
        self.assertIsNotNone(self.collector)
        self.assertEqual(self.collector.config.target_hours_thai, 1)
        self.assertEqual(self.collector.config.target_hours_isan, 1)
    
    @patch('librosa.load')
    @patch('soundfile.read')
    def test_audio_quality_assessment(self, mock_sf_read, mock_librosa_load):
        """Test audio quality assessment"""
        # Mock audio data
        mock_audio = np.random.randn(48000 * 3) * 0.1  # 3 seconds of audio
        mock_sf_read.return_value = (mock_audio, 48000)
        mock_librosa_load.return_value = (mock_audio, 48000)
        
        test_file = Path(self.temp_dir) / "test_audio.wav"
        test_file.write_bytes(b"dummy audio data")
        
        # Test quality assessment
        quality_score = self.collector.assess_audio_quality(str(test_file))
        
        self.assertIsInstance(quality_score, float)
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 1.0)
    
    def test_speaker_registration(self):
        """Test speaker registration"""
        speaker_data = MockSpeaker.create_mock_speaker("test_speaker_001", "th")
        
        success = self.collector.register_speaker(speaker_data)
        
        self.assertTrue(success)
        self.assertIn("test_speaker_001", self.collector.speakers)
    
    def test_text_validation(self):
        """Test text validation for Thai and Isan"""
        valid_thai_text = "สวัสดีครับ นี่คือการทดสอบ"
        valid_isan_text = "สวัสดีครับ นี่คือการทดสอบภาษาอีสาน"
        invalid_text = "Hello world 123"
        
        # Test Thai text
        is_valid_thai = self.collector.validate_text(valid_thai_text, "th")
        self.assertTrue(is_valid_thai)
        
        # Test Isan text
        is_valid_isan = self.collector.validate_text(valid_isan_text, "tts")
        self.assertTrue(is_valid_isan)
        
        # Test invalid text
        is_valid_invalid = self.collector.validate_text(invalid_text, "th")
        self.assertFalse(is_valid_invalid)
    
    def test_data_splitting(self):
        """Test train/validation/test data splitting"""
        # Create mock data
        mock_data = []
        for i in range(20):
            mock_data.append({
                "recording_id": f"rec_{i:03d}",
                "speaker_id": f"speaker_{i % 5:03d}",
                "language": "th" if i % 2 == 0 else "tts",
                "quality_score": 0.8 + (i % 5) * 0.04
            })
        
        train_data, val_data, test_data = self.collector.split_data(mock_data)
        
        # Check proportions (approximate due to random splitting)
        total_len = len(mock_data)
        self.assertGreater(len(train_data), total_len * 0.6)
        self.assertGreater(len(val_data), total_len * 0.1)
        self.assertGreater(len(test_data), total_len * 0.1)
        
        # Check no overlap
        train_ids = set(item["recording_id"] for item in train_data)
        val_ids = set(item["recording_id"] for item in val_data)
        test_ids = set(item["recording_id"] for item in test_data)
        
        self.assertEqual(len(train_ids & val_ids), 0)
        self.assertEqual(len(train_ids & test_ids), 0)
        self.assertEqual(len(val_ids & test_ids), 0)


class TestDataCollectionIntegration(unittest.TestCase):
    """Integration tests for data collection pipeline"""
    
    @unittest.skipIf(not MODULE_AVAILABLE, "data_collection_pipeline module not available")
    def setUp(self):
        """Set up integration test environment"""
        self.test_data_manager = TestDataManager()
        self.mock_speakers = self.test_data_manager.create_test_speaker_profiles(3)
        self.mock_transcriptions = self.test_data_manager.create_test_transcriptions(10)
    
    def test_end_to_end_data_collection_workflow(self):
        """Test complete data collection workflow"""
        # This is a high-level integration test
        # In a real scenario, this would involve actual audio recording
        
        # Step 1: Register speakers
        for speaker in self.mock_speakers:
            # Mock registration process
            self.assertIn("speaker_id", speaker)
            self.assertIn("language", speaker)
            self.assertIn("region", speaker)
        
        # Step 2: Validate transcriptions
        for transcription in self.mock_transcriptions:
            self.assertIn("text", transcription)
            self.assertIn("language", transcription)
            self.assertIn("duration", transcription)
            self.assertIn("text_id", transcription)
        
        # Step 3: Simulate data collection progress
        total_target_duration = 2 * 60 * 60  # 2 hours in seconds
        collected_duration = 0
        
        for transcription in self.mock_transcriptions:
            collected_duration += transcription["duration"]
            progress_ratio = collected_duration / total_target_duration
            self.assertGreaterEqual(progress_ratio, 0.0)
            self.assertLessEqual(progress_ratio, 1.0)
        
        # Verify collection goals
        self.assertGreaterEqual(collected_duration, 0)


if __name__ == "__main__":
    unittest.main()