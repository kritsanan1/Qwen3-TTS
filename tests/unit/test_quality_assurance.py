"""
Unit tests for Thai-Isan Quality Assurance System
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import test utilities
from tests.utils import MockConfig, TestDataManager, MockAudioData

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

try:
    import pythainlp
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False

try:
    import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False

# Mock modules that might not be available
if not TORCH_AVAILABLE:
    torch = Mock()
    torch.nn = Mock()
    torch.tensor = lambda x: x

if not LIBROSA_AVAILABLE:
    librosa = Mock()
    librosa.load = Mock(return_value=(np.array([0.1, 0.2, 0.3]), 48000))
    librosa.effects.trim = Mock(return_value=(np.array([0.1, 0.2, 0.3]), np.array([0, 1, 2])))
    librosa.feature.mfcc = Mock(return_value=np.random.randn(13, 100))
    librosa.core.piptrack = Mock(return_value=(np.array([440]), np.array([0.8])))

if not SOUNDFILE_AVAILABLE:
    sf = Mock()
    sf.read = Mock(return_value=(np.array([0.1, 0.2, 0.3]), 48000))
    sf.write = Mock()

if not PYTHAINLP_AVAILABLE:
    pythainlp = Mock()
    pythainlp.tokenize = Mock()
    pythainlp.tokenize.word_tokenize = Mock(return_value=['test'])
    pythainlp.corpus = Mock()
    pythainlp.corpus.thai_words = Mock(return_value=['test'])
    pythainlp.transliterate = Mock()
    pythainlp.transliterate.romanize = Mock(return_value='test')

if not PESQ_AVAILABLE:
    pesq = Mock()
    pesq.pesq = Mock(return_value=3.0)

if not STOI_AVAILABLE:
    stoi = Mock()
    stoi.stoi = Mock(return_value=0.8)

# Now import the modules to test
with patch.dict('sys.modules', {
    'torch': torch,
    'librosa': librosa,
    'soundfile': sf,
    'pythainlp': pythainlp,
    'pythainlp.tokenize': pythainlp.tokenize if not PYTHAINLP_AVAILABLE else __import__('pythainlp.tokenize'),
    'pythainlp.corpus': pythainlp.corpus if not PYTHAINLP_AVAILABLE else __import__('pythainlp.corpus'),
    'pythainlp.transliterate': pythainlp.transliterate if not PYTHAINLP_AVAILABLE else __import__('pythainlp.transliterate'),
    'pesq': pesq if not PESQ_AVAILABLE else __import__('pesq'),
    'stoi': stoi if not STOI_AVAILABLE else __import__('stoi'),
    'pystoi': stoi,
    'matplotlib.pyplot': Mock(),
    'seaborn': Mock(),
    'sklearn.metrics': Mock()
}):
    try:
        from quality_assurance_system import (
            QualityMetrics, SubjectiveScores, LanguageSpecificMetrics, ThaiIsanQualityAssessor
        )
        MODULE_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import quality_assurance_system: {e}")
        MODULE_AVAILABLE = False


class TestQualityMetrics(unittest.TestCase):
    """Test QualityMetrics dataclass"""
    
    def test_quality_metrics_creation(self):
        """Test QualityMetrics creation with valid data"""
        metrics = QualityMetrics(
            mel_cepstral_distortion=5.2,
            fundamental_frequency_correlation=0.85,
            spectral_convergence=0.78,
            signal_to_noise_ratio=32.5,
            total_harmonic_distortion=0.02,
            pesq_score=3.2,
            stoi_score=0.88,
            tone_accuracy=0.94,
            phoneme_accuracy=0.93,
            phoneme_error_rate=0.07,
            pace_naturalness=0.82,
            pitch_naturalness=0.79,
            rhythm_accuracy=0.85,
            regional_authenticity=0.88,
            cultural_appropriateness=0.91,
            native_speaker_similarity=0.87
        )
        
        self.assertEqual(metrics.mel_cepstral_distortion, 5.2)
        self.assertEqual(metrics.fundamental_frequency_correlation, 0.85)
        self.assertEqual(metrics.pesq_score, 3.2)
        self.assertEqual(metrics.tone_accuracy, 0.94)
        self.assertEqual(metrics.regional_authenticity, 0.88)
    
    def test_quality_metrics_validation(self):
        """Test QualityMetrics validation"""
        # Test with invalid correlation value
        with self.assertRaises(ValueError):
            QualityMetrics(
                mel_cepstral_distortion=5.2,
                fundamental_frequency_correlation=1.5,  # Invalid: should be -1 to 1
                spectral_convergence=0.78,
                signal_to_noise_ratio=32.5,
                total_harmonic_distortion=0.02,
                pesq_score=3.2,
                stoi_score=0.88,
                tone_accuracy=0.94,
                phoneme_accuracy=0.93,
                phoneme_error_rate=0.07,
                pace_naturalness=0.82,
                pitch_naturalness=0.79,
                rhythm_accuracy=0.85,
                regional_authenticity=0.88,
                cultural_appropriateness=0.91,
                native_speaker_similarity=0.87
            )


class TestSubjectiveScores(unittest.TestCase):
    """Test SubjectiveScores dataclass"""
    
    def test_subjective_scores_creation(self):
        """Test SubjectiveScores creation with valid data"""
        scores = SubjectiveScores(
            naturalness=4.2,
            intelligibility=4.5,
            pleasantness=4.0,
            similarity=4.3,
            overall_quality=4.2,
            tone_naturalness=4.1,
            accent_authenticity=4.4,
            cultural_appropriateness=4.6
        )
        
        self.assertEqual(scores.naturalness, 4.2)
        self.assertEqual(scores.intelligibility, 4.5)
        self.assertEqual(scores.overall_quality, 4.2)
        self.assertEqual(scores.accent_authenticity, 4.4)
    
    def test_subjective_scores_validation(self):
        """Test SubjectiveScores validation"""
        # Test with invalid score value
        with self.assertRaises(ValueError):
            SubjectiveScores(
                naturalness=6.0,  # Invalid: should be 1-5
                intelligibility=4.5,
                pleasantness=4.0,
                similarity=4.3,
                overall_quality=4.2,
                tone_naturalness=4.1,
                accent_authenticity=4.4,
                cultural_appropriateness=4.6
            )


class TestLanguageSpecificMetrics(unittest.TestCase):
    """Test LanguageSpecificMetrics dataclass"""
    
    def test_language_specific_metrics_creation(self):
        """Test LanguageSpecificMetrics creation with valid data"""
        metrics = LanguageSpecificMetrics(
            thai_tone_accuracy=0.96,
            thai_consonant_accuracy=0.94,
            thai_vowel_accuracy=0.95,
            thai_final_consonant_accuracy=0.93,
            isan_tone_accuracy=0.94,
            isan_dialect_authenticity=0.88,
            isan_lexical_accuracy=0.92,
            isan_phonological_faithfulness=0.90
        )
        
        self.assertEqual(metrics.thai_tone_accuracy, 0.96)
        self.assertEqual(metrics.thai_consonant_accuracy, 0.94)
        self.assertEqual(metrics.isan_tone_accuracy, 0.94)
        self.assertEqual(metrics.isan_dialect_authenticity, 0.88)


@unittest.skipIf(not MODULE_AVAILABLE, "quality_assurance_system module not available")
class TestThaiIsanQualityAssessor(unittest.TestCase):
    """Test ThaiIsanQualityAssessor class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.assessor = ThaiIsanQualityAssessor()
        self.test_data_manager = TestDataManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_assessor_initialization(self):
        """Test quality assessor initialization"""
        self.assertIsNotNone(self.assessor)
        self.assertTrue(hasattr(self.assessor, 'evaluate_audio_quality'))
        self.assertTrue(hasattr(self.assessor, 'evaluate_tone_accuracy'))
        self.assertTrue(hasattr(self.assessor, 'evaluate_phoneme_accuracy'))
    
    @patch('librosa.load')
    @patch('librosa.feature.mfcc')
    def test_mfcc_extraction(self, mock_mfcc, mock_load):
        """Test MFCC feature extraction"""
        # Mock audio data
        mock_audio = np.random.randn(48000 * 3) * 0.1  # 3 seconds of audio
        mock_load.return_value = (mock_audio, 48000)
        mock_mfcc.return_value = np.random.randn(13, 300)  # 13 MFCCs, 300 frames
        
        test_file = Path(self.temp_dir) / "test_audio.wav"
        test_file.write_bytes(b"dummy audio data")
        
        # Test MFCC extraction
        mfcc_features = self.assessor.extract_mfcc_features(str(test_file))
        
        self.assertIsNotNone(mfcc_features)
        self.assertEqual(mfcc_features.shape[0], 13)  # 13 MFCC coefficients
    
    def test_tone_accuracy_evaluation(self):
        """Test tone accuracy evaluation"""
        # Mock reference and synthesized tones
        reference_tones = np.array([1, 2, 3, 4, 5, 1, 2, 3])  # Thai tones
        synthesized_tones = np.array([1, 2, 3, 4, 5, 1, 2, 3])  # Perfect match
        
        tone_accuracy = self.assessor.evaluate_tone_accuracy(reference_tones, synthesized_tones)
        
        self.assertAlmostEqual(tone_accuracy, 1.0, places=2)
        
        # Test with some errors
        synthesized_tones_with_errors = np.array([1, 2, 4, 4, 5, 2, 2, 3])  # Some errors
        tone_accuracy_with_errors = self.assessor.evaluate_tone_accuracy(
            reference_tones, synthesized_tones_with_errors
        )
        
        self.assertLess(tone_accuracy_with_errors, 1.0)
        self.assertGreater(tone_accuracy_with_errors, 0.0)
    
    def test_phoneme_accuracy_evaluation(self):
        """Test phoneme accuracy evaluation"""
        # Mock reference and synthesized phonemes
        reference_phonemes = ['s', 'a', 'r', 'a', 's', 'a', 't', 'h', 'i']
        synthesized_phonemes = ['s', 'a', 'r', 'a', 's', 'a', 't', 'h', 'i']  # Perfect match
        
        phoneme_accuracy = self.assessor.evaluate_phoneme_accuracy(reference_phonemes, synthesized_phonemes)
        
        self.assertAlmostEqual(phoneme_accuracy, 1.0, places=2)
        
        # Test with some errors
        synthesized_phonemes_with_errors = ['s', 'a', 'l', 'a', 's', 'a', 't', 'h', 'i']  # One error
        phoneme_accuracy_with_errors = self.assessor.evaluate_phoneme_accuracy(
            reference_phonemes, synthesized_phonemes_with_errors
        )
        
        self.assertLess(phoneme_accuracy_with_errors, 1.0)
        self.assertGreater(phoneme_accuracy_with_errors, 0.8)
    
    def test_overall_quality_evaluation(self):
        """Test overall quality evaluation"""
        # Mock audio file path
        mock_audio_path = Path(self.temp_dir) / "mock_audio.wav"
        mock_audio_path.write_bytes(b"dummy audio data")
        
        # Mock reference text and language
        reference_text = "สวัสดีครับ นี่คือการทดสอบ"
        language = "th"
        
        # Mock synthesized audio (in real test, would be actual audio)
        mock_synthesized_audio = MockAudioData.generate_sine_wave(3.0)
        
        # Test overall evaluation (mock implementation)
        with patch.object(self.assessor, 'extract_mfcc_features', return_value=np.random.randn(13, 300)):
            with patch.object(self.assessor, 'evaluate_tone_accuracy', return_value=0.94):
                with patch.object(self.assessor, 'evaluate_phoneme_accuracy', return_value=0.93):
                    quality_metrics = self.assessor.evaluate_overall_quality(
                        mock_synthesized_audio, reference_text, language
                    )
        
        self.assertIsInstance(quality_metrics, QualityMetrics)
        self.assertGreaterEqual(quality_metrics.tone_accuracy, 0.0)
        self.assertLessEqual(quality_metrics.tone_accuracy, 1.0)
        self.assertGreaterEqual(quality_metrics.phoneme_accuracy, 0.0)
        self.assertLessEqual(quality_metrics.phoneme_accuracy, 1.0)
    
    def test_subjective_evaluation(self):
        """Test subjective evaluation scoring"""
        # Mock evaluation data
        evaluation_data = {
            "naturalness_ratings": [4, 5, 4, 3, 4, 5, 4, 4],
            "intelligibility_ratings": [5, 5, 4, 4, 5, 5, 4, 5],
            "pleasantness_ratings": [4, 4, 3, 4, 4, 5, 3, 4],
            "similarity_ratings": [4, 5, 4, 4, 4, 5, 4, 4],
            "authenticity_ratings": [4, 5, 4, 3, 4, 5, 4, 4]
        }
        
        subjective_scores = self.assessor.evaluate_subjective_scores(evaluation_data)
        
        self.assertIsInstance(subjective_scores, SubjectiveScores)
        self.assertGreaterEqual(subjective_scores.naturalness, 1.0)
        self.assertLessEqual(subjective_scores.naturalness, 5.0)
        self.assertGreaterEqual(subjective_scores.intelligibility, 1.0)
        self.assertLessEqual(subjective_scores.intelligibility, 5.0)


class TestQualityMetricsCalculation(unittest.TestCase):
    """Test quality metrics calculation functions"""
    
    def test_mel_cepstral_distortion_calculation(self):
        """Test MCD calculation"""
        # Mock MFCC features
        ref_mfcc = np.random.randn(13, 100)
        synth_mfcc = ref_mfcc + np.random.randn(13, 100) * 0.1  # Slight difference
        
        mcd = ThaiIsanQualityAssessor.calculate_mcd(ref_mfcc, synth_mfcc)
        
        self.assertGreater(mcd, 0.0)  # MCD should be positive
        self.assertLess(mcd, 10.0)  # MCD should be reasonable for similar audio
    
    def test_signal_to_noise_ratio_calculation(self):
        """Test SNR calculation"""
        # Mock signal and noise
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 48000))  # 1 second sine wave
        noise = np.random.randn(48000) * 0.01  # Low noise
        noisy_signal = signal + noise
        
        snr = ThaiIsanQualityAssessor.calculate_snr(signal, noisy_signal)
        
        self.assertGreater(snr, 0.0)  # SNR should be positive
        self.assertGreater(snr, 10.0)  # Should be good SNR for this case


class TestQualityAssuranceIntegration(unittest.TestCase):
    """Integration tests for quality assurance system"""
    
    @unittest.skipIf(not MODULE_AVAILABLE, "quality_assurance_system module not available")
    def setUp(self):
        """Set up integration test environment"""
        self.assessor = ThaiIsanQualityAssessor()
        self.test_data_manager = TestDataManager()
    
    def test_quality_benchmarking(self):
        """Test quality benchmarking against standards"""
        # Mock quality metrics
        quality_metrics = MockConfig.create_mock_quality_metrics()
        
        # Test benchmarking
        benchmark_results = self.assessor.benchmark_quality(quality_metrics)
        
        self.assertIn("overall_score", benchmark_results)
        self.assertIn("tone_accuracy_pass", benchmark_results)
        self.assertIn("phoneme_accuracy_pass", benchmark_results)
        self.assertIn("pesq_score_pass", benchmark_results)
        
        # Check that benchmark results are boolean
        self.assertIsInstance(benchmark_results["tone_accuracy_pass"], bool)
        self.assertIsInstance(benchmark_results["phoneme_accuracy_pass"], bool)
    
    def test_multi_language_quality_comparison(self):
        """Test quality comparison between Thai and Isan"""
        # Mock quality metrics for both languages
        thai_metrics = MockConfig.create_mock_quality_metrics()
        isan_metrics = MockConfig.create_mock_quality_metrics()
        
        # Adjust Isan metrics slightly
        isan_metrics["tone_accuracy"] = 0.92
        isan_metrics["pesq_score"] = 3.0
        
        # Compare quality
        comparison = self.assessor.compare_language_quality(thai_metrics, isan_metrics)
        
        self.assertIn("thai_advantage", comparison)
        self.assertIn("isan_advantage", comparison)
        self.assertIn("quality_gap", comparison)
        
        # Thai should generally have slightly better metrics in this mock
        self.assertGreater(comparison["thai_advantage"], 0)


if __name__ == "__main__":
    unittest.main()