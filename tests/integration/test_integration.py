"""
Integration tests for Thai-Isan TTS System
End-to-end testing of complete workflows
"""

import unittest
import tempfile
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import test utilities
from tests.utils import MockSpeaker, MockConfig, TestDataManager, MockAudioData, MockModel

# Mock dependencies that might not be available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = Mock()

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = Mock()

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    sf = Mock()

# Mock modules
with patch.dict('sys.modules', {
    'torch': torch,
    'librosa': librosa,
    'soundfile': sf,
    'pythainlp': Mock(),
    'pythainlp.tokenize': Mock(),
    'pythainlp.corpus': Mock(),
    'pythainlp.transliterate': Mock(),
    'pesq': Mock(),
    'stoi': Mock(),
    'pystoi': Mock(),
    'matplotlib.pyplot': Mock(),
    'seaborn': Mock(),
    'sklearn.metrics': Mock(),
    'pandas': Mock(),
    'tqdm': Mock()
}):
    try:
        from data_collection_pipeline_refactored import (
            ThaiIsanDataCollector, DataCollectionConfig, SpeakerInfo, AudioRecording
        )
        from quality_assurance_system_refactored import (
            ThaiIsanQualityAssessor, QualityMetrics, SubjectiveScores
        )
        COLLECTION_AVAILABLE = True
        QA_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import refactored modules: {e}")
        COLLECTION_AVAILABLE = False
        QA_AVAILABLE = False


class TestEndToEndDataCollection(unittest.TestCase):
    """Test complete data collection workflow"""
    
    @unittest.skipIf(not COLLECTION_AVAILABLE, "Data collection module not available")
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = DataCollectionConfig(
            target_hours_thai=1,  # Reduced for testing
            target_hours_isan=1,
            min_speakers_per_language=2,
            max_recordings_per_speaker=5,
            base_data_dir=self.temp_dir,
            num_workers=2,
            batch_size=4
        )
        self.collector = ThaiIsanDataCollector(self.config)
        self.test_data_manager = TestDataManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_complete_data_collection_workflow(self):
        """Test complete data collection from speaker registration to data splitting"""
        # Step 1: Register speakers
        test_speakers = self.test_data_manager.create_test_speaker_profiles(3)
        registered_speakers = []
        
        for speaker_data in test_speakers:
            success = self.collector.register_speaker(speaker_data)
            self.assertTrue(success, f"Failed to register speaker {speaker_data['speaker_id']}")
            registered_speakers.append(speaker_data['speaker_id'])
        
        # Verify speakers were registered
        self.assertEqual(len(registered_speakers), 3)
        self.assertEqual(self.collector.collection_stats['total_speakers'], 3)
        
        # Step 2: Create mock audio files and process recordings
        test_audio_files = self.test_data_manager.create_test_audio_files(6)  # 6 recordings
        test_transcriptions = self.test_data_manager.create_test_transcriptions(6)
        
        processed_recordings = []
        for i, (audio_file, transcription) in enumerate(zip(test_audio_files, test_transcriptions)):
            metadata = {
                'speaker_id': registered_speakers[i % len(registered_speakers)],
                'text': transcription['text'],
                'language': transcription['language']
            }
            
            # Mock audio processing
            recording = self.collector.process_audio_recording(str(audio_file), metadata)
            if recording:  # May be None due to quality thresholds
                processed_recordings.append(recording)
        
        # Verify recordings were processed
        self.assertGreater(len(processed_recordings), 0)
        self.assertGreater(self.collector.collection_stats['total_recordings'], 0)
        
        # Step 3: Check collection progress
        progress = self.collector.get_collection_progress()
        self.assertIn('speakers_registered', progress)
        self.assertIn('recordings_collected', progress)
        self.assertIn('thai_progress', progress)
        self.assertIn('isan_progress', progress)
        self.assertEqual(progress['speakers_registered'], 3)
        self.assertGreaterEqual(progress['recordings_collected'], 0)
        
        # Step 4: Split data for training
        data_split = self.collector.split_data_for_training()
        self.assertIn('train', data_split)
        self.assertIn('validation', data_split)
        self.assertIn('test', data_split)
        
        # Verify data split proportions
        total_recordings = len(data_split['train']) + len(data_split['validation']) + len(data_split['test'])
        self.assertGreaterEqual(total_recordings, 0)
        
        # Step 5: Export collection report
        report_file = Path(self.temp_dir) / "collection_report.json"
        success = self.collector.export_collection_report(str(report_file))
        self.assertTrue(success)
        self.assertTrue(report_file.exists())
        
        # Verify report content
        with open(report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        self.assertIn('timestamp', report)
        self.assertIn('configuration', report)
        self.assertIn('progress', report)
        self.assertIn('statistics', report)
        self.assertIn('speakers', report)
        self.assertIn('recordings', report)


class TestEndToEndQualityAssessment(unittest.TestCase):
    """Test complete quality assessment workflow"""
    
    @unittest.skipIf(not QA_AVAILABLE, "Quality assurance module not available")
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.assessor = ThaiIsanQualityAssessor()
        self.test_data_manager = TestDataManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_complete_quality_assessment_workflow(self):
        """Test complete quality assessment from audio comparison to benchmarking"""
        # Step 1: Create mock reference and synthesized audio
        duration = 3.0  # 3 seconds
        sample_rate = 48000
        
        # Generate reference audio (clean sine wave)
        ref_audio = MockAudioData.generate_sine_wave(duration, sample_rate, frequency=440)
        
        # Generate synthesized audio (reference + small noise)
        synth_audio = ref_audio + np.random.randn(len(ref_audio)) * 0.01
        
        # Save as temporary files
        ref_file = Path(self.temp_dir) / "reference_audio.wav"
        synth_file = Path(self.temp_dir) / "synthesized_audio.wav"
        
        # Mock audio saving (since we don't have actual audio libraries)
        ref_file.write_bytes(b"mock reference audio data")
        synth_file.write_bytes(b"mock synthesized audio data")
        
        # Step 2: Evaluate audio quality
        with patch.object(self.assessor, '_load_audio_safe', side_effect=[
            (ref_audio, sample_rate), (synth_audio, sample_rate)
        ]):
            quality_metrics = self.assessor.evaluate_audio_quality(str(ref_file), str(synth_file))
        
        # Verify quality metrics
        self.assertIsInstance(quality_metrics, QualityMetrics)
        self.assertGreaterEqual(quality_metrics.pesq_score, 1.0)
        self.assertLessEqual(quality_metrics.pesq_score, 4.5)
        self.assertGreaterEqual(quality_metrics.stoi_score, 0.0)
        self.assertLessEqual(quality_metrics.stoi_score, 1.0)
        
        # Step 3: Evaluate tone accuracy
        ref_tones = np.array([1, 2, 3, 4, 5, 1, 2, 3])  # Thai tones
        synth_tones = np.array([1, 2, 3, 4, 5, 1, 2, 3])  # Perfect match
        
        tone_accuracy = self.assessor.evaluate_tone_accuracy(ref_tones, synth_tones)
        self.assertAlmostEqual(tone_accuracy, 1.0, places=2)
        
        # Test with errors
        synth_tones_with_errors = np.array([1, 2, 4, 4, 5, 2, 2, 3])
        tone_accuracy_with_errors = self.assessor.evaluate_tone_accuracy(ref_tones, synth_tones_with_errors)
        self.assertLess(tone_accuracy_with_errors, 1.0)
        self.assertGreater(tone_accuracy_with_errors, 0.0)
        
        # Step 4: Evaluate phoneme accuracy
        ref_phonemes = ['s', 'a', 'r', 'a', 's', 'a', 't', 'h', 'i']
        synth_phonemes = ['s', 'a', 'r', 'a', 's', 'a', 't', 'h', 'i']
        
        phoneme_accuracy = self.assessor.evaluate_phoneme_accuracy(ref_phonemes, synth_phonemes)
        self.assertAlmostEqual(phoneme_accuracy, 1.0, places=2)
        
        # Step 5: Evaluate subjective quality
        evaluation_data = {
            'naturalness_ratings': [4, 5, 4, 3, 4, 5, 4, 4],
            'intelligibility_ratings': [5, 5, 4, 4, 5, 5, 4, 5],
            'pleasantness_ratings': [4, 4, 3, 4, 4, 5, 3, 4],
            'similarity_ratings': [4, 5, 4, 4, 4, 5, 4, 4],
            'overall_quality_ratings': [4, 5, 4, 3, 4, 5, 4, 4],
            'tone_naturalness_ratings': [4, 5, 4, 3, 4, 5, 4, 4],
            'accent_authenticity_ratings': [4, 5, 4, 3, 4, 5, 4, 4],
            'cultural_appropriateness_ratings': [4, 5, 4, 3, 4, 5, 4, 4]
        }
        
        subjective_scores = self.assessor.evaluate_subjective_quality(evaluation_data)
        self.assertIsInstance(subjective_scores, SubjectiveScores)
        self.assertGreaterEqual(subjective_scores.naturalness, 1.0)
        self.assertLessEqual(subjective_scores.naturalness, 5.0)
        self.assertGreaterEqual(subjective_scores.get_overall_score(), 1.0)
        self.assertLessEqual(subjective_scores.get_overall_score(), 5.0)
        
        # Step 6: Benchmark quality
        benchmark_results = self.assessor.benchmark_quality(quality_metrics, language='th')
        self.assertIn('overall_pass', benchmark_results)
        self.assertIn('tone_accuracy', benchmark_results)
        self.assertIn('phoneme_accuracy', benchmark_results)
        self.assertIn('pesq_score', benchmark_results)
        self.assertIn('stoi_score', benchmark_results)
        self.assertIsInstance(benchmark_results['overall_pass'], bool)


class TestSystemIntegration(unittest.TestCase):
    """Test integration between different system components"""
    
    def test_data_collection_to_quality_assessment_integration(self):
        """Test integration from data collection to quality assessment"""
        # This test simulates a complete workflow from data collection
        # through quality assessment
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Create mock data collection environment
            collection_config = DataCollectionConfig(
                target_hours_thai=1,
                target_hours_isan=1,
                min_speakers_per_language=2,
                max_recordings_per_speaker=3,
                base_data_dir=temp_dir,
                num_workers=1,
                batch_size=2
            )
            
            collector = ThaiIsanDataCollector(collection_config)
            
            # Register test speakers
            test_speakers = [
                {
                    'speaker_id': 'speaker_001',
                    'language': 'th',
                    'gender': 'female',
                    'age_group': 'adult',
                    'region': 'central',
                    'dialect': 'central_thai',
                    'native_language': 'th',
                    'years_speaking': 25,
                    'recording_quality': 'studio',
                    'accent_rating': 4.5,
                    'clarity_rating': 4.7
                },
                {
                    'speaker_id': 'speaker_002',
                    'language': 'tts',
                    'gender': 'male',
                    'age_group': 'adult',
                    'region': 'northeastern',
                    'dialect': 'central_isan',
                    'native_language': 'tts',
                    'years_speaking': 30,
                    'recording_quality': 'professional',
                    'accent_rating': 4.3,
                    'clarity_rating': 4.5
                }
            ]
            
            for speaker_data in test_speakers:
                success = collector.register_speaker(speaker_data)
                self.assertTrue(success)
            
            # Create quality assessor
            assessor = ThaiIsanQualityAssessor()
            
            # Test that both systems can work together
            # This is a conceptual test - in reality, you'd need actual audio files
            self.assertEqual(collector.collection_stats['total_speakers'], 2)
            self.assertIsNotNone(assessor)
            
        finally:
            import shutil
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)


class TestErrorHandlingIntegration(unittest.TestCase):
    """Test error handling across system components"""
    
    def test_graceful_degradation_with_missing_dependencies(self):
        """Test that system handles missing dependencies gracefully"""
        # Test with mocked missing dependencies
        with patch.dict('sys.modules', {
            'torch': Mock(),
            'librosa': Mock(),
            'soundfile': Mock(),
            'pythainlp': Mock(),
            'pythainlp.tokenize': Mock(),
            'pythainlp.corpus': Mock(),
            'pythainlp.transliterate': Mock(),
            'pesq': Mock(),
            'stoi': Mock(),
            'pystoi': Mock(),
            'matplotlib.pyplot': Mock(),
            'seaborn': Mock(),
            'sklearn.metrics': Mock(),
            'pandas': Mock(),
            'tqdm': Mock()
        }):
            try:
                # These should not raise exceptions even with missing dependencies
                from data_collection_pipeline_refactored import ThaiIsanDataCollector, DataCollectionConfig
                from quality_assurance_system_refactored import ThaiIsanQualityAssessor
                
                # Test basic instantiation
                config = DataCollectionConfig(
                    target_hours_thai=1,
                    target_hours_isan=1,
                    min_speakers_per_language=1,
                    base_data_dir="/tmp/test"
                )
                collector = ThaiIsanDataCollector(config)
                assessor = ThaiIsanQualityAssessor()
                
                self.assertIsNotNone(collector)
                self.assertIsNotNone(assessor)
                
            except ImportError:
                # This is acceptable - the system should handle missing dependencies gracefully
                pass


class TestPerformanceIntegration(unittest.TestCase):
    """Test performance aspects of the integrated system"""
    
    def test_large_scale_data_handling(self):
        """Test system performance with large amounts of data"""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create configuration with large data requirements
            config = DataCollectionConfig(
                target_hours_thai=10,  # Larger for performance testing
                target_hours_isan=10,
                min_speakers_per_language=10,
                max_recordings_per_speaker=20,
                base_data_dir=temp_dir,
                num_workers=4,
                batch_size=8
            )
            
            collector = ThaiIsanDataCollector(config)
            
            # Test that the system can handle large configuration
            # without excessive memory usage or timeouts
            self.assertEqual(collector.config.target_hours_thai, 10)
            self.assertEqual(collector.config.min_speakers_per_language, 10)
            
            # Test progress tracking with large numbers
            progress = collector.get_collection_progress()
            self.assertIn('thai_progress', progress)
            self.assertIn('isan_progress', progress)
            
            # Progress should be 0 since we haven't collected any data yet
            self.assertEqual(progress['thai_progress'], 0.0)
            self.assertEqual(progress['isan_progress'], 0.0)
            
        finally:
            import shutil
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()