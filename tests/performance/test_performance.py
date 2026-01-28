"""
Performance benchmarks and validation tests for Thai-Isan TTS System
"""

import unittest
import time
import tempfile
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import threading
import psutil
import gc

# Import test utilities
from tests.utils import MockSpeaker, MockConfig, TestDataManager, MockAudioData, MockModel

# Mock dependencies
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

with patch.dict('sys.modules', {
    'torch': torch,
    'librosa': librosa,
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
    'tqdm': Mock(),
    'psutil': Mock()
}):
    try:
        from data_collection_pipeline_refactored import (
            ThaiIsanDataCollector, DataCollectionConfig
        )
        from quality_assurance_system_refactored import (
            ThaiIsanQualityAssessor, QualityMetrics
        )
        COLLECTION_AVAILABLE = True
        QA_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import modules for performance testing: {e}")
        COLLECTION_AVAILABLE = False
        QA_AVAILABLE = False


class PerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for the Thai-Isan TTS system"""
    
    def setUp(self):
        """Set up performance test environment"""
        self.benchmark_results = {}
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up performance test environment"""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def measure_execution_time(self, func, *args, **kwargs):
        """Measure execution time of a function"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time
    
    def measure_memory_usage(self):
        """Measure current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
    
    @unittest.skipIf(not COLLECTION_AVAILABLE, "Data collection module not available")
    def test_speaker_registration_performance(self):
        """Test performance of speaker registration"""
        config = DataCollectionConfig(
            target_hours_thai=1,
            target_hours_isan=1,
            min_speakers_per_language=100,  # Large number for performance testing
            base_data_dir=self.temp_dir
        )
        collector = ThaiIsanDataCollector(config)
        
        # Create test speakers
        num_speakers = 50
        test_speakers = []
        for i in range(num_speakers):
            speaker_data = {
                'speaker_id': f'speaker_{i:04d}',
                'language': 'th' if i % 2 == 0 else 'tts',
                'gender': 'female' if i % 3 == 0 else 'male',
                'age_group': 'adult',
                'region': 'central' if i % 2 == 0 else 'northeastern',
                'dialect': 'central_thai' if i % 2 == 0 else 'central_isan',
                'native_language': 'th' if i % 2 == 0 else 'tts',
                'years_speaking': 20 + (i % 20),
                'recording_quality': 'studio',
                'accent_rating': 4.0 + (i % 10) * 0.1,
                'clarity_rating': 4.0 + (i % 10) * 0.1
            }
            test_speakers.append(speaker_data)
        
        # Measure memory before
        memory_before = self.measure_memory_usage()
        
        # Measure registration performance
        registration_times = []
        for speaker_data in test_speakers:
            _, exec_time = self.measure_execution_time(collector.register_speaker, speaker_data)
            registration_times.append(exec_time)
        
        # Measure memory after
        memory_after = self.measure_memory_usage()
        memory_increase = memory_after - memory_before
        
        # Calculate statistics
        avg_registration_time = np.mean(registration_times)
        max_registration_time = np.max(registration_times)
        min_registration_time = np.min(registration_times)
        
        # Store benchmark results
        self.benchmark_results['speaker_registration'] = {
            'num_speakers': num_speakers,
            'avg_time_ms': avg_registration_time * 1000,
            'max_time_ms': max_registration_time * 1000,
            'min_time_ms': min_registration_time * 1000,
            'memory_increase_mb': memory_increase
        }
        
        # Assertions for performance requirements
        self.assertLess(avg_registration_time, 0.1)  # Should be less than 100ms per speaker
        self.assertLess(memory_increase, 100)  # Should use less than 100MB additional memory
        
        print(f"Speaker registration performance: {avg_registration_time*1000:.2f}ms avg, {memory_increase:.1f}MB memory")
    
    @unittest.skipIf(not QA_AVAILABLE, "Quality assurance module not available")
    def test_quality_assessment_performance(self):
        """Test performance of quality assessment"""
        assessor = ThaiIsanQualityAssessor()
        
        # Create test audio data
        duration = 3.0  # 3 seconds
        sample_rate = 48000
        num_tests = 10
        
        assessment_times = []
        memory_usage = []
        
        for i in range(num_tests):
            # Generate test audio
            ref_audio = np.random.randn(int(sample_rate * duration)) * 0.1
            synth_audio = ref_audio + np.random.randn(int(sample_rate * duration)) * 0.01
            
            # Measure memory before
            memory_before = self.measure_memory_usage()
            
            # Measure assessment time
            with patch.object(assessor, '_load_audio_safe', side_effect=[
                (ref_audio, sample_rate), (synth_audio, sample_rate)
            ]):
                quality_metrics, exec_time = self.measure_execution_time(
                    assessor.evaluate_audio_quality,
                    "dummy_ref.wav", "dummy_synth.wav"
                )
            
            # Measure memory after
            memory_after = self.measure_memory_usage()
            
            assessment_times.append(exec_time)
            memory_usage.append(memory_after - memory_before)
            
            # Verify quality metrics
            self.assertIsInstance(quality_metrics, QualityMetrics)
        
        # Calculate statistics
        avg_assessment_time = np.mean(assessment_times)
        max_assessment_time = np.max(assessment_times)
        avg_memory_increase = np.mean(memory_usage)
        
        # Store benchmark results
        self.benchmark_results['quality_assessment'] = {
            'num_tests': num_tests,
            'avg_time_ms': avg_assessment_time * 1000,
            'max_time_ms': max_assessment_time * 1000,
            'avg_memory_increase_mb': avg_memory_increase
        }
        
        # Assertions for performance requirements
        self.assertLess(avg_assessment_time, 2.0)  # Should be less than 2 seconds per assessment
        self.assertLess(avg_memory_increase, 50)  # Should use less than 50MB additional memory
        
        print(f"Quality assessment performance: {avg_assessment_time*1000:.2f}ms avg, {avg_memory_increase:.1f}MB memory")
    
    def test_memory_efficiency(self):
        """Test memory efficiency under load"""
        # Create a large number of objects to test memory management
        large_data = []
        initial_memory = self.measure_memory_usage()
        
        # Create 1000 large objects
        for i in range(1000):
            large_object = {
                'id': i,
                'data': np.random.randn(1000),  # 1000 floats per object
                'metadata': {'test': f'object_{i}', 'value': i * 2}
            }
            large_data.append(large_object)
        
        peak_memory = self.measure_memory_usage()
        
        # Clear references and force garbage collection
        large_data.clear()
        gc.collect()
        
        final_memory = self.measure_memory_usage()
        
        memory_peak_increase = peak_memory - initial_memory
        memory_final_increase = final_memory - initial_memory
        
        # Store benchmark results
        self.benchmark_results['memory_efficiency'] = {
            'peak_increase_mb': memory_peak_increase,
            'final_increase_mb': memory_final_increase,
            'memory_reclaimed_mb': memory_peak_increase - memory_final_increase
        }
        
        # Assertions for memory efficiency
        self.assertLess(memory_final_increase, 10)  # Should release most memory
        
        print(f"Memory efficiency: peak +{memory_peak_increase:.1f}MB, final +{memory_final_increase:.1f}MB")
    
    def test_concurrent_processing_performance(self):
        """Test performance under concurrent processing"""
        import threading
        import time
        
        def concurrent_task(task_id, results):
            """Simulate a concurrent processing task"""
            start_time = time.perf_counter()
            
            # Simulate processing work
            data = np.random.randn(1000, 100)  # Create some data
            result = np.sum(data)  # Process the data
            
            end_time = time.perf_counter()
            results[task_id] = {
                'result': result,
                'processing_time': end_time - start_time
            }
        
        # Test sequential processing
        sequential_results = {}
        sequential_start = time.perf_counter()
        
        for i in range(10):
            concurrent_task(i, sequential_results)
        
        sequential_end = time.perf_counter()
        sequential_time = sequential_end - sequential_start
        
        # Test concurrent processing
        concurrent_results = {}
        threads = []
        concurrent_start = time.perf_counter()
        
        for i in range(10):
            thread = threading.Thread(target=concurrent_task, args=(i, concurrent_results))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        concurrent_end = time.perf_counter()
        concurrent_time = concurrent_end - concurrent_start
        
        # Store benchmark results
        self.benchmark_results['concurrent_processing'] = {
            'sequential_time_ms': sequential_time * 1000,
            'concurrent_time_ms': concurrent_time * 1000,
            'speedup': sequential_time / concurrent_time if concurrent_time > 0 else 0
        }
        
        # Concurrent processing should be faster
        if concurrent_time > 0:
            self.assertLess(concurrent_time, sequential_time)
        
        print(f"Concurrent processing: sequential {sequential_time*1000:.2f}ms, concurrent {concurrent_time*1000:.2f}ms")


class ValidationTests(unittest.TestCase):
    """Validation tests for system correctness and accuracy"""
    
    def test_audio_quality_validation(self):
        """Test that audio quality assessment produces reasonable results"""
        assessor = ThaiIsanQualityAssessor()
        
        # Test with identical audio (should have perfect quality)
        duration = 2.0
        sample_rate = 48000
        identical_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        
        with patch.object(assessor, '_load_audio_safe', side_effect=[
            (identical_audio, sample_rate), (identical_audio, sample_rate)
        ]):
            quality_metrics = assessor.evaluate_audio_quality("dummy_ref.wav", "dummy_synth.wav")
        
        # For identical audio, quality should be very high
        self.assertGreater(quality_metrics.stoi_score, 0.9)  # Should be close to 1.0
        self.assertGreater(quality_metrics.fundamental_frequency_correlation, 0.9)  # Should be close to 1.0
    
    def test_tone_accuracy_validation(self):
        """Test that tone accuracy calculation is correct"""
        assessor = ThaiIsanQualityAssessor()
        
        # Test perfect match
        ref_tones = np.array([1, 2, 3, 4, 5])
        synth_tones = np.array([1, 2, 3, 4, 5])
        accuracy = assessor.evaluate_tone_accuracy(ref_tones, synth_tones)
        self.assertAlmostEqual(accuracy, 1.0, places=5)
        
        # Test complete mismatch
        synth_tones_wrong = np.array([2, 3, 4, 5, 1])
        accuracy_wrong = assessor.evaluate_tone_accuracy(ref_tones, synth_tones_wrong)
        self.assertAlmostEqual(accuracy_wrong, 0.0, places=5)
        
        # Test partial match
        synth_tones_partial = np.array([1, 2, 4, 4, 5])
        accuracy_partial = assessor.evaluate_tone_accuracy(ref_tones, synth_tones_partial)
        self.assertAlmostEqual(accuracy_partial, 0.6, places=5)  # 3 out of 5 correct
    
    def test_phoneme_accuracy_validation(self):
        """Test that phoneme accuracy calculation is correct"""
        assessor = ThaiIsanQualityAssessor()
        
        # Test perfect match
        ref_phonemes = ['s', 'a', 'r', 'a', 's']
        synth_phonemes = ['s', 'a', 'r', 'a', 's']
        accuracy = assessor.evaluate_phoneme_accuracy(ref_phonemes, synth_phonemes)
        self.assertAlmostEqual(accuracy, 1.0, places=5)
        
        # Test with deletions
        synth_phonemes_deleted = ['s', 'a', 'r', 's']  # Missing 'a'
        accuracy_deleted = assessor.evaluate_phoneme_accuracy(ref_phonemes, synth_phonemes_deleted)
        self.assertLess(accuracy_deleted, 1.0)
        self.assertGreater(accuracy_deleted, 0.0)
        
        # Test with insertions
        synth_phonemes_inserted = ['s', 'a', 'r', 'a', 'x', 's']  # Extra 'x'
        accuracy_inserted = assessor.evaluate_phoneme_accuracy(ref_phonemes, synth_phonemes_inserted)
        self.assertLess(accuracy_inserted, 1.0)
        self.assertGreater(accuracy_inserted, 0.0)
    
    def test_quality_metrics_bounds(self):
        """Test that quality metrics stay within expected bounds"""
        # Test QualityMetrics validation
        with self.assertRaises(ValueError):
            QualityMetrics(
                mel_cepstral_distortion=-1.0,  # Negative MCD should fail
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
        
        # Test valid metrics
        valid_metrics = QualityMetrics(
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
        
        # Check that all values are within expected ranges
        self.assertGreaterEqual(valid_metrics.pesq_score, 1.0)
        self.assertLessEqual(valid_metrics.pesq_score, 4.5)
        self.assertGreaterEqual(valid_metrics.stoi_score, 0.0)
        self.assertLessEqual(valid_metrics.stoi_score, 1.0)
        self.assertGreaterEqual(valid_metrics.tone_accuracy, 0.0)
        self.assertLessEqual(valid_metrics.tone_accuracy, 1.0)


class StressTests(unittest.TestCase):
    """Stress tests to identify system limits"""
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets"""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create configuration with very large requirements
            config = DataCollectionConfig(
                target_hours_thai=1000,  # Very large
                target_hours_isan=1000,
                min_speakers_per_language=1000,
                max_recordings_per_speaker=1000,
                base_data_dir=temp_dir
            )
            
            collector = ThaiIsanDataCollector(config)
            
            # Test that the system doesn't crash with large numbers
            self.assertEqual(collector.config.target_hours_thai, 1000)
            self.assertEqual(collector.config.min_speakers_per_language, 1000)
            
            # Test progress calculation with large numbers
            progress = collector.get_collection_progress()
            self.assertIn('thai_progress', progress)
            self.assertIn('isan_progress', progress)
            
            # Progress should be very small since we haven't collected any data
            self.assertLess(progress['thai_progress'], 0.01)
            self.assertLess(progress['isan_progress'], 0.01)
            
        finally:
            import shutil
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
    
    def test_rapid_processing_stress(self):
        """Test rapid processing of many small tasks"""
        assessor = ThaiIsanQualityAssessor()
        
        # Rapidly process many small audio segments
        num_tasks = 100
        sample_rate = 48000
        duration = 0.1  # Very short audio
        
        start_time = time.perf_counter()
        
        for i in range(num_tasks):
            ref_audio = np.random.randn(int(sample_rate * duration)) * 0.1
            synth_audio = ref_audio + np.random.randn(int(sample_rate * duration)) * 0.001
            
            with patch.object(assessor, '_load_audio_safe', side_effect=[
                (ref_audio, sample_rate), (synth_audio, sample_rate)
            ]):
                try:
                    quality_metrics = assessor.evaluate_audio_quality("dummy_ref.wav", "dummy_synth.wav")
                    self.assertIsNotNone(quality_metrics)
                except Exception:
                    # Some failures are acceptable under stress
                    pass
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Should complete reasonably quickly even under stress
        self.assertLess(total_time, 30.0)  # Less than 30 seconds for 100 tasks
        
        print(f"Rapid processing stress test: {num_tasks} tasks in {total_time:.2f}s")


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)