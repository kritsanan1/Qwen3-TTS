"""
Thai-Isan Speech Data Collection Pipeline
Comprehensive data collection system for gathering 100+ hours of Thai and Isan speech
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
import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from sklearn.model_selection import train_test_split

# Thai language processing
import pythainlp
from pythainlp.tokenize import word_tokenize, syllable_tokenize
from pythainlp.corpus import thai_words

logger = logging.getLogger(__name__)

@dataclass
class SpeakerInfo:
    """Information about a speaker"""
    speaker_id: str
    language: str  # 'th' or 'tts' (Isan)
    gender: str  # 'male', 'female'
    age_group: str  # 'young', 'adult', 'senior'
    region: str  # 'central', 'northeastern', 'northern', 'southern'
    dialect: str  # Specific dialect variant
    native_language: str
    years_speaking: int
    recording_quality: str  # 'studio', 'professional', 'consumer'
    accent_rating: float  # 1-5 scale
    clarity_rating: float  # 1-5 scale

@dataclass
class AudioRecording:
    """Individual audio recording metadata"""
    recording_id: str
    speaker_id: str
    text: str
    language: str
    duration: float
    sample_rate: int
    bit_depth: int
    channels: int
    file_size: int
    file_path: str
    quality_score: float
    timestamp: datetime
    background_noise_level: float
    signal_to_noise_ratio: float

@dataclass
class DataCollectionConfig:
    """Configuration for data collection"""
    # Target requirements
    target_hours_thai: int = 100
    target_hours_isan: int = 100
    min_speakers_per_language: int = 50
    max_recordings_per_speaker: int = 200
    
    # Audio quality settings
    target_sample_rate: int = 48000
    target_bit_depth: int = 24
    min_duration_seconds: float = 2.0
    max_duration_seconds: float = 15.0
    target_snr_db: float = 30.0
    
    # Quality thresholds
    min_quality_score: float = 0.8
    max_background_noise: float = -40.0  # dB
    min_clarity_rating: float = 4.0
    
    # Storage settings
    base_data_dir: str = "./data"
    backup_enabled: bool = True
    compression_format: str = "flac"
    
    # Processing settings
    num_workers: int = 8
    batch_size: int = 32
    validation_split: float = 0.1
    test_split: float = 0.1

class ThaiIsanDataCollector:
    """
    Comprehensive data collection system for Thai and Isan speech data
    """
    
    def __init__(self, config: DataCollectionConfig = None):
        self.config = config or DataCollectionConfig()
        self.speakers: Dict[str, SpeakerInfo] = {}
        self.recordings: Dict[str, AudioRecording] = {}
        self.language_stats = {'th': {'hours': 0, 'speakers': 0, 'recordings': 0},
                            'tts': {'hours': 0, 'speakers': 0, 'recordings': 0}}
        
        # Setup directories
        self.setup_directories()
        
        # Initialize language-specific text corpora
        self.thai_texts = self.load_thai_text_corpus()
        self.isan_texts = self.load_isan_text_corpus()
        
        logger.info(f"Initialized Thai-Isan data collector targeting "
                   f"{self.config.target_hours_thai}h Thai + "
                   f"{self.config.target_hours_isan}h Isan speech")
    
    def setup_directories(self):
        """Create necessary directory structure"""
        dirs = [
            self.config.base_data_dir,
            f"{self.config.base_data_dir}/speakers",
            f"{self.config.base_data_dir}/recordings/thai",
            f"{self.config.base_data_dir}/recordings/isan",
            f"{self.config.base_data_dir}/texts",
            f"{self.config.base_data_dir}/metadata",
            f"{self.config.base_data_dir}/quality_reports",
            f"{self.config.base_data_dir}/backups"
        ]
        
        for directory in dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_thai_text_corpus(self) -> List[str]:
        """Load Thai text corpus for recording prompts"""
        # This would typically load from external files
        # For now, return a diverse set of Thai sentences
        return [
            "สวัสดีครับ ผมชื่อสมชาย",
            "วันนี้อากาศดีมาก",
            "คุณกินข้าวหรือยังครับ",
            "ผมมาจากกรุงเทพฯ",
            "นี่คือบ้านของผม",
            "ขอบคุณมากครับ",
            "ลาก่อนนะครับ",
            "เชิญทางนี้ครับ",
            "อันนี้ราคาเท่าไหร่ครับ",
            "ผมไม่เข้าใจที่คุณพูด",
            "กรุณาพูดใหม่ได้ไหมครับ",
            "วันนี้เป็นวันพระ",
            "ผมต้องไปทำงานแล้ว",
            "คุณพูดภาษาไทยได้ไหม",
            "นี่คือเพื่อนของผม"
        ]
    
    def load_isan_text_corpus(self) -> List[str]:
        """Load Isan text corpus for recording prompts"""
        # Isan (Northeastern Thai) sentences with authentic vocabulary
        return [
            "สบายดีบ่ คุณสบายดีบ่",
            "เฮาไปกินข้าวกันเด้อ",
            "บ่เข้าใจที่คุณพูด",
            "เฮามาจากอีสาน",
            "นี่คือบ้านของเฮา",
            "ขอบคุณหลายๆ",
            "ลาก่อนนำเด้อ",
            "เชิญทางนี้เด้อ",
            "อันนี้ราคาเท่าใด",
            "เฮาบ่เข้าใจภาษาของคุณ",
            "กรุณาพูดใหม่ได้บ่",
            "วันนี้เป็นวันพระ",
            "เฮาต้องไปทำนาแล้ว",
            "คุณพูดภาษาอีสานได้บ่",
            "นี่คือเพื่อนของเฮา"
        ]
    
    def register_speaker(self, speaker_info: SpeakerInfo) -> str:
        """Register a new speaker in the system"""
        speaker_id = speaker_info.speaker_id
        self.speakers[speaker_id] = speaker_info
        
        # Create speaker directory
        speaker_dir = f"{self.config.base_data_dir}/speakers/{speaker_id}"
        Path(speaker_dir).mkdir(parents=True, exist_ok=True)
        
        # Save speaker metadata
        speaker_file = f"{speaker_dir}/speaker_info.json"
        with open(speaker_file, 'w', encoding='utf-8') as f:
            json.dump({
                'speaker_id': speaker_info.speaker_id,
                'language': speaker_info.language,
                'gender': speaker_info.gender,
                'age_group': speaker_info.age_group,
                'region': speaker_info.region,
                'dialect': speaker_info.dialect,
                'native_language': speaker_info.native_language,
                'years_speaking': speaker_info.years_speaking,
                'recording_quality': speaker_info.recording_quality,
                'accent_rating': speaker_info.accent_rating,
                'clarity_rating': speaker_info.clarity_rating
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Registered speaker {speaker_id} ({speaker_info.language})")
        self.language_stats[speaker_info.language]['speakers'] += 1
        return speaker_id
    
    def record_audio_sample(self, speaker_id: str, text: str, language: str) -> Optional[AudioRecording]:
        """
        Record a single audio sample from a speaker
        This is a template method - in practice, this would interface with actual recording hardware
        """
        if speaker_id not in self.speakers:
            logger.error(f"Speaker {speaker_id} not registered")
            return None
        
        speaker = self.speakers[speaker_id]
        
        # Generate unique recording ID
        recording_id = f"{speaker_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Simulate recording process (in practice, this would capture real audio)
        duration = np.random.uniform(self.config.min_duration_seconds, 
                                   self.config.max_duration_seconds)
        
        # Generate synthetic audio for demonstration (replace with actual recording)
        sample_rate = self.config.target_sample_rate
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create synthetic speech-like signal
        frequency = 200 if speaker.gender == 'male' else 250
        audio_signal = np.sin(2 * np.pi * frequency * t) * 0.5
        
        # Add some harmonics to make it more speech-like
        audio_signal += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
        audio_signal += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)
        
        # Add slight noise
        noise = np.random.normal(0, 0.01, len(audio_signal))
        audio_signal += noise
        
        # Apply fade in/out
        fade_length = int(0.1 * sample_rate)
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)
        audio_signal[:fade_length] *= fade_in
        audio_signal[-fade_length:] *= fade_out
        
        # Calculate quality metrics
        quality_score = self.calculate_audio_quality(audio_signal, sample_rate)
        background_noise = self.estimate_background_noise(audio_signal)
        snr = self.calculate_snr(audio_signal, noise)
        
        # Save audio file
        language_dir = 'thai' if language == 'th' else 'isan'
        audio_path = f"{self.config.base_data_dir}/recordings/{language_dir}/{recording_id}.wav"
        
        # Ensure directory exists
        Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as WAV file
        sf.write(audio_path, audio_signal, sample_rate)
        
        # Create recording object
        recording = AudioRecording(
            recording_id=recording_id,
            speaker_id=speaker_id,
            text=text,
            language=language,
            duration=duration,
            sample_rate=sample_rate,
            bit_depth=self.config.target_bit_depth,
            channels=1,
            file_size=os.path.getsize(audio_path),
            file_path=audio_path,
            quality_score=quality_score,
            timestamp=datetime.now(),
            background_noise_level=background_noise,
            signal_to_noise_ratio=snr
        )
        
        # Validate quality
        if self.validate_recording_quality(recording):
            self.recordings[recording_id] = recording
            self.language_stats[language]['recordings'] += 1
            self.language_stats[language]['hours'] += duration / 3600
            
            logger.info(f"Recorded sample {recording_id} "
                       f"({duration:.1f}s, quality: {quality_score:.2f})")
            return recording
        else:
            # Remove low-quality recording
            if os.path.exists(audio_path):
                os.remove(audio_path)
            logger.warning(f"Rejected low-quality recording {recording_id}")
            return None
    
    def calculate_audio_quality(self, audio_signal: np.ndarray, sample_rate: int) -> float:
        """Calculate audio quality score (0-1)"""
        # Simple quality metrics
        signal_power = np.mean(audio_signal ** 2)
        
        # Check for clipping
        clipping_ratio = np.sum(np.abs(audio_signal) > 0.95) / len(audio_signal)
        
        # Check dynamic range
        dynamic_range = np.max(audio_signal) - np.min(audio_signal)
        
        # Calculate quality score
        quality_score = 1.0
        
        # Penalize clipping
        if clipping_ratio > 0.01:
            quality_score *= (1 - clipping_ratio)
        
        # Penalize low dynamic range
        if dynamic_range < 0.1:
            quality_score *= 0.8
        
        # Penalize low signal power
        if signal_power < 0.01:
            quality_score *= 0.7
        
        return max(0.0, min(1.0, quality_score))
    
    def estimate_background_noise(self, audio_signal: np.ndarray) -> float:
        """Estimate background noise level in dB"""
        # Simple noise estimation (would be more sophisticated in practice)
        noise_level = np.std(audio_signal) * 0.1  # Rough estimate
        return 20 * np.log10(noise_level + 1e-10)  # Convert to dB
    
    def calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """Calculate signal-to-noise ratio in dB"""
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            return 10 * np.log10(signal_power / noise_power)
        return 60.0  # High SNR if no noise
    
    def validate_recording_quality(self, recording: AudioRecording) -> bool:
        """Validate if recording meets quality standards"""
        return (recording.quality_score >= self.config.min_quality_score and
                recording.background_noise_level <= self.config.max_background_noise and
                self.config.min_duration_seconds <= recording.duration <= self.config.max_duration_seconds)
    
    def collect_speaker_data(self, speaker_id: str, num_recordings: int = None) -> int:
        """
        Collect multiple recordings from a single speaker
        """
        if speaker_id not in self.speakers:
            logger.error(f"Speaker {speaker_id} not registered")
            return 0
        
        speaker = self.speakers[speaker_id]
        language = speaker.language
        
        # Determine number of recordings
        if num_recordings is None:
            num_recordings = min(50, self.config.max_recordings_per_speaker)
        
        # Get appropriate text corpus
        texts = self.thai_texts if language == 'th' else self.isan_texts
        
        successful_recordings = 0
        
        for i in range(num_recordings):
            # Select text (cycle through corpus)
            text = texts[i % len(texts)]
            
            # Record audio sample
            recording = self.record_audio_sample(speaker_id, text, language)
            
            if recording is not None:
                successful_recordings += 1
                
                # Save metadata
                self.save_recording_metadata(recording)
        
        logger.info(f"Collected {successful_recordings}/{num_recordings} recordings "
                   f"from speaker {speaker_id}")
        
        return successful_recordings
    
    def save_recording_metadata(self, recording: AudioRecording):
        """Save recording metadata to disk"""
        metadata_path = f"{self.config.base_data_dir}/metadata/{recording.recording_id}.json"
        
        metadata = {
            'recording_id': recording.recording_id,
            'speaker_id': recording.speaker_id,
            'text': recording.text,
            'language': recording.language,
            'duration': recording.duration,
            'sample_rate': recording.sample_rate,
            'bit_depth': recording.bit_depth,
            'channels': recording.channels,
            'file_size': recording.file_size,
            'file_path': recording.file_path,
            'quality_score': recording.quality_score,
            'timestamp': recording.timestamp.isoformat(),
            'background_noise_level': recording.background_noise_level,
            'signal_to_noise_ratio': recording.signal_to_noise_ratio
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def generate_comprehensive_dataset(self) -> Dict[str, int]:
        """
        Generate a comprehensive dataset with multiple speakers
        """
        logger.info("Starting comprehensive dataset generation...")
        
        # Register sample speakers (in practice, these would be real speakers)
        self.register_sample_speakers()
        
        # Collect data from each speaker
        results = {'th': 0, 'tts': 0}
        
        for speaker_id in self.speakers:
            speaker = self.speakers[speaker_id]
            num_recorded = self.collect_speaker_data(speaker_id, 25)  # 25 recordings per speaker
            results[speaker.language] += num_recorded
        
        # Generate comprehensive statistics
        self.generate_dataset_statistics()
        
        logger.info(f"Dataset generation complete. Results: {results}")
        return results
    
    def register_sample_speakers(self):
        """Register sample speakers for demonstration"""
        # Thai speakers
        thai_speakers = [
            SpeakerInfo("th_male_001", "th", "male", "adult", "central", "standard", "thai", 30, "professional", 4.5, 4.8),
            SpeakerInfo("th_female_001", "th", "female", "adult", "central", "standard", "thai", 28, "professional", 4.7, 4.9),
            SpeakerInfo("th_male_002", "th", "male", "young", "northern", "northern", "thai", 22, "consumer", 4.2, 4.5),
            SpeakerInfo("th_female_002", "th", "female", "senior", "southern", "southern", "thai", 55, "professional", 4.3, 4.6),
        ]
        
        # Isan speakers
        isan_speakers = [
            SpeakerInfo("tts_male_001", "tts", "male", "adult", "northeastern", "central_isan", "isan", 35, "professional", 4.6, 4.7),
            SpeakerInfo("tts_female_001", "tts", "female", "adult", "northeastern", "central_isan", "isan", 32, "professional", 4.8, 4.9),
            SpeakerInfo("tts_male_002", "tts", "male", "young", "northeastern", "northern_isan", "isan", 25, "consumer", 4.1, 4.4),
            SpeakerInfo("tts_female_002", "tts", "female", "adult", "northeastern", "southern_isan", "isan", 38, "professional", 4.4, 4.7),
        ]
        
        for speaker in thai_speakers + isan_speakers:
            self.register_speaker(speaker)
    
    def generate_dataset_statistics(self):
        """Generate comprehensive dataset statistics"""
        stats = {
            'dataset_info': {
                'created_at': datetime.now().isoformat(),
                'total_speakers': len(self.speakers),
                'total_recordings': len(self.recordings),
                'total_hours': sum(self.language_stats[lang]['hours'] for lang in self.language_stats)
            },
            'language_breakdown': self.language_stats,
            'speaker_demographics': self.analyze_speaker_demographics(),
            'quality_metrics': self.analyze_quality_metrics(),
            'recording_statistics': self.analyze_recording_statistics()
        }
        
        # Save statistics
        stats_path = f"{self.config.base_data_dir}/dataset_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dataset statistics saved to {stats_path}")
        return stats
    
    def analyze_speaker_demographics(self) -> Dict:
        """Analyze speaker demographics"""
        demographics = {
            'gender_distribution': {'male': 0, 'female': 0},
            'age_distribution': {'young': 0, 'adult': 0, 'senior': 0},
            'region_distribution': {},
            'quality_distribution': {}
        }
        
        for speaker in self.speakers.values():
            demographics['gender_distribution'][speaker.gender] += 1
            demographics['age_distribution'][speaker.age_group] += 1
            
            region = speaker.region
            if region not in demographics['region_distribution']:
                demographics['region_distribution'][region] = 0
            demographics['region_distribution'][region] += 1
        
        return demographics
    
    def analyze_quality_metrics(self) -> Dict:
        """Analyze quality metrics across the dataset"""
        if not self.recordings:
            return {}
        
        quality_scores = [r.quality_score for r in self.recordings.values()]
        background_noise = [r.background_noise_level for r in self.recordings.values()]
        snr_values = [r.signal_to_noise_ratio for r in self.recordings.values()]
        
        return {
            'quality_score_stats': {
                'mean': float(np.mean(quality_scores)),
                'std': float(np.std(quality_scores)),
                'min': float(np.min(quality_scores)),
                'max': float(np.max(quality_scores))
            },
            'background_noise_stats': {
                'mean': float(np.mean(background_noise)),
                'std': float(np.std(background_noise)),
                'min': float(np.min(background_noise)),
                'max': float(np.max(background_noise))
            },
            'snr_stats': {
                'mean': float(np.mean(snr_values)),
                'std': float(np.std(snr_values)),
                'min': float(np.min(snr_values)),
                'max': float(np.max(snr_values))
            }
        }
    
    def analyze_recording_statistics(self) -> Dict:
        """Analyze recording statistics"""
        if not self.recordings:
            return {}
        
        durations = [r.duration for r in self.recordings.values()]
        languages = [r.language for r in self.recordings.values()]
        
        language_durations = {}
        for lang in set(languages):
            lang_durations = [r.duration for r in self.recordings.values() if r.language == lang]
            language_durations[lang] = {
                'total_hours': sum(lang_durations) / 3600,
                'count': len(lang_durations),
                'avg_duration': float(np.mean(lang_durations)),
                'std_duration': float(np.std(lang_durations))
            }
        
        return {
            'duration_stats': {
                'mean': float(np.mean(durations)),
                'std': float(np.std(durations)),
                'min': float(np.min(durations)),
                'max': float(np.max(durations))
            },
            'language_breakdown': language_durations,
            'total_hours': sum(durations) / 3600
        }
    
    def create_training_splits(self) -> Dict[str, List[str]]:
        """Create train/validation/test splits for model training"""
        recording_ids = list(self.recordings.keys())
        
        # Group by speaker to ensure speaker independence in splits
        speaker_recordings = {}
        for rec_id in recording_ids:
            speaker_id = self.recordings[rec_id].speaker_id
            if speaker_id not in speaker_recordings:
                speaker_recordings[speaker_id] = []
            speaker_recordings[speaker_id].append(rec_id)
        
        # Split speakers (not individual recordings)
        speakers = list(speaker_recordings.keys())
        train_speakers, temp_speakers = train_test_split(speakers, test_size=0.2, random_state=42)
        val_speakers, test_speakers = train_test_split(temp_speakers, test_size=0.5, random_state=42)
        
        # Create recording splits
        train_recordings = []
        val_recordings = []
        test_recordings = []
        
        for speaker in train_speakers:
            train_recordings.extend(speaker_recordings[speaker])
        for speaker in val_speakers:
            val_recordings.extend(speaker_recordings[speaker])
        for speaker in test_speakers:
            test_recordings.extend(speaker_recordings[speaker])
        
        splits = {
            'train': train_recordings,
            'validation': val_recordings,
            'test': test_recordings,
            'metadata': {
                'num_train_speakers': len(train_speakers),
                'num_val_speakers': len(val_speakers),
                'num_test_speakers': len(test_speakers),
                'num_train_recordings': len(train_recordings),
                'num_val_recordings': len(val_recordings),
                'num_test_recordings': len(test_recordings)
            }
        }
        
        # Save splits
        splits_path = f"{self.config.base_data_dir}/training_splits.json"
        with open(splits_path, 'w', encoding='utf-8') as f:
            json.dump(splits, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created training splits: {splits['metadata']}")
        return splits

def main():
    """Main function to demonstrate data collection"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create data collector
    config = DataCollectionConfig(
        target_hours_thai=100,
        target_hours_isan=100,
        min_speakers_per_language=50,
        num_workers=8
    )
    
    collector = ThaiIsanDataCollector(config)
    
    # Generate comprehensive dataset
    results = collector.generate_comprehensive_dataset()
    print(f"Dataset generation results: {results}")
    
    # Create training splits
    splits = collector.create_training_splits()
    print(f"Training splits created: {splits['metadata']}")
    
    # Display final statistics
    print(f"\nFinal dataset statistics:")
    print(f"Total speakers: {len(collector.speakers)}")
    print(f"Total recordings: {len(collector.recordings)}")
    print(f"Total hours: {collector.language_stats['th']['hours'] + collector.language_stats['tts']['hours']:.1f}")
    print(f"Thai hours: {collector.language_stats['th']['hours']:.1f}")
    print(f"Isan hours: {collector.language_stats['tts']['hours']:.1f}")

if __name__ == "__main__":
    main()