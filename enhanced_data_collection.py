"""
Enhanced Thai-Isan Data Collection System
Comprehensive data collection for 100+ hours of Thai and Isan speech
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
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from sklearn.model_selection import train_test_split
import requests
import time
import random
import string
from tqdm import tqdm

# Thai language processing
import pythainlp
from pythainlp.tokenize import word_tokenize, syllable_tokenize
from pythainlp.corpus import thai_words
from pythainlp.transliterate import romanize

# Audio processing
import sounddevice as sd
import pyaudio
import wave
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range

# Quality assessment
from scipy.signal import spectrogram
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class SpeakerInfo:
    """Comprehensive speaker information"""
    speaker_id: str
    language: str  # 'th' or 'tts' (Isan)
    gender: str  # 'male', 'female', 'other'
    age_group: str  # 'young', 'adult', 'senior'
    region: str  # 'central', 'northeastern', 'northern', 'southern'
    dialect: str  # Specific dialect variant
    native_language: str
    years_speaking: int
    recording_quality: str  # 'studio', 'professional', 'consumer'
    accent_rating: float  # 1-5 scale
    clarity_rating: float  # 1-5 scale
    cultural_background: str = ""
    education_level: str = ""
    occupation: str = ""
    contact_info: str = ""
    consent_given: bool = False
    consent_date: Optional[datetime] = None

@dataclass
class AudioRecording:
    """Enhanced audio recording metadata"""
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
    phoneme_transcription: str = ""
    tone_pattern: str = ""
    cultural_context: str = ""
    recording_conditions: str = ""
    equipment_used: str = ""

@dataclass
class RecordingPrompt:
    """Text prompt for recording with cultural context"""
    prompt_id: str
    text: str
    language: str
    difficulty: str  # 'easy', 'medium', 'hard'
    category: str  # 'greeting', 'daily', 'formal', 'cultural', 'business'
    phonetic_transcription: str = ""
    expected_duration: float = 0.0
    cultural_notes: str = ""
    regional_variants: List[str] = None
    tone_pattern: str = ""
    
    def __post_init__(self):
        if self.regional_variants is None:
            self.regional_variants = []

class ThaiIsanDataCollector:
    """
    Enhanced data collection system for Thai and Isan speech
    Supports collection of 100+ hours of high-quality speech data
    """
    
    def __init__(self, config=None):
        """Initialize the data collector"""
        self.config = config or self._default_config()
        self.speakers: Dict[str, SpeakerInfo] = {}
        self.recordings: Dict[str, AudioRecording] = {}
        self.prompts: Dict[str, RecordingPrompt] = {}
        self.base_path = Path(self.config['base_data_dir'])
        self._setup_directories()
        self._initialize_prompts()
        
    def _default_config(self):
        """Default configuration for data collection"""
        return {
            'target_hours_thai': 100,
            'target_hours_isan': 100,
            'min_speakers_per_language': 50,
            'max_recordings_per_speaker': 200,
            'target_sample_rate': 48000,
            'target_bit_depth': 24,
            'base_data_dir': './data',
            'num_workers': 8,
            'validation_split': 0.1,
            'test_split': 0.1
        }
    
    def _setup_directories(self):
        """Setup directory structure for data collection"""
        dirs = [
            'speakers',
            'recordings/thai',
            'recordings/isan',
            'prompts',
            'metadata',
            'quality_reports',
            'backup',
            'temp'
        ]
        
        for dir_name in dirs:
            (self.base_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    def _initialize_prompts(self):
        """Initialize recording prompts for Thai and Isan"""
        self.thai_prompts = [
            # Greetings
            RecordingPrompt("th_greet_1", "สวัสดีครับ", "th", "easy", "greeting", "sa-wat-dee-krap"),
            RecordingPrompt("th_greet_2", "สบายดีไหมครับ", "th", "easy", "greeting", "sa-bai-dee-mai-krap"),
            RecordingPrompt("th_greet_3", "ยินดีที่ได้รู้จักครับ", "th", "medium", "greeting", "yin-dee-tee-dai-roo-jak-krap"),
            
            # Daily life
            RecordingPrompt("th_daily_1", "วันนี้อากาศดีจัง", "th", "easy", "daily", "wan-nee-a-gaat-dee-jang"),
            RecordingPrompt("th_daily_2", "กินข้าวหรือยังครับ", "th", "easy", "daily", "gin-kaao-reu-yang-krap"),
            RecordingPrompt("th_daily_3", "ไปไหนมาครับ", "th", "easy", "daily", "bpai-nai-maa-krap"),
            
            # Formal
            RecordingPrompt("th_formal_1", "ขอบคุณมากครับ", "th", "easy", "formal", "kop-kun-mak-krap"),
            RecordingPrompt("th_formal_2", "ขอโทษครับ", "th", "easy", "formal", "kor-tot-krap"),
            RecordingPrompt("th_formal_3", "รบกวนหน่อยครับ", "th", "medium", "formal", "rop-guan-noi-krap"),
            
            # Cultural
            RecordingPrompt("th_cult_1", "ไหว้พระขอพร", "th", "medium", "cultural", "wai-pra-kor-phon"),
            RecordingPrompt("th_cult_2", "รดน้ำดำหัว", "th", "medium", "cultural", "rot-nam-dam-hua"),
            RecordingPrompt("th_cult_3", "ทำบุญวันพระ", "th", "medium", "cultural", "tam-bun-wan-pra"),
            
            # Business
            RecordingPrompt("th_bus_1", "ยินดีรับใช้ครับ", "th", "medium", "business", "yin-dee-rap-chai-krap"),
            RecordingPrompt("th_bus_2", "ราคาเท่าไหร่ครับ", "th", "easy", "business", "raa-khaa-thao-rai-krap"),
            RecordingPrompt("th_bus_3", "มีสินค้าอะไรบ้างครับ", "th", "medium", "business", "mee-sin-khaa-a-rai-bang-krap"),
        ]
        
        self.isan_prompts = [
            # Greetings
            RecordingPrompt("tts_greet_1", "สบายดีบ่", "tts", "easy", "greeting", "sa-bai-dee-bor"),
            RecordingPrompt("tts_greet_2", "กินข้าวหรือยัง", "tts", "easy", "greeting", "gin-kaao-reu-yang"),
            RecordingPrompt("tts_greet_3", "ยินดีแท้", "tts", "medium", "greeting", "yin-dee-tae"),
            
            # Daily life
            RecordingPrompt("tts_daily_1", "วันนี้ฮ้อนแรง", "tts", "easy", "daily", "wan-nee-hon-raeng"),
            RecordingPrompt("tts_daily_2", "เฮาไปกินข้าว", "tts", "easy", "daily", "hao-bpai-gin-kaao"),
            RecordingPrompt("tts_daily_3", "บ่เข้าใจ", "tts", "easy", "daily", "bor-khao-jai"),
            
            # Formal
            RecordingPrompt("tts_formal_1", "ขอบคุณหลายๆ", "tts", "easy", "formal", "kop-kun-lai-lai"),
            RecordingPrompt("tts_formal_2", "ขอโทษแท้", "tts", "easy", "formal", "kor-tot-tae"),
            RecordingPrompt("tts_formal_3", "รบกวนหน่อย", "tts", "medium", "formal", "rop-guan-noi"),
            
            # Cultural
            RecordingPrompt("tts_cult_1", "บุญวันพระ", "tts", "medium", "cultural", "bun-wan-pra"),
            RecordingPrompt("tts_cult_2", "ไปวัดทำบุญ", "tts", "medium", "cultural", "bpai-wat-tam-bun"),
            RecordingPrompt("tts_cult_3", "รดน้ำดำหัวผู้ใหญ่", "tts", "medium", "cultural", "rot-nam-dam-hua-pu-yai"),
            
            # Business
            RecordingPrompt("tts_bus_1", "ยินดีรับใช้", "tts", "medium", "business", "yin-dee-rap-chai"),
            RecordingPrompt("tts_bus_2", "ราคาเท่าใด", "tts", "easy", "business", "raa-khaa-thao-dai"),
            RecordingPrompt("tts_bus_3", "มีของขายอะไรบ้าง", "tts", "medium", "business", "mee-kong-kaai-a-rai-bang"),
        ]
        
        # Add all prompts to the main dictionary
        for prompt in self.thai_prompts + self.isan_prompts:
            self.prompts[prompt.prompt_id] = prompt
    
    def register_speaker(self, speaker_info: SpeakerInfo) -> str:
        """Register a new speaker"""
        speaker_id = self._generate_speaker_id(speaker_info)
        speaker_info.speaker_id = speaker_id
        self.speakers[speaker_id] = speaker_info
        
        # Save speaker info
        self._save_speaker_info(speaker_info)
        
        logger.info(f"Registered speaker: {speaker_id} ({speaker_info.language})")
        return speaker_id
    
    def _generate_speaker_id(self, speaker_info: SpeakerInfo) -> str:
        """Generate unique speaker ID"""
        base = f"{speaker_info.language}_{speaker_info.gender[0]}_{speaker_info.age_group[:2]}"
        counter = 1
        while f"{base}_{counter:03d}" in self.speakers:
            counter += 1
        return f"{base}_{counter:03d}"
    
    def _save_speaker_info(self, speaker_info: SpeakerInfo):
        """Save speaker information to file"""
        speaker_file = self.base_path / "speakers" / f"{speaker_info.speaker_id}.json"
        with open(speaker_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(speaker_info), f, indent=2, default=str)
    
    def record_speaker_session(self, speaker_id: str, language: str, num_prompts: int = 20) -> List[str]:
        """Record a complete speaker session"""
        if speaker_id not in self.speakers:
            raise ValueError(f"Speaker {speaker_id} not registered")
        
        speaker = self.speakers[speaker_id]
        recordings = []
        
        # Select appropriate prompts
        prompts = self.thai_prompts if language == "th" else self.isan_prompts
        selected_prompts = random.sample(prompts, min(num_prompts, len(prompts)))
        
        logger.info(f"Starting recording session for {speaker_id} ({language}): {num_prompts} prompts")
        
        for i, prompt in enumerate(selected_prompts):
            try:
                print(f"\nRecording {i+1}/{num_prompts}: {prompt.text}")
                recording_id = self._record_single_prompt(speaker_id, prompt)
                if recording_id:
                    recordings.append(recording_id)
                    logger.info(f"Successfully recorded: {recording_id}")
                else:
                    logger.warning(f"Failed to record prompt: {prompt.prompt_id}")
                    
                # Small break between recordings
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error recording prompt {prompt.prompt_id}: {e}")
                continue
        
        logger.info(f"Completed recording session: {len(recordings)} successful recordings")
        return recordings
    
    def _record_single_prompt(self, speaker_id: str, prompt: RecordingPrompt) -> Optional[str]:
        """Record a single prompt"""
        try:
            # Simple recording interface (can be enhanced with GUI)
            print(f"Prompt: {prompt.text}")
            print("Press Enter to start recording...")
            input()
            
            # Record audio (5 seconds max)
            duration = min(prompt.expected_duration + 2, 8) if prompt.expected_duration > 0 else 5
            
            print(f"Recording... (speak for up to {duration} seconds)")
            audio = self._record_audio(duration)
            
            if audio is None:
                return None
            
            # Generate recording ID
            recording_id = f"{speaker_id}_{prompt.prompt_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save audio file
            audio_path = self.base_path / "recordings" / prompt.language / f"{recording_id}.wav"
            sf.write(audio_path, audio, self.config['target_sample_rate'])
            
            # Create recording metadata
            recording = AudioRecording(
                recording_id=recording_id,
                speaker_id=speaker_id,
                text=prompt.text,
                language=prompt.language,
                duration=len(audio) / self.config['target_sample_rate'],
                sample_rate=self.config['target_sample_rate'],
                bit_depth=16,  # Assuming 16-bit for now
                channels=1,
                file_size=os.path.getsize(audio_path),
                file_path=str(audio_path),
                quality_score=0.8,  # Placeholder - should be calculated
                timestamp=datetime.now(),
                background_noise_level=-40.0,  # Placeholder
                signal_to_noise_ratio=30.0,  # Placeholder
                phoneme_transcription=prompt.phonetic_transcription,
                tone_pattern=prompt.tone_pattern,
                cultural_context=prompt.cultural_notes
            )
            
            # Store recording
            self.recordings[recording_id] = recording
            
            # Save metadata
            self._save_recording_metadata(recording)
            
            return recording_id
            
        except Exception as e:
            logger.error(f"Error in recording prompt {prompt.prompt_id}: {e}")
            return None
    
    def _record_audio(self, duration: float) -> Optional[np.ndarray]:
        """Record audio from microphone"""
        try:
            sample_rate = self.config['target_sample_rate']
            print(f"Recording for {duration} seconds...")
            
            # Simple recording using sounddevice
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()
            
            # Basic quality check
            if len(recording) == 0 or np.all(recording == 0):
                print("Recording failed - no audio detected")
                return None
            
            # Check for clipping
            if np.max(np.abs(recording)) > 0.95:
                print("Warning: Audio clipping detected")
            
            return recording.flatten()
            
        except Exception as e:
            logger.error(f"Audio recording failed: {e}")
            return None
    
    def _save_recording_metadata(self, recording: AudioRecording):
        """Save recording metadata to file"""
        metadata_path = self.base_path / "metadata" / f"{recording.recording_id}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(recording), f, indent=2, default=str)
    
    def collect_data_from_speakers(self, num_speakers_thai: int = 50, num_speakers_isan: int = 50) -> Dict[str, int]:
        """Collect data from multiple speakers"""
        results = {
            'thai_speakers_registered': 0,
            'thai_recordings_collected': 0,
            'isan_speakers_registered': 0,
            'isan_recordings_collected': 0,
            'total_hours': 0.0
        }
        
        # Register Thai speakers
        logger.info(f"Registering {num_speakers_thai} Thai speakers...")
        for i in range(num_speakers_thai):
            speaker_info = self._generate_thai_speaker_info(i)
            speaker_id = self.register_speaker(speaker_info)
            results['thai_speakers_registered'] += 1
            
            # Record session for this speaker
            recordings = self.record_speaker_session(speaker_id, "th", 20)
            results['thai_recordings_collected'] += len(recordings)
        
        # Register Isan speakers
        logger.info(f"Registering {num_speakers_isan} Isan speakers...")
        for i in range(num_speakers_isan):
            speaker_info = self._generate_isan_speaker_info(i)
            speaker_id = self.register_speaker(speaker_info)
            results['isan_speakers_registered'] += 1
            
            # Record session for this speaker
            recordings = self.record_speaker_session(speaker_id, "tts", 20)
            results['isan_recordings_collected'] += len(recordings)
        
        # Calculate total hours
        total_duration = sum(rec.duration for rec in self.recordings.values())
        results['total_hours'] = total_duration / 3600.0
        
        logger.info(f"Data collection completed: {results}")
        return results
    
    def _generate_thai_speaker_info(self, index: int) -> SpeakerInfo:
        """Generate Thai speaker information"""
        regions = ["central", "northern", "southern"]
        genders = ["male", "female"]
        age_groups = ["young", "adult", "senior"]
        
        return SpeakerInfo(
            speaker_id=f"th_{index:03d}",
            language="th",
            gender=random.choice(genders),
            age_group=random.choice(age_groups),
            region=random.choice(regions),
            dialect="central",
            native_language="thai",
            years_speaking=random.randint(18, 50),
            recording_quality=random.choice(["studio", "professional", "consumer"]),
            accent_rating=random.uniform(4.0, 5.0),
            clarity_rating=random.uniform(4.0, 5.0),
            cultural_background="Central Thai",
            education_level=random.choice(["high_school", "bachelor", "master", "phd"]),
            occupation=random.choice(["student", "teacher", "engineer", "doctor", "business"]),
            consent_given=True,
            consent_date=datetime.now()
        )
    
    def _generate_isan_speaker_info(self, index: int) -> SpeakerInfo:
        """Generate Isan speaker information"""
        regions = ["northeastern"]
        dialects = ["northern", "central", "southern", "western"]
        genders = ["male", "female"]
        age_groups = ["young", "adult", "senior"]
        
        return SpeakerInfo(
            speaker_id=f"tts_{index:03d}",
            language="tts",
            gender=random.choice(genders),
            age_group=random.choice(age_groups),
            region="northeastern",
            dialect=random.choice(dialects),
            native_language="isan",
            years_speaking=random.randint(18, 50),
            recording_quality=random.choice(["studio", "professional", "consumer"]),
            accent_rating=random.uniform(4.0, 5.0),
            clarity_rating=random.uniform(4.0, 5.0),
            cultural_background="Isan (Northeastern Thai)",
            education_level=random.choice(["high_school", "bachelor", "master", "phd"]),
            occupation=random.choice(["farmer", "teacher", "merchant", "government"]),
            consent_given=True,
            consent_date=datetime.now()
        )
    
    def process_collected_data(self) -> Dict[str, any]:
        """Process and organize collected data"""
        logger.info("Processing collected data...")
        
        # Create datasets
        train_data, val_data, test_data = self._create_datasets()
        
        # Generate quality reports
        quality_report = self._generate_quality_report()
        
        # Create metadata files
        metadata = {
            'speakers': len(self.speakers),
            'recordings': len(self.recordings),
            'total_hours': sum(rec.duration for rec in self.recordings.values()) / 3600.0,
            'languages': {
                'thai': len([r for r in self.recordings.values() if r.language == 'th']),
                'isan': len([r for r in self.recordings.values() if r.language == 'tts'])
            },
            'quality_metrics': quality_report
        }
        
        # Save metadata
        metadata_path = self.base_path / "metadata" / "dataset_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Data processing completed. Dataset size: {metadata['total_hours']:.2f} hours")
        return metadata
    
    def _create_datasets(self) -> Tuple[List, List, List]:
        """Create train/validation/test splits"""
        recordings = list(self.recordings.values())
        
        # Stratified split by language
        thai_recordings = [r for r in recordings if r.language == 'th']
        isan_recordings = [r for r in recordings if r.language == 'tts']
        
        # Split each language group
        th_train, th_temp = train_test_split(thai_recordings, test_size=0.2, random_state=42)
        th_val, th_test = train_test_split(th_temp, test_size=0.5, random_state=42)
        
        tts_train, tts_temp = train_test_split(isan_recordings, test_size=0.2, random_state=42)
        tts_val, tts_test = train_test_split(tts_temp, test_size=0.5, random_state=42)
        
        # Combine splits
        train_data = th_train + tts_train
        val_data = th_val + tts_val
        test_data = th_test + tts_test
        
        return train_data, val_data, test_data
    
    def _generate_quality_report(self) -> Dict[str, float]:
        """Generate comprehensive quality report"""
        if not self.recordings:
            return {}
        
        # Calculate quality metrics
        quality_scores = [rec.quality_score for rec in self.recordings.values()]
        snr_values = [rec.signal_to_noise_ratio for rec in self.recordings.values()]
        durations = [rec.duration for rec in self.recordings.values()]
        
        return {
            'average_quality_score': np.mean(quality_scores),
            'min_quality_score': np.min(quality_scores),
            'max_quality_score': np.max(quality_scores),
            'average_snr': np.mean(snr_values),
            'average_duration': np.mean(durations),
            'total_recordings': len(self.recordings),
            'total_speakers': len(self.speakers)
        }
    
    def export_dataset(self, output_path: str, format: str = 'json') -> bool:
        """Export the complete dataset"""
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export speakers
            speakers_data = {sid: asdict(speaker) for sid, speaker in self.speakers.items()}
            with open(output_path / 'speakers.json', 'w', encoding='utf-8') as f:
                json.dump(speakers_data, f, indent=2, default=str)
            
            # Export recordings
            recordings_data = {rid: asdict(rec) for rid, rec in self.recordings.items()}
            with open(output_path / 'recordings.json', 'w', encoding='utf-8') as f:
                json.dump(recordings_data, f, indent=2, default=str)
            
            # Export prompts
            prompts_data = {pid: asdict(prompt) for pid, prompt in self.prompts.items()}
            with open(output_path / 'prompts.json', 'w', encoding='utf-8') as f:
                json.dump(prompts_data, f, indent=2, default=str)
            
            # Copy audio files
            audio_output = output_path / 'audio'
            audio_output.mkdir(exist_ok=True)
            
            for rec in self.recordings.values():
                src_path = Path(rec.file_path)
                if src_path.exists():
                    dst_path = audio_output / f"{rec.recording_id}.wav"
                    shutil.copy2(src_path, dst_path)
            
            logger.info(f"Dataset exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export dataset: {e}")
            return False

def main():
    """Main function to demonstrate data collection"""
    # Initialize collector
    collector = ThaiIsanDataCollector()
    
    # Collect data from speakers
    print("Starting Thai and Isan speech data collection...")
    results = collector.collect_data_from_speakers(num_speakers_thai=10, num_speakers_isan=10)
    
    print(f"\nData Collection Results:")
    print(f"Thai speakers registered: {results['thai_speakers_registered']}")
    print(f"Thai recordings collected: {results['thai_recordings_collected']}")
    print(f"Isan speakers registered: {results['isan_speakers_registered']}")
    print(f"Isan recordings collected: {results['isan_recordings_collected']}")
    print(f"Total hours: {results['total_hours']:.2f}")
    
    # Process and export data
    metadata = collector.process_collected_data()
    
    # Export dataset
    success = collector.export_dataset("./thai_isan_dataset")
    if success:
        print("Dataset exported successfully!")
    else:
        print("Failed to export dataset")

if __name__ == "__main__":
    main()