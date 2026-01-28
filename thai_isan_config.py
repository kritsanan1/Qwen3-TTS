"""
Thai-Isan TTS System Configuration
Comprehensive configuration for Thai and Isan speech synthesis system
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import os

@dataclass
class DataCollectionConfig:
    """Configuration for data collection pipeline"""
    
    # Target requirements
    target_hours_thai: int = 100
    target_hours_isan: int = 100
    min_speakers_per_language: int = 50
    max_recordings_per_speaker: int = 200
    min_recordings_per_speaker: int = 50
    
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
    min_speaker_age: int = 18
    max_speaker_age: int = 65
    
    # Storage settings
    base_data_dir: str = "./data"
    backup_enabled: bool = True
    compression_format: str = "flac"
    max_file_size_mb: int = 100
    
    # Processing settings
    num_workers: int = 8
    batch_size: int = 32
    validation_split: float = 0.1
    test_split: float = 0.1
    
    # Language-specific settings
    thai_regions: List[str] = field(default_factory=lambda: ["central", "northern", "southern"])
    isan_dialects: List[str] = field(default_factory=lambda: ["northern", "central", "southern", "western"])
    
    # Recording prompts
    prompt_categories: List[str] = field(default_factory=lambda: ["greeting", "daily", "formal", "cultural", "business"])
    prompt_difficulties: List[str] = field(default_factory=lambda: ["easy", "medium", "hard"])

@dataclass
class SpeakerRecordingConfig:
    """Configuration for speaker recording interface"""
    
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
    supported_languages: List[str] = field(default_factory=lambda: ["th", "tts"])
    
    # Recording session settings
    max_session_duration: int = 120  # minutes
    break_interval: int = 30  # minutes
    max_consecutive_failures: int = 5

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    
    # Model architecture
    model_name: str = "thai-isan-tts"
    base_model_path: str = "Qwen/Qwen3-TTS"
    vocab_size: int = 151936
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    intermediate_size: int = 22016
    
    # Language-specific settings
    num_languages: int = 2
    num_thai_tones: int = 5
    num_isan_tones: int = 6
    phoneme_vocab_size: int = 128
    
    # Training parameters
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Training schedule
    num_epochs: int = 100
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 50
    
    # Loss weights
    thai_loss_weight: float = 1.0
    isan_loss_weight: float = 1.2  # Higher weight for minority language
    tone_loss_weight: float = 2.0
    phoneme_loss_weight: float = 1.5
    
    # Data augmentation
    pitch_shift_range: float = 0.2
    time_stretch_range: float = 0.1
    noise_addition_prob: float = 0.3
    volume_augmentation: bool = True
    
    # Quality targets
    target_tone_accuracy: float = 0.95
    target_phoneme_accuracy: float = 0.95
    target_mos_score: float = 4.0
    
    # Hardware settings
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Distributed training
    use_distributed: bool = False
    num_gpus: int = 1
    local_rank: int = 0
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.001

@dataclass
class QualityAssuranceConfig:
    """Configuration for quality assurance system"""
    
    # Evaluation metrics
    enable_objective_metrics: bool = True
    enable_subjective_metrics: bool = True
    enable_perceptual_metrics: bool = True
    enable_cultural_metrics: bool = True
    
    # Quality thresholds
    min_tone_accuracy: float = 0.95
    min_phoneme_accuracy: float = 0.95
    min_pesq_score: float = 3.0
    min_stoi_score: float = 0.8
    min_mos_score: float = 4.0
    
    # Cultural authenticity
    min_regional_authenticity: float = 0.8
    min_cultural_appropriateness: float = 0.8
    min_native_speaker_similarity: float = 0.75
    
    # Testing settings
    test_languages: List[str] = field(default_factory=lambda: ["th", "tts"])
    test_speakers_per_language: int = 10
    test_utterances_per_speaker: int = 20
    
    # Evaluation protocol
    enable_native_speaker_evaluation: bool = True
    enable_linguist_evaluation: bool = True
    enable_automated_testing: bool = True
    
    # Reporting
    generate_detailed_reports: bool = True
    generate_summary_reports: bool = True
    enable_wandb_logging: bool = True
    
@dataclass
class DeploymentConfig:
    """Configuration for production deployment"""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 300
    
    # Model settings
    model_path: str = "./models/thai_isan_tts"
    model_format: str = "onnx"  # or "torch", "tensorrt"
    enable_model_caching: bool = True
    max_model_cache_size: int = 2
    
    # Performance settings
    batch_inference: bool = True
    max_batch_size: int = 8
    enable_streaming: bool = True
    streaming_chunk_size: int = 1024
    
    # Quality settings
    enable_quality_control: bool = True
    quality_threshold: float = 0.8
    enable_automatic_retry: bool = True
    max_retry_attempts: int = 3
    
    # Security settings
    enable_authentication: bool = False
    api_key_required: bool = False
    rate_limiting: bool = True
    max_requests_per_minute: int = 60
    
    # Monitoring
    enable_logging: bool = True
    enable_metrics: bool = True
    enable_health_checks: bool = True
    health_check_interval: int = 30
    
    # Scaling
    enable_auto_scaling: bool = True
    min_instances: int = 1
    max_instances: int = 10
    scaling_threshold: float = 0.8

@dataclass
class PhonemeMappingConfig:
    """Configuration for phoneme mapping between Thai and Isan"""
    
    # Phoneme conversion rules
    enable_automatic_conversion: bool = True
    preserve_phonological_distinctions: bool = True
    maintain_tone_correspondence: bool = True
    
    # Thai phonemes
    thai_consonants: List[str] = field(default_factory=lambda: [
        "p", "t", "k", "ʔ", "b", "d", "m", "n", "ŋ", "ŋ", "r", "l", "j", "w", "s", "h", "f", "tɕ", "tɕʰ", "ph", "th", "kh"
    ])
    thai_vowels: List[str] = field(default_factory=lambda: [
        "a", "i", "u", "e", "ɛ", "o", "ɔ", "ɯ", "ɤ", "ə"
    ])
    thai_tones: List[str] = field(default_factory=lambda: ["mid", "low", "falling", "high", "rising"])
    
    # Isan phonemes
    isan_consonants: List[str] = field(default_factory=lambda: [
        "p", "t", "k", "ʔ", "b", "d", "m", "n", "ŋ", "r", "l", "j", "w", "s", "h", "f", "ɲ"
    ])
    isan_vowels: List[str] = field(default_factory=lambda: [
        "a", "i", "u", "e", "ɛ", "o", "ɔ", "ɨ", "ə"
    ])
    isan_tones: List[str] = field(default_factory=lambda: ["mid", "low", "falling", "high", "rising", "high_falling"])
    
    # Conversion mappings
    thai_to_isan_consonants: Dict[str, str] = field(default_factory=lambda: {
        "tɕʰ": "s",      # ช้าง → ส้าง
        "r": ["h", "l"],  # ร้อน → ฮ้อน
        "ɯ": "ɨ",        # Centralization
        "ɤ": "ə",        # Centralization
    })

@dataclass
class ThaiIsanTTSConfig:
    """Main configuration class for Thai-Isan TTS system"""
    
    # Sub-configurations
    data_collection: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    speaker_recording: SpeakerRecordingConfig = field(default_factory=SpeakerRecordingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quality_assurance: QualityAssuranceConfig = field(default_factory=QualityAssuranceConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    phoneme_mapping: PhonemeMappingConfig = field(default_factory=PhonemeMappingConfig)
    
    # System settings
    system_name: str = "Thai-Isan TTS System"
    version: str = "1.0.0"
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Paths
    base_dir: str = "./thai_isan_tts"
    data_dir: str = "./data"
    models_dir: str = "./models"
    logs_dir: str = "./logs"
    output_dir: str = "./output"
    
    # Language settings
    supported_languages: List[str] = field(default_factory=lambda: ["th", "tts"])
    default_language: str = "th"
    enable_auto_detection: bool = True
    
    # Performance settings
    enable_gpu_acceleration: bool = True
    enable_cpu_optimization: bool = True
    enable_memory_optimization: bool = True
    max_memory_usage_gb: int = 8
    
    # Quality settings
    target_tone_accuracy: float = 0.95
    target_phoneme_accuracy: float = 0.95
    target_mos_score: float = 4.0
    target_processing_speed: float = 0.2  # seconds per second of audio

# Default configuration instance
DEFAULT_CONFIG = ThaiIsanTTSConfig()

def load_config(config_path: str = None) -> ThaiIsanTTSConfig:
    """Load configuration from file or create default"""
    if config_path and os.path.exists(config_path):
        # Load from YAML/JSON file
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        # Convert to dataclass (simplified version)
        return ThaiIsanTTSConfig(**config_dict)
    return DEFAULT_CONFIG

def save_config(config: ThaiIsanTTSConfig, config_path: str):
    """Save configuration to file"""
    import yaml
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Convert dataclass to dict
    config_dict = {}
    for key, value in config.__dict__.items():
        if hasattr(value, '__dict__'):
            config_dict[key] = value.__dict__
        else:
            config_dict[key] = value
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)