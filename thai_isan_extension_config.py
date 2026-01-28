"""
Thai-Isan Extension Configuration for Qwen3-TTS
Integration with existing Qwen3-TTS architecture
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import json
import os
from pathlib import Path

from transformers.configuration_utils import PretrainedConfig


@dataclass
class ThaiIsanExtensionConfig:
    """Configuration for Thai-Isan language extension to Qwen3-TTS"""
    
    # Language settings
    supported_languages: List[str] = field(default_factory=lambda: ["th", "tts"])
    default_language: str = "th"
    
    # Phoneme inventory
    thai_phonemes: List[str] = field(default_factory=lambda: [
        # Consonants
        'p', 't', 'k', 'ʔ', 'b', 'd', 'f', 's', 'h', 'v',
        'm', 'n', 'ŋ', 'j', 'w', 'l', 'r',
        'pʰ', 'tʰ', 'kʰ', 'tɕ', 'tɕʰ',
        # Vowels
        'a', 'i', 'u', 'e', 'ɛ', 'o', 'ɔ', 'ɯ', 'ɤ',
        'aː', 'iː', 'uː', 'eː', 'ɛː', 'oː', 'ɔː', 'ɯː', 'ɤː',
        # Diphthongs
        'ia', 'ua', 'ɯa', 'ai', 'au', 'oi', 'ɔi', 'ɛu', 'iu'
    ])
    
    isan_phonemes: List[str] = field(default_factory=lambda: [
        # Consonants (with Isan-specific substitutions)
        'p', 't', 'k', 'ʔ', 'b', 'd', 'f', 's', 'h', 'v', 'ʋ',
        'm', 'n', 'ŋ', 'j', 'ɲ', 'w', 'l', 'h',  # /r/ → /h/ or /l/
        'pʰ', 'tʰ', 'kʰ', 'tɕ', 's',  # /tɕʰ/ → /s/
        # Vowels (with centralization)
        'a', 'i', 'u', 'e', 'ɛ', 'o', 'ɔ', 'ɨ', 'ə',
        'aː', 'iː', 'uː', 'eː', 'ɛː', 'oː', 'ɔː', 'ɨː', 'əː',
        # Diphthongs
        'ia', 'ua', 'ɨa', 'ai', 'au', 'oi', 'ɔi', 'ɛu', 'iu'
    ])
    
    # Tone systems
    thai_tones: List[str] = field(default_factory=lambda: [
        "mid", "low", "falling", "high", "rising"
    ])
    
    isan_tones: List[str] = field(default_factory=lambda: [
        "mid", "low", "falling", "high", "rising", "high_falling"
    ])
    
    # Regional Isan variants
    isan_dialects: List[str] = field(default_factory=lambda: [
        "northern",    # Loei, Nong Khai (influenced by Luang Prabang)
        "central",     # Khon Kaen, Kalasin (standard Isan)
        "southern",    # Ubon, Sisaket (influenced by Pakse)
        "western"      # Nakhon Phanom, Sakon Nakhon (close to Lao)
    ])
    
    # Phoneme mapping rules
    thai_to_isan_phoneme_mapping: Dict[str, Union[str, List[str]]] = field(default_factory=lambda: {
        # Consonant substitutions
        "tɕʰ": "s",      # Thai ช → Isan s
        "ʃ": "s",        # Allophone
        "r": ["h", "l"],  # Thai r → Isan h/l
        "j": "j",        # Context-dependent (see below)
        "ɯ": "ɨ",        # Centralization
        "ɤ": "ə",        # Centralization
    })
    
    # Context-dependent phoneme rules
    context_dependent_rules: Dict[str, Dict] = field(default_factory=lambda: {
        "j_to_ɲ": {
            "condition": "etymological_ɲ",
            "input": "j",
            "output": "ɲ",
            "examples": ["ยาก" -> "หญาก"]
        },
        "w_to_ʋ": {
            "condition": "vientiane_dialect",
            "input": "w",
            "output": "ʋ",
            "probability": 0.7
        }
    })
    
    # Tone mapping between Thai and Isan
    thai_to_isan_tone_mapping: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "vientiane_dialect": {
            "high_class": {
                "live_syllable": "high_falling",
                "dead_short": "mid",
                "dead_long": "mid",
                "with_mai_ek": "mid",
                "with_mai_tho": "high_falling"
            },
            "mid_class": {
                "live_syllable": "mid",
                "dead_short": "low",
                "dead_long": "low",
                "with_mai_ek": "low",
                "with_mai_tho": "high_falling"
            },
            "low_class": {
                "live_syllable": "rising",
                "dead_short": "high_falling",
                "dead_long": "high_falling",
                "with_mai_ek": "high_falling",
                "with_mai_tho": "rising"
            }
        }
    })
    
    # Vocabulary differences
    thai_to_isan_vocabulary: Dict[str, str] = field(default_factory=lambda: {
        # Pronouns
        "เรา": "เฮา",      # we/us
        "คุณ": "เจ้า",     # you (formal)
        "ฉัน": "ข้า",      # I (female)
        "ผม": "ข้า",       # I (male)
        
        # Negation
        "ไม่": "บ่",       # not
        "ไม่ได้": "บ่ได้",   # cannot
        
        # Common words
        "กิน": "แซบ",      # eat
        "สวย": "งาม",      # beautiful
        "ไป": "ไป๋",       # go
        "มา": "มา๋",       # come
        
        # Final particles
        "ครับ": "เด้อ",    # male polite
        "ค่ะ": "เด้อ",     # female polite
    })
    
    # Training configuration
    training_config: Dict = field(default_factory=lambda: {
        "batch_size": 32,
        "learning_rate": 2e-5,
        "num_epochs": 100,
        "warmup_steps": 1000,
        "gradient_accumulation_steps": 4,
        "loss_weights": {
            "thai": 1.0,
            "isan": 1.2,  # Higher for minority language
            "tone": 2.0   # Emphasize tone accuracy
        },
        "data_ratios": {
            "thai": 0.6,
            "isan": 0.4
        }
    })
    
    # Audio processing
    audio_config: Dict = field(default_factory=lambda: {
        "sample_rate": 24000,
        "hop_length": 300,  # 12Hz tokenizer
        "win_length": 1200,
        "n_mels": 128,
        "n_fft": 2048,
        "mel_fmin": 0,
        "mel_fmax": 12000
    })
    
    # Evaluation settings
    evaluation_config: Dict = field(default_factory=lambda: {
        "tone_accuracy_threshold": 0.95,
        "phoneme_error_rate_threshold": 0.05,
        "mos_target": 4.0,
        "subjective_evaluation": True,
        "native_speaker_validation": True,
        "regional_accuracy_check": True
    })


class Qwen3ThaiIsanConfig(PretrainedConfig):
    """HuggingFace-compatible config for Thai-Isan extension"""
    
    model_type = "qwen3_thai_isan"
    
    def __init__(
        self,
        thai_isan_config: Optional[ThaiIsanExtensionConfig] = None,
        vocab_size: int = 151936,  # Extended from base Qwen3-TTS
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 32,
        intermediate_size: int = 22016,
        max_position_embeddings: int = 32768,
        **kwargs
    ):
        # Initialize base config
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            **kwargs
        )
        
        # Thai-Isan specific configuration
        self.thai_isan_config = thai_isan_config or ThaiIsanExtensionConfig()
        
        # Language-specific parameters
        self.num_languages = len(self.thai_isan_config.supported_languages)
        self.num_thai_tones = self.thai_isan_config.thai_tones
        self.num_isan_tones = self.thai_isan_config.isan_tones
        self.isan_dialects = self.thai_isan_config.isan_dialects
        
        # Training configuration
        self.training = self.thai_isan_config.training_config
        self.audio = self.thai_isan_config.audio_config
        self.evaluation = self.thai_isan_config.evaluation_config


def get_phoneme_inventory(language: str = "th") -> List[str]:
    """Get phoneme inventory for specified language"""
    config = ThaiIsanExtensionConfig()
    
    if language == "th":
        return config.thai_phonemes
    elif language == "tts":
        return config.isan_phonemes
    else:
        raise ValueError(f"Unsupported language: {language}")


def get_tone_system(language: str = "th") -> List[str]:
    """Get tone system for specified language"""
    config = ThaiIsanExtensionConfig()
    
    if language == "th":
        return config.thai_tones
    elif language == "tts":
        return config.isan_tones
    else:
        raise ValueError(f"Unsupported language: {language}")


def convert_thai_to_isan_tone(
    thai_tone: str,
    consonant_class: str,
    syllable_type: str,
    dialect: str = "vientiane_dialect"
) -> str:
    """Convert Thai tone to Isan equivalent"""
    config = ThaiIsanExtensionConfig()
    
    mapping = config.thai_to_isan_tone_mapping.get(dialect, {})
    class_mapping = mapping.get(consonant_class, {})
    
    return class_mapping.get(syllable_type, thai_tone)  # Default to original


def get_vocabulary_difference(thai_word: str) -> str:
    """Get Isan equivalent for Thai word"""
    config = ThaiIsanExtensionConfig()
    
    return config.thai_to_isan_vocabulary.get(thai_word, thai_word)


def is_supported_language(language_code: str) -> bool:
    """Check if language is supported"""
    config = ThaiIsanExtensionConfig()
    return language_code in config.supported_languages


def get_training_config() -> Dict:
    """Get training configuration"""
    config = ThaiIsanExtensionConfig()
    return config.training_config


def get_audio_config() -> Dict:
    """Get audio processing configuration"""
    config = ThaiIsanExtensionConfig()
    return config.audio_config


def get_evaluation_config() -> Dict:
    """Get evaluation configuration"""
    config = ThaiIsanExtensionConfig()
    return config.evaluation_config


# Example usage
if __name__ == "__main__":
    # Test configuration
    config = ThaiIsanExtensionConfig()
    
    print("Supported languages:", config.supported_languages)
    print("Thai tones:", config.thai_tones)
    print("Isan tones:", config.isan_tones)
    print("Isan dialects:", config.isan_dialects)
    
    # Test phoneme inventory
    thai_phonemes = get_phoneme_inventory("th")
    isan_phonemes = get_phoneme_inventory("tts")
    
    print(f"Thai phonemes: {len(thai_phonemes)}")
    print(f"Isan phonemes: {len(isan_phonemes)}")
    
    # Test tone conversion
    isan_tone = convert_thai_to_isan_tone("mid", "high_class", "live_syllable")
    print(f"Thai mid tone → Isan: {isan_tone}")
    
    # Test vocabulary conversion
    isan_word = get_vocabulary_difference("เรา")
    print(f"Thai 'เรา' → Isan: {isan_word}")
    
    # Test HuggingFace config
    hf_config = Qwen3ThaiIsanConfig()
    print(f"HF Config vocab size: {hf_config.vocab_size}")
    print(f"Number of languages: {hf_config.num_languages}")
    
    # Test training config
    train_config = get_training_config()
    print(f"Batch size: {train_config['batch_size']}")
    print(f"Learning rate: {train_config['learning_rate']}")
    
    # Test evaluation config
    eval_config = get_evaluation_config()
    print(f"Tone accuracy threshold: {eval_config['tone_accuracy_threshold']}")
    print(f"MOS target: {eval_config['mos_target']}")