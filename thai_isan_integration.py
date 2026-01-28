"""
Thai-Isan TTS Integration Module
Main integration point for Thai-Isan TTS functionality
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import json
import os
from pathlib import Path
import logging

# Import Thai-Isan modules
from thai_isan_extension_config import Qwen3ThaiIsanConfig, ThaiIsanExtensionConfig
from thai_isan_phoneme_mapping import (
    thai_to_isan_phoneme_conversion,
    thai_to_isan_tone_conversion,
    thai_to_isan_word_conversion,
    THAI_TONE_CLASSES,
    ISAN_TONE_RULES
)
from thai_isan_training_pipeline import ThaiIsanTTSModel, ThaiIsanDataset
from thai_isan_evaluation import ThaiIsanTTSEvaluator

logger = logging.getLogger(__name__)


class Qwen3ThaiIsanTTS:
    """
    Main Thai-Isan TTS model extending Qwen3-TTS
    
    This class provides a unified interface for Thai and Isan speech synthesis
    with automatic language detection, phoneme mapping, and tone conversion.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Qwen3ThaiIsanConfig] = None,
        device: str = "auto"
    ):
        """
        Initialize Thai-Isan TTS model
        
        Args:
            model_path: Path to pre-trained model
            config: Model configuration
            device: Computing device ('auto', 'cpu', 'cuda')
        """
        self.config = config or Qwen3ThaiIsanConfig()
        self.device = self._get_device(device)
        
        # Initialize base TTS model
        self.base_model = self._load_base_model(model_path)
        
        # Language-specific components
        self.language_detector = LanguageDetector(self.config)
        self.phoneme_mapper = PhonemeMapper(self.config)
        self.tone_converter = ToneConverter(self.config)
        
        # Evaluation
        self.evaluator = ThaiIsanTTSEvaluator(self.config.thai_isan_config)
        
        logger.info(f"Thai-Isan TTS model initialized on {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Get computing device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_base_model(self, model_path: Optional[str]) -> nn.Module:
        """Load base Qwen3-TTS model"""
        if model_path and os.path.exists(model_path):
            # Load pre-trained model
            # This would load the actual Qwen3-TTS model
            model = ThaiIsanTTSModel(self.config)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            return model
        else:
            # Initialize new model
            model = ThaiIsanTTSModel(self.config)
            model.to(self.device)
            return model
    
    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        speaker: str = "default",
        prosody: Optional[Dict] = None,
        return_evaluation: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Synthesize speech from text
        
        Args:
            text: Input text
            language: Language code ('th', 'tts', or None for auto-detection)
            speaker: Speaker ID
            prosody: Prosody control parameters
            return_evaluation: Whether to return evaluation metrics
        
        Returns:
            Synthesized audio (and evaluation metrics if requested)
        """
        
        # Auto-detect language if not specified
        if language is None:
            language = self.language_detector.detect_language(text)
        
        # Validate language
        if language not in self.config.thai_isan_config.supported_languages:
            raise ValueError(f"Unsupported language: {language}. Supported: {self.config.thai_isan_config.supported_languages}")
        
        # Language-specific preprocessing
        if language == "tts":
            # Isan synthesis
            processed_text = self._preprocess_isan_text(text)
            phonemes = self.phoneme_mapper.map_thai_to_isan_phonemes(processed_text)
            tones = self.tone_converter.convert_thai_to_isan_tones(processed_text)
        else:
            # Thai synthesis
            processed_text = self._preprocess_thai_text(text)
            phonemes = self.phoneme_mapper.map_thai_phonemes(processed_text)
            tones = self.tone_converter.get_thai_tones(processed_text)
        
        # Apply prosody modifications
        if prosody:
            phonemes, tones = self._apply_prosody(phonemes, tones, prosody)
        
        # Synthesize speech
        with torch.no_grad():
            audio = self.base_model.synthesize(
                text=processed_text,
                phonemes=phonemes,
                tones=tones,
                speaker=speaker,
                language=language
            )
        
        # Post-processing
        audio = self._postprocess_audio(audio, prosody)
        
        # Return evaluation if requested
        if return_evaluation:
            evaluation = self.evaluator.evaluate(
                synthesized_speech=audio,
                reference_speech=None,
                text=text,
                language=language
            )
            return audio, evaluation
        
        return audio
    
    def _preprocess_isan_text(self, text: str) -> str:
        """Preprocess Isan text"""
        # Convert Thai vocabulary to Isan equivalents
        isan_text = thai_to_isan_word_conversion(text)
        
        # Handle regional variations
        # This could include dialect-specific processing
        
        return isan_text
    
    def _preprocess_thai_text(self, text: str) -> str:
        """Preprocess Thai text"""
        # Standard Thai text preprocessing
        # This could include text normalization, number handling, etc.
        
        return text
    
    def _apply_prosody(self, phonemes: List, tones: List, prosody: Dict) -> Tuple[List, List]:
        """Apply prosody modifications"""
        # Modify phonemes and tones based on prosody parameters
        if "speed" in prosody:
            # Adjust duration
            speed_factor = prosody["speed"]
            phonemes = self._adjust_duration(phonemes, speed_factor)
            tones = self._adjust_duration(tones, speed_factor)
        
        if "pitch" in prosody:
            # Adjust pitch
            pitch_shift = prosody["pitch"]
            tones = self._adjust_pitch(tones, pitch_shift)
        
        if "emotion" in prosody:
            # Apply emotional prosody
            emotion = prosody["emotion"]
            phonemes, tones = self._apply_emotion(phonemes, tones, emotion)
        
        return phonemes, tones
    
    def _postprocess_audio(self, audio: np.ndarray, prosody: Optional[Dict]) -> np.ndarray:
        """Post-process synthesized audio"""
        # Apply audio effects based on prosody
        if prosody and "volume" in prosody:
            audio = audio * prosody["volume"]
        
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return audio
    
    def _adjust_duration(self, sequence: List, factor: float) -> List:
        """Adjust sequence duration"""
        # Simple duration adjustment (real implementation would be more sophisticated)
        if factor > 1.0:
            # Speed up - remove elements
            new_length = int(len(sequence) / factor)
            indices = np.linspace(0, len(sequence) - 1, new_length, dtype=int)
            return [sequence[i] for i in indices]
        else:
            # Slow down - repeat elements
            new_length = int(len(sequence) * (1.0 / factor))
            indices = np.linspace(0, len(sequence) - 1, new_length, dtype=int)
            return [sequence[i] for i in indices]
    
    def _adjust_pitch(self, tones: List, shift: float) -> List:
        """Adjust pitch of tones"""
        # Apply pitch shift to tone sequence
        # This would modify the tonal patterns
        return tones
    
    def _apply_emotion(self, phonemes: List, tones: List, emotion: str) -> Tuple[List, List]:
        """Apply emotional prosody"""
        # Modify phonemes and tones based on emotion
        emotion_mappings = {
            "happy": {"pitch_range": 1.2, "speed": 1.1, "energy": 1.3},
            "sad": {"pitch_range": 0.8, "speed": 0.9, "energy": 0.7},
            "excited": {"pitch_range": 1.4, "speed": 1.2, "energy": 1.5},
            "neutral": {"pitch_range": 1.0, "speed": 1.0, "energy": 1.0}
        }
        
        if emotion in emotion_mappings:
            mapping = emotion_mappings[emotion]
            # Apply emotion-specific modifications
            # This is simplified - real implementation would be more sophisticated
            
        return phonemes, tones
    
    def get_available_speakers(self, language: str = "th") -> List[str]:
        """Get available speakers for a language"""
        # This would return actual speaker IDs from the model
        if language == "th":
            return ["th_male_central", "th_female_central", "th_male_northern", "th_female_southern"]
        elif language == "tts":
            return ["tts_male_northeastern", "tts_female_northeastern", "tts_male_vientiane", "tts_female_luang_prabang"]
        else:
            return ["default"]
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return self.config.thai_isan_config.supported_languages
    
    def evaluate_text(self, text: str, language: str) -> Dict[str, float]:
        """Evaluate text characteristics"""
        
        # Language-specific text evaluation
        if language == "tts":
            # Check for Isan-specific vocabulary
            isan_words = 0
            total_words = len(text.split())
            
            for word in text.split():
                if word in self.config.thai_isan_config.thai_to_isan_vocabulary.values():
                    isan_words += 1
            
            vocab_score = isan_words / total_words if total_words > 0 else 0.0
        else:
            vocab_score = 1.0
        
        # Text complexity
        avg_word_length = np.mean([len(word) for word in text.split()]) if text.split() else 0
        sentence_count = text.count('。') + text.count('．') + text.count('.') + 1
        
        # Tone complexity
        tone_marks = sum(text.count(mark) for mark in ['่', '้', '๊', '๋'])
        
        return {
            "vocabulary_score": vocab_score,
            "avg_word_length": avg_word_length,
            "sentence_count": sentence_count,
            "tone_markers": tone_marks,
            "character_count": len(text)
        }


class LanguageDetector:
    """Language detection for Thai and Isan text"""
    
    def __init__(self, config: Qwen3ThaiIsanConfig):
        self.config = config
        self.isan_vocabulary = set(config.thai_isan_config.thai_to_isan_vocabulary.values())
    
    def detect_language(self, text: str) -> str:
        """Detect whether text is Thai or Isan"""
        
        # Count Isan-specific vocabulary
        isan_words = 0
        total_words = len(text.split())
        
        for word in text.split():
            if word in self.isan_vocabulary:
                isan_words += 1
        
        # Decision based on vocabulary ratio
        if total_words > 0:
            isan_ratio = isan_words / total_words
            if isan_ratio > 0.3:  # Threshold for Isan detection
                return "tts"
        
        # Default to Thai
        return "th"
    
    def get_language_confidence(self, text: str) -> float:
        """Get confidence score for language detection"""
        detected_lang = self.detect_language(text)
        
        # Simple confidence based on vocabulary match
        if detected_lang == "tts":
            isan_words = sum(1 for word in text.split() if word in self.isan_vocabulary)
            total_words = len(text.split())
            confidence = min(1.0, isan_words / total_words) if total_words > 0 else 0.0
        else:
            confidence = 0.8  # Default confidence for Thai
        
        return confidence


class PhonemeMapper:
    """Phoneme mapping between Thai and Isan"""
    
    def __init__(self, config: Qwen3ThaiIsanConfig):
        self.config = config
        self.mapping_rules = config.thai_isan_config.thai_to_isan_phoneme_mapping
    
    def map_thai_to_isan_phonemes(self, text: str) -> List[str]:
        """Map Thai phonemes to Isan equivalents"""
        
        # This would involve sophisticated phoneme extraction and mapping
        # Placeholder implementation
        
        # Extract phonemes from text (simplified)
        phonemes = self._extract_phonemes(text)
        
        # Apply mapping rules
        isan_phonemes = []
        for phoneme in phonemes:
            if phoneme in self.mapping_rules:
                mapped = self.mapping_rules[phoneme]
                if isinstance(mapped, list):
                    # Context-dependent selection
                    isan_phonemes.append(mapped[0])  # Simplified
                else:
                    isan_phonemes.append(mapped)
            else:
                isan_phonemes.append(phoneme)
        
        return isan_phonemes
    
    def map_thai_phonemes(self, text: str) -> List[str]:
        """Map Thai text to Thai phonemes"""
        # Extract Thai phonemes
        return self._extract_phonemes(text)
    
    def _extract_phonemes(self, text: str) -> List[str]:
        """Extract phonemes from text (simplified)"""
        # This would use a proper grapheme-to-phoneme converter
        # Placeholder implementation
        
        phonemes = []
        for char in text:
            if char.strip():
                # Map character to phoneme (simplified)
                phoneme = self._character_to_phoneme(char)
                if phoneme:
                    phonemes.append(phoneme)
        
        return phonemes
    
    def _character_to_phoneme(self, char: str) -> Optional[str]:
        """Convert character to phoneme (simplified)"""
        # This would use a proper G2P model
        # Placeholder mapping
        
        char_phoneme_map = {
            'ก': 'k', 'ข': 'kʰ', 'ค': 'kʰ', 'ง': 'ŋ',
            'จ': 'tɕ', 'ช': 'tɕʰ', 'ซ': 's', 'ญ': 'j',
            'ด': 'd', 'ต': 't', 'ถ': 'tʰ', 'ท': 'tʰ',
            'น': 'n', 'บ': 'b', 'ป': 'p', 'ผ': 'pʰ',
            'พ': 'pʰ', 'ฟ': 'f', 'ม': 'm', 'ย': 'j',
            'ร': 'r', 'ล': 'l', 'ว': 'w', 'ส': 's',
            'ห': 'h', 'อ': 'ʔ'
        }
        
        return char_phoneme_map.get(char, char)


class ToneConverter:
    """Tone conversion between Thai and Isan"""
    
    def __init__(self, config: Qwen3ThaiIsanConfig):
        self.config = config
        self.thai_tones = config.thai_tones
        self.isan_tones = config.isan_tones
        self.tone_mapping = config.thai_isan_config.thai_to_isan_tone_mapping
    
    def convert_thai_to_isan_tones(self, text: str, dialect: str = "vientiane_dialect") -> List[str]:
        """Convert Thai tones to Isan equivalents"""
        
        # Parse Thai tones from text
        thai_tones = self.get_thai_tones(text)
        
        # Convert to Isan tones
        isan_tones = []
        for tone_info in thai_tones:
            thai_tone = tone_info["tone"]
            consonant_class = tone_info["consonant_class"]
            syllable_type = tone_info["syllable_type"]
            
            isan_tone = thai_to_isan_tone_conversion(
                thai_tone, consonant_class, syllable_type, dialect
            )
            isan_tones.append(isan_tone)
        
        return isan_tones
    
    def get_thai_tones(self, text: str) -> List[Dict]:
        """Extract Thai tones from text"""
        
        tones = []
        
        # Parse syllables and extract tone information
        syllables = self._parse_syllables(text)
        
        for syllable in syllables:
            tone_info = self._analyze_syllable_tone(syllable)
            tones.append(tone_info)
        
        return tones
    
    def _parse_syllables(self, text: str) -> List[str]:
        """Parse text into syllables (simplified)"""
        # This would use a proper Thai syllable parser
        # Placeholder implementation
        
        # Split by spaces and some punctuation
        import re
        syllables = re.findall(r'[ก-ฮะาำิีึืุูเแโใไๆ็่้๊๋์]+', text)
        
        return syllables
    
    def _analyze_syllable_tone(self, syllable: str) -> Dict:
        """Analyze tone information for a syllable"""
        
        # Get consonant class
        consonant_class = self._get_consonant_class(syllable)
        
        # Get tone mark
        tone_mark = self._get_tone_mark(syllable)
        
        # Determine syllable type
        syllable_type = self._get_syllable_type(syllable)
        
        # Determine tone based on rules
        if tone_mark:
            tone = self._get_tone_from_mark(tone_mark, consonant_class)
        else:
            tone = self._get_tone_from_rules(consonant_class, syllable_type)
        
        return {
            "tone": tone,
            "consonant_class": consonant_class,
            "syllable_type": syllable_type,
            "tone_mark": tone_mark
        }
    
    def _get_consonant_class(self, syllable: str) -> str:
        """Get consonant class for syllable"""
        # Get initial consonant
        initial_consonant = self._get_initial_consonant(syllable)
        
        # Find consonant class
        for class_name, consonants in THAI_TONE_CLASSES.items():
            if initial_consonant in consonants:
                return class_name
        
        return "mid_class"  # Default
    
    def _get_initial_consonant(self, syllable: str) -> str:
        """Get initial consonant of syllable"""
        # This would use proper Thai syllable parsing
        # Simplified implementation
        for char in syllable:
            if char in [c for consonants in THAI_TONE_CLASSES.values() for c in consonants]:
                return char
        return "ก"  # Default
    
    def _get_tone_mark(self, syllable: str) -> Optional[str]:
        """Get tone mark from syllable"""
        for mark in ['่', '้', '๊', '๋']:
            if mark in syllable:
                return mark
        return None
    
    def _get_syllable_type(self, syllable: str) -> str:
        """Determine syllable type (live/dead, short/long)"""
        # This would use proper Thai syllable analysis
        # Simplified implementation
        
        # Check for final consonant
        has_final = any(char in 'กขคงจชซญฎดตถทธนบปผพฟภมยรลวศษสหฬฮ' for char in syllable[1:])
        
        # Check vowel length (simplified)
        long_vowels = ['า', 'ี', 'ื', 'ู', 'เ', 'แ', 'โ', 'ใ', 'ไ']
        has_long_vowel = any(vowel in syllable for vowel in long_vowels)
        
        if has_final:
            return "dead_short" if not has_long_vowel else "dead_long"
        else:
            return "live" if has_long_vowel else "live_short"
    
    def _get_tone_from_mark(self, tone_mark: str, consonant_class: str) -> str:
        """Get tone from tone mark and consonant class"""
        # This would use proper Thai tone rules
        # Simplified implementation
        
        tone_mark_map = {
            '่': "low",
            '้': "falling",
            '๊': "high",
            '๋': "rising"
        }
        
        return tone_mark_map.get(tone_mark, "mid")
    
    def _get_tone_from_rules(self, consonant_class: str, syllable_type: str) -> str:
        """Get tone from consonant class and syllable type"""
        # Use Thai tone rules
        rules = THAI_TONE_RULES.get(consonant_class, {})
        return rules.get(syllable_type, "mid")


def create_thai_isan_tts(
    model_path: Optional[str] = None,
    device: str = "auto",
    **kwargs
) -> Qwen3ThaiIsanTTS:
    """
    Create Thai-Isan TTS model
    
    Args:
        model_path: Path to pre-trained model
        device: Computing device
        **kwargs: Additional arguments
    
    Returns:
        Thai-Isan TTS model instance
    """
    return Qwen3ThaiIsanTTS(model_path=model_path, device=device, **kwargs)


def train_thai_isan_tts(
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    **kwargs
) -> str:
    """
    Train Thai-Isan TTS model
    
    Args:
        train_data_path: Training data directory
        val_data_path: Validation data directory
        output_dir: Output directory for trained model
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        **kwargs: Additional training arguments
    
    Returns:
        Path to trained model
    """
    
    from thai_isan_training_pipeline import ThaiIsanTrainer, ThaiIsanDataset, create_training_args
    from transformers import TrainingArguments
    
    # Create datasets
    train_dataset = ThaiIsanDataset(train_data_path, tokenizer=None, language="both")
    val_dataset = ThaiIsanDataset(val_data_path, tokenizer=None, language="both")
    
    # Create model
    config = Qwen3ThaiIsanConfig()
    model = ThaiIsanTTSModel(config)
    
    # Create trainer
    trainer = ThaiIsanTrainer(config)
    
    # Training arguments
    training_args = create_training_args(config)
    training_args.num_train_epochs = num_epochs
    training_args.per_device_train_batch_size = batch_size
    training_args.per_device_eval_batch_size = batch_size
    training_args.learning_rate = learning_rate
    training_args.output_dir = output_dir
    
    # Train model
    # This would implement the actual training loop
    logger.info(f"Training Thai-Isan TTS model for {num_epochs} epochs")
    logger.info(f"Output directory: {output_dir}")
    
    # Placeholder for training implementation
    model_path = os.path.join(output_dir, "thai_isan_tts_model.pth")
    
    return model_path


# Example usage and testing
if __name__ == "__main__":
    # Create model
    print("Creating Thai-Isan TTS model...")
    model = create_thai_isan_tts()
    
    # Test synthesis
    print("\nTesting Thai synthesis...")
    thai_text = "สวัสดีครับ น้ำเย็นไหม"
    thai_audio = model.synthesize(thai_text, language="th")
    print(f"Thai text: {thai_text}")
    print(f"Audio shape: {thai_audio.shape}")
    print(f"Audio duration: {len(thai_audio) / 24000:.2f} seconds")
    
    print("\nTesting Isan synthesis...")
    isan_text = "สบายดีบ่ เฮาไปกินข้าว"
    isan_audio = model.synthesize(isan_text, language="tts")
    print(f"Isan text: {isan_text}")
    print(f"Audio shape: {isan_audio.shape}")
    print(f"Audio duration: {len(isan_audio) / 24000:.2f} seconds")
    
    # Test evaluation
    print("\nEvaluating synthesis...")
    audio, evaluation = model.synthesize(thai_text, language="th", return_evaluation=True)
    print(f"Tone accuracy: {evaluation.language_specific_metrics.get('tone_accuracy', 0):.3f}")
    print(f"Naturalness: {evaluation.subjective_scores.get('naturalness', 0):.3f}")
    
    # Test language detection
    print("\nTesting language detection...")
    test_texts = [
        ("สวัสดีครับ", "th"),
        ("สบายดีบ่", "tts"),
        ("เฮาไปกินข้าว", "tts"),
        ("น้ำเย็นไหม", "th")
    ]
    
    for text, expected_lang in test_texts:
        detected_lang = model.language_detector.detect_language(text)
        confidence = model.language_detector.get_language_confidence(text)
        print(f"Text: '{text}' -> Detected: {detected_lang} (confidence: {confidence:.3f}), Expected: {expected_lang}")
    
    print("\nThai-Isan TTS demonstration completed!")


# Export main classes
__all__ = [
    'Qwen3ThaiIsanTTS',
    'ThaiIsanExtensionConfig',
    'Qwen3ThaiIsanConfig',
    'create_thai_isan_tts',
    'train_thai_isan_tts'
]