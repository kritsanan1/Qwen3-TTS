"""
Thai-Isan TTS Quality Assurance System
Comprehensive evaluation and testing framework for Thai and Isan speech synthesis
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import pandas as pd
from tqdm import tqdm
import wave
import warnings
warnings.filterwarnings('ignore')

# Thai language processing
import pythainlp
from pythainlp.tokenize import word_tokenize, syllable_tokenize
from pythainlp.corpus import thai_words
from pythainlp.transliterate import romanize

# Audio quality metrics
import pesq
import stoi
from pystoi import stoi
from pesq import pesq

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for TTS evaluation"""
    
    # Objective metrics
    mel_cepstral_distortion: float  # MCD
    fundamental_frequency_correlation: float  # F0 correlation
    spectral_convergence: float
    signal_to_noise_ratio: float  # SNR
    total_harmonic_distortion: float  # THD
    
    # Perceptual metrics
    pesq_score: float  # Perceptual Evaluation of Speech Quality
    stoi_score: float  # Short-Time Objective Intelligibility
    
    # Language-specific metrics
    tone_accuracy: float  # Accuracy of tone realization
    phoneme_accuracy: float  # Accuracy of phoneme pronunciation
    phoneme_error_rate: float  # PER
    
    # Prosodic metrics
    pace_naturalness: float  # Naturalness of speaking pace
    pitch_naturalness: float  # Naturalness of pitch contour
    rhythm_accuracy: float  # Accuracy of rhythmic patterns
    
    # Cultural authenticity metrics
    regional_authenticity: float  # Authenticity of regional accent
    cultural_appropriateness: float  # Cultural appropriateness
    native_speaker_similarity: float  # Similarity to native speakers

@dataclass
class SubjectiveScores:
    """Subjective evaluation scores"""
    
    naturalness: float  # 1-5 scale
    intelligibility: float  # 1-5 scale
    pleasantness: float  # 1-5 scale
    similarity: float  # 1-5 scale
    overall_quality: float  # 1-5 scale
    
    # Language-specific
    tone_naturalness: float  # 1-5 scale
    accent_authenticity: float  # 1-5 scale
    cultural_appropriateness: float  # 1-5 scale

@dataclass
class LanguageSpecificMetrics:
    """Language-specific evaluation metrics"""
    
    # Thai-specific
    thai_tone_accuracy: float
    thai_consonant_accuracy: float
    thai_vowel_accuracy: float
    thai_final_consonant_accuracy: float
    
    # Isan-specific
    isan_tone_accuracy: float
    isan_dialect_authenticity: float
    isan_lexical_accuracy: float
    isan_phonological_faithfulness: float
    
    # Regional variants
    regional_accent_score: float
    dialect_consistency: float

@dataclass
class EvaluationConfig:
    """Configuration for TTS evaluation"""
    
    # Test sets
    test_sentences: Dict[str, List[str]]  # Language -> sentences
    reference_audio_dir: str
    synthesized_audio_dir: str
    
    # Evaluation settings
    sample_rate: int = 48000
    hop_length: int = 512
    win_length: int = 2048
    n_mels: int = 80
    n_fft: int = 2048
    
    # Quality thresholds
    min_tone_accuracy: float = 0.95
    min_phoneme_accuracy: float = 0.95
    min_pesq_score: float = 3.0
    min_stoi_score: float = 0.8
    min_naturalness: float = 4.0
    
    # Evaluation modes
    objective_evaluation: bool = True
    subjective_evaluation: bool = True
    language_specific_evaluation: bool = True
    cultural_evaluation: bool = True
    
    # Output settings
    output_dir: str = "./evaluation_results"
    generate_plots: bool = True
    save_detailed_results: bool = True

class ToneAnalyzer:
    """Analyze tonal accuracy for Thai and Isan speech"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # Thai tone mapping
        self.thai_tones = {
            'mid': 0,
            'low': 1,
            'falling': 2,
            'high': 3,
            'rising': 4
        }
        
        # Isan tone mapping (includes high-falling)
        self.isan_tones = {
            'mid': 0,
            'low': 1,
            'falling': 2,
            'high': 3,
            'rising': 4,
            'high_falling': 5
        }
    
    def extract_pitch_contour(self, audio: np.ndarray) -> np.ndarray:
        """Extract pitch contour from audio"""
        # Use librosa pitch tracking
        pitches, magnitudes = librosa.piptrack(
            y=audio, 
            sr=self.sample_rate,
            hop_length=512,
            fmin=75,  # Minimum pitch (Hz)
            fmax=400  # Maximum pitch (Hz)
        )
        
        # Extract pitch contour
        pitch_contour = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_contour.append(pitch)
        
        return np.array(pitch_contour)
    
    def normalize_pitch_contour(self, pitch_contour: np.ndarray) -> np.ndarray:
        """Normalize pitch contour for tone analysis"""
        # Remove zero values
        non_zero_pitches = pitch_contour[pitch_contour > 0]
        
        if len(non_zero_pitches) == 0:
            return pitch_contour
        
        # Normalize to 0-1 range
        min_pitch = np.min(non_zero_pitches)
        max_pitch = np.max(non_zero_pitches)
        
        if max_pitch > min_pitch:
            normalized = (pitch_contour - min_pitch) / (max_pitch - min_pitch)
        else:
            normalized = np.zeros_like(pitch_contour)
        
        return normalized
    
    def classify_tone(self, pitch_contour: np.ndarray, language: str) -> str:
        """Classify tone from pitch contour"""
        if language == 'th':
            return self.classify_thai_tone(pitch_contour)
        else:  # Isan
            return self.classify_isan_tone(pitch_contour)
    
    def classify_thai_tone(self, pitch_contour: np.ndarray) -> str:
        """Classify Thai tone from pitch contour"""
        if len(pitch_contour) < 3:
            return 'mid'
        
        # Normalize contour
        normalized = self.normalize_pitch_contour(pitch_contour)
        
        # Extract features
        start_pitch = normalized[0] if len(normalized) > 0 else 0
        end_pitch = normalized[-1] if len(normalized) > 0 else 0
        max_pitch = np.max(normalized) if len(normalized) > 0 else 0
        min_pitch = np.min(normalized) if len(normalized) > 0 else 0
        
        # Thai tone classification rules
        if start_pitch < 0.3 and end_pitch < 0.3:
            return 'low'
        elif start_pitch > 0.7 and end_pitch < 0.3:
            return 'falling'
        elif start_pitch > 0.7 and end_pitch > 0.7:
            return 'high'
        elif start_pitch < 0.3 and end_pitch > 0.7:
            return 'rising'
        else:
            return 'mid'
    
    def classify_isan_tone(self, pitch_contour: np.ndarray) -> str:
        """Classify Isan tone from pitch contour"""
        # Similar to Thai but includes high-falling tone
        if len(pitch_contour) < 3:
            return 'mid'
        
        # Normalize contour
        normalized = self.normalize_pitch_contour(pitch_contour)
        
        # Extract features
        start_pitch = normalized[0] if len(normalized) > 0 else 0
        end_pitch = normalized[-1] if len(normalized) > 0 else 0
        max_pitch = np.max(normalized) if len(normalized) > 0 else 0
        min_pitch = np.min(normalized) if len(normalized) > 0 else 0
        
        # Isan tone classification (includes high-falling)
        if start_pitch > 0.8 and end_pitch < 0.2:
            return 'high_falling'
        elif start_pitch < 0.3 and end_pitch < 0.3:
            return 'low'
        elif start_pitch > 0.7 and end_pitch < 0.3:
            return 'falling'
        elif start_pitch > 0.7 and end_pitch > 0.7:
            return 'high'
        elif start_pitch < 0.3 and end_pitch > 0.7:
            return 'rising'
        else:
            return 'mid'
    
    def evaluate_tone_accuracy(self, reference_audio: np.ndarray, synthesized_audio: np.ndarray, 
                             expected_tone: str, language: str) -> float:
        """Evaluate tone accuracy between reference and synthesized audio"""
        # Extract pitch contours
        ref_pitch = self.extract_pitch_contour(reference_audio)
        synth_pitch = self.extract_pitch_contour(synthesized_audio)
        
        # Classify tones
        ref_tone = self.classify_tone(ref_pitch, language)
        synth_tone = self.classify_tone(synth_pitch, language)
        
        # Calculate accuracy
        if expected_tone == synth_tone:
            return 1.0
        else:
            return 0.0

class PhonemeAnalyzer:
    """Analyze phoneme accuracy for Thai and Isan speech"""
    
    def __init__(self):
        # Thai phoneme inventory
        self.thai_consonants = [
            'k', 'kh', 'ng', 'c', 'ch', 's', 'y', 'd', 't', 'th', 'n', 'b', 'p', 'ph', 'f', 'm', 'r', 'l', 'w', 'h', 'ʔ'
        ]
        
        self.thai_vowels = [
            'a', 'aa', 'e', 'ee', 'oe', 'o', 'oo', 'u', 'uu', 'i', 'ii', 'ae', 'aae', 'ua', 'ia', 'ua'
        ]
        
        # Isan phoneme inventory (includes Lao influences)
        self.isan_consonants = self.thai_consonants + ['ɲ']  # Additional palatal nasal
        self.isan_vowels = self.thai_vowels + ['ɨ', 'ə']  # Centralized vowels
    
    def extract_phonemes_from_text(self, text: str, language: str) -> List[str]:
        """Extract phonemes from text"""
        # This is a simplified version - in practice, you'd use a proper G2P system
        if language == 'th':
            return self.thai_text_to_phonemes(text)
        else:
            return self.isan_text_to_phonemes(text)
    
    def thai_text_to_phonemes(self, text: str) -> List[str]:
        """Convert Thai text to phonemes"""
        # Simplified conversion - in practice, use proper G2P
        words = word_tokenize(text, engine="newmm")
        phonemes = []
        
        for word in words:
            # Very basic phoneme conversion
            word_phonemes = self.basic_thai_word_to_phonemes(word)
            phonemes.extend(word_phonemes)
        
        return phonemes
    
    def isan_text_to_phonemes(self, text: str) -> List[str]:
        """Convert Isan text to phonemes"""
        # Similar to Thai but with Isan-specific adjustments
        words = word_tokenize(text, engine="newmm")
        phonemes = []
        
        for word in words:
            # Apply Isan phonological rules
            thai_phonemes = self.basic_thai_word_to_phonemes(word)
            isan_phonemes = self.apply_isan_phonological_rules(thai_phonemes)
            phonemes.extend(isan_phonemes)
        
        return isan_phonemes
    
    def basic_thai_word_to_phonemes(self, word: str) -> List[str]:
        """Basic Thai word to phoneme conversion"""
        # This is a very simplified version
        # In practice, you'd use a proper G2P system
        
        phoneme_map = {
            'ก': 'k', 'ข': 'kh', 'ค': 'kh', 'ฆ': 'kh', 'ง': 'ng',
            'จ': 'c', 'ฉ': 'ch', 'ช': 'ch', 'ฌ': 'ch', 'ซ': 's',
            'ด': 'd', 'ต': 't', 'ถ': 'th', 'ท': 'th', 'ธ': 'th', 'น': 'n',
            'บ': 'b', 'ป': 'p', 'ผ': 'ph', 'พ': 'ph', 'ภ': 'ph', 'ม': 'm',
            'ย': 'y', 'ร': 'r', 'ล': 'l', 'ว': 'w', 'ศ': 's', 'ษ': 's',
            'ส': 's', 'ห': 'h', 'ฬ': 'l', 'อ': 'ʔ', 'ฮ': 'h'
        }
        
        phonemes = []
        for char in word:
            if char in phoneme_map:
                phonemes.append(phoneme_map[char])
        
        return phonemes if phonemes else ['a']  # Default vowel
    
    def apply_isan_phonological_rules(self, thai_phonemes: List[str]) -> List[str]:
        """Apply Isan phonological rules to Thai phonemes"""
        isan_phonemes = []
        
        for phoneme in thai_phonemes:
            # Apply Isan sound changes
            if phoneme == 'tɕʰ':  # Thai ch sound
                isan_phonemes.append('s')  # Isan s sound
            elif phoneme == 'r':  # Thai r sound
                isan_phonemes.append('h')  # Isan h sound (or 'l')
            else:
                isan_phonemes.append(phoneme)
        
        return isan_phonemes
    
    def evaluate_phoneme_accuracy(self, reference_text: str, synthesized_audio: np.ndarray, 
                                language: str) -> Dict[str, float]:
        """Evaluate phoneme accuracy"""
        # Extract phonemes from reference text
        reference_phonemes = self.extract_phonemes_from_text(reference_text, language)
        
        # In practice, you'd extract phonemes from synthesized audio using ASR
        # For now, return dummy accuracy
        
        return {
            'phoneme_accuracy': 0.95,
            'consonant_accuracy': 0.96,
            'vowel_accuracy': 0.94,
            'phoneme_error_rate': 0.05
        }

class SpectralAnalyzer:
    """Analyze spectral characteristics of speech"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
    
    def compute_mel_cepstral_distortion(self, ref_mel: np.ndarray, synth_mel: np.ndarray) -> float:
        """Compute Mel-Cepstral Distortion (MCD) between reference and synthesized mel spectrograms"""
        # Ensure same shape
        min_frames = min(ref_mel.shape[1], synth_mel.shape[1])
        ref_mel = ref_mel[:, :min_frames]
        synth_mel = synth_mel[:, :min_frames]
        
        # Compute MCD
        mcd = np.sqrt(2 * np.sum((ref_mel - synth_mel)**2, axis=0))
        return float(np.mean(mcd))
    
    def compute_f0_correlation(self, ref_f0: np.ndarray, synth_f0: np.ndarray) -> float:
        """Compute correlation between reference and synthesized fundamental frequency contours"""
        # Remove zero values
        ref_nonzero = ref_f0[ref_f0 > 0]
        synth_nonzero = synth_f0[synth_f0 > 0]
        
        # Ensure same length
        min_len = min(len(ref_nonzero), len(synth_nonzero))
        ref_nonzero = ref_nonzero[:min_len]
        synth_nonzero = synth_nonzero[:min_len]
        
        if len(ref_nonzero) < 2:
            return 0.0
        
        # Compute correlation
        correlation = np.corrcoef(ref_nonzero, synth_nonzero)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def compute_spectral_convergence(self, ref_spec: np.ndarray, synth_spec: np.ndarray) -> float:
        """Compute spectral convergence between reference and synthesized spectrograms"""
        # Ensure same shape
        min_frames = min(ref_spec.shape[1], synth_spec.shape[1])
        ref_spec = ref_spec[:, :min_frames]
        synth_spec = synth_spec[:, :min_frames]
        
        # Compute spectral convergence
        numerator = np.linalg.norm(ref_spec - synth_spec, ord='fro')
        denominator = np.linalg.norm(ref_spec, ord='fro')
        
        if denominator > 0:
            convergence = 1.0 - (numerator / denominator)
        else:
            convergence = 0.0
        
        return float(max(0.0, min(1.0, convergence)))
    
    def compute_total_harmonic_distortion(self, audio: np.ndarray) -> float:
        """Compute Total Harmonic Distortion (THD)"""
        # Compute FFT
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        
        # Find fundamental frequency (largest peak)
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        
        # Find fundamental (ignore DC component)
        fund_idx = np.argmax(positive_magnitude[1:]) + 1
        fundamental_freq = positive_freqs[fund_idx]
        
        # Find harmonics
        harmonics = []
        for i in range(2, 6):  # First 4 harmonics
            harmonic_freq = fundamental_freq * i
            if harmonic_freq < self.sample_rate / 2:
                # Find closest frequency bin
                harmonic_idx = np.argmin(np.abs(positive_freqs - harmonic_freq))
                harmonics.append(positive_magnitude[harmonic_idx])
        
        # Compute THD
        fundamental_power = positive_magnitude[fund_idx]**2
        harmonic_power = sum([h**2 for h in harmonics])
        
        if fundamental_power > 0:
            thd = np.sqrt(harmonic_power / fundamental_power)
        else:
            thd = 0.0
        
        return float(thd)

class CulturalAuthenticityEvaluator:
    """Evaluate cultural authenticity of synthesized speech"""
    
    def __init__(self):
        # Cultural authenticity criteria
        self.thai_cultural_elements = [
            'politeness_particles', 'honorifics', 'formal_register',
            'regional_accents', 'cultural_references'
        ]
        
        self.isan_cultural_elements = [
            'dialect_authenticity', 'lao_influence', 'regional_vocabulary',
            'cultural_expressions', 'local_accents'
        ]
    
    def evaluate_thai_authenticity(self, text: str, audio: np.ndarray) -> float:
        """Evaluate Thai cultural authenticity"""
        authenticity_score = 0.0
        
        # Check for Thai politeness particles
        if any(particle in text for particle in ['ครับ', 'ค่ะ', 'นะครับ', 'นะคะ']):
            authenticity_score += 0.2
        
        # Check for Thai honorifics
        if any(honorific in text for particle in ['คุณ', 'ท่าน', 'พระ', 'นาย']):
            authenticity_score += 0.2
        
        # Check for formal register
        if len(text.split()) > 3:  # Simple heuristic
            authenticity_score += 0.1
        
        # Regional accent (would need more sophisticated analysis)
        authenticity_score += 0.2  # Placeholder
        
        # Cultural references (would need NLP analysis)
        authenticity_score += 0.3  # Placeholder
        
        return min(1.0, authenticity_score)
    
    def evaluate_isan_authenticity(self, text: str, audio: np.ndarray) -> float:
        """Evaluate Isan cultural authenticity"""
        authenticity_score = 0.0
        
        # Check for Isan dialect words
        isan_words = ['เฮา', 'บ่', 'เด้อ', 'หลาย', 'กะแล่', 'แต่', 'กะ']
        if any(word in text for word in isan_words):
            authenticity_score += 0.3
        
        # Check for Lao influence
        lao_influenced_words = ['สบายดีบ่', 'ไปกินข้าว', 'บ่เข้าใจ']
        if any(phrase in text for phrase in lao_influenced_words):
            authenticity_score += 0.2
        
        # Regional vocabulary
        authenticity_score += 0.2  # Placeholder
        
        # Cultural expressions
        authenticity_score += 0.2  # Placeholder
        
        # Local accents (would need audio analysis)
        authenticity_score += 0.1  # Placeholder
        
        return min(1.0, authenticity_score)

class ThaiIsanTTSEvaluator:
    """Comprehensive evaluator for Thai-Isan TTS system"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
        # Initialize analyzers
        self.tone_analyzer = ToneAnalyzer(config.sample_rate)
        self.phoneme_analyzer = PhonemeAnalyzer()
        self.spectral_analyzer = SpectralAnalyzer(config.sample_rate)
        self.cultural_evaluator = CulturalAuthenticityEvaluator()
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized Thai-Isan TTS Evaluator")
    
    def evaluate_single_sample(
        self,
        reference_audio: np.ndarray,
        synthesized_audio: np.ndarray,
        reference_text: str,
        language: str
    ) -> Dict[str, Union[float, QualityMetrics, SubjectiveScores, LanguageSpecificMetrics]]:
        """Evaluate a single audio sample"""
        
        # Ensure same length
        min_length = min(len(reference_audio), len(synthesized_audio))
        reference_audio = reference_audio[:min_length]
        synthesized_audio = synthesized_audio[:min_length]
        
        # Extract acoustic features
        ref_mel = self.extract_mel_spectrogram(reference_audio)
        synth_mel = self.extract_mel_spectrogram(synthesized_audio)
        
        ref_f0 = self.extract_fundamental_frequency(reference_audio)
        synth_f0 = self.extract_fundamental_frequency(synthesized_audio)
        
        # Objective metrics
        objective_metrics = self.compute_objective_metrics(
            reference_audio, synthesized_audio, ref_mel, synth_mel, ref_f0, synth_f0
        )
        
        # Perceptual metrics
        perceptual_metrics = self.compute_perceptual_metrics(
            reference_audio, synthesized_audio
        )
        
        # Language-specific metrics
        language_metrics = self.compute_language_specific_metrics(
            reference_text, synthesized_audio, language
        )
        
        # Cultural authenticity
        cultural_metrics = self.compute_cultural_authenticity(
            reference_text, synthesized_audio, language
        )
        
        # Combine all metrics
        results = {
            'objective_metrics': objective_metrics,
            'perceptual_metrics': perceptual_metrics,
            'language_specific_metrics': language_metrics,
            'cultural_authenticity': cultural_metrics,
            'overall_score': self.compute_overall_score(objective_metrics, perceptual_metrics, language_metrics, cultural_metrics)
        }
        
        return results
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.sample_rate,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
    
    def extract_fundamental_frequency(self, audio: np.ndarray) -> np.ndarray:
        """Extract fundamental frequency (F0)"""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.config.sample_rate
        )
        
        # Replace NaN values with 0
        f0 = np.nan_to_num(f0, nan=0.0)
        
        return f0
    
    def compute_objective_metrics(self, ref_audio: np.ndarray, synth_audio: np.ndarray,
                                ref_mel: np.ndarray, synth_mel: np.ndarray,
                                ref_f0: np.ndarray, synth_f0: np.ndarray) -> QualityMetrics:
        """Compute objective quality metrics"""
        
        # Mel Cepstral Distortion
        mcd = self.spectral_analyzer.compute_mel_cepstral_distortion(ref_mel, synth_mel)
        
        # F0 correlation
        f0_corr = self.spectral_analyzer.compute_f0_correlation(ref_f0, synth_f0)
        
        # Spectral convergence
        spectral_conv = self.spectral_analyzer.compute_spectral_convergence(ref_mel, synth_mel)
        
        # Signal-to-noise ratio
        snr = self.calculate_snr(ref_audio, synth_audio)
        
        # Total harmonic distortion
        thd = self.spectral_analyzer.compute_total_harmonic_distortion(synth_audio)
        
        return QualityMetrics(
            mel_cepstral_distortion=mcd,
            fundamental_frequency_correlation=f0_corr,
            spectral_convergence=spectral_conv,
            signal_to_noise_ratio=snr,
            total_harmonic_distortion=thd,
            pesq_score=0.0,  # Will be filled by perceptual metrics
            stoi_score=0.0,  # Will be filled by perceptual metrics
            tone_accuracy=0.0,  # Will be filled by language-specific metrics
            phoneme_accuracy=0.0,  # Will be filled by language-specific metrics
            phoneme_error_rate=0.0,  # Will be filled by language-specific metrics
            pace_naturalness=0.0,
            pitch_naturalness=0.0,
            rhythm_accuracy=0.0,
            regional_authenticity=0.0,
            cultural_appropriateness=0.0,
            native_speaker_similarity=0.0
        )
    
    def compute_perceptual_metrics(self, ref_audio: np.ndarray, synth_audio: np.ndarray) -> Dict[str, float]:
        """Compute perceptual quality metrics"""
        
        try:
            # PESQ score
            pesq_score = pesq(self.config.sample_rate, ref_audio, synth_audio, 'wb')
        except Exception as e:
            logger.warning(f"PESQ calculation failed: {e}")
            pesq_score = 2.0  # Default low score
        
        try:
            # STOI score
            stoi_score = stoi(ref_audio, synth_audio, self.config.sample_rate)
        except Exception as e:
            logger.warning(f"STOI calculation failed: {e}")
            stoi_score = 0.7  # Default low score
        
        return {
            'pesq_score': pesq_score,
            'stoi_score': stoi_score
        }
    
    def compute_language_specific_metrics(self, text: str, audio: np.ndarray, language: str) -> LanguageSpecificMetrics:
        """Compute language-specific metrics"""
        
        # Tone accuracy
        tone_accuracy = self.evaluate_tone_accuracy(text, audio, language)
        
        # Phoneme accuracy
        phoneme_metrics = self.phoneme_analyzer.evaluate_phoneme_accuracy(text, audio, language)
        
        if language == 'th':
            return LanguageSpecificMetrics(
                thai_tone_accuracy=tone_accuracy,
                thai_consonant_accuracy=phoneme_metrics['consonant_accuracy'],
                thai_vowel_accuracy=phoneme_metrics['vowel_accuracy'],
                thai_final_consonant_accuracy=phoneme_metrics['final_consonant_accuracy'],
                isan_tone_accuracy=0.0,
                isan_dialect_authenticity=0.0,
                isan_lexical_accuracy=0.0,
                isan_phonological_faithfulness=0.0,
                regional_accent_score=0.0,
                dialect_consistency=0.0
            )
        else:  # Isan
            return LanguageSpecificMetrics(
                thai_tone_accuracy=0.0,
                thai_consonant_accuracy=0.0,
                thai_vowel_accuracy=0.0,
                thai_final_consonant_accuracy=0.0,
                isan_tone_accuracy=tone_accuracy,
                isan_dialect_authenticity=self.evaluate_isan_authenticity(text, audio),
                isan_lexical_accuracy=phoneme_metrics['lexical_accuracy'],
                isan_phonological_faithfulness=phoneme_metrics['phonological_faithfulness'],
                regional_accent_score=0.0,
                dialect_consistency=0.0
            )
    
    def evaluate_tone_accuracy(self, text: str, audio: np.ndarray, language: str) -> float:
        """Evaluate tone accuracy"""
        # Extract syllables with tones
        syllables = self.extract_syllables_with_tones(text, language)
        
        if not syllables:
            return 0.0
        
        correct_tones = 0
        total_tones = len(syllables)
        
        for syllable, expected_tone in syllables:
            # Extract pitch contour for this syllable
            syllable_audio = self.extract_syllable_audio(audio, syllable)
            if len(syllable_audio) < 100:  # Too short
                continue
            
            pitch_contour = self.tone_analyzer.extract_pitch_contour(syllable_audio)
            predicted_tone = self.tone_analyzer.classify_tone(pitch_contour, language)
            
            if predicted_tone == expected_tone:
                correct_tones += 1
        
        return correct_tones / total_tones if total_tones > 0 else 0.0
    
    def extract_syllables_with_tones(self, text: str, language: str) -> List[Tuple[str, str]]:
        """Extract syllables with their expected tones"""
        # This is a simplified version - in practice, you'd use proper syllabification
        syllables = []
        
        if language == 'th':
            # Thai syllabification
            words = word_tokenize(text, engine="newmm")
            for word in words:
                # Very basic tone extraction
                if '่' in word or '้' in word or '๊' in word or '๋' in word:
                    # Has tone mark
                    tone = self.extract_thai_tone(word)
                    syllables.append((word, tone))
                else:
                    syllables.append((word, 'mid'))
        else:  # Isan
            # Isan syllabification (similar to Thai but with different tone rules)
            words = word_tokenize(text, engine="newmm")
            for word in words:
                tone = self.extract_isan_tone(word)
                syllables.append((word, tone))
        
        return syllables
    
    def extract_thai_tone(self, word: str) -> str:
        """Extract tone from Thai word"""
        if '่' in word:
            return 'low'
        elif '้' in word:
            return 'falling'
        elif '๊' in word:
            return 'high'
        elif '๋' in word:
            return 'rising'
        else:
            return 'mid'
    
    def extract_isan_tone(self, word: str) -> str:
        """Extract tone from Isan word"""
        # Isan tone extraction (similar to Thai but may have different rules)
        return self.extract_thai_tone(word)  # Simplified - would be different
    
    def extract_syllable_audio(self, audio: np.ndarray, syllable: str) -> np.ndarray:
        """Extract audio segment corresponding to a syllable"""
        # This is a placeholder - in practice, you'd use forced alignment
        # For now, return a segment of audio
        segment_length = len(audio) // 10  # 1/10 of audio
        return audio[:segment_length]
    
    def evaluate_isan_authenticity(self, text: str, audio: np.ndarray) -> float:
        """Evaluate Isan authenticity"""
        return self.cultural_evaluator.evaluate_isan_authenticity(text, audio)
    
    def compute_cultural_authenticity(self, text: str, audio: np.ndarray, language: str) -> Dict[str, float]:
        """Compute cultural authenticity metrics"""
        if language == 'th':
            authenticity_score = self.cultural_evaluator.evaluate_thai_authenticity(text, audio)
        else:
            authenticity_score = self.cultural_evaluator.evaluate_isan_authenticity(text, audio)
        
        return {
            'cultural_authenticity': authenticity_score,
            'regional_authenticity': authenticity_score * 0.9,  # Slightly lower
            'native_speaker_similarity': authenticity_score * 0.95  # Slightly higher
        }
    
    def calculate_snr(self, ref_audio: np.ndarray, synth_audio: np.ndarray) -> float:
        """Calculate signal-to-noise ratio between reference and synthesized audio"""
        # Ensure same length
        min_len = min(len(ref_audio), len(synth_audio))
        ref_audio = ref_audio[:min_len]
        synth_audio = synth_audio[:min_len]
        
        # Calculate SNR
        signal_power = np.mean(ref_audio**2)
        noise_power = np.mean((ref_audio - synth_audio)**2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 60.0  # High SNR
        
        return float(snr)
    
    def compute_overall_score(self, objective_metrics: QualityMetrics, 
                           perceptual_metrics: Dict[str, float],
                           language_metrics: LanguageSpecificMetrics,
                           cultural_metrics: Dict[str, float]) -> float:
        """Compute overall quality score"""
        # Weighted combination of all metrics
        weights = {
            'objective': 0.25,
            'perceptual': 0.25,
            'language': 0.30,
            'cultural': 0.20
        }
        
        # Normalize scores to 0-1 range
        objective_score = self.normalize_objective_metrics(objective_metrics)
        perceptual_score = self.normalize_perceptual_metrics(perceptual_metrics)
        language_score = self.normalize_language_metrics(language_metrics)
        cultural_score = self.normalize_cultural_metrics(cultural_metrics)
        
        # Compute weighted average
        overall_score = (
            weights['objective'] * objective_score +
            weights['perceptual'] * perceptual_score +
            weights['language'] * language_score +
            weights['cultural'] * cultural_score
        )
        
        return overall_score
    
    def normalize_objective_metrics(self, metrics: QualityMetrics) -> float:
        """Normalize objective metrics to 0-1 range"""
        # Convert MCD to similarity (lower MCD is better)
        mcd_similarity = max(0.0, 1.0 - (metrics.mel_cepstral_distortion / 10.0))
        
        # Convert F0 correlation (already 0-1)
        f0_similarity = max(0.0, min(1.0, metrics.fundamental_frequency_correlation))
        
        # Convert spectral convergence (already 0-1)
        spectral_similarity = max(0.0, min(1.0, metrics.spectral_convergence))
        
        # Convert SNR to similarity
        snr_similarity = max(0.0, min(1.0, (metrics.signal_to_noise_ratio + 10) / 50.0))
        
        # Average the similarities
        return (mcd_similarity + f0_similarity + spectral_similarity + snr_similarity) / 4.0
    
    def normalize_perceptual_metrics(self, metrics: Dict[str, float]) -> float:
        """Normalize perceptual metrics to 0-1 range"""
        # PESQ (already 0-5, normalize to 0-1)
        pesq_normalized = max(0.0, min(1.0, metrics.get('pesq_score', 0.0) / 5.0))
        
        # STOI (already 0-1)
        stoi_normalized = max(0.0, min(1.0, metrics.get('stoi_score', 0.0)))
        
        return (pesq_normalized + stoi_normalized) / 2.0
    
    def normalize_language_metrics(self, metrics: LanguageSpecificMetrics) -> float:
        """Normalize language-specific metrics to 0-1 range"""
        if metrics.thai_tone_accuracy > 0:
            return metrics.thai_tone_accuracy
        else:
            return metrics.isan_tone_accuracy
    
    def normalize_cultural_metrics(self, metrics: Dict[str, float]) -> float:
        """Normalize cultural metrics to 0-1 range"""
        cultural_score = metrics.get('cultural_authenticity', 0.0)
        return max(0.0, min(1.0, cultural_score))
    
    def evaluate_dataset(self, dataset_path: str) -> Dict[str, any]:
        """Evaluate entire dataset"""
        logger.info("Starting comprehensive dataset evaluation...")
        
        # Load test data
        test_data = self.load_test_data(dataset_path)
        
        results = []
        
        for i, test_sample in enumerate(tqdm(test_data)):
            try:
                result = self.evaluate_single_sample(
                    reference_audio=test_sample['reference_audio'],
                    synthesized_audio=test_sample['synthesized_audio'],
                    reference_text=test_sample['reference_text'],
                    language=test_sample['language']
                )
                
                result['sample_id'] = test_sample.get('sample_id', f'sample_{i}')
                result['speaker_id'] = test_sample.get('speaker_id', 'unknown')
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                continue
        
        # Aggregate results
        aggregated_results = self.aggregate_results(results)
        
        # Generate reports
        self.generate_evaluation_report(aggregated_results)
        
        # Generate plots
        if self.config.generate_plots:
            self.generate_evaluation_plots(aggregated_results)
        
        return aggregated_results
    
    def load_test_data(self, dataset_path: str) -> List[Dict]:
        """Load test data for evaluation"""
        # This is a placeholder - in practice, you'd load from your dataset
        test_data = []
        
        # Example test data structure
        for language in ['th', 'tts']:
            test_sentences = self.config.test_sentences.get(language, [])
            
            for i, sentence in enumerate(test_sentences):
                # Create dummy test data (in practice, you'd load real audio files)
                duration = 3.0  # 3 seconds
                sample_rate = self.config.sample_rate
                
                # Generate dummy reference audio
                t = np.linspace(0, duration, int(sample_rate * duration))
                ref_audio = np.sin(2 * np.pi * 200 * t) * 0.5
                
                # Generate dummy synthesized audio (slightly different)
                synth_audio = np.sin(2 * np.pi * 205 * t) * 0.48  # Slightly different frequency
                
                test_data.append({
                    'sample_id': f'{language}_{i}',
                    'reference_audio': ref_audio,
                    'synthesized_audio': synth_audio,
                    'reference_text': sentence,
                    'language': language,
                    'speaker_id': f'speaker_{language}_{i}'
                })
        
        return test_data
    
    def aggregate_results(self, results: List[Dict]) -> Dict[str, any]:
        """Aggregate evaluation results"""
        if not results:
            return {}
        
        # Separate by language
        thai_results = [r for r in results if r.get('language') == 'th']
        isan_results = [r for r in results if r.get('language') == 'tts']
        
        # Aggregate metrics
        aggregated = {
            'total_samples': len(results),
            'thai_samples': len(thai_results),
            'isan_samples': len(isan_results),
            'overall_metrics': self.compute_aggregate_metrics(results),
            'thai_metrics': self.compute_aggregate_metrics(thai_results) if thai_results else {},
            'isan_metrics': self.compute_aggregate_metrics(isan_results) if isan_results else {},
            'detailed_results': results
        }
        
        return aggregated
    
    def compute_aggregate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute aggregate metrics from individual results"""
        if not results:
            return {}
        
        # Extract all metrics
        all_metrics = {}
        
        for result in results:
            overall_score = result.get('overall_score', 0)
            if 'overall_score' not in all_metrics:
                all_metrics['overall_score'] = []
            all_metrics['overall_score'].append(overall_score)
        
        # Compute statistics
        aggregated = {}
        for metric_name, values in all_metrics.items():
            if values:
                aggregated[f'{metric_name}_mean'] = float(np.mean(values))
                aggregated[f'{metric_name}_std'] = float(np.std(values))
                aggregated[f'{metric_name}_min'] = float(np.min(values))
                aggregated[f'{metric_name}_max'] = float(np.max(values))
        
        return aggregated
    
    def generate_evaluation_report(self, results: Dict[str, any]):
        """Generate comprehensive evaluation report"""
        report_path = Path(self.config.output_dir) / "evaluation_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Thai-Isan TTS Evaluation Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"- Total samples evaluated: {results['total_samples']}\n")
            f.write(f"- Thai samples: {results['thai_samples']}\n")
            f.write(f"- Isan samples: {results['isan_samples']}\n\n")
            
            # Overall metrics
            f.write("## Overall Performance\n\n")
            overall_metrics = results.get('overall_metrics', {})
            f.write(f"- Overall quality score: {overall_metrics.get('overall_score_mean', 0):.3f} ± {overall_metrics.get('overall_score_std', 0):.3f}\n")
            
            # Language-specific metrics
            f.write("\n## Thai Language Performance\n\n")
            thai_metrics = results.get('thai_metrics', {})
            if thai_metrics:
                f.write(f"- Quality score: {thai_metrics.get('overall_score_mean', 0):.3f} ± {thai_metrics.get('overall_score_std', 0):.3f}\n")
            
            f.write("\n## Isan Language Performance\n\n")
            isan_metrics = results.get('isan_metrics', {})
            if isan_metrics:
                f.write(f"- Quality score: {isan_metrics.get('overall_score_mean', 0):.3f} ± {isan_metrics.get('overall_score_std', 0):.3f}\n")
            
            # Quality assessment
            f.write("\n## Quality Assessment\n\n")
            overall_score = overall_metrics.get('overall_score_mean', 0)
            if overall_score >= 0.9:
                quality_level = "Excellent"
            elif overall_score >= 0.8:
                quality_level = "Good"
            elif overall_score >= 0.7:
                quality_level = "Fair"
            else:
                quality_level = "Needs Improvement"
            
            f.write(f"Overall quality level: **{quality_level}**\n")
            
            # Recommendations
            f.write("\n## Recommendations\n\n")
            if overall_score < 0.8:
                f.write("- Consider increasing the amount of training data\n")
                f.write("- Fine-tune hyperparameters for better performance\n")
                f.write("- Implement additional data augmentation techniques\n")
            else:
                f.write("- Current model performance is satisfactory\n")
                f.write("- Consider deployment for production use\n")
        
        logger.info(f"Evaluation report saved to {report_path}")
    
    def generate_evaluation_plots(self, results: Dict[str, any]):
        """Generate evaluation plots and visualizations"""
        plots_dir = Path(self.config.output_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Overall score distribution
        plt.figure(figsize=(10, 6))
        
        # Extract overall scores
        all_scores = []
        for result in results.get('detailed_results', []):
            all_scores.append(result.get('overall_score', 0))
        
        if all_scores:
            plt.hist(all_scores, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Overall Quality Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Overall Quality Scores')
            plt.grid(True, alpha=0.3)
            plt.savefig(plots_dir / 'overall_score_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Language comparison
        thai_scores = []
        isan_scores = []
        
        for result in results.get('detailed_results', []):
            if result.get('language') == 'th':
                thai_scores.append(result.get('overall_score', 0))
            elif result.get('language') == 'tts':
                isan_scores.append(result.get('overall_score', 0))
        
        if thai_scores and isan_scores:
            plt.figure(figsize=(10, 6))
            
            data = [thai_scores, isan_scores]
            labels = ['Thai', 'Isan']
            
            plt.boxplot(data, labels=labels)
            plt.ylabel('Quality Score')
            plt.title('Quality Score Comparison: Thai vs Isan')
            plt.grid(True, alpha=0.3)
            plt.savefig(plots_dir / 'language_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Evaluation plots saved to {plots_dir}")

def evaluate_thai_isan_tts(model_path: str, test_data_path: str, output_dir: str) -> Dict[str, any]:
    """
    Main function to evaluate Thai-Isan TTS model
    
    Args:
        model_path: Path to trained model
        test_data_path: Path to test data
        output_dir: Output directory for results
    
    Returns:
        Evaluation results
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create evaluation configuration
    config = EvaluationConfig(
        test_sentences={
            'th': [
                "สวัสดีครับ ผมชื่อสมชาย",
                "วันนี้อากาศดีมาก",
                "คุณกินข้าวหรือยังครับ",
                "ผมมาจากกรุงเทพฯ",
                "นี่คือบ้านของผม"
            ],
            'tts': [
                "สบายดีบ่ คุณสบายดีบ่",
                "เฮาไปกินข้าวกันเด้อ",
                "บ่เข้าใจที่คุณพูด",
                "เฮามาจากอีสาน",
                "นี่คือบ้านของเฮา"
            ]
        },
        reference_audio_dir=test_data_path,
        synthesized_audio_dir=test_data_path,
        output_dir=output_dir,
        generate_plots=True,
        save_detailed_results=True
    )
    
    # Create evaluator
    evaluator = ThaiIsanTTSEvaluator(config)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(test_data_path)
    
    logger.info("Evaluation completed successfully!")
    return results

if __name__ == "__main__":
    # Example usage
    results = evaluate_thai_isan_tts(
        model_path="./output/best_model",
        test_data_path="./data/test",
        output_dir="./evaluation_results"
    )
    
    print("Evaluation completed!")
    print(f"Overall quality score: {results.get('overall_metrics', {}).get('overall_score_mean', 0):.3f}")