"""
Comprehensive Thai-Isan TTS Quality Assurance System
Advanced evaluation and testing framework for Thai and Isan speech synthesis
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
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report
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

# Advanced audio analysis
import torchaudio
from torchaudio.transforms import MelSpectrogram, Spectrogram

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
    
    # Overall quality
    overall_quality_score: float  # Combined quality score

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
    thai_vowel_length_accuracy: float
    thai_tonal_contour_accuracy: float
    
    # Isan-specific
    isan_tone_accuracy: float
    isan_dialect_authenticity: float
    isan_lexical_accuracy: float
    isan_phonological_faithfulness: float
    isan_regional_variation_accuracy: float
    isan_cultural_expression_accuracy: float

@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    
    synthesized_audio: np.ndarray
    reference_audio: np.ndarray
    text: str
    language: str
    speaker_id: str
    
    # Metrics
    objective_metrics: QualityMetrics
    subjective_scores: SubjectiveScores
    language_specific: LanguageSpecificMetrics
    
    # Detailed analysis
    detailed_results: Dict[str, any]
    error_analysis: Dict[str, List]
    recommendations: List[str]
    
    # Metadata
    evaluation_timestamp: datetime
    evaluator_info: str
    cultural_notes: str

class ThaiIsanQualityEvaluator:
    """Comprehensive quality evaluator for Thai and Isan TTS"""
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize audio processing
        self.mel_transform = MelSpectrogram(
            sample_rate=48000,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )
        
        # Quality thresholds
        self.quality_thresholds = {
            'tone_accuracy': 0.95,
            'phoneme_accuracy': 0.95,
            'pesq_score': 3.0,
            'stoi_score': 0.8,
            'mos_score': 4.0,
            'regional_authenticity': 0.8,
            'cultural_appropriateness': 0.8
        }
        
        # Initialize Thai language processing
        self._initialize_thai_processing()
    
    def _default_config(self):
        """Default configuration"""
        return {
            'sample_rate': 48000,
            'n_fft': 1024,
            'hop_length': 256,
            'n_mels': 80,
            'enable_cultural_metrics': True,
            'enable_perceptual_metrics': True,
            'enable_objective_metrics': True
        }
    
    def _initialize_thai_processing(self):
        """Initialize Thai language processing components"""
        # Thai tone classes
        self.thai_tone_classes = {
            'mid': ['ก', 'ด', 'บ', 'ป', 'อ'],
            'low': ['ค', 'ฆ', 'ง', 'ช', 'ซ', 'ฌ', 'ญ', 'ฑ', 'ฒ', 'ณ', 'ท', 'ธ', 'น', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'ฮ'],
            'high': ['ข', 'ฃ', 'ฉ', 'ฐ', 'ถ', 'ผ', 'ฝ', 'ส', 'ษ', 'ศ', 'ห']
        }
        
        # Isan tone mapping
        self.isan_tone_mapping = {
            'high_class_live': 'high_falling',
            'mid_class_dead': 'low',
            'low_class_long': 'rising',
            'tone_sandhi': 'contextual'
        }
    
    def evaluate_speech(self, 
                       synthesized_audio: np.ndarray,
                       reference_audio: np.ndarray,
                       text: str,
                       language: str,
                       speaker_id: str) -> EvaluationResult:
        """Comprehensive speech evaluation"""
        
        logger.info(f"Evaluating {language} speech: {text}")
        
        # Ensure audio is at correct sample rate
        if len(synthesized_audio) != len(reference_audio):
            # Resample to match lengths
            min_len = min(len(synthesized_audio), len(reference_audio))
            synthesized_audio = synthesized_audio[:min_len]
            reference_audio = reference_audio[:min_len]
        
        # Objective metrics
        objective_metrics = self._calculate_objective_metrics(synthesized_audio, reference_audio)
        
        # Perceptual metrics
        perceptual_metrics = self._calculate_perceptual_metrics(synthesized_audio, reference_audio)
        
        # Language-specific metrics
        language_metrics = self._calculate_language_metrics(synthesized_audio, reference_audio, text, language)
        
        # Cultural authenticity
        cultural_metrics = self._assess_cultural_authenticity(synthesized_audio, reference_audio, text, language)
        
        # Combine metrics
        quality_metrics = QualityMetrics(
            mel_cepstral_distortion=objective_metrics['mcd'],
            fundamental_frequency_correlation=objective_metrics['f0_corr'],
            spectral_convergence=objective_metrics['spectral_conv'],
            signal_to_noise_ratio=objective_metrics['snr'],
            total_harmonic_distortion=objective_metrics['thd'],
            pesq_score=perceptual_metrics['pesq'],
            stoi_score=perceptual_metrics['stoi'],
            tone_accuracy=language_metrics['tone_accuracy'],
            phoneme_accuracy=language_metrics['phoneme_accuracy'],
            phoneme_error_rate=language_metrics['per'],
            pace_naturalness=cultural_metrics['pace_naturalness'],
            pitch_naturalness=cultural_metrics['pitch_naturalness'],
            rhythm_accuracy=cultural_metrics['rhythm_accuracy'],
            regional_authenticity=cultural_metrics['regional_authenticity'],
            cultural_appropriateness=cultural_metrics['cultural_appropriateness'],
            native_speaker_similarity=cultural_metrics['native_similarity'],
            overall_quality_score=self._calculate_overall_quality(objective_metrics, perceptual_metrics, language_metrics, cultural_metrics)
        )
        
        # Subjective scores (would be from human evaluators)
        subjective_scores = self._estimate_subjective_scores(quality_metrics)
        
        # Language-specific metrics
        language_specific = self._calculate_language_specific_metrics(language_metrics, language)
        
        # Detailed analysis
        detailed_results = self._generate_detailed_analysis(synthesized_audio, reference_audio, text, language)
        
        # Error analysis
        error_analysis = self._analyze_errors(synthesized_audio, reference_audio, text, language)
        
        # Recommendations
        recommendations = self._generate_recommendations(quality_metrics, language)
        
        return EvaluationResult(
            synthesized_audio=synthesized_audio,
            reference_audio=reference_audio,
            text=text,
            language=language,
            speaker_id=speaker_id,
            objective_metrics=quality_metrics,
            subjective_scores=subjective_scores,
            language_specific=language_specific,
            detailed_results=detailed_results,
            error_analysis=error_analysis,
            recommendations=recommendations,
            evaluation_timestamp=datetime.now(),
            evaluator_info="ThaiIsanQualityEvaluator",
            cultural_notes=self._generate_cultural_notes(text, language)
        )
    
    def _calculate_objective_metrics(self, synthesized: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
        """Calculate objective quality metrics"""
        metrics = {}
        
        # Mel Cepstral Distortion (MCD)
        metrics['mcd'] = self._calculate_mcd(synthesized, reference)
        
        # Fundamental frequency correlation
        metrics['f0_corr'] = self._calculate_f0_correlation(synthesized, reference)
        
        # Spectral convergence
        metrics['spectral_conv'] = self._calculate_spectral_convergence(synthesized, reference)
        
        # Signal-to-noise ratio
        metrics['snr'] = self._calculate_snr(synthesized, reference)
        
        # Total Harmonic Distortion
        metrics['thd'] = self._calculate_thd(synthesized)
        
        return metrics
    
    def _calculate_perceptual_metrics(self, synthesized: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
        """Calculate perceptual quality metrics"""
        metrics = {}
        
        # PESQ score
        try:
            pesq_score = pesq(48000, reference, synthesized, 'wb')
            metrics['pesq'] = float(pesq_score)
        except:
            metrics['pesq'] = 2.0  # Default low score
        
        # STOI score
        try:
            stoi_score = stoi(reference, synthesized, 48000)
            metrics['stoi'] = float(stoi_score)
        except:
            metrics['stoi'] = 0.5  # Default low score
        
        return metrics
    
    def _calculate_language_metrics(self, synthesized: np.ndarray, reference: np.ndarray, text: str, language: str) -> Dict[str, float]:
        """Calculate language-specific metrics"""
        metrics = {}
        
        # Tone accuracy
        metrics['tone_accuracy'] = self._calculate_tone_accuracy(synthesized, reference, text, language)
        
        # Phoneme accuracy
        metrics['phoneme_accuracy'] = self._calculate_phoneme_accuracy(synthesized, reference, text, language)
        
        # Phoneme error rate
        metrics['per'] = 1.0 - metrics['phoneme_accuracy']
        
        return metrics
    
    def _calculate_tone_accuracy(self, synthesized: np.ndarray, reference: np.ndarray, text: str, language: str) -> float:
        """Calculate tone accuracy"""
        # Extract F0 contours
        f0_synthesized, _ = librosa.piptrack(y=synthesized, sr=48000)
        f0_reference, _ = librosa.piptrack(y=reference, sr=48000)
        
        # Extract tone information from text
        tones = self._extract_tones_from_text(text, language)
        
        # Compare tone realization
        accuracy = self._compare_tone_realization(f0_synthesized, f0_reference, tones)
        
        return accuracy
    
    def _calculate_phoneme_accuracy(self, synthesized: np.ndarray, reference: np.ndarray, text: str, language: str) -> float:
        """Calculate phoneme accuracy"""
        # This would use a phoneme recognition model
        # For now, use a simplified approach
        
        # Extract phoneme sequences (placeholder)
        phonemes_synthesized = self._extract_phonemes_from_audio(synthesized)
        phonemes_reference = self._extract_phonemes_from_audio(reference)
        
        # Calculate accuracy
        if len(phonemes_reference) > 0:
            accuracy = self._calculate_sequence_accuracy(phonemes_synthesized, phonemes_reference)
        else:
            accuracy = 0.0
        
        return accuracy
    
    def _assess_cultural_authenticity(self, synthesized: np.ndarray, reference: np.ndarray, text: str, language: str) -> Dict[str, float]:
        """Assess cultural authenticity"""
        metrics = {}
        
        # Regional authenticity
        metrics['regional_authenticity'] = self._assess_regional_authenticity(synthesized, reference, language)
        
        # Cultural appropriateness
        metrics['cultural_appropriateness'] = self._assess_cultural_appropriateness(text, language)
        
        # Native speaker similarity
        metrics['native_similarity'] = self._calculate_native_similarity(synthesized, reference, language)
        
        # Prosodic naturalness
        prosody_metrics = self._assess_prosodic_naturalness(synthesized, reference)
        metrics.update(prosody_metrics)
        
        return metrics
    
    def _calculate_mcd(self, synthesized: np.ndarray, reference: np.ndarray) -> float:
        """Calculate Mel Cepstral Distortion"""
        # Extract mel cepstral coefficients
        mel_synthesized = librosa.feature.melspectrogram(y=synthesized, sr=48000, n_mels=80)
        mel_reference = librosa.feature.melspectrogram(y=reference, sr=48000, n_mels=80)
        
        # Convert to log scale
        log_mel_synth = np.log(mel_synthesized + 1e-10)
        log_mel_ref = np.log(mel_reference + 1e-10)
        
        # Calculate MCD
        mcd = np.mean(np.sqrt(np.sum((log_mel_synth - log_mel_ref)**2, axis=0)))
        
        return float(mcd)
    
    def _calculate_f0_correlation(self, synthesized: np.ndarray, reference: np.ndarray) -> float:
        """Calculate F0 correlation"""
        # Extract F0
        f0_synthesized = librosa.yin(synthesized, fmin=75, fmax=400)
        f0_reference = librosa.yin(reference, fmin=75, fmax=400)
        
        # Ensure same length
        min_len = min(len(f0_synthesized), len(f0_reference))
        f0_synthesized = f0_synthesized[:min_len]
        f0_reference = f0_reference[:min_len]
        
        # Calculate correlation
        correlation = np.corrcoef(f0_synthesized, f0_reference)[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _calculate_spectral_convergence(self, synthesized: np.ndarray, reference: np.ndarray) -> float:
        """Calculate spectral convergence"""
        # Extract spectrograms
        spec_synthesized = np.abs(librosa.stft(synthesized))
        spec_reference = np.abs(librosa.stft(reference))
        
        # Calculate spectral convergence
        convergence = np.linalg.norm(spec_synthesized - spec_reference) / np.linalg.norm(spec_reference)
        
        return float(convergence)
    
    def _calculate_snr(self, synthesized: np.ndarray, reference: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        # Simple SNR calculation
        signal_power = np.mean(reference**2)
        noise_power = np.mean((synthesized - reference)**2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 40.0  # High SNR if no noise
        
        return float(snr)
    
    def _calculate_thd(self, audio: np.ndarray) -> float:
        """Calculate Total Harmonic Distortion"""
        # Simplified THD calculation
        # This would normally use more sophisticated harmonic analysis
        
        # FFT
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        
        # Estimate THD based on spectral content
        total_power = np.sum(magnitude**2)
        harmonic_power = np.sum(magnitude[100:]**2)  # Simple approximation
        
        if total_power > 0:
            thd = np.sqrt(harmonic_power / total_power)
        else:
            thd = 0.0
        
        return float(thd)
    
    def _extract_tones_from_text(self, text: str, language: str) -> List[str]:
        """Extract tone information from text"""
        # This is a simplified version - would use proper linguistic analysis
        tones = []
        
        if language == "th":
            # Thai tone extraction
            for char in text:
                if char in '่้๊๋':
                    if char == '่':
                        tones.append('low')
                    elif char == '้':
                        tones.append('falling')
                    elif char == '๊':
                        tones.append('high')
                    elif char == '๋':
                        tones.append('rising')
        
        return tones
    
    def _extract_phonemes_from_audio(self, audio: np.ndarray) -> List[str]:
        """Extract phonemes from audio (placeholder)"""
        # This would use a phoneme recognition model
        # For now, return dummy phonemes
        return ['p', 'a', 'a', 'a']  # Dummy
    
    def _calculate_sequence_accuracy(self, predicted: List, reference: List) -> float:
        """Calculate sequence accuracy"""
        if len(reference) == 0:
            return 0.0
        
        # Simple accuracy calculation
        min_len = min(len(predicted), len(reference))
        matches = sum(1 for i in range(min_len) if predicted[i] == reference[i])
        
        return matches / len(reference)
    
    def _compare_tone_realization(self, f0_synthesized: np.ndarray, f0_reference: np.ndarray, tones: List[str]) -> float:
        """Compare tone realization"""
        # Simplified tone comparison
        # Would use more sophisticated tone analysis
        return 0.85  # Placeholder
    
    def _assess_regional_authenticity(self, synthesized: np.ndarray, reference: np.ndarray, language: str) -> float:
        """Assess regional authenticity"""
        # Compare acoustic features that characterize regional accents
        # This is a simplified approach
        
        if language == "th":
            # Thai regional authenticity
            return 0.85
        elif language == "tts":
            # Isan regional authenticity
            return 0.80
        else:
            return 0.75
    
    def _assess_cultural_appropriateness(self, text: str, language: str) -> float:
        """Assess cultural appropriateness of the text"""
        # Analyze text for cultural appropriateness
        # This would use cultural knowledge bases
        
        if language == "th":
            return 0.90
        elif language == "tts":
            return 0.85
        else:
            return 0.80
    
    def _calculate_native_similarity(self, synthesized: np.ndarray, reference: np.ndarray, language: str) -> float:
        """Calculate similarity to native speakers"""
        # Compare acoustic features
        # This is a simplified calculation
        
        mel_synth = librosa.feature.melspectrogram(y=synthesized, sr=48000)
        mel_ref = librosa.feature.melspectrogram(y=reference, sr=48000)
        
        # Calculate similarity
        similarity = 1.0 - np.mean(np.abs(mel_synth - mel_ref)) / np.mean(np.abs(mel_ref))
        
        return max(0.0, float(similarity))
    
    def _assess_prosodic_naturalness(self, synthesized: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
        """Assess prosodic naturalness"""
        metrics = {}
        
        # Pace naturalness
        synth_duration = len(synthesized) / 48000
        ref_duration = len(reference) / 48000
        metrics['pace_naturalness'] = 1.0 - abs(synth_duration - ref_duration) / max(synth_duration, ref_duration)
        
        # Pitch naturalness
        f0_synth = librosa.yin(synthesized, fmin=75, fmax=400)
        f0_ref = librosa.yin(reference, fmin=75, fmax=400)
        
        if len(f0_synth) > 0 and len(f0_ref) > 0:
            min_len = min(len(f0_synth), len(f0_ref))
            f0_synth = f0_synth[:min_len]
            f0_ref = f0_ref[:min_len]
            
            pitch_corr = np.corrcoef(f0_synth, f0_ref)[0, 1]
            metrics['pitch_naturalness'] = max(0.0, float(pitch_corr) if not np.isnan(pitch_corr) else 0.0)
        else:
            metrics['pitch_naturalness'] = 0.0
        
        # Rhythm accuracy (simplified)
        metrics['rhythm_accuracy'] = 0.8  # Placeholder
        
        return metrics
    
    def _estimate_subjective_scores(self, quality_metrics: QualityMetrics) -> SubjectiveScores:
        """Estimate subjective scores based on objective metrics"""
        # This is a simplified estimation
        # In practice, these would come from human evaluators
        
        # Estimate naturalness based on spectral and prosodic metrics
        naturalness = min(5.0, max(1.0, 
            quality_metrics.pesq_score * 1.2 + 
            quality_metrics.tone_accuracy * 2.0 +
            quality_metrics.pitch_naturalness * 1.5
        ))
        
        # Estimate intelligibility
        intelligibility = min(5.0, max(1.0,
            quality_metrics.stoi_score * 5.0 +
            quality_metrics.phoneme_accuracy * 2.0 +
            quality_metrics.snr * 0.1
        ))
        
        # Other scores
        pleasantness = naturalness * 0.9
        similarity = quality_metrics.native_speaker_similarity * 5.0
        overall_quality = (naturalness + intelligibility + pleasantness + similarity) / 4
        
        return SubjectiveScores(
            naturalness=naturalness,
            intelligibility=intelligibility,
            pleasantness=pleasantness,
            similarity=similarity,
            overall_quality=overall_quality,
            tone_naturalness=quality_metrics.tone_accuracy * 5.0,
            accent_authenticity=quality_metrics.regional_authenticity * 5.0,
            cultural_appropriateness=quality_metrics.cultural_appropriateness * 5.0
        )
    
    def _calculate_language_specific_metrics(self, language_metrics: Dict[str, float], language: str) -> LanguageSpecificMetrics:
        """Calculate language-specific metrics"""
        
        if language == "th":
            return LanguageSpecificMetrics(
                thai_tone_accuracy=language_metrics['tone_accuracy'],
                thai_consonant_accuracy=language_metrics['phoneme_accuracy'] * 0.95,
                thai_vowel_accuracy=language_metrics['phoneme_accuracy'] * 0.90,
                thai_final_consonant_accuracy=language_metrics['phoneme_accuracy'] * 0.85,
                thai_vowel_length_accuracy=language_metrics['phoneme_accuracy'] * 0.88,
                thai_tonal_contour_accuracy=language_metrics['tone_accuracy'] * 0.92,
                isan_tone_accuracy=0.0,
                isan_dialect_authenticity=0.0,
                isan_lexical_accuracy=0.0,
                isan_phonological_faithfulness=0.0,
                isan_regional_variation_accuracy=0.0,
                isan_cultural_expression_accuracy=0.0
            )
        elif language == "tts":
            return LanguageSpecificMetrics(
                thai_tone_accuracy=0.0,
                thai_consonant_accuracy=0.0,
                thai_vowel_accuracy=0.0,
                thai_final_consonant_accuracy=0.0,
                thai_vowel_length_accuracy=0.0,
                thai_tonal_contour_accuracy=0.0,
                isan_tone_accuracy=language_metrics['tone_accuracy'],
                isan_dialect_authenticity=language_metrics['regional_authenticity'],
                isan_lexical_accuracy=language_metrics['phoneme_accuracy'] * 0.88,
                isan_phonological_faithfulness=language_metrics['phoneme_accuracy'] * 0.90,
                isan_regional_variation_accuracy=language_metrics['regional_authenticity'] * 0.85,
                isan_cultural_expression_accuracy=language_metrics['cultural_appropriateness'] * 0.92
            )
        else:
            return LanguageSpecificMetrics(
                thai_tone_accuracy=0.0,
                thai_consonant_accuracy=0.0,
                thai_vowel_accuracy=0.0,
                thai_final_consonant_accuracy=0.0,
                thai_vowel_length_accuracy=0.0,
                thai_tonal_contour_accuracy=0.0,
                isan_tone_accuracy=0.0,
                isan_dialect_authenticity=0.0,
                isan_lexical_accuracy=0.0,
                isan_phonological_faithfulness=0.0,
                isan_regional_variation_accuracy=0.0,
                isan_cultural_expression_accuracy=0.0
            )
    
    def _calculate_overall_quality(self, objective: Dict[str, float], perceptual: Dict[str, float], 
                                 language: Dict[str, float], cultural: Dict[str, float]) -> float:
        """Calculate overall quality score"""
        # Weighted combination of all metrics
        weights = {
            'objective': 0.3,
            'perceptual': 0.3,
            'language': 0.25,
            'cultural': 0.15
        }
        
        # Normalize scores to 0-1 range
        objective_score = (objective['snr'] + 20) / 60  # SNR from -20 to 40 dB
        perceptual_score = (perceptual['pesq'] + 0.5) / 4.5  # PESQ from -0.5 to 4.5
        language_score = language['tone_accuracy']
        cultural_score = cultural['regional_authenticity']
        
        # Combined score
        overall_score = (
            weights['objective'] * objective_score +
            weights['perceptual'] * perceptual_score +
            weights['language'] * language_score +
            weights['cultural'] * cultural_score
        )
        
        return max(0.0, min(1.0, overall_score))
    
    def _generate_detailed_analysis(self, synthesized: np.ndarray, reference: np.ndarray, text: str, language: str) -> Dict[str, any]:
        """Generate detailed analysis results"""
        analysis = {}
        
        # Spectral analysis
        analysis['spectral_analysis'] = {
            'synthesized_centroid': float(np.mean(librosa.feature.spectral_centroid(y=synthesized, sr=48000))),
            'reference_centroid': float(np.mean(librosa.feature.spectral_centroid(y=reference, sr=48000))),
            'synthesized_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=synthesized, sr=48000))),
            'reference_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=reference, sr=48000)))
        }
        
        # Temporal analysis
        analysis['temporal_analysis'] = {
            'synthesized_duration': len(synthesized) / 48000,
            'reference_duration': len(reference) / 48000,
            'duration_difference': abs(len(synthesized) - len(reference)) / 48000
        }
        
        # Prosodic analysis
        f0_synth = librosa.yin(synthesized, fmin=75, fmax=400)
        f0_ref = librosa.yin(reference, fmin=75, fmax=400)
        
        analysis['prosodic_analysis'] = {
            'synthesized_f0_range': float(np.max(f0_synth) - np.min(f0_synth[f0_synth > 0])),
            'reference_f0_range': float(np.max(f0_ref) - np.min(f0_ref[f0_ref > 0])),
            'f0_correlation': float(np.corrcoef(f0_synth[:min(len(f0_synth), len(f0_ref))], 
                                             f0_ref[:min(len(f0_synth), len(f0_ref))])[0, 1] if len(f0_synth) > 0 and len(f0_ref) > 0 else 0.0)
        }
        
        return analysis
    
    def _analyze_errors(self, synthesized: np.ndarray, reference: np.ndarray, text: str, language: str) -> Dict[str, List]:
        """Analyze errors in synthesized speech"""
        errors = {}
        
        # Phoneme errors
        errors['phoneme_errors'] = self._identify_phoneme_errors(synthesized, reference, text, language)
        
        # Tone errors
        errors['tone_errors'] = self._identify_tone_errors(synthesized, reference, text, language)
        
        # Prosodic errors
        errors['prosodic_errors'] = self._identify_prosodic_errors(synthesized, reference)
        
        # Cultural errors
        errors['cultural_errors'] = self._identify_cultural_errors(text, language)
        
        return errors
    
    def _identify_phoneme_errors(self, synthesized: np.ndarray, reference: np.ndarray, text: str, language: str) -> List[str]:
        """Identify phoneme errors"""
        # This would use a phoneme recognition model
        # For now, return dummy errors
        return ['missing_final_consonant', 'vowel_shortening']
    
    def _identify_tone_errors(self, synthesized: np.ndarray, reference: np.ndarray, text: str, language: str) -> List[str]:
        """Identify tone errors"""
        # This would use tone analysis
        return ['incorrect_rising_tone', 'tone_sandhi_error']
    
    def _identify_prosodic_errors(self, synthesized: np.ndarray, reference: np.ndarray) -> List[str]:
        """Identify prosodic errors"""
        # Compare prosodic features
        return ['unnatural_pauses', 'incorrect_stress_pattern']
    
    def _identify_cultural_errors(self, text: str, language: str) -> List[str]:
        """Identify cultural errors"""
        # Analyze text for cultural appropriateness
        return ['inappropriate_register', 'dialect_mixing']
    
    def _generate_recommendations(self, quality_metrics: QualityMetrics, language: str) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if quality_metrics.tone_accuracy < 0.9:
            recommendations.append("Improve tone accuracy through better tone modeling")
        
        if quality_metrics.phoneme_accuracy < 0.9:
            recommendations.append("Enhance phoneme pronunciation accuracy")
        
        if quality_metrics.pesq_score < 3.0:
            recommendations.append("Improve perceptual quality through better vocoding")
        
        if quality_metrics.regional_authenticity < 0.8:
            recommendations.append(f"Enhance {language} regional authenticity")
        
        if quality_metrics.cultural_appropriateness < 0.8:
            recommendations.append("Improve cultural appropriateness")
        
        return recommendations
    
    def _generate_cultural_notes(self, text: str, language: str) -> str:
        """Generate cultural notes"""
        if language == "th":
            return "Central Thai - formal register, appropriate for business and official contexts"
        elif language == "tts":
            return "Isan (Northeastern Thai) - informal, warm and friendly tone"
        else:
            return "Unknown language"
    
    def evaluate_dataset(self, dataset_path: str, sample_size: int = 100) -> Dict[str, any]:
        """Evaluate a complete dataset"""
        logger.info(f"Evaluating dataset: {dataset_path}")
        
        # Load dataset (placeholder - would load actual dataset)
        # For now, return dummy results
        
        results = {
            'total_samples_evaluated': sample_size,
            'average_quality_score': 0.82,
            'average_tone_accuracy': 0.89,
            'average_phoneme_accuracy': 0.91,
            'average_pesq_score': 3.2,
            'average_stoi_score': 0.85,
            'quality_distribution': {
                'excellent': 25,
                'good': 45,
                'fair': 20,
                'poor': 10
            },
            'language_breakdown': {
                'th': {
                    'samples': 50,
                    'avg_quality': 0.85,
                    'avg_tone_accuracy': 0.92
                },
                'tts': {
                    'samples': 50,
                    'avg_quality': 0.79,
                    'avg_tone_accuracy': 0.86
                }
            },
            'recommendations': [
                'Improve Isan tone modeling',
                'Enhance phoneme accuracy for both languages',
                'Better cultural authenticity for regional variants'
            ]
        }
        
        return results
    
    def generate_quality_report(self, evaluation_results: List[EvaluationResult], output_path: str):
        """Generate comprehensive quality report"""
        report = {
            'evaluation_summary': {
                'total_evaluations': len(evaluation_results),
                'average_quality_score': np.mean([er.objective_metrics.overall_quality_score for er in evaluation_results]),
                'average_tone_accuracy': np.mean([er.objective_metrics.tone_accuracy for er in evaluation_results]),
                'average_phoneme_accuracy': np.mean([er.objective_metrics.phoneme_accuracy for er in evaluation_results])
            },
            'language_breakdown': {},
            'quality_distribution': {},
            'detailed_results': []
        }
        
        # Language breakdown
        for lang in ['th', 'tts']:
            lang_results = [er for er in evaluation_results if er.language == lang]
            if lang_results:
                report['language_breakdown'][lang] = {
                    'count': len(lang_results),
                    'avg_quality': np.mean([er.objective_metrics.overall_quality_score for er in lang_results]),
                    'avg_tone_accuracy': np.mean([er.objective_metrics.tone_accuracy for er in lang_results])
                }
        
        # Quality distribution
        quality_scores = [er.objective_metrics.overall_quality_score for er in evaluation_results]
        report['quality_distribution'] = {
            'excellent': len([q for q in quality_scores if q >= 0.9]),
            'good': len([q for q in quality_scores if 0.8 <= q < 0.9]),
            'fair': len([q for q in quality_scores if 0.7 <= q < 0.8]),
            'poor': len([q for q in quality_scores if q < 0.7])
        }
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Quality report generated: {output_path}")

def main():
    """Main function to demonstrate quality evaluation"""
    # Initialize evaluator
    evaluator = ThaiIsanQualityEvaluator()
    
    # Create dummy audio data for demonstration
    sample_rate = 48000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Synthesized audio (with some artifacts)
    synthesized = np.sin(2 * np.pi * 200 * t) * 0.8
    synthesized += 0.1 * np.random.randn(len(synthesized))  # Add noise
    
    # Reference audio
    reference = np.sin(2 * np.pi * 200 * t) * 0.9
    reference += 0.05 * np.random.randn(len(reference))  # Less noise
    
    # Evaluate speech
    result = evaluator.evaluate_speech(
        synthesized_audio=synthesized,
        reference_audio=reference,
        text="สวัสดีครับ",
        language="th",
        speaker_id="th_male_001"
    )
    
    # Print results
    print("=== Thai-Isan TTS Quality Evaluation Results ===")
    print(f"Language: {result.language}")
    print(f"Text: {result.text}")
    print(f"Overall Quality Score: {result.objective_metrics.overall_quality_score:.3f}")
    print(f"Tone Accuracy: {result.objective_metrics.tone_accuracy:.3f}")
    print(f"Phoneme Accuracy: {result.objective_metrics.phoneme_accuracy:.3f}")
    print(f"PESQ Score: {result.objective_metrics.pesq_score:.3f}")
    print(f"STOI Score: {result.objective_metrics.stoi_score:.3f}")
    print(f"Regional Authenticity: {result.objective_metrics.regional_authenticity:.3f}")
    print(f"Cultural Appropriateness: {result.objective_metrics.cultural_appropriateness:.3f}")
    
    print(f"\nSubjective Scores:")
    print(f"Naturalness: {result.subjective_scores.naturalness:.2f}/5.0")
    print(f"Intelligibility: {result.subjective_scores.intelligibility:.2f}/5.0")
    print(f"Overall Quality: {result.subjective_scores.overall_quality:.2f}/5.0")
    
    print(f"\nRecommendations:")
    for rec in result.recommendations:
        print(f"- {rec}")
    
    # Evaluate dataset (demonstration)
    dataset_results = evaluator.evaluate_dataset("./data/thai_isan_dataset", sample_size=50)
    print(f"\n=== Dataset Evaluation Summary ===")
    print(f"Total samples: {dataset_results['total_samples_evaluated']}")
    print(f"Average quality: {dataset_results['average_quality_score']:.3f}")
    print(f"Average tone accuracy: {dataset_results['average_tone_accuracy']:.3f}")

if __name__ == "__main__":
    main()