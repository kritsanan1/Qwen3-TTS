"""
Thai-Isan TTS Evaluation Framework
Comprehensive evaluation metrics for Thai and Isan speech synthesis
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import librosa
import soundfile as sf
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import jiwer
from dataclasses import dataclass
from collections import defaultdict

from thai_isan_extension_config import (
    ThaiIsanExtensionConfig,
    get_tone_system,
    get_evaluation_config
)
from thai_isan_phoneme_mapping import (
    THAI_TONE_CLASSES,
    THAI_TONE_RULES,
    ISAN_TONE_RULES,
    ISAN_VOCABULARY_DIFFERENCES
)


@dataclass
class EvaluationMetrics:
    """Container for evaluation results"""
    objective_metrics: Dict[str, float]
    subjective_scores: Dict[str, float]
    language_specific_metrics: Dict[str, float]
    detailed_results: Dict[str, List]


class ThaiIsanTTSEvaluator:
    """Comprehensive evaluator for Thai-Isan TTS"""
    
    def __init__(self, config: Optional[ThaiIsanExtensionConfig] = None):
        self.config = config or ThaiIsanExtensionConfig()
        self.eval_config = get_evaluation_config()
        
        # Initialize evaluation components
        self.tone_analyzer = ToneAccuracyAnalyzer(self.config)
        self.phoneme_analyzer = PhonemeAccuracyAnalyzer(self.config)
        self.speech_quality_analyzer = SpeechQualityAnalyzer(self.config)
        self.naturalness_evaluator = NaturalnessEvaluator(self.config)
        
    def evaluate(
        self,
        synthesized_speech: np.ndarray,
        reference_speech: np.ndarray,
        text: str,
        language: str = "th",
        sample_rate: int = 24000
    ) -> EvaluationMetrics:
        """Comprehensive evaluation of synthesized speech"""
        
        # Objective metrics
        objective_metrics = self.compute_objective_metrics(
            synthesized_speech, reference_speech, sample_rate
        )
        
        # Language-specific metrics
        language_metrics = self.compute_language_specific_metrics(
            synthesized_speech, text, language, sample_rate
        )
        
        # Subjective metrics (simulated with models)
        subjective_metrics = self.estimate_subjective_scores(
            synthesized_speech, reference_speech, language
        )
        
        # Detailed results
        detailed_results = self.gather_detailed_results(
            synthesized_speech, reference_speech, text, language
        )
        
        return EvaluationMetrics(
            objective_metrics=objective_metrics,
            subjective_scores=subjective_metrics,
            language_specific_metrics=language_metrics,
            detailed_results=detailed_results
        )
    
    def compute_objective_metrics(
        self,
        synthesized: np.ndarray,
        reference: np.ndarray,
        sample_rate: int
    ) -> Dict[str, float]:
        """Compute standard objective metrics"""
        
        # Ensure same length
        min_len = min(len(synthesized), len(reference))
        synthesized = synthesized[:min_len]
        reference = reference[:min_len]
        
        # Mel-Cepstral Distortion (MCD)
        mcd = self.compute_mcd(synthesized, reference, sample_rate)
        
        # F0 correlation
        f0_corr = self.compute_f0_correlation(synthesized, reference, sample_rate)
        
        # Spectral features
        spectral_metrics = self.compute_spectral_metrics(synthesized, reference, sample_rate)
        
        # Energy correlation
        energy_corr = self.compute_energy_correlation(synthesized, reference)
        
        return {
            "mcd": mcd,
            "f0_correlation": f0_corr,
            "spectral_convergence": spectral_metrics["spectral_convergence"],
            "energy_correlation": energy_corr,
            "snr": self.compute_snr(synthesized, reference)
        }
    
    def compute_language_specific_metrics(
        self,
        synthesized: np.ndarray,
        text: str,
        language: str,
        sample_rate: int
    ) -> Dict[str, float]:
        """Compute language-specific metrics"""
        
        # Tone accuracy
        tone_accuracy = self.tone_analyzer.analyze_tone_accuracy(
            synthesized, text, language, sample_rate
        )
        
        # Phoneme accuracy
        phoneme_accuracy = self.phoneme_analyzer.analyze_phoneme_accuracy(
            synthesized, text, language, sample_rate
        )
        
        # Regional accent accuracy (for Isan)
        if language == "tts":
            regional_accuracy = self.analyze_regional_accuracy(
                synthesized, text, sample_rate
            )
        else:
            regional_accuracy = 0.0
        
        # Vocabulary appropriateness
        vocab_score = self.analyze_vocabulary_appropriateness(text, language)
        
        return {
            "tone_accuracy": tone_accuracy,
            "phoneme_accuracy": phoneme_accuracy,
            "regional_accuracy": regional_accuracy,
            "vocabulary_score": vocab_score
        }
    
    def estimate_subjective_scores(
        self,
        synthesized: np.ndarray,
        reference: Optional[np.ndarray],
        language: str
    ) -> Dict[str, float]:
        """Estimate subjective scores using predictive models"""
        
        # Naturalness prediction
        naturalness = self.naturalness_evaluator.predict_naturalness(
            synthesized, language
        )
        
        # Intelligibility prediction
        intelligibility = self.predict_intelligibility(synthesized, language)
        
        # Authenticity (cultural appropriateness)
        authenticity = self.predict_authenticity(synthesized, language)
        
        # Emotional expressiveness
        emotion_score = self.predict_emotion_score(synthesized, language)
        
        return {
            "naturalness": naturalness,
            "intelligibility": intelligibility,
            "authenticity": authenticity,
            "emotion_score": emotion_score
        }
    
    def compute_mcd(self, synthesized: np.ndarray, reference: np.ndarray, sr: int) -> float:
        """Compute Mel-Cepstral Distortion"""
        
        # Extract MFCC features
        mfcc_syn = librosa.feature.mfcc(y=synthesized, sr=sr, n_mfcc=13)
        mfcc_ref = librosa.feature.mfcc(y=reference, sr=sr, n_mfcc=13)
        
        # Dynamic time warping alignment
        from dtw import dtw
        distance, _, _, _ = dtw(mfcc_syn.T, mfcc_ref.T)
        
        # Normalize by number of frames
        mcd = distance / len(mfcc_syn.T)
        
        return float(mcd)
    
    def compute_f0_correlation(self, synthesized: np.ndarray, reference: np.ndarray, sr: int) -> float:
        """Compute F0 correlation"""
        
        # Extract F0
        f0_syn, _ = librosa.piptrack(y=synthesized, sr=sr)
        f0_ref, _ = librosa.piptrack(y=reference, sr=sr)
        
        # Get dominant F0 track
        f0_syn_track = f0_syn[f0_syn > 0].mean() if f0_syn[f0_syn > 0].size > 0 else 0
        f0_ref_track = f0_ref[f0_ref > 0].mean() if f0_ref[f0_ref > 0].size > 0 else 0
        
        # Compute correlation
        if f0_syn_track > 0 and f0_ref_track > 0:
            correlation = 1 - abs(f0_syn_track - f0_ref_track) / max(f0_syn_track, f0_ref_track)
        else:
            correlation = 0.0
        
        return float(correlation)
    
    def compute_spectral_metrics(self, synthesized: np.ndarray, reference: np.ndarray, sr: int) -> Dict[str, float]:
        """Compute spectral convergence and other spectral metrics"""
        
        # Compute spectrograms
        D_syn = librosa.stft(synthesized)
        D_ref = librosa.stft(reference)
        
        # Spectral convergence
        mag_syn = np.abs(D_syn)
        mag_ref = np.abs(D_ref)
        
        # Normalize
        mag_syn_norm = mag_syn / (mag_syn.max() + 1e-8)
        mag_ref_norm = mag_ref / (mag_ref.max() + 1e-8)
        
        # Spectral convergence (1 - L2 distance)
        spectral_conv = 1 - np.linalg.norm(mag_syn_norm - mag_ref_norm) / np.linalg.norm(mag_ref_norm)
        
        return {
            "spectral_convergence": float(spectral_conv)
        }
    
    def compute_energy_correlation(self, synthesized: np.ndarray, reference: np.ndarray) -> float:
        """Compute energy correlation"""
        
        # Compute frame energies
        frame_length = 1024
        hop_length = 512
        
        energy_syn = librosa.feature.rms(y=synthesized, frame_length=frame_length, hop_length=hop_length)[0]
        energy_ref = librosa.feature.rms(y=reference, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Normalize
        energy_syn_norm = (energy_syn - energy_syn.mean()) / (energy_syn.std() + 1e-8)
        energy_ref_norm = (energy_ref - energy_ref.mean()) / (energy_ref.std() + 1e-8)
        
        # Align lengths
        min_len = min(len(energy_syn_norm), len(energy_ref_norm))
        energy_syn_norm = energy_syn_norm[:min_len]
        energy_ref_norm = energy_ref_norm[:min_len]
        
        # Compute correlation
        correlation = np.corrcoef(energy_syn_norm, energy_ref_norm)[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def compute_snr(self, synthesized: np.ndarray, reference: np.ndarray) -> float:
        """Compute Signal-to-Noise Ratio"""
        
        # Ensure same length
        min_len = min(len(synthesized), len(reference))
        synthesized = synthesized[:min_len]
        reference = reference[:min_len]
        
        # Compute SNR
        signal_power = np.mean(reference ** 2)
        noise_power = np.mean((synthesized - reference) ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')
        
        return float(snr)
    
    def analyze_regional_accuracy(self, synthesized: np.ndarray, text: str, sr: int) -> float:
        """Analyze regional accent accuracy for Isan"""
        
        # This would involve sophisticated accent classification
        # Placeholder implementation
        
        # Extract prosodic features
        f0, voiced_flag, voiced_probs = librosa.pyin(
            synthesized, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        
        # Analyze F0 patterns (simplified)
        if f0[voiced_flag].size > 0:
            f0_mean = np.mean(f0[voiced_flag])
            f0_std = np.std(f0[voiced_flag])
            
            # Simple heuristic for Isan accent (higher F0 variation)
            if f0_std > 0.3 * f0_mean:
                regional_score = 0.8
            else:
                regional_score = 0.6
        else:
            regional_score = 0.5
        
        return regional_score
    
    def analyze_vocabulary_appropriateness(self, text: str, language: str) -> float:
        """Check if vocabulary is appropriate for the language"""
        
        if language == "tts":
            # Check for Isan-specific vocabulary
            isan_words = 0
            total_words = len(text.split())
            
            for word in text.split():
                if word in ISAN_VOCABULARY_DIFFERENCES.values():
                    isan_words += 1
            
            vocab_score = isan_words / total_words if total_words > 0 else 0.0
        else:
            vocab_score = 1.0  # Assume appropriate for Thai
        
        return vocab_score
    
    def predict_intelligibility(self, synthesized: np.ndarray, language: str) -> float:
        """Predict intelligibility score"""
        
        # Extract clarity features
        spectral_centroid = librosa.feature.spectral_centroid(y=synthesized)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=synthesized)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(synthesized)[0]
        
        # Simple heuristic based on feature variation
        clarity_score = (
            np.std(spectral_centroid) / (np.mean(spectral_centroid) + 1e-8) +
            np.std(spectral_rolloff) / (np.mean(spectral_rolloff) + 1e-8) +
            np.std(zero_crossing_rate) / (np.mean(zero_crossing_rate) + 1e-8)
        ) / 3
        
        # Convert to 1-5 scale
        intelligibility = min(5.0, max(1.0, clarity_score * 5))
        
        return intelligibility
    
    def predict_authenticity(self, synthesized: np.ndarray, language: str) -> float:
        """Predict cultural authenticity"""
        
        # Analyze prosodic patterns
        f0, voiced_flag, voiced_probs = librosa.pyin(
            synthesized, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        
        if f0[voiced_flag].size > 0:
            # Analyze tonal patterns
            f0_contour = f0[voiced_flag]
            
            # Simple authenticity heuristic based on tonal variation
            if language == "tts":
                # Isan should have more tonal variation
                tonal_variation = np.std(f0_contour) / (np.mean(f0_contour) + 1e-8)
                authenticity = min(5.0, max(1.0, tonal_variation * 8))
            else:
                authenticity = 4.0  # Default for Thai
        else:
            authenticity = 3.0
        
        return authenticity
    
    def predict_emotion_score(self, synthesized: np.ndarray, language: str) -> float:
        """Predict emotional expressiveness"""
        
        # Extract emotion-related features
        energy = librosa.feature.rms(y=synthesized)[0]
        f0, voiced_flag, voiced_probs = librosa.pyin(
            synthesized, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        
        # Simple emotion heuristic based on energy and F0 variation
        if energy.size > 0 and f0[voiced_flag].size > 0:
            energy_variation = np.std(energy) / (np.mean(energy) + 1e-8)
            f0_variation = np.std(f0[voiced_flag]) / (np.mean(f0[voiced_flag]) + 1e-8)
            
            emotion_score = (energy_variation + f0_variation) / 2 * 5
            emotion_score = min(5.0, max(1.0, emotion_score))
        else:
            emotion_score = 3.0
        
        return emotion_score
    
    def gather_detailed_results(
        self,
        synthesized: np.ndarray,
        reference: Optional[np.ndarray],
        text: str,
        language: str
    ) -> Dict[str, List]:
        """Gather detailed analysis results"""
        
        results = {
            "frame_analysis": [],
            "tone_analysis": [],
            "phoneme_errors": [],
            "prosodic_features": []
        }
        
        # Frame-level analysis
        # This would include detailed frame-by-frame analysis
        
        # Tone analysis
        if language in ["th", "tts"]:
            tone_analysis = self.tone_analyzer.detailed_tone_analysis(
                synthesized, text, language
            )
            results["tone_analysis"] = tone_analysis
        
        # Phoneme error analysis
        phoneme_errors = self.phoneme_analyzer.detailed_phoneme_analysis(
            synthesized, text, language
        )
        results["phoneme_errors"] = phoneme_errors
        
        # Prosodic feature analysis
        prosodic_features = self.extract_prosodic_features(synthesized)
        results["prosodic_features"] = prosodic_features
        
        return results
    
    def extract_prosodic_features(self, audio: np.ndarray) -> List[Dict]:
        """Extract prosodic features for analysis"""
        
        features = []
        
        # F0 extraction
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        
        # Energy
        energy = librosa.feature.rms(y=audio)[0]
        
        # Speaking rate (simplified)
        tempo, _ = librosa.beat.beat_track(y=audio)
        
        # Segment into voiced regions
        voiced_regions = []
        current_start = None
        
        for i, is_voiced in enumerate(voiced_flag):
            if is_voiced and current_start is None:
                current_start = i
            elif not is_voiced and current_start is not None:
                voiced_regions.append({
                    "start": current_start,
                    "end": i,
                    "f0_mean": np.mean(f0[current_start:i][f0[current_start:i] > 0]),
                    "f0_std": np.std(f0[current_start:i][f0[current_start:i] > 0])
                })
                current_start = None
        
        # Create feature summary
        if voiced_regions:
            avg_f0_mean = np.mean([r["f0_mean"] for r in voiced_regions if not np.isnan(r["f0_mean"])])
            avg_f0_std = np.mean([r["f0_std"] for r in voiced_regions if not np.isnan(r["f0_std"])])
            
            features.append({
                "type": "prosodic_summary",
                "avg_f0_mean": avg_f0_mean,
                "avg_f0_std": avg_f0_std,
                "tempo": tempo,
                "energy_mean": np.mean(energy),
                "energy_std": np.std(energy),
                "voiced_regions": len(voiced_regions)
            })
        
        return features


class ToneAccuracyAnalyzer:
    """Specialized analyzer for tone accuracy in Thai and Isan"""
    
    def __init__(self, config: ThaiIsanExtensionConfig):
        self.config = config
        
    def analyze_tone_accuracy(self, audio: np.ndarray, text: str, language: str, sr: int) -> float:
        """Analyze tone accuracy for the given audio"""
        
        # Extract F0 contour
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        
        # Parse expected tones from text
        expected_tones = self.parse_expected_tones(text, language)
        
        # Extract actual tones from audio
        actual_tones = self.extract_tones_from_audio(f0, voiced_flag, language)
        
        # Compare and compute accuracy
        accuracy = self.compare_tones(expected_tones, actual_tones)
        
        return accuracy
    
    def parse_expected_tones(self, text: str, language: str) -> List[str]:
        """Parse expected tones from text"""
        # This would involve sophisticated text analysis
        # Placeholder implementation
        
        if language == "th":
            tones = self.parse_thai_tones(text)
        elif language == "tts":
            tones = self.parse_isan_tones(text)
        else:
            tones = []
        
        return tones
    
    def parse_thai_tones(self, text: str) -> List[str]:
        """Parse Thai tones from text"""
        tones = []
        
        # Simple tone parsing based on tone marks
        for char in text:
            if char == '่':
                tones.append("low")
            elif char == '้':
                tones.append("falling")
            elif char == '๊':
                tones.append("high")
            elif char == '๋':
                tones.append("rising")
            else:
                # Default tone based on consonant class
                if self.get_consonant_class(char) in THAI_TONE_CLASSES["mid_class"]:
                    tones.append("mid")
                else:
                    tones.append("mid")  # Simplified
        
        return tones
    
    def parse_isan_tones(self, text: str) -> List[str]:
        """Parse Isan tones from text"""
        # Similar to Thai but with different mapping
        return self.parse_thai_tones(text)  # Simplified
    
    def extract_tones_from_audio(self, f0: np.ndarray, voiced_flag: np.ndarray, language: str) -> List[str]:
        """Extract tones from F0 contour"""
        tones = []
        
        # Segment into syllable-sized chunks (simplified)
        for i, is_voiced in enumerate(voiced_flag):
            if is_voiced and f0[i] > 0:
                # Classify F0 into tone categories
                tone = self.classify_f0_to_tone(f0[i], language)
                tones.append(tone)
        
        return tones
    
    def classify_f0_to_tone(self, f0_value: float, language: str) -> str:
        """Classify F0 value to tone category"""
        # This is a simplified classification
        # Real implementation would use trained models
        
        if language == "th":
            # Thai tone classification
            if f0_value < 150:
                return "low"
            elif f0_value < 200:
                return "mid"
            elif f0_value < 250:
                return "falling"
            elif f0_value < 300:
                return "rising"
            else:
                return "high"
        else:
            # Isan tone classification
            return "mid"  # Simplified
    
    def compare_tones(self, expected: List[str], actual: List[str]) -> float:
        """Compare expected and actual tones"""
        if not expected or not actual:
            return 0.0
        
        # Align sequences (simplified)
        min_len = min(len(expected), len(actual))
        expected = expected[:min_len]
        actual = actual[:min_len]
        
        # Compute accuracy
        correct = sum(1 for e, a in zip(expected, actual) if e == a)
        accuracy = correct / len(expected) if len(expected) > 0 else 0.0
        
        return accuracy
    
    def get_consonant_class(self, char: str) -> str:
        """Get consonant class for Thai character"""
        for class_name, consonants in THAI_TONE_CLASSES.items():
            if char in consonants:
                return class_name
        return "mid_class"  # Default
    
    def detailed_tone_analysis(self, audio: np.ndarray, text: str, language: str) -> List[Dict]:
        """Perform detailed tone analysis"""
        
        # Extract F0
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        
        # Analyze each voiced segment
        analysis_results = []
        
        for i, is_voiced in enumerate(voiced_flag):
            if is_voiced and f0[i] > 0:
                result = {
                    "frame": i,
                    "f0": f0[i],
                    "voiced_probability": voiced_probs[i],
                    "predicted_tone": self.classify_f0_to_tone(f0[i], language)
                }
                analysis_results.append(result)
        
        return analysis_results


class PhonemeAccuracyAnalyzer:
    """Analyzer for phoneme accuracy"""
    
    def __init__(self, config: ThaiIsanExtensionConfig):
        self.config = config
    
    def analyze_phoneme_accuracy(self, audio: np.ndarray, text: str, language: str, sr: int) -> float:
        """Analyze phoneme accuracy"""
        
        # This would involve ASR or forced alignment
        # Placeholder implementation
        
        # Simulate phoneme accuracy based on audio quality
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        
        # Quality indicators
        spectral_clarity = np.std(spectral_centroid) / (np.mean(spectral_centroid) + 1e-8)
        rolloff_stability = 1.0 / (np.std(spectral_rolloff) + 1e-8)
        
        # Estimate phoneme accuracy based on quality
        phoneme_accuracy = min(1.0, (spectral_clarity + rolloff_stability) / 2)
        
        return phoneme_accuracy
    
    def detailed_phoneme_analysis(self, audio: np.ndarray, text: str, language: str) -> List[Dict]:
        """Perform detailed phoneme-level analysis"""
        
        # This would involve detailed phoneme recognition
        # Placeholder implementation
        
        phoneme_results = []
        
        # Simulate phoneme errors
        for i, char in enumerate(text):
            if char.strip():  # Non-space characters
                phoneme_result = {
                    "position": i,
                    "character": char,
                    "predicted_phoneme": f"phoneme_{i}",
                    "confidence": np.random.uniform(0.7, 0.95),  # Simulated confidence
                    "error_type": np.random.choice(["substitution", "deletion", "insertion", "correct"], p=[0.1, 0.05, 0.05, 0.8])
                }
                phoneme_results.append(phoneme_result)
        
        return phoneme_results


class SpeechQualityAnalyzer:
    """Analyzer for general speech quality"""
    
    def __init__(self, config: ThaiIsanExtensionConfig):
        self.config = config
    
    def analyze_quality(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze speech quality"""
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        
        # Temporal features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Quality metrics
        quality_metrics = {
            "spectral_clarity": np.std(spectral_centroid) / (np.mean(spectral_centroid) + 1e-8),
            "spectral_stability": 1.0 / (np.std(spectral_rolloff) + 1e-8),
            "bandwidth_consistency": 1.0 / (np.std(spectral_bandwidth) + 1e-8),
            "temporal_regularity": 1.0 / (np.std(zero_crossing_rate) + 1e-8)
        }
        
        return quality_metrics


class NaturalnessEvaluator:
    """Evaluator for naturalness of synthesized speech"""
    
    def __init__(self, config: ThaiIsanExtensionConfig):
        self.config = config
    
    def predict_naturalness(self, audio: np.ndarray, language: str) -> float:
        """Predict naturalness score (1-5)"""
        
        # Extract features relevant to naturalness
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        
        energy = librosa.feature.rms(y=audio)[0]
        
        # Naturalness indicators
        if f0[voiced_flag].size > 0 and energy.size > 0:
            # F0 variation (natural speech has variation)
            f0_variation = np.std(f0[voiced_flag]) / (np.mean(f0[voiced_flag]) + 1e-8)
            
            # Energy variation
            energy_variation = np.std(energy) / (np.mean(energy) + 1e-8)
            
            # Voicing consistency
            voicing_ratio = np.sum(voiced_flag) / len(voiced_flag)
            
            # Predict naturalness based on features
            naturalness = (
                min(1.0, f0_variation * 3) * 0.4 +
                min(1.0, energy_variation * 2) * 0.3 +
                voicing_ratio * 0.3
            ) * 5  # Scale to 1-5
            
            naturalness = min(5.0, max(1.0, naturalness))
        else:
            naturalness = 3.0  # Default
        
        return naturalness


# Test sentences for evaluation
TEST_SENTENCES = {
    "th": [
        {"text": "สวัสดีครับ", "description": "Greeting with polite particle"},
        {"text": "น้ำเย็น", "description": "Cold water"},
        {"text": "ไปกินข้าว", "description": "Go eat rice"},
        {"text": "ร้อนมาก", "description": "Very hot"},
        {"text": "ขอบคุณมาก", "description": "Thank you very much"}
    ],
    "tts": [
        {"text": "สบายดีบ่", "description": "How are you? (Isan)"},
        {"text": "เฮาไปกินข้าว", "description": "We go eat rice (Isan)"},
        {"text": "ฮ้อนมากบ่", "description": "Very hot (Isan)"},
        {"text": "เจ้าไปไส", "description": "Where are you going? (Isan)"},
        {"text": "ขอบคุณเด้อ", "description": "Thank you (Isan polite)"}
    ]
}


def evaluate_thai_isan_tts(
    synthesized_audio: np.ndarray,
    reference_audio: Optional[np.ndarray],
    text: str,
    language: str = "th",
    sample_rate: int = 24000
) -> Dict[str, float]:
    """
    Convenience function to evaluate Thai-Isan TTS
    
    Args:
        synthesized_audio: Generated speech audio
        reference_audio: Reference speech audio (optional)
        text: Input text
        language: Language code ("th" or "tts")
        sample_rate: Audio sample rate
    
    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = ThaiIsanTTSEvaluator()
    
    results = evaluator.evaluate(
        synthesized_audio,
        reference_audio,
        text,
        language,
        sample_rate
    )
    
    # Combine all metrics
    all_metrics = {}
    all_metrics.update(results.objective_metrics)
    all_metrics.update(results.subjective_scores)
    all_metrics.update(results.language_specific_metrics)
    
    return all_metrics


# Example usage
if __name__ == "__main__":
    # Create test audio (synthetic)
    sample_rate = 24000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Synthetic speech-like signal
    synthesized = np.sin(2 * np.pi * 440 * t) * np.exp(-t * 2)
    synthesized += 0.1 * np.random.randn(len(synthesized))
    
    reference = np.sin(2 * np.pi * 440 * t) * np.exp(-t * 2)
    
    # Test evaluation
    text = "สวัสดีครับ"
    language = "th"
    
    metrics = evaluate_thai_isan_tts(
        synthesized,
        reference,
        text,
        language,
        sample_rate
    )
    
    print("Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Test with Isan
    text_isan = "สบายดีบ่"
    language_isan = "tts"
    
    metrics_isan = evaluate_thai_isan_tts(
        synthesized,
        reference,
        text_isan,
        language_isan,
        sample_rate
    )
    
    print("\nIsan Evaluation Results:")
    for metric, value in metrics_isan.items():
        print(f"{metric}: {value:.4f}")