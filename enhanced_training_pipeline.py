"""
Advanced Thai-Isan TTS Model Training Pipeline
Comprehensive fine-tuning pipeline for bilingual Thai-Isan speech synthesis
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler, WeightedRandomSampler
from transformers import (
    TrainingArguments, 
    Trainer, 
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    AutoTokenizer,
    AutoModel
)
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import json
import os
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import wandb
from tqdm import tqdm
import librosa
import soundfile as sf
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchaudio
from torchaudio.transforms import MelSpectrogram, InverseMelScale

# Thai language processing
import pythainlp
from pythainlp.tokenize import word_tokenize, syllable_tokenize
from pythainlp.transliterate import romanize
from pythainlp.corpus import thai_words

# Custom modules
from thai_isan_config import TrainingConfig
from enhanced_data_collection import AudioRecording, SpeakerInfo

logger = logging.getLogger(__name__)

@dataclass
class ThaiIsanAudioFeatures:
    """Audio features for Thai and Isan speech"""
    mel_spectrogram: torch.Tensor
    fundamental_frequency: torch.Tensor
    energy: torch.Tensor
    phoneme_sequence: torch.Tensor
    tone_sequence: torch.Tensor
    duration: torch.Tensor
    speaker_embedding: torch.Tensor

@dataclass
class TrainingBatch:
    """Training batch data"""
    text_tokens: torch.Tensor
    text_lengths: torch.Tensor
    audio_features: ThaiIsanAudioFeatures
    audio_lengths: torch.Tensor
    speaker_ids: torch.Tensor
    language_ids: torch.Tensor
    tone_labels: torch.Tensor
    phoneme_labels: torch.Tensor

class ThaiIsanDataset(Dataset):
    """Dataset for Thai and Isan speech training"""
    
    def __init__(self, 
                 recordings: List[AudioRecording],
                 speakers: Dict[str, SpeakerInfo],
                 tokenizer: AutoTokenizer,
                 config: TrainingConfig,
                 is_training: bool = True):
        self.recordings = recordings
        self.speakers = speakers
        self.tokenizer = tokenizer
        self.config = config
        self.is_training = is_training
        
        # Audio processing
        self.mel_transform = MelSpectrogram(
            sample_rate=config.target_sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )
        
        # Preprocess data
        self.processed_data = self._preprocess_data()
        
    def _preprocess_data(self) -> List[Dict]:
        """Preprocess all recordings"""
        processed = []
        
        for recording in tqdm(self.recordings, desc="Preprocessing"):
            try:
                # Load audio
                audio, sr = librosa.load(recording.file_path, sr=self.config.target_sample_rate)
                
                # Extract features
                features = self._extract_audio_features(audio, sr)
                
                # Process text
                text_tokens = self._process_text(recording.text, recording.language)
                
                # Get speaker info
                speaker = self.speakers.get(recording.speaker_id)
                if not speaker:
                    continue
                
                # Create processed sample
                sample = {
                    'recording_id': recording.recording_id,
                    'text_tokens': text_tokens,
                    'audio_features': features,
                    'speaker_id': recording.speaker_id,
                    'language': recording.language,
                    'duration': recording.duration,
                    'quality_score': recording.quality_score
                }
                
                processed.append(sample)
                
            except Exception as e:
                logger.warning(f"Failed to process recording {recording.recording_id}: {e}")
                continue
        
        logger.info(f"Preprocessed {len(processed)} recordings")
        return processed
    
    def _extract_audio_features(self, audio: np.ndarray, sr: int) -> ThaiIsanAudioFeatures:
        """Extract audio features"""
        # Mel spectrogram
        mel_spec = self.mel_transform(torch.from_numpy(audio).float())
        
        # Fundamental frequency (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        f0 = torch.from_numpy(np.nan_to_num(f0, nan=0.0)).float()
        
        # Energy
        energy = torch.sqrt(torch.mean(mel_spec**2, dim=0))
        
        # Duration
        duration = torch.tensor(len(audio) / sr).float()
        
        # Placeholder for phoneme and tone sequences (would be extracted from text)
        phoneme_seq = torch.zeros(100).long()  # Max length
        tone_seq = torch.zeros(100).long()
        
        # Speaker embedding (placeholder)
        speaker_embedding = torch.randn(256)  # Would be learned
        
        return ThaiIsanAudioFeatures(
            mel_spectrogram=mel_spec,
            fundamental_frequency=f0,
            energy=energy,
            phoneme_sequence=phoneme_seq,
            tone_sequence=tone_seq,
            duration=duration,
            speaker_embedding=speaker_embedding
        )
    
    def _process_text(self, text: str, language: str) -> torch.Tensor:
        """Process text with language-specific handling"""
        if language == "th":
            # Thai text processing
            tokens = word_tokenize(text, engine='newmm')
        elif language == "tts":
            # Isan text processing (similar to Thai but with adjustments)
            tokens = word_tokenize(text, engine='newmm')
        else:
            tokens = list(text)
        
        # Tokenize with the model's tokenizer
        text_input = self.tokenizer(
            ' '.join(tokens),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        return text_input['input_ids'].squeeze(0)
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        sample = self.processed_data[idx]
        
        # Create training batch
        return TrainingBatch(
            text_tokens=sample['text_tokens'],
            text_lengths=torch.tensor(len(sample['text_tokens'])),
            audio_features=sample['audio_features'],
            audio_lengths=torch.tensor(sample['audio_features'].mel_spectrogram.shape[1]),
            speaker_ids=torch.tensor(hash(sample['speaker_id']) % 1000),
            language_ids=torch.tensor(0 if sample['language'] == 'th' else 1),
            tone_labels=torch.zeros(100),  # Placeholder
            phoneme_labels=torch.zeros(100)  # Placeholder
        )

class ThaiIsanTTSModel(nn.Module):
    """Advanced Thai-Isan TTS Model based on Qwen3-TTS architecture"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Text encoder (based on Qwen3-TTS)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=0.1
            ),
            num_layers=config.num_hidden_layers // 2
        )
        
        # Language-specific encoders
        self.thai_encoder = nn.Linear(config.hidden_size, config.hidden_size)
        self.isan_encoder = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Multi-scale duration predictor
        self.duration_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 1),
            nn.ReLU()
        )
        
        # Pitch predictor for tone handling
        self.pitch_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, config.num_isan_tones if config.num_isan_tones > config.num_thai_tones else config.num_thai_tones)
        )
        
        # Mel decoder
        self.mel_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=0.1
            ),
            num_layers=config.num_hidden_layers // 2
        )
        
        # Output layers
        self.mel_projection = nn.Linear(config.hidden_size, 80)  # 80 mel channels
        self.energy_projection = nn.Linear(config.hidden_size, 1)
        self.f0_projection = nn.Linear(config.hidden_size, 1)
        
        # Speaker embedding
        self.speaker_embedding = nn.Embedding(1000, config.hidden_size)
        
        # Language embedding
        self.language_embedding = nn.Embedding(config.num_languages, config.hidden_size)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(1000, config.hidden_size)
        
    def _create_positional_encoding(self, max_len: int, d_model: int):
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, batch: TrainingBatch) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Text encoding
        text_features = self.text_encoder(batch.text_tokens)
        
        # Add positional encoding
        seq_len = text_features.size(1)
        text_features = text_features + self.positional_encoding[:, :seq_len, :].to(text_features.device)
        
        # Language-specific processing
        if batch.language_ids[0] == 0:  # Thai
            text_features = self.thai_encoder(text_features)
        else:  # Isan
            text_features = self.isan_encoder(text_features)
        
        # Add speaker and language embeddings
        speaker_emb = self.speaker_embedding(batch.speaker_ids).unsqueeze(1)
        language_emb = self.language_embedding(batch.language_ids).unsqueeze(1)
        
        text_features = text_features + speaker_emb + language_emb
        
        # Duration prediction
        durations = self.duration_predictor(text_features).squeeze(-1)
        
        # Pitch prediction for tones
        pitch_logits = self.pitch_predictor(text_features)
        
        # Expand text features based on predicted durations
        expanded_features = self._expand_features(text_features, durations)
        
        # Mel decoding
        mel_features = self.mel_decoder(expanded_features, expanded_features)
        
        # Output projections
        mel_output = self.mel_projection(mel_features)
        energy_output = self.energy_projection(mel_features)
        f0_output = self.f0_projection(mel_features)
        
        return {
            'mel_output': mel_output,
            'energy_output': energy_output,
            'f0_output': f0_output,
            'durations': durations,
            'pitch_logits': pitch_logits
        }
    
    def _expand_features(self, features: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        """Expand features based on predicted durations"""
        batch_size, seq_len, hidden_size = features.size()
        
        expanded = []
        for b in range(batch_size):
            expanded_batch = []
            for t in range(seq_len):
                duration = int(durations[b, t].item())
                if duration > 0:
                    expanded_batch.append(features[b, t:t+1, :].repeat(duration, 1))
            
            if expanded_batch:
                expanded_seq = torch.cat(expanded_batch, dim=0)
                expanded.append(expanded_seq.unsqueeze(0))
        
        # Pad sequences to same length
        if expanded:
            max_len = max(seq.size(1) for seq in expanded)
            padded = []
            for seq in expanded:
                if seq.size(1) < max_len:
                    padding = torch.zeros(1, max_len - seq.size(1), hidden_size, device=seq.device)
                    seq = torch.cat([seq, padding], dim=1)
                padded.append(seq)
            
            return torch.cat(padded, dim=0)
        else:
            return torch.zeros(batch_size, 1, hidden_size, device=features.device)

class ThaiIsanTrainer:
    """Advanced trainer for Thai-Isan TTS model"""
    
    def __init__(self, model: ThaiIsanTTSModel, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize wandb for experiment tracking
        if config.enable_wandb_logging:
            wandb.init(project="thai-isn-tts", config=config.__dict__)
        
        # Loss functions
        self.mel_loss = nn.MSELoss()
        self.energy_loss = nn.MSELoss()
        self.f0_loss = nn.MSELoss()
        self.duration_loss = nn.MSELoss()
        self.pitch_loss = nn.CrossEntropyLoss()
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.num_epochs * 1000  # Placeholder
        )
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            outputs = self.model(batch)
            
            # Calculate losses
            loss = self._calculate_loss(outputs, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to wandb
            if self.config.enable_wandb_logging and num_batches % 10 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'train_loss': avg_loss}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Calculate loss
                loss = self._calculate_loss(outputs, batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Log to wandb
        if self.config.enable_wandb_logging:
            wandb.log({'val_loss': avg_loss})
        
        return {'val_loss': avg_loss}
    
    def _move_batch_to_device(self, batch: TrainingBatch) -> TrainingBatch:
        """Move batch to device"""
        return TrainingBatch(
            text_tokens=batch.text_tokens.to(self.device),
            text_lengths=batch.text_lengths.to(self.device),
            audio_features=batch.audio_features,  # Keep as is for now
            audio_lengths=batch.audio_lengths.to(self.device),
            speaker_ids=batch.speaker_ids.to(self.device),
            language_ids=batch.language_ids.to(self.device),
            tone_labels=batch.tone_labels.to(self.device),
            phoneme_labels=batch.phoneme_labels.to(self.device)
        )
    
    def _calculate_loss(self, outputs: Dict[str, torch.Tensor], batch: TrainingBatch) -> torch.Tensor:
        """Calculate total loss"""
        # Mel loss
        mel_loss = self.mel_loss(outputs['mel_output'], batch.audio_features.mel_spectrogram)
        
        # Energy loss
        energy_loss = self.energy_loss(outputs['energy_output'].squeeze(-1), batch.audio_features.energy)
        
        # F0 loss
        f0_loss = self.f0_loss(outputs['f0_output'].squeeze(-1), batch.audio_features.fundamental_frequency)
        
        # Duration loss
        duration_loss = self.duration_loss(outputs['durations'], torch.ones_like(outputs['durations']))
        
        # Pitch loss (for tone accuracy)
        pitch_loss = self.pitch_loss(outputs['pitch_logits'], batch.tone_labels[:outputs['pitch_logits'].size(1)])
        
        # Weighted total loss
        total_loss = (
            mel_loss * 1.0 +
            energy_loss * 0.5 +
            f0_loss * 0.5 +
            duration_loss * 0.3 +
            pitch_loss * self.config.tone_loss_weight
        )
        
        return total_loss
    
    def save_checkpoint(self, epoch: int, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded: {path}")
        return checkpoint['epoch']

def main():
    """Main training function"""
    # Initialize configuration
    config = TrainingConfig()
    
    # Create model
    model = ThaiIsanTTSModel(config)
    
    # Create trainer
    trainer = ThaiIsanTrainer(model, config)
    
    # Create datasets (placeholder - would load from actual data)
    dummy_recordings = []
    dummy_speakers = {}
    
    # Create dummy tokenizer (would use actual tokenizer)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    train_dataset = ThaiIsanDataset(dummy_recordings, dummy_speakers, tokenizer, config, is_training=True)
    val_dataset = ThaiIsanDataset(dummy_recordings, dummy_speakers, tokenizer, config, is_training=False)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_dataloader)
        
        # Validate
        val_metrics = trainer.validate(val_dataloader)
        
        logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}, Val Loss: {val_metrics['val_loss']:.4f}")
        
        # Save best checkpoint
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            trainer.save_checkpoint(epoch, "./models/thai_isan_tts_best.pt")
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(epoch, f"./models/thai_isan_tts_epoch_{epoch + 1}.pt")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()