"""
Thai-Isan TTS Model Training Pipeline
Advanced fine-tuning pipeline for bilingual Thai-Isan speech synthesis
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import (
    TrainingArguments, 
    Trainer, 
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
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
from sklearn.metrics import accuracy_score, mean_squared_error
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Thai language processing
import pythainlp
from pythainlp.tokenize import word_tokenize, syllable_tokenize
from pythainlp.transliterate import romanize

# Custom modules
from data_collection_pipeline import ThaiIsanDataCollector, AudioRecording, SpeakerInfo
from thai_isan_phoneme_mapping import (
    thai_to_isan_phoneme_conversion,
    thai_to_isan_tone_conversion,
    THAI_TONE_CLASSES,
    ISAN_TONE_RULES,
    ThaiIsanPhonemeMapper
)

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Advanced training configuration"""
    
    # Model architecture
    model_name: str = "thai-isan-tts"
    base_model_path: str = "Qwen/Qwen3-TTS"  # Base Qwen3-TTS model
    vocab_size: int = 151936
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    intermediate_size: int = 22016
    
    # Language-specific settings
    num_languages: int = 2  # Thai, Isan
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
    dataloader_num_workers: int = 8
    pin_memory: bool = True
    
    # Distributed training
    distributed_training: bool = False
    local_rank: int = -1
    world_size: int = 1
    
    # Output settings
    output_dir: str = "./output"
    logging_dir: str = "./logs"
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Weights & Biases logging
    use_wandb: bool = True
    wandb_project: str = "thai-isan-tts"
    wandb_entity: Optional[str] = None

class ThaiIsanTTSDataset(Dataset):
    """Advanced dataset for Thai-Isan TTS training"""
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        config: TrainingConfig = None,
        phoneme_mapper: ThaiIsanPhonemeMapper = None
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.config = config or TrainingConfig()
        self.phoneme_mapper = phoneme_mapper or ThaiIsanPhonemeMapper()
        
        # Load data
        self.recordings = self.load_recordings()
        self.speakers = self.load_speakers()
        
        # Initialize audio processing
        self.sample_rate = 48000
        self.hop_length = 512
        self.win_length = 2048
        self.n_mels = 80
        self.n_fft = 2048
        
        logger.info(f"Loaded {len(self.recordings)} recordings for {split} split")
    
    def load_recordings(self) -> Dict[str, AudioRecording]:
        """Load recording metadata"""
        recordings = {}
        
        # Load splits
        splits_path = self.data_path / "training_splits.json"
        if splits_path.exists():
            with open(splits_path, 'r', encoding='utf-8') as f:
                splits = json.load(f)
                recording_ids = splits.get(self.split, [])
        else:
            # If no splits exist, use all recordings
            recording_ids = []
            metadata_dir = self.data_path / "metadata"
            for metadata_file in metadata_dir.glob("*.json"):
                recording_ids.append(metadata_file.stem)
        
        # Load recording metadata
        for recording_id in recording_ids:
            metadata_path = self.data_path / "metadata" / f"{recording_id}.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    recording = AudioRecording(
                        recording_id=data['recording_id'],
                        speaker_id=data['speaker_id'],
                        text=data['text'],
                        language=data['language'],
                        duration=data['duration'],
                        sample_rate=data['sample_rate'],
                        bit_depth=data['bit_depth'],
                        channels=data['channels'],
                        file_size=data['file_size'],
                        file_path=data['file_path'],
                        quality_score=data['quality_score'],
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        background_noise_level=data['background_noise_level'],
                        signal_to_noise_ratio=data['signal_to_noise_ratio']
                    )
                    recordings[recording_id] = recording
        
        return recordings
    
    def load_speakers(self) -> Dict[str, SpeakerInfo]:
        """Load speaker information"""
        speakers = {}
        speakers_dir = self.data_path / "speakers"
        
        for speaker_dir in speakers_dir.iterdir():
            if speaker_dir.is_dir():
                speaker_info_path = speaker_dir / "speaker_info.json"
                if speaker_info_path.exists():
                    with open(speaker_info_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        speaker = SpeakerInfo(
                            speaker_id=data['speaker_id'],
                            language=data['language'],
                            gender=data['gender'],
                            age_group=data['age_group'],
                            region=data['region'],
                            dialect=data['dialect'],
                            native_language=data['native_language'],
                            years_speaking=data['years_speaking'],
                            recording_quality=data['recording_quality'],
                            accent_rating=data['accent_rating'],
                            clarity_rating=data['clarity_rating']
                        )
                        speakers[speaker.speaker_id] = speaker
        
        return speakers
    
    def __len__(self):
        return len(self.recordings)
    
    def __getitem__(self, idx):
        recording_ids = list(self.recordings.keys())
        recording_id = recording_ids[idx]
        recording = self.recordings[recording_id]
        
        # Load audio
        audio_path = recording.file_path
        if not os.path.exists(audio_path):
            # Try relative path
            audio_path = self.data_path / "recordings" / recording.language / f"{recording_id}.wav"
        
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            # Return dummy data
            audio = np.zeros(self.sample_rate * 3)  # 3 seconds of silence
            sr = self.sample_rate
        
        # Get speaker info
        speaker = self.speakers.get(recording.speaker_id)
        if speaker is None:
            # Create dummy speaker
            speaker = SpeakerInfo(
                speaker_id=recording.speaker_id,
                language=recording.language,
                gender="male",
                age_group="adult",
                region="central" if recording.language == "th" else "northeastern",
                dialect="standard",
                native_language=recording.language,
                years_speaking=25,
                recording_quality="professional",
                accent_rating=4.0,
                clarity_rating=4.0
            )
        
        # Process text
        text = recording.text
        language = recording.language
        
        # Convert to phonemes
        phonemes = self.text_to_phonemes(text, language)
        
        # Extract acoustic features
        mel_spectrogram = self.extract_mel_spectrogram(audio)
        
        # Data augmentation for training
        if self.split == "train" and self.config:
            audio = self.apply_data_augmentation(audio)
            mel_spectrogram = self.extract_mel_spectrogram(audio)
        
        # Create sample dictionary
        sample = {
            'recording_id': recording_id,
            'text': text,
            'phonemes': phonemes,
            'language': language,
            'speaker_id': recording.speaker_id,
            'speaker_gender': speaker.gender,
            'speaker_age_group': speaker.age_group,
            'speaker_region': speaker.region,
            'speaker_dialect': speaker.dialect,
            'audio': audio.astype(np.float32),
            'mel_spectrogram': mel_spectrogram.astype(np.float32),
            'duration': len(audio) / self.sample_rate,
            'sample_rate': self.sample_rate
        }
        
        return sample
    
    def text_to_phonemes(self, text: str, language: str) -> List[str]:
        """Convert text to phonemes"""
        # This is a simplified version - in practice, you'd use a proper G2P system
        if language == "th":
            # Thai phonemization
            words = word_tokenize(text, engine="newmm")
            phonemes = []
            for word in words:
                # Simplified phoneme conversion
                phoneme = self.phoneme_mapper.thai_word_to_phonemes(word)
                phonemes.extend(phoneme)
        else:  # Isan
            # Isan phonemization
            words = word_tokenize(text, engine="newmm")
            phonemes = []
            for word in words:
                # Convert to Isan phonemes
                thai_phonemes = self.phoneme_mapper.thai_word_to_phonemes(word)
                isan_phonemes = self.phoneme_mapper.thai_to_isan_phonemes(thai_phonemes)
                phonemes.extend(isan_phonemes)
        
        return phonemes
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram features"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
    
    def apply_data_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """Apply data augmentation techniques"""
        augmented = audio.copy()
        
        # Pitch shifting
        if np.random.random() < 0.3:
            n_steps = np.random.uniform(-self.config.pitch_shift_range, 
                                      self.config.pitch_shift_range)
            augmented = librosa.effects.pitch_shift(
                augmented, sr=self.sample_rate, n_steps=n_steps
            )
        
        # Time stretching
        if np.random.random() < 0.2:
            rate = np.random.uniform(1 - self.config.time_stretch_range, 
                                   1 + self.config.time_stretch_range)
            augmented = librosa.effects.time_stretch(augmented, rate=rate)
        
        # Noise addition
        if np.random.random() < self.config.noise_addition_prob:
            noise = np.random.normal(0, 0.01, len(augmented))
            augmented = augmented + noise
        
        # Volume variation
        if self.config.volume_augmentation and np.random.random() < 0.4:
            volume_factor = np.random.uniform(0.7, 1.3)
            augmented = augmented * volume_factor
        
        return augmented

class ThaiIsanTTSModel(nn.Module):
    """Advanced Thai-Isan TTS model with bilingual support"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Language embedding
        self.language_embedding = nn.Embedding(config.num_languages, config.hidden_size)
        
        # Speaker embedding
        self.speaker_embedding = nn.Embedding(1000, config.hidden_size)  # Max 1000 speakers
        
        # Phoneme encoder
        self.phoneme_embedding = nn.Embedding(config.phoneme_vocab_size, config.hidden_size)
        
        # Tone encoder
        self.tone_embedding = nn.Embedding(max(config.num_thai_tones, config.num_isan_tones), 
                                         config.hidden_size)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=0.1,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers // 2)
        
        # Prosody predictor
        self.prosody_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 3)  # pitch, energy, duration
        )
        
        # Duration predictor
        self.duration_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        # Mel decoder
        self.mel_decoder = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Output projection
        self.mel_projection = nn.Linear(config.hidden_size * 2, 80)  # 80 mel channels
        
        # Post-net for mel refinement
        self.post_net = nn.Sequential(
            nn.Conv1d(80, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(512, 80, kernel_size=5, padding=2)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        phonemes: torch.Tensor,
        language_ids: torch.Tensor,
        speaker_ids: torch.Tensor,
        tone_ids: torch.Tensor = None,
        mel_targets: torch.Tensor = None,
        duration_targets: torch.Tensor = None
    ):
        """Forward pass through the model"""
        batch_size, seq_len = phonemes.shape
        
        # Embeddings
        phoneme_embeds = self.phoneme_embedding(phonemes)
        language_embeds = self.language_embedding(language_ids).unsqueeze(1).expand(-1, seq_len, -1)
        speaker_embeds = self.speaker_embedding(speaker_ids).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine embeddings
        hidden_states = phoneme_embeds + language_embeds + speaker_embeds
        
        # Add tone embeddings if provided
        if tone_ids is not None:
            tone_embeds = self.tone_embedding(tone_ids)
            hidden_states = hidden_states + tone_embeds
        
        # Encode
        hidden_states = self.encoder(hidden_states)
        
        # Predict prosody features
        prosody_features = self.prosody_predictor(hidden_states)
        pitch_pred, energy_pred, duration_pred = torch.chunk(prosody_features, 3, dim=-1)
        
        # Refine duration prediction
        if duration_targets is not None:
            duration_pred = duration_targets
        
        # Expand hidden states based on duration
        expanded_states = self.expand_hidden_states(hidden_states, duration_pred.squeeze(-1))
        
        # Decode to mel spectrogram
        mel_outputs, _ = self.mel_decoder(expanded_states)
        mel_outputs = self.mel_projection(mel_outputs)
        
        # Apply post-net
        mel_outputs_post = mel_outputs + self.post_net(mel_outputs.transpose(1, 2)).transpose(1, 2)
        
        outputs = {
            'mel_outputs': mel_outputs,
            'mel_outputs_post': mel_outputs_post,
            'pitch_prediction': pitch_pred,
            'energy_prediction': energy_pred,
            'duration_prediction': duration_pred
        }
        
        return outputs
    
    def expand_hidden_states(self, hidden_states: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        """Expand hidden states based on predicted durations"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        expanded_list = []
        
        for b in range(batch_size):
            expanded_seq = []
            for t in range(seq_len):
                duration = int(durations[b, t].item())
                if duration > 0:
                    expanded_seq.append(hidden_states[b, t:t+1].repeat(duration, 1))
            
            if expanded_seq:
                expanded_states = torch.cat(expanded_seq, dim=0)
            else:
                expanded_states = hidden_states[b]
            
            expanded_list.append(expanded_states.unsqueeze(0))
        
        # Pad sequences to same length
        max_len = max(seq.shape[1] for seq in expanded_list)
        padded_list = []
        
        for seq in expanded_list:
            if seq.shape[1] < max_len:
                padding = torch.zeros(1, max_len - seq.shape[1], hidden_size, device=seq.device)
                seq = torch.cat([seq, padding], dim=1)
            padded_list.append(seq)
        
        return torch.cat(padded_list, dim=0)

class ThaiIsanTrainer:
    """Advanced trainer for Thai-Isan TTS model"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.optimizer = None
        self.scheduler = None
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=config.__dict__
            )
        
        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.logging_dir).mkdir(parents=True, exist_ok=True)
    
    def setup_model(self):
        """Initialize and setup model"""
        self.model = ThaiIsanTTSModel(self.config)
        self.model.to(self.device)
        
        # Setup distributed training if needed
        if self.config.distributed_training:
            self.model = DDP(self.model, device_ids=[self.config.local_rank])
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def setup_data(self, data_path: str):
        """Setup training and validation data"""
        # Create datasets
        self.train_dataset = ThaiIsanTTSDataset(
            data_path=data_path,
            split="train",
            config=self.config
        )
        
        self.val_dataset = ThaiIsanTTSDataset(
            data_path=data_path,
            split="validation",
            config=self.config
        )
        
        self.test_dataset = ThaiIsanTTSDataset(
            data_path=data_path,
            split="test",
            config=self.config
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self.collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self.collate_fn
        )
        
        logger.info(f"Data loaded - Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def collate_fn(self, batch):
        """Custom collate function for variable-length sequences"""
        # Extract sequences
        texts = [item['text'] for item in batch]
        phonemes = [item['phonemes'] for item in batch]
        languages = [item['language'] for item in batch]
        speakers = [item['speaker_id'] for item in batch]
        audios = [item['audio'] for item in batch]
        mel_specs = [item['mel_spectrogram'] for item in batch]
        
        # Convert to tensors
        # This is simplified - in practice, you'd pad sequences properly
        batch_dict = {
            'texts': texts,
            'phonemes': phonemes,
            'languages': languages,
            'speakers': speakers,
            'audios': audios,
            'mel_spectrograms': mel_specs
        }
        
        return batch_dict
    
    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        # Separate parameters for different learning rates
        language_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'language' in name:
                language_params.append(param)
            else:
                other_params.append(param)
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            [
                {'params': language_params, 'lr': self.config.learning_rate * 0.1},  # Lower LR for language embeddings
                {'params': other_params, 'lr': self.config.learning_rate}
            ],
            weight_decay=self.config.weight_decay
        )
        
        # Create scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
    
    def compute_loss(self, outputs, targets, language_ids):
        """Compute multi-task loss"""
        # Mel spectrogram loss
        mel_loss = nn.MSELoss()(outputs['mel_outputs'], targets['mel_spectrograms'])
        mel_post_loss = nn.MSELoss()(outputs['mel_outputs_post'], targets['mel_spectrograms'])
        
        # Prosody prediction loss
        pitch_loss = nn.MSELoss()(outputs['pitch_prediction'], targets.get('pitch', torch.zeros_like(outputs['pitch_prediction'])))
        energy_loss = nn.MSELoss()(outputs['energy_prediction'], targets.get('energy', torch.zeros_like(outputs['energy_prediction'])))
        
        # Duration prediction loss
        duration_loss = nn.MSELoss()(outputs['duration_prediction'], targets.get('duration', torch.zeros_like(outputs['duration_prediction'])))
        
        # Language-specific loss weighting
        thai_weight = self.config.thai_loss_weight
        isan_weight = self.config.isan_loss_weight
        
        # Apply language-specific weights
        total_loss = 0
        for i, lang_id in enumerate(language_ids):
            if lang_id == 0:  # Thai
                weight = thai_weight
            else:  # Isan
                weight = isan_weight
            
            total_loss += weight * (mel_loss + mel_post_loss + 
                                  self.config.tone_loss_weight * pitch_loss +
                                  self.config.phoneme_loss_weight * energy_loss +
                                  duration_loss)
        
        return total_loss / len(language_ids)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            # This is simplified - in practice, you'd properly handle device placement
            
            # Forward pass
            outputs = self.model(
                phonemes=batch['phonemes'],
                language_ids=batch['languages'],
                speaker_ids=batch['speakers']
            )
            
            # Compute loss
            loss = self.compute_loss(outputs, batch, batch['languages'])
            
            # Backward pass
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Logging
            if batch_idx % self.config.logging_steps == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                if self.config.use_wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/epoch': epoch
                    })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {'loss': avg_loss}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}"):
                # Forward pass
                outputs = self.model(
                    phonemes=batch['phonemes'],
                    language_ids=batch['languages'],
                    speaker_ids=batch['speakers']
                )
                
                # Compute loss
                loss = self.compute_loss(outputs, batch, batch['languages'])
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Log validation metrics
        logger.info(f"Validation Epoch {epoch}, Loss: {avg_loss:.4f}")
        if self.config.use_wandb:
            wandb.log({
                'val/loss': avg_loss,
                'val/epoch': epoch
            })
        
        return {'loss': avg_loss}
    
    def train(self, data_path: str):
        """Main training loop"""
        # Setup
        self.setup_model()
        self.setup_data(data_path)
        self.setup_optimizer_and_scheduler()
        
        logger.info("Starting training...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Save checkpoint
            if epoch % self.config.save_steps == 0:
                checkpoint_path = f"{self.config.output_dir}/checkpoint_epoch_{epoch}"
                self.save_checkpoint(checkpoint_path, epoch, val_metrics['loss'])
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save best model
                best_model_path = f"{self.config.output_dir}/best_model"
                self.save_checkpoint(best_model_path, epoch, val_metrics['loss'])
            else:
                patience_counter += 1
                
            if patience_counter >= 10:  # Early stopping patience
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

def train_thai_isan_tts(
    data_path: str = "./data",
    output_dir: str = "./output",
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 2e-5
) -> str:
    """
    Train Thai-Isan TTS model
    
    Args:
        data_path: Path to training data
        output_dir: Output directory for model checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    
    Returns:
        Path to best model checkpoint
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create training configuration
    config = TrainingConfig(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_wandb=True
    )
    
    # Create trainer
    trainer = ThaiIsanTrainer(config)
    
    # Train model
    trainer.train(data_path)
    
    # Return path to best model
    best_model_path = f"{output_dir}/best_model"
    return best_model_path

if __name__ == "__main__":
    # Example usage
    best_model = train_thai_isan_tts(
        data_path="./data",
        output_dir="./output",
        num_epochs=50,
        batch_size=16,
        learning_rate=2e-5
    )
    print(f"Training completed. Best model saved at: {best_model}")