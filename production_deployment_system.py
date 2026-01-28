"""
Production-Ready Thai-Isan TTS Deployment System
Enterprise-grade deployment infrastructure for Thai and Isan speech synthesis
"""

import os
import json
import logging
import asyncio
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import psutil

# Web framework
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Model serving
import onnxruntime as ort
import torch
from transformers import AutoTokenizer, AutoModel

# Audio processing
import librosa
import soundfile as sf
import io
import base64

# Monitoring and logging
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import logging.handlers
from pythonjsonlogger import jsonlogger

# Cache
import redis
from functools import lru_cache

# Security
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta

# Configuration
from thai_isan_config import DeploymentConfig

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('thai_isan_tts_requests_total', 'Total requests', ['language', 'status'])
REQUEST_DURATION = Histogram('thai_isan_tts_request_duration_seconds', 'Request duration', ['language'])
ACTIVE_CONNECTIONS = Gauge('thai_isan_tts_active_connections', 'Active connections')
MODEL_LOAD_TIME = Histogram('thai_isan_tts_model_load_time_seconds', 'Model load time')
CACHE_HIT_RATE = Counter('thai_isan_tts_cache_hits_total', 'Cache hits', ['type'])
CACHE_MISS_RATE = Counter('thai_isan_tts_cache_misses_total', 'Cache misses', ['type'])

@dataclass
class ModelConfig:
    """Model configuration for deployment"""
    model_path: str
    tokenizer_path: str
    language: str
    device: str
    batch_size: int
    max_length: int
    sample_rate: int
    precision: str
    enable_optimization: bool

@dataclass
class RequestMetadata:
    """Request metadata for tracking"""
    request_id: str
    client_ip: str
    language: str
    text_length: int
    timestamp: datetime
    processing_time: float
    cache_hit: bool
    quality_score: float
    error: Optional[str] = None

class AudioRequest(BaseModel):
    """Audio synthesis request"""
    text: str = Field(..., min_length=1, max_length=500, description="Text to synthesize")
    language: str = Field(default="th", regex="^(th|tts)$", description="Language code (th=Thai, tts=Isan)")
    speaker_id: Optional[str] = Field(default=None, description="Speaker ID for voice cloning")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    pitch: float = Field(default=0.0, ge=-1.0, le=1.0, description="Pitch adjustment")
    emotion: Optional[str] = Field(default="neutral", description="Emotion style")
    return_format: str = Field(default="wav", regex="^(wav|mp3|ogg)$", description="Audio format")
    quality: str = Field(default="high", regex="^(low|medium|high)$", description="Quality level")

class AudioResponse(BaseModel):
    """Audio synthesis response"""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(..., description="Audio sample rate")
    duration: float = Field(..., description="Audio duration in seconds")
    language: str = Field(..., description="Detected/synthesized language")
    speaker_id: str = Field(..., description="Speaker ID used")
    quality_score: float = Field(..., description="Quality assessment score")
    processing_time: float = Field(..., description="Processing time in seconds")
    request_id: str = Field(..., description="Unique request ID")

class BatchAudioRequest(BaseModel):
    """Batch audio synthesis request"""
    requests: List[AudioRequest] = Field(..., min_items=1, max_items=100, description="List of synthesis requests")
    priority: str = Field(default="normal", regex="^(low|normal|high)$", description="Processing priority")
    callback_url: Optional[str] = Field(default=None, description="Callback URL for async processing")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    uptime: float
    system_info: Dict[str, any]
    model_status: Dict[str, str]

class ThaiIsanTTSModel:
    """Optimized Thai-Isan TTS model for production"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        
        # Audio processing
        self.sample_rate = config.sample_rate
        self.mel_transform = self._create_mel_transform()
        
        # Performance optimization
        self.warmup_model()
        
        logger.info(f"Model loaded: {config.language} on {self.device}")
    
    def _load_model(self):
        """Load optimized model"""
        try:
            if self.config.precision == "fp16":
                # Load FP16 model for faster inference
                model = torch.load(self.config.model_path, map_location=self.device)
                model = model.half()
            elif self.config.precision == "int8":
                # Load INT8 quantized model
                model = torch.load(self.config.model_path, map_location=self.device)
                model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
            else:
                # Load full precision model
                model = torch.load(self.config.model_path, map_location=self.device)
            
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_tokenizer(self):
        """Load tokenizer"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _create_mel_transform(self):
        """Create mel spectrogram transform"""
        return MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )
    
    def warmup_model(self):
        """Warmup model for faster inference"""
        logger.info("Warming up model...")
        with torch.no_grad():
            # Dummy input for warmup
            dummy_text = "สวัสดีครับ" if self.config.language == "th" else "สบายดีบ่"
            _ = self.synthesize(dummy_text)
        logger.info("Model warmup completed")
    
    @torch.no_grad()
    def synthesize(self, text: str, **kwargs) -> Tuple[np.ndarray, float]:
        """Synthesize speech from text"""
        start_time = time.time()
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        # Generate audio
        audio_output = self.model(inputs['input_ids'])
        
        # Postprocess audio
        audio = self._postprocess_audio(audio_output, **kwargs)
        
        # Calculate quality score
        quality_score = self._assess_quality(audio)
        
        processing_time = time.time() - start_time
        
        return audio, quality_score
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for synthesis"""
        # Clean text
        text = text.strip()
        
        # Language-specific preprocessing
        if self.config.language == "th":
            # Thai text preprocessing
            text = self._preprocess_thai_text(text)
        elif self.config.language == "tts":
            # Isan text preprocessing
            text = self._preprocess_isan_text(text)
        
        return text
    
    def _preprocess_thai_text(self, text: str) -> str:
        """Preprocess Thai text"""
        # Normalize Thai text
        # This would include tone normalization, character normalization, etc.
        return text
    
    def _preprocess_isan_text(self, text: str) -> str:
        """Preprocess Isan text"""
        # Convert Thai script to Isan pronunciation
        # Handle Isan-specific phonological processes
        return text
    
    def _postprocess_audio(self, audio_output: torch.Tensor, **kwargs) -> np.ndarray:
        """Postprocess generated audio"""
        # Convert to numpy
        audio = audio_output.cpu().numpy().squeeze()
        
        # Apply speed adjustment if needed
        speed = kwargs.get('speed', 1.0)
        if speed != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=speed)
        
        # Apply pitch adjustment if needed
        pitch = kwargs.get('pitch', 0.0)
        if pitch != 0.0:
            n_steps = pitch * 12  # Convert to semitones
            audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio)) * 0.95
        
        return audio
    
    def _assess_quality(self, audio: np.ndarray) -> float:
        """Assess audio quality"""
        # Simple quality metrics
        # In production, this would use more sophisticated quality assessment
        
        # Check for clipping
        if np.max(np.abs(audio)) > 0.95:
            return 0.7
        
        # Check for silence
        if np.mean(audio**2) < 0.001:
            return 0.3
        
        # Check dynamic range
        dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / (np.sqrt(np.mean(audio**2)) + 1e-10))
        if dynamic_range < 10:
            return 0.6
        
        return 0.9  # Good quality

class AudioCache:
    """LRU cache for audio synthesis results"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.lock = threading.Lock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Tuple[np.ndarray, float]]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                # Check TTL
                if time.time() - self.access_times[key] < self.ttl:
                    self.access_times[key] = time.time()  # Update access time
                    CACHE_HIT_RATE.labels(type='audio').inc()
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.access_times[key]
            
            CACHE_MISS_RATE.labels(type='audio').inc()
            return None
    
    def put(self, key: str, value: Tuple[np.ndarray, float]):
        """Put item in cache"""
        with self.lock:
            # Remove oldest if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times, key=self.access_times.get)
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def _cleanup_loop(self):
        """Background cleanup of expired items"""
        while True:
            time.sleep(300)  # Cleanup every 5 minutes
            
            with self.lock:
                current_time = time.time()
                expired_keys = [
                    key for key, access_time in self.access_times.items()
                    if current_time - access_time > self.ttl
                ]
                
                for key in expired_keys:
                    del self.cache[key]
                    del self.access_times[key]

class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # client_id: [timestamps]
        self.lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        with self.lock:
            now = time.time()
            
            # Clean old requests
            if client_id in self.requests:
                self.requests[client_id] = [
                    ts for ts in self.requests[client_id]
                    if now - ts < 60
                ]
            else:
                self.requests[client_id] = []
            
            # Check limit
            if len(self.requests[client_id]) >= self.requests_per_minute:
                return False
            
            # Add current request
            self.requests[client_id].append(now)
            return True
    
    def get_wait_time(self, client_id: str) -> float:
        """Get wait time until next request is allowed"""
        with self.lock:
            if client_id not in self.requests or not self.requests[client_id]:
                return 0.0
            
            oldest_request = min(self.requests[client_id])
            return max(0.0, 60 - (time.time() - oldest_request))

class AuthenticationManager:
    """API authentication manager"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.api_keys = {}  # In production, this would be in a database
        self.http_bearer = HTTPBearer()
    
    def create_api_key(self, user_id: str) -> str:
        """Create new API key"""
        api_key = secrets.token_urlsafe(32)
        self.api_keys[api_key] = {
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'is_active': True
        }
        return api_key
    
    def verify_api_key(self, api_key: str) -> bool:
        """Verify API key"""
        return api_key in self.api_keys and self.api_keys[api_key]['is_active']
    
    def create_access_token(self, api_key: str) -> str:
        """Create JWT access token"""
        if not self.verify_api_key(api_key):
            raise ValueError("Invalid API key")
        
        payload = {
            'api_key': api_key,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> bool:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return self.verify_api_key(payload['api_key'])
        except jwt.ExpiredSignatureError:
            return False
        except jwt.InvalidTokenError:
            return False

class ThaiIsanTTSServer:
    """Production Thai-Isan TTS server"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.app = FastAPI(
            title="Thai-Isan TTS API",
            description="Production-ready Thai and Isan text-to-speech synthesis",
            version="1.0.0"
        )
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize components
        self.models = {}
        self.cache = AudioCache()
        self.rate_limiter = RateLimiter(config.max_requests_per_minute)
        self.auth_manager = AuthenticationManager(config.jwt_secret_key if hasattr(config, 'jwt_secret_key') else "secret")
        
        # Request tracking
        self.request_queue = queue.Queue()
        self.active_requests = {}
        self.request_history = []
        
        # Performance monitoring
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        
        # Setup routes
        self._setup_routes()
        
        # Start background threads
        self._start_background_threads()
        
        logger.info("Thai-Isan TTS server initialized")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now(),
                version="1.0.0",
                uptime=uptime,
                system_info={
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage('/').percent
                },
                model_status={
                    lang: "loaded" for lang in self.models.keys()
                }
            )
        
        @self.app.post("/synthesize", response_model=AudioResponse)
        async def synthesize_audio(request: AudioRequest, background_tasks: BackgroundTasks):
            """Synthesize audio from text"""
            request_id = secrets.token_urlsafe(16)
            
            # Rate limiting
            if not self.rate_limiter.is_allowed("anonymous"):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            # Check cache
            cache_key = self._generate_cache_key(request.text, request.language, request.speaker_id)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                audio_data, quality_score = cached_result
                cache_hit = True
            else:
                # Synthesize audio
                audio_data, quality_score = self._synthesize_audio(request)
                cache_hit = False
                
                # Cache result
                background_tasks.add_task(self.cache.put, cache_key, (audio_data, quality_score))
            
            # Convert to base64
            audio_base64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
            
            # Track request
            metadata = RequestMetadata(
                request_id=request_id,
                client_ip="anonymous",
                language=request.language,
                text_length=len(request.text),
                timestamp=datetime.now(),
                processing_time=0.0,  # Would be measured
                cache_hit=cache_hit,
                quality_score=quality_score
            )
            background_tasks.add_task(self._track_request, metadata)
            
            # Update metrics
            REQUEST_COUNT.labels(language=request.language, status="success").inc()
            
            return AudioResponse(
                audio_data=audio_base64,
                sample_rate=48000,
                duration=len(audio_data) / 48000,
                language=request.language,
                speaker_id=request.speaker_id or "default",
                quality_score=quality_score,
                processing_time=0.0,
                request_id=request_id
            )
        
        @self.app.post("/synthesize/batch")
        async def synthesize_batch(request: BatchAudioRequest):
            """Batch synthesis endpoint"""
            # Process batch requests
            results = []
            
            for audio_request in request.requests:
                try:
                    audio_data, quality_score = self._synthesize_audio(audio_request)
                    audio_base64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
                    
                    results.append({
                        "audio_data": audio_base64,
                        "sample_rate": 48000,
                        "duration": len(audio_data) / 48000,
                        "language": audio_request.language,
                        "speaker_id": audio_request.speaker_id or "default",
                        "quality_score": quality_score,
                        "status": "success"
                    })
                except Exception as e:
                    results.append({
                        "status": "error",
                        "error": str(e)
                    })
            
            return {"results": results}
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Prometheus metrics endpoint"""
            return Response(generate_latest(), media_type="text/plain")
    
    def _generate_cache_key(self, text: str, language: str, speaker_id: Optional[str]) -> str:
        """Generate cache key"""
        key_data = f"{text}_{language}_{speaker_id or 'default'}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _synthesize_audio(self, request: AudioRequest) -> Tuple[np.ndarray, float]:
        """Synthesize audio from request"""
        # Get model for language
        if request.language not in self.models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Language {request.language} not supported"
            )
        
        model = self.models[request.language]
        
        # Synthesize audio
        audio_data, quality_score = model.synthesize(
            request.text,
            speed=request.speed,
            pitch=request.pitch,
            emotion=request.emotion
        )
        
        return audio_data, quality_score
    
    def _track_request(self, metadata: RequestMetadata):
        """Track request metadata"""
        self.request_history.append(metadata)
        
        # Keep only recent history
        if len(self.request_history) > 10000:
            self.request_history = self.request_history[-10000:]
    
    def _start_background_threads(self):
        """Start background processing threads"""
        # Request processing thread
        processing_thread = threading.Thread(target=self._process_request_queue, daemon=True)
        processing_thread.start()
        
        # Metrics collection thread
        metrics_thread = threading.Thread(target=self._collect_metrics, daemon=True)
        metrics_thread.start()
    
    def _process_request_queue(self):
        """Process requests from queue"""
        while True:
            try:
                request_data = self.request_queue.get(timeout=1)
                # Process request
                self._process_request(request_data)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing request: {e}")
    
    def _collect_metrics(self):
        """Collect system metrics"""
        while True:
            try:
                # Update Prometheus metrics
                ACTIVE_CONNECTIONS.set(len(self.active_requests))
                
                time.sleep(60)  # Collect every minute
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
    
    def add_model(self, language: str, model: ThaiIsanTTSModel):
        """Add language model"""
        self.models[language] = model
        logger.info(f"Added model for language: {language}")
    
    def start_server(self):
        """Start the server"""
        logger.info(f"Starting Thai-Isan TTS server on {self.config.host}:{self.config.port}")
        
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            workers=self.config.workers,
            log_level="info"
        )

# Production configuration
PRODUCTION_CONFIG = DeploymentConfig(
    host="0.0.0.0",
    port=8000,
    workers=4,
    timeout=300,
    model_path="./models/thai_isan_tts",
    model_format="onnx",
    enable_model_caching=True,
    max_model_cache_size=2,
    batch_inference=True,
    max_batch_size=8,
    enable_streaming=True,
    streaming_chunk_size=1024,
    enable_quality_control=True,
    quality_threshold=0.8,
    enable_automatic_retry=True,
    max_retry_attempts=3,
    enable_authentication=False,
    api_key_required=False,
    rate_limiting=True,
    max_requests_per_minute=60,
    enable_logging=True,
    enable_metrics=True,
    enable_health_checks=True,
    health_check_interval=30,
    enable_auto_scaling=True,
    min_instances=1,
    max_instances=10,
    scaling_threshold=0.8
)

def create_production_server(config: DeploymentConfig = PRODUCTION_CONFIG) -> ThaiIsanTTSServer:
    """Create production server instance"""
    
    # Initialize models
    thai_model = ThaiIsanTTSModel(ModelConfig(
        model_path="./models/thai_model.pt",
        tokenizer_path="Qwen/Qwen3-TTS",
        language="th",
        device="cuda",
        batch_size=8,
        max_length=512,
        sample_rate=48000,
        precision="fp16",
        enable_optimization=True
    ))
    
    isan_model = ThaiIsanTTSModel(ModelConfig(
        model_path="./models/isan_model.pt",
        tokenizer_path="Qwen/Qwen3-TTS",
        language="tts",
        device="cuda",
        batch_size=8,
        max_length=512,
        sample_rate=48000,
        precision="fp16",
        enable_optimization=True
    ))
    
    # Create server
    server = ThaiIsanTTSServer(config)
    
    # Add models
    server.add_model("th", thai_model)
    server.add_model("tts", isan_model)
    
    return server

def main():
    """Main function to start production server"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Thai-Isan TTS production server...")
    
    # Create and start server
    server = create_production_server()
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

if __name__ == "__main__":
    main()