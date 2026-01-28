"""
Thai-Isan TTS System - Complete Integration
Master script to run the entire Thai and Isan TTS system
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

# Import all system components
from thai_isan_config import DEFAULT_CONFIG, load_config, save_config
from enhanced_data_collection import ThaiIsanDataCollector
from professional_recording_system import ProfessionalAudioRecorder, RecordingInterfaceConfig
from enhanced_training_pipeline import ThaiIsanTTSModel, ThaiIsanTrainer, TrainingConfig
from comprehensive_quality_assurance import ThaiIsanQualityEvaluator
from production_deployment_system import create_production_server, PRODUCTION_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('thai_isan_tts.log')
    ]
)

logger = logging.getLogger(__name__)

class ThaiIsanTTSSystem:
    """Complete Thai-Isan TTS system orchestrator"""
    
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.components = {}
        self.status = "initialized"
        
    def run_data_collection(self, num_thai_speakers=50, num_isan_speakers=50):
        """Run data collection phase"""
        logger.info("Starting data collection phase...")
        
        try:
            collector = ThaiIsanDataCollector(self.config.data_collection)
            
            # Collect data from speakers
            results = collector.collect_data_from_speakers(
                num_speakers_thai=num_thai_speakers,
                num_speakers_isan=num_isan_speakers
            )
            
            # Process and export dataset
            metadata = collector.process_collected_data()
            success = collector.export_dataset("./thai_isan_dataset")
            
            logger.info(f"Data collection completed: {results}")
            logger.info(f"Dataset exported: {success}")
            
            self.components['data_collector'] = collector
            return results
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            raise
    
    def run_recording_system(self):
        """Run speaker recording system"""
        logger.info("Starting speaker recording system...")
        
        try:
            config = RecordingInterfaceConfig(
                sample_rate=self.config.speaker_recording.sample_rate,
                show_waveform=self.config.speaker_recording.show_waveform,
                show_spectrum=self.config.speaker_recording.show_spectrum,
                real_time_monitoring=self.config.speaker_recording.real_time_monitoring
            )
            
            recorder = ProfessionalAudioRecorder(config)
            recorder.run_gui_interface()
            
            self.components['recorder'] = recorder
            logger.info("Recording system completed")
            
        except Exception as e:
            logger.error(f"Recording system failed: {e}")
            raise
    
    def run_training(self, train_data_path, val_data_path, output_dir):
        """Run model training"""
        logger.info("Starting model training...")
        
        try:
            # Configure training
            training_config = TrainingConfig(
                num_epochs=self.config.training.num_epochs,
                batch_size=self.config.training.batch_size,
                learning_rate=self.config.training.learning_rate,
                target_tone_accuracy=self.config.training.target_tone_accuracy,
                target_phoneme_accuracy=self.config.training.target_phoneme_accuracy
            )
            
            # Create model and trainer
            model = ThaiIsanTTSModel(training_config)
            trainer = ThaiIsanTrainer(model, training_config)
            
            # Note: In a real implementation, you would load actual data here
            # For now, this is a placeholder
            logger.info("Training model... (placeholder - implement actual training)")
            
            # Save trained model
            model_path = os.path.join(output_dir, "thai_isan_tts_model.pt")
            trainer.save_checkpoint(0, model_path)
            
            self.components['model'] = model
            self.components['trainer'] = trainer
            
            logger.info(f"Training completed. Model saved to: {model_path}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def run_quality_evaluation(self, model_path, test_data_path):
        """Run quality evaluation"""
        logger.info("Starting quality evaluation...")
        
        try:
            evaluator = ThaiIsanQualityEvaluator(self.config.quality_assurance)
            
            # Evaluate dataset
            results = evaluator.evaluate_dataset(test_data_path, sample_size=100)
            
            # Generate quality report
            report_path = "./quality_report.json"
            evaluator.generate_quality_report([], report_path)
            
            logger.info(f"Quality evaluation completed: {results}")
            logger.info(f"Quality report saved: {report_path}")
            
            self.components['evaluator'] = evaluator
            return results
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            raise
    
    def run_production_server(self, host="0.0.0.0", port=8000):
        """Run production server"""
        logger.info(f"Starting production server on {host}:{port}...")
        
        try:
            # Configure deployment
            deployment_config = PRODUCTION_CONFIG
            deployment_config.host = host
            deployment_config.port = port
            
            # Create and start server
            server = create_production_server(deployment_config)
            server.start_server()
            
        except Exception as e:
            logger.error(f"Production server failed: {e}")
            raise
    
    def run_complete_pipeline(self, skip_phases=None):
        """Run complete TTS pipeline"""
        logger.info("Starting complete Thai-Isan TTS pipeline...")
        
        skip_phases = skip_phases or []
        
        # Phase 1: Data Collection
        if 'data_collection' not in skip_phases:
            logger.info("=== Phase 1: Data Collection ===")
            self.run_data_collection()
        
        # Phase 2: Speaker Recording
        if 'recording' not in skip_phases:
            logger.info("=== Phase 2: Speaker Recording ===")
            self.run_recording_system()
        
        # Phase 3: Model Training
        if 'training' not in skip_phases:
            logger.info("=== Phase 3: Model Training ===")
            self.run_training(
                train_data_path="./thai_isan_dataset/train",
                val_data_path="./thai_isan_dataset/val",
                output_dir="./models"
            )
        
        # Phase 4: Quality Evaluation
        if 'evaluation' not in skip_phases:
            logger.info("=== Phase 4: Quality Evaluation ===")
            self.run_quality_evaluation(
                model_path="./models/thai_isan_tts_model.pt",
                test_data_path="./thai_isan_dataset/test"
            )
        
        # Phase 5: Production Deployment
        if 'deployment' not in skip_phases:
            logger.info("=== Phase 5: Production Deployment ===")
            self.run_production_server()
        
        logger.info("Complete Thai-Isan TTS pipeline finished successfully!")
        self.status = "completed"

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Thai-Isan TTS System")
    parser.add_argument("--mode", choices=["complete", "data", "recording", "training", "evaluation", "deployment"], 
                       default="complete", help="Run mode")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--skip", nargs="+", choices=["data_collection", "recording", "training", "evaluation", "deployment"],
                       help="Phases to skip")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--thai-speakers", type=int, default=50, help="Number of Thai speakers")
    parser.add_argument("--isan-speakers", type=int, default=50, help="Number of Isan speakers")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = DEFAULT_CONFIG
    
    # Create system
    system = ThaiIsanTTSSystem(config)
    
    try:
        if args.mode == "complete":
            system.run_complete_pipeline(skip_phases=args.skip or [])
        elif args.mode == "data":
            system.run_data_collection(num_thai_speakers=args.thai_speakers, num_isan_speakers=args.isan_speakers)
        elif args.mode == "recording":
            system.run_recording_system()
        elif args.mode == "training":
            system.run_training(
                train_data_path="./thai_isan_dataset/train",
                val_data_path="./thai_isan_dataset/val",
                output_dir="./models"
            )
        elif args.mode == "evaluation":
            system.run_quality_evaluation(
                model_path="./models/thai_isan_tts_model.pt",
                test_data_path="./thai_isan_dataset/test"
            )
        elif args.mode == "deployment":
            system.run_production_server(host=args.host, port=args.port)
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()