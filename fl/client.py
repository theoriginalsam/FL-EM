import requests
import numpy as np
import logging
import json
import os
import time
from requests.exceptions import RequestException
from models.unet import build_unet, compile_model
from models.metrics import ForestChangeMetrics, weighted_binary_crossentropy
from data.data_loader import get_regional_data
from data.preprocessing import BalancedPatchGenerator, DataGenerator
from config.settings import MODEL_CONFIG, FL_CONFIG
import tensorflow as tf
from utils.visualization import FLVisualizer
from datetime import datetime

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        return super(NumpyEncoder, self).default(obj)

class FLClient:
    def __init__(self, region_id, server_url):
        """Initialize FL Client"""
        self.region_id = region_id
        self.batch_size = MODEL_CONFIG['BATCH_SIZE']
        self.server_url = server_url
        self.current_round = 0
        
        # Initialize components first
        self.metrics_calculator = ForestChangeMetrics()
        self.patch_generator = BalancedPatchGenerator()
        self.training_history = []
        self.visualizer = FLVisualizer()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize data
        self.logger.info("Loading data during initialization...")
        self._initialize_data()
        self.logger.info("Data initialization completed")
        
        # Calculate class weights
        self.class_weights = self._calculate_class_weights()
        
        # Initialize model
        self._initialize_model()

        if self.load_latest_checkpoint():
            self.logger.info("Resumed from checkpoint")
        # Setup other components
        self.metrics_calculator = ForestChangeMetrics()
        self.patch_generator = BalancedPatchGenerator()
        self.training_history = []
        self.visualizer = FLVisualizer()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(f'fl_client_{self.region_id}')
        self.logger.setLevel(logging.INFO)
        
        os.makedirs('logs', exist_ok=True)
        fh = logging.FileHandler(f'logs/client_{self.region_id}.log')
        ch = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def _calculate_class_weights(self):
        """Calculate class weights from training data"""
        if not hasattr(self, 'y'):
            raise ValueError("Data must be initialized before calculating class weights")
        
        class_weights = {
            0: len(self.y) / (2 * np.sum(self.y == 0)),
            1: len(self.y) / (2 * np.sum(self.y == 1))
        }
        self.logger.info(f"Calculated class weights: {class_weights}")
        return class_weights
    
    def _initialize_model(self):
        """Initialize and compile model"""
        self.model = build_unet()
        
        # Setup callbacks
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,        # Reduced from 10
                restore_best_weights=True,
                   # Minimum change to qualify as an improvement
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'checkpoints/client_{self.region_id}/best_model.keras',
                save_best_only=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,       # Reduced from 5
                min_lr=1e-5
            )
        ]   
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(threshold=0.5),
                tf.keras.metrics.Precision(thresholds=0.5),
                tf.keras.metrics.Recall(thresholds=0.5),
                tf.keras.metrics.AUC()
            ]
        )

    
    def _initialize_data(self):
        """Initialize and split data with caching"""
        cached_data_path = f'cached_data/client_{self.region_id}_data.npz'
        
        try:
            # Check if cached data exists
            if os.path.exists(cached_data_path):
                self.logger.info("Loading data from cache...")
                cached_data = np.load(cached_data_path, allow_pickle=True)
                self.X = cached_data['X']
                self.y = cached_data['y']
                self.train_X = cached_data['train_X']
                self.train_y = cached_data['train_y']
                self.val_X = cached_data['val_X']
                self.val_y = cached_data['val_y']
                
            else:
                self.logger.info("Processing data from scratch...")
                # Load and process raw data
                all_images, all_masks = get_regional_data(f"region_{self.region_id}")
                self.X, self.y = self.patch_generator.create_balanced_patches(all_images, all_masks)
                
                # Split data
                val_size = int(0.2 * len(self.X))
                self.train_X = self.X[:-val_size]
                self.train_y = self.y[:-val_size]
                self.val_X = self.X[-val_size:]
                self.val_y = self.y[-val_size:]
                
                # Save processed data
                os.makedirs('cached_data', exist_ok=True)
                np.savez(cached_data_path,
                        X=self.X, y=self.y,
                        train_X=self.train_X, train_y=self.train_y,
                        val_X=self.val_X, val_y=self.val_y)
                self.logger.info(f"Cached data saved to {cached_data_path}")
            
            # Create data generators
            self.train_gen = DataGenerator(
                self.train_X, self.train_y, 
                batch_size=self.batch_size
            )
            self.val_gen = DataGenerator(
                self.val_X, self.val_y, 
                batch_size=self.batch_size
            )
            
            self.logger.info(f"Data split - Training: {len(self.train_X)}, Validation: {len(self.val_X)}")
            
        except Exception as e:
            self.logger.error(f"Error initializing data: {str(e)}")
            raise

    def _send_update_with_retry(self, update_data, max_retries=3):
      
        for attempt in range(max_retries):
            try:
                # Convert to JSON using NumpyEncoder
                json_data = json.dumps(update_data, cls=NumpyEncoder)
                
                response = requests.post(
                    f"{self.server_url}/update",
                    data=json_data,  # Use the converted json_data
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"Retry {attempt + 1}/{max_retries} after error: {str(e)}")
                time.sleep(5 * (attempt + 1))
    def train_local(self, epochs=FL_CONFIG['LOCAL_EPOCHS']):
        """Train model on local data"""
        try:
            self.logger.info(f"Starting local training for round {self.current_round}")
            
            history = self.model.fit(
                self.train_gen,
                validation_data=self.val_gen,
                epochs=epochs,
                callbacks=self.callbacks,
                verbose=1
            )
            
            # Save checkpoint
            self.save_checkpoint()


            
            # Convert history to regular Python types
            history_dict = {}
            for key, value in history.history.items():
                history_dict[key] = [float(v) for v in value]
            
            self.training_history.append(history_dict)
            self.logger.info("Local training completed")
            return history_dict
        
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
    def save_history(self, save_path='results/clients'):
        """Save training history"""
        try:
            save_dir = os.path.join(save_path, f'client_{self.region_id}')
            os.makedirs(save_dir, exist_ok=True)
            
            history_path = os.path.join(save_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2, cls=NumpyEncoder)
                
            self.logger.info(f"Training history saved to {history_path}")
            
            if hasattr(self, 'visualizer'):
                self.visualizer.plot_training_metrics(
                    self.training_history[-1], 
                    client_id=self.region_id
                )
                
        except Exception as e:
            self.logger.error(f"Error saving history: {str(e)}")
            raise
    def save_checkpoint(self):
        """Save comprehensive checkpoint"""
        try:
            checkpoint_dir = f'checkpoints/client_{self.region_id}/round_{self.current_round}'
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_data = {
                'round': self.current_round,
                'model_path': os.path.join(checkpoint_dir, 'model.keras'),
                'history': self.training_history,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save model
            self.model.save(checkpoint_data['model_path'])
            
            # Save checkpoint metadata
            with open(os.path.join(checkpoint_dir, 'checkpoint.json'), 'w') as f:
                json.dump(checkpoint_data, f)
                
            self.logger.info(f"Saved checkpoint for round {self.current_round}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")

    def load_latest_checkpoint(self):
        """Load most recent checkpoint"""
        checkpoint_base = f'checkpoints/client_{self.region_id}'
        if not os.path.exists(checkpoint_base):
            self.logger.info("No checkpoint directory found")
            return False
                
        # Find all round directories
        round_dirs = [d for d in os.listdir(checkpoint_base) 
                    if d.startswith('round_')]
        if not round_dirs:
            self.logger.info("No round directories found")
            return False
                
        # Get the latest round number
        rounds = [int(d.split('_')[-1]) for d in round_dirs]
        if not rounds:
            self.logger.info("No valid round numbers found")
            return False
                
        latest_round = max(rounds)
        checkpoint_dir = f'{checkpoint_base}/round_{latest_round}'
        model_path = os.path.join(checkpoint_dir, 'model.keras')
            
        if not os.path.exists(model_path):
            self.logger.info(f"No model file found at {model_path}")
            return False
                
        # Load model from the latest checkpoint
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.current_round = latest_round
            self.logger.info(f"Successfully loaded checkpoint from round {self.current_round}")
            return True
        
        return False
    def get_metrics(self, X_val=None, y_val=None):
        """Calculate metrics on validation data"""
        try:
            # Use stored validation data if none provided
            if X_val is None:
                X_val = self.val_X
            if y_val is None:
                y_val = self.val_y

            predictions = self.model.predict(X_val, verbose=0)
            metrics = self.metrics_calculator.calculate_metrics_dict(y_val, predictions)
            self.logger.info(f"Metrics calculated: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def update(self):
        try:
            # Get server status first
            response = requests.get(f"{self.server_url}/status", timeout=30)
            server_status = response.json()
            server_round = server_status.get('current_round', 0)

             # Check if training is complete
            if server_status.get('training_completed', False) or self.current_round >= FL_CONFIG['ROUNDS']:
                self.logger.info(f"Training completed after {self.current_round} rounds")
                self.save_history()
                return True, {"training_completed": True}

            # Check if we need to catch up to server round
            if server_round > self.current_round:
                self.logger.info(f"Catching up from round {self.current_round} to {server_round}")
                self.current_round = server_round
                self.last_trained_round = None  # Reset to force training for new round
            
            # Only train if we haven't trained for current round
            if not hasattr(self, 'last_trained_round') or self.last_trained_round != self.current_round:
                self.logger.info(f"Training for round {self.current_round}")
                history = self.train_local()
                metrics = self.get_metrics()
                self.last_trained_round = self.current_round
                
                update_data = {
                    'client_id': self.region_id,
                    'weights': [w.tolist() for w in self.model.get_weights()],
                    'metrics': metrics,
                    'history': history,
                    'data_size': len(self.X),
                    'round': self.current_round
                }
                
                response = self._send_update_with_retry(update_data)
                
                if response.get('result', {}).get('action') == 'proceed':
                    self.current_round += 1
                    self.last_trained_round = None  # Reset for next round
                    self.logger.info(f"Proceeding to round {self.current_round}")
                else:
                    self.logger.info(f"Waiting for other clients in round {self.current_round}")
                    time.sleep(30)
            else:
                self.logger.info(f"Checking status for round {self.current_round}")
                time.sleep(30)
                
            return True, response
                
        except Exception as e:
            self.logger.error(f"Error in update: {str(e)}")
            return False, str(e)
    def run(self, total_rounds=None):
        """Run FL client for specified number of rounds"""
        if total_rounds is None:
            total_rounds = FL_CONFIG['ROUNDS']
        
        try:
            while self.current_round < total_rounds:
                success, result = self.update()
                
                if not success:
                    self.logger.error("Failed to complete round")
                    break
                    
                if result.get('training_completed', False):
                    self.logger.info("Training completed successfully")
                    break
                    
            self.save_history()
            
        except Exception as e:
            self.logger.error(f"Error in run: {str(e)}")
        finally:
            self.logger.info("Client shutting down")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FL Client')
    parser.add_argument('--region', type=int, required=True, help='Region ID for this client')
    parser.add_argument('--server', type=str, default='http://localhost:5000', help='FL server URL')
    parser.add_argument('--rounds', type=int, default=FL_CONFIG['ROUNDS'], help='Number of training rounds')
    
    args = parser.parse_args()
    
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Create and run client
    client = FLClient(args.region, args.server)
    client.run(args.rounds)