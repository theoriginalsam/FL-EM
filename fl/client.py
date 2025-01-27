import requests
import numpy as np
import logging
import json
import os
import time
from datetime import datetime
import tensorflow as tf
from models.unet import build_unet
from data.data_loader import get_regional_data
from data.preprocessing import create_balanced_patches_with_oversampling
from config.settings import MODEL_CONFIG, FL_CONFIG

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
        self.region_id = region_id
        self.batch_size = MODEL_CONFIG['BATCH_SIZE']
        self.server_url = server_url
        self.current_round = 0
        self.training_history = []
        
        # Setup logging
        self._setup_logging()
        
        # Initialize data and model
        self.logger.info("Loading data during initialization...")
        self._initialize_data()
        self.logger.info("Data initialization completed")
        self._initialize_model()

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

    def _initialize_data(self):
        """Initialize data with caching"""
        cached_data_path = f'cached_data/client_{self.region_id}_data.npz'
        
        try:
            if os.path.exists(cached_data_path):
                self.logger.info("Loading data from cache...")
                cached_data = np.load(cached_data_path, allow_pickle=True)
                self.train_X = cached_data['train_X']
                self.train_y = cached_data['train_y']
                self.val_X = cached_data['val_X']
                self.val_y = cached_data['val_y']
                
            else:
                self.logger.info("Processing data from scratch...")
                # Load and process raw data
                all_images, all_masks = get_regional_data(f"region_{self.region_id}")
                
                # Use balanced patch creation
                X_balanced, y_balanced = create_balanced_patches_with_oversampling(
                    all_images, all_masks,
                    minority_samples=1000,
                    majority_samples=1000
                )
                
                # Split data
                train_size = int(0.8 * len(X_balanced))
                train_size = (train_size // self.batch_size) * self.batch_size
                
                self.train_X = X_balanced[:train_size]
                self.train_y = y_balanced[:train_size]
                self.val_X = X_balanced[train_size:]
                self.val_y = y_balanced[train_size:]
                
                # Save processed data
                os.makedirs('cached_data', exist_ok=True)
                np.savez(cached_data_path,
                        train_X=self.train_X, train_y=self.train_y,
                        val_X=self.val_X, val_y=self.val_y)
                self.logger.info(f"Cached data saved to {cached_data_path}")
            
            # Create tf.data.Dataset after loading/creating data
            self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_X, self.train_y))
            self.train_dataset = self.train_dataset.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            
            self.val_dataset = tf.data.Dataset.from_tensor_slices((self.val_X, self.val_y))
            self.val_dataset = self.val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            
            self.logger.info(f"Data split - Training: {len(self.train_X)}, Validation: {len(self.val_X)}")
            
        except Exception as e:
            self.logger.error(f"Error initializing data: {str(e)}")
            raise
    def _initialize_model(self):
        """Initialize model matching centralized setup"""
        self.model = build_unet()
        
        # Match centralized callbacks exactly
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'checkpoints/client_{self.region_id}/best_model.keras',
                save_best_only=True
            )
        ]
        
        # Match centralized compilation
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

    def train_local(self, epochs=FL_CONFIG['LOCAL_EPOCHS']):
        """Train model matching centralized approach"""
        try:
            self.logger.info(f"Starting local training for round {self.current_round}")
            
            history = self.model.fit(
                self.train_dataset,
                validation_data=self.val_dataset,
                epochs=epochs,
                callbacks=self.callbacks,
                verbose=1
            )
            
            self.save_checkpoint()
            
            history_dict = {}
            for key, value in history.history.items():
                history_dict[key] = [float(v) for v in value]
            
            self.training_history.append(history_dict)
            self.logger.info("Local training completed")
            return history_dict
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def _send_update_with_retry(self, update_data, max_retries=3):
        for attempt in range(max_retries):
            try:
                json_data = json.dumps(update_data, cls=NumpyEncoder)
                response = requests.post(
                    f"{self.server_url}/update",
                    data=json_data,
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

    def save_checkpoint(self):
        """Save model checkpoint"""
        try:
            checkpoint_dir = f'checkpoints/client_{self.region_id}/round_{self.current_round}'
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            self.model.save(os.path.join(checkpoint_dir, 'model.keras'))
            self.logger.info(f"Saved checkpoint for round {self.current_round}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")

    def get_metrics(self):
        """Calculate metrics on validation data"""
        try:
            predictions = self.model.predict(self.val_dataset, verbose=0)
            
            # Calculate standard metrics
            results = {}
            for name, value in zip(self.model.metrics_names, self.model.evaluate(self.val_dataset, verbose=0)):
                results[name] = float(value)
                
            # Calculate IoU
            val_y = np.concatenate([y for x, y in self.val_dataset], axis=0)
            predictions = (predictions > 0.5).astype(np.float32)  # Threshold predictions
            
            intersection = np.sum(predictions * val_y)
            union = np.sum(predictions) + np.sum(val_y) - intersection
            iou = intersection / (union + 1e-7)  # Add small epsilon to avoid division by zero
            
            results['iou'] = float(iou)
            
            self.logger.info(f"Metrics calculated: {results}")
            return results
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise
    # Add this function to your client.py file
    

    def update(self):
        try:
            while True:  # Keep checking until successful update
                # Get server status
                response = requests.get(f"{self.server_url}/status", timeout=30)
                server_status = response.json()
                server_round = server_status.get('current_round', 0)

                # Check if we need to catch up with server round
                if server_round > self.current_round:
                    self.logger.info(f"Catching up from round {self.current_round} to {server_round}")
                    self.current_round = server_round
                    self.last_trained_round = None  # Force training for new round

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
                        'data_size': len(self.train_X),
                        'round': self.current_round
                    }
                    
                    server_response = self._send_update_with_retry(update_data)
                    
                    if server_response.get('result', {}).get('action') == 'proceed':
                        self.current_round += 1
                        self.last_trained_round = None
                        self.logger.info(f"Proceeding to round {self.current_round}")
                        return True, server_status  # Successfully completed round
                    elif server_response.get('result', {}).get('action') == 'wait':
                        self.logger.info(f"Waiting for other clients in round {self.current_round}")
                        time.sleep(30)
                        continue  # Go back to checking server status
                else:
                    self.logger.info(f"Already trained for round {self.current_round}, waiting for other clients")
                    time.sleep(30)
                    continue  # Go back to checking server status
                    
        except Exception as e:
            self.logger.error(f"Error in update: {str(e)}")
            return False, str(e)
        

    def format_status(status):
        """Format status output to be more readable"""
        formatted = f"""
    FL Training Status:
    ------------------
    Round: {status['current_round']}/{status['total_rounds']}
    Active Clients: {status['active_clients']}/{status['required_clients']}

    Metrics History:
    ---------------"""
        
        for round_data in status.get('metrics_history', []):
            formatted += f"""
    Round {round_data['round']}:
    - Aggregated Loss: {round_data['aggregated_loss']:.4f}
    - Aggregated Accuracy: {round_data['aggregated_accuracy']:.4f}
    - Number of Clients: {round_data['n_clients']}
    """
        return formatted

    def run(self, total_rounds=None):
        """Run FL client for specified number of rounds"""
        if total_rounds is None:
            total_rounds = FL_CONFIG['ROUNDS']
        
        try:
            while self.current_round < total_rounds:
                success, status = self.update()
                
                if not success:
                    self.logger.error("Failed to complete round")
                    break
                    
                # Add formatted status logging here
                self.logger.info(format_status(status))
                    
                if status.get('training_completed', False):
                    self.logger.info("Training completed successfully")
                    break
                
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