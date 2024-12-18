import requests
import numpy as np
import logging
import json
import os
import time
from requests.exceptions import RequestException
from models.unet import build_unet, compile_model
from models.metrics import ForestChangeMetrics
from data.data_loader import get_regional_data
from data.preprocessing import BalancedPatchGenerator, DataGenerator
from config.settings import MODEL_CONFIG, FL_CONFIG
import tensorflow as tf

from utils.visualization import FLVisualizer

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

# Keep all the imports and NumpyEncoder class the same...

class FLClient:
    def __init__(self, region_id, server_url):
        """Initialize FL Client"""
        self.region_id = region_id
        self.batch_size = MODEL_CONFIG['BATCH_SIZE']
        self.server_url = server_url
        self.model = build_unet()
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
        self.metrics_calculator = ForestChangeMetrics()
        self.patch_generator = BalancedPatchGenerator()
        self.training_history = []
        self.visualizer = FLVisualizer()
        
        # Setup logging
        self.logger = logging.getLogger(f'fl_client_{region_id}')
        self.logger.setLevel(logging.INFO)
        
        os.makedirs('logs', exist_ok=True)
        fh = logging.FileHandler(f'logs/client_{region_id}.log')
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # Load data once during initialization
        self.logger.info("Loading data during initialization...")
        self._initialize_data()
        self.logger.info("Data initialization completed")
        
    def _initialize_data(self):
        """Initialize data once during startup"""
        try:
            # Load raw data
            all_images, all_masks = get_regional_data(f"region_{self.region_id}")
            
            # Create balanced patches
            self.X, self.y = self.patch_generator.create_balanced_patches(all_images, all_masks)
            
            # Split into train and validation sets
            val_size = int(0.2 * len(self.X))
            self.train_X = self.X[:-val_size]
            self.train_y = self.y[:-val_size]
            self.val_X = self.X[-val_size:]
            self.val_y = self.y[-val_size:]
            
            # Create data generators
            self.train_gen = DataGenerator(
                self.train_X, self.train_y, 
                batch_size=self.batch_size
            )
            self.val_gen = DataGenerator(
                self.val_X, self.val_y, 
                batch_size=self.batch_size
            )
            
            self.logger.info(f"Data split - Training samples: {len(self.train_X)}, Validation samples: {len(self.val_X)}")
            
        except Exception as e:
            self.logger.error(f"Error initializing data: {str(e)}")
            raise
        
    def train_local(self, epochs=FL_CONFIG['LOCAL_EPOCHS']):
        """Train model on local data"""
        try:
            self.logger.info("Starting local training")
            
            # Use pre-initialized data generators
            history = self.model.fit(
                self.train_gen,
                validation_data=self.val_gen,
                epochs=epochs,
                verbose=1,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                ]
            )
            
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
        """Perform one round of federated learning"""
        try:
            # Get global model
            self.logger.info(f"Requesting model from {self.server_url}/get_model")
            response = requests.get(f"{self.server_url}/get_model", timeout=30)
            
            if response.status_code != 200:
                raise RequestException(f"Server returned status code: {response.status_code}")
            
            response_data = response.json()
            
            # Strict check for training completion
            if (response_data.get('status') == 'completed' or 
                response_data.get('training_completed', False)):
                self.logger.info("Training completed signal received from server")
                self.save_history()  # Save history before exiting
                # Exit the program completely
                self.logger.info("Shutting down client...")
                os._exit(0)  # Force exit to prevent any further execution
            
            # Continue with normal update process only if training is not completed
            global_weights = response_data['weights']
            self.model.set_weights([np.array(w) for w in global_weights])
            
            # Train locally using pre-loaded data
            history = self.train_local()
            metrics = self.get_metrics()
            
            # Prepare update data
            update_data = {
                'client_id': self.region_id,
                'weights': [w.tolist() for w in self.model.get_weights()],
                'metrics': metrics,
                'history': history,
                'data_size': int(len(self.X))
            }
            
            # Send update to server
            self.logger.info("Sending update to server")
            response = requests.post(
                f"{self.server_url}/update",
                data=json.dumps(update_data, cls=NumpyEncoder),
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code != 200:
                raise RequestException(f"Failed to send update: {response.status_code}")
            
            response_data = response.json()
            
            # Check for training completion in update response
            if (response_data.get('status') == 'completed' or 
                response_data.get('training_completed', False)):
                self.logger.info("Training completed signal received in update response")
                self.save_history()  # Save history before exiting
                # Exit the program completely
                self.logger.info("Shutting down client...")
                os._exit(0)  # Force exit to prevent any further execution
                
            return True, response_data
            
        except RequestException as e:
            self.logger.error(f"Network error: {str(e)}")
            return False, str(e)
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return False, str(e)

    def run(self, total_rounds=None):
        """Run FL client for specified number of rounds"""
        if total_rounds is None:
            total_rounds = FL_CONFIG['ROUNDS']
            
        retry_delay = 10
        max_retries = 3
        current_round = 0
        
        try:
            while current_round < total_rounds:
                retries = 0
                while retries < max_retries:
                    self.logger.info(f"\nStarting round {current_round + 1}/{total_rounds}")
                    success, result = self.update()
                    
                    if not success:
                        retries += 1
                        if retries < max_retries:
                            self.logger.warning(f"Retry {retries}/{max_retries} after {retry_delay} seconds")
                            time.sleep(retry_delay)
                        else:
                            self.logger.error(f"Failed to complete round after {max_retries} attempts")
                            return
                    else:
                        current_round += 1
                        break
                
                if current_round >= total_rounds:
                    self.logger.info("Maximum rounds reached")
                    self.save_history()
                    break
                    
        except Exception as e:
            self.logger.error(f"Unexpected error in run: {str(e)}")
        finally:
            self.logger.info("Training complete!")
    def save_history(self, save_path='results/clients'):
        """Save training history"""
        try:
            save_dir = os.path.join(save_path, f'client_{self.region_id}')
            os.makedirs(save_dir, exist_ok=True)
            
            history_path = os.path.join(save_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2, cls=NumpyEncoder)
                
            self.logger.info(f"Training history saved to {history_path}")
            self.visualizer.plot_training_metrics(
            self.training_history[-1], 
            client_id=self.region_id
        )
            
        except Exception as e:
            self.logger.error(f"Error saving history: {str(e)}")
            raise

if __name__ == '__main__':
    import argparse
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Run FL Client')
    parser.add_argument('--region', type=int, required=True,
                      help='Region ID for this client')
    parser.add_argument('--server', type=str, default='http://localhost:5000',
                      help='FL server URL')
    parser.add_argument('--rounds', type=int, default=FL_CONFIG['ROUNDS'],
                      help='Number of training rounds')
    
    args = parser.parse_args()
    
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Create and run client
    client = FLClient(args.region, args.server)
    client.run(args.rounds)