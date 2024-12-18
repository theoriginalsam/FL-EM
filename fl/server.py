from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from models.unet import build_unet
from models.metrics import ForestChangeMetrics
from config.settings import FL_CONFIG
import os
import json
import logging
import time
from datetime import datetime
from utils.visualization import FLVisualizer

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fl_server')

class FLServer:
    _instance = None

    @staticmethod
    def get_instance():
        if FLServer._instance is None:
            FLServer._instance = FLServer()
        return FLServer._instance

    def __init__(self):
        if FLServer._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            FLServer._instance = self
            # Initialize variables
            self.global_model = build_unet()
            self.current_round = 0
            self.client_updates = {}
            self.metrics_history = []
            self.total_rounds = FL_CONFIG['ROUNDS']
            self.min_clients = FL_CONFIG['MIN_CLIENTS']
            self.active_clients = set()
            self.training_completed = False
            self.round_start_time = time.time()
            self.visualizer = FLVisualizer()
            self.visualization_generated = False
            
            # Enhanced training statistics
            self.training_stats = {
                'loss': [],
                'accuracy': [],
                'iou': [],
                'communication_cost': [],  # in MB
                'round_duration': [],      # in seconds
                'client_participation': [] # clients per round
            }
            
            # Performance metrics
            self.aggregation_times = []
            self.client_response_times = {}
            
            logger.info("FL Server initialized successfully")

    def calculate_model_size(self, weights):
        """Calculate size of model weights in MB"""
        total_size = sum(w.nbytes for w in weights)
        return total_size / (1024 * 1024)  # Convert to MB

    def get_status(self):
        """Get enhanced server status"""
        return {
            'current_round': self.current_round,
            'total_rounds': self.total_rounds,
            'active_clients': len(self.active_clients),
            'current_clients': len(self.client_updates),
            'required_clients': self.min_clients,
            'training_stats': self.training_stats,
            'metrics_history': self.metrics_history,
            'training_completed': self.training_completed,
            'visualization_generated': self.visualization_generated,
            'performance_metrics': {
                'avg_aggregation_time': np.mean(self.aggregation_times) if self.aggregation_times else 0,
                'avg_communication_cost': np.mean(self.training_stats['communication_cost']) if self.training_stats['communication_cost'] else 0,
                'avg_round_duration': np.mean(self.training_stats['round_duration']) if self.training_stats['round_duration'] else 0
            }
        }

    def get_client_metrics(self, client_id):
        """Get metrics for a specific client"""
        client_metrics = []
        for round_metrics in self.metrics_history:
            if client_id in [update.get('client_id') for update in round_metrics.get('metrics', [])]:
                client_metrics.append({
                    'round': round_metrics['round'],
                    'metrics': next(m for m in round_metrics['metrics'] if m.get('client_id') == client_id)
                })
        return client_metrics

    def update_global_model(self, client_update):
        """Process client updates with enhanced tracking"""
        logger.info(f"Processing update from client {client_update['client_id']} (Round {self.current_round}/{self.total_rounds})")

        if self.training_completed:
            return False, {"message": "Training has already completed", "training_completed": True}

        # Track client activity and timing
        client_id = client_update['client_id']
        self.active_clients.add(client_id)
        self.client_updates[client_id] = client_update
        
        # Calculate communication cost
        weights_size = self.calculate_model_size(
            [np.array(w) for w in client_update['weights']]
        )
        self.training_stats['communication_cost'].append(weights_size)

        # Update metrics
        if 'metrics' in client_update:
            self._update_training_stats(client_update)

        # Check if enough clients for aggregation
        if len(self.client_updates) >= self.min_clients:
            agg_start_time = time.time()
            
            # Perform FedAvg
            weights = [update['weights'] for update in self.client_updates.values()]
            metrics = [update['metrics'] for update in self.client_updates.values()]
            
            avg_weights = []
            for layer_weights in zip(*weights):
                avg_weights.append(np.mean(layer_weights, axis=0))
            
            # Update global model
            self.global_model.set_weights([np.array(w) for w in avg_weights])
            
            # Track timing and metrics
            agg_time = time.time() - agg_start_time
            round_time = time.time() - self.round_start_time
            
            self.aggregation_times.append(agg_time)
            self.training_stats['round_duration'].append(round_time)
            self.training_stats['client_participation'].append(len(self.client_updates))
            
            # Record round metrics
            self.metrics_history.append({
                'round': self.current_round,
                'metrics': metrics,
                'aggregation_time': agg_time,
                'round_duration': round_time,
                'num_clients': len(self.client_updates)
            })
            
            # Prepare for next round
            self.current_round += 1
            self.round_start_time = time.time()
            self.client_updates.clear()
            
            # Check if training is complete
            if self.current_round >= self.total_rounds:
                logger.info("Training completed successfully!")
                self.training_completed = True
                try:
                    logger.info("Generating training visualization report...")
                    self.create_training_report()
                    self.visualization_generated = True
                    logger.info("Training visualization report created successfully")
                except Exception as e:
                    logger.error(f"Error creating visualization report: {str(e)}")
                
                return True, {
                    "round": self.current_round,
                    "metrics": metrics,
                    "training_completed": True,
                    "visualization_generated": self.visualization_generated
                }
            
            logger.info(f"Round {self.current_round} completed successfully")
            return True, {
                "round": self.current_round,
                "metrics": metrics,
                "training_completed": False
            }
        
        return False, {
            "message": f"Waiting for more clients ({len(self.client_updates)}/{self.min_clients})"
        }

    def _update_training_stats(self, client_update):
        """Update training statistics with client metrics"""
        metrics = client_update['metrics']
        history = client_update.get('history', {})
        
        if len(self.training_stats['iou']) <= self.current_round:
            self.training_stats['iou'].append(metrics.get('iou', 0))
        
        if history:
            if len(self.training_stats['loss']) <= self.current_round:
                self.training_stats['loss'].append(
                    history.get('loss', [0])[-1] if 'loss' in history else 0
                )
            if len(self.training_stats['accuracy']) <= self.current_round:
                self.training_stats['accuracy'].append(
                    history.get('binary_accuracy', [0])[-1] if 'binary_accuracy' in history else 0
                )

    def create_training_report(self):
        """Create visualization report after training"""
        metrics = {
            'metrics_history': self.metrics_history,
            'communication_costs': self.training_stats['communication_cost'],
            'aggregation_times': self.aggregation_times
        }
        
        client_metrics = {
            client_id: self.get_client_metrics(client_id)
            for client_id in self.active_clients
        }
        
        self.visualizer.create_visualization_report(metrics, client_metrics)

def create_app():
    """Create Flask app with enhanced error handling and logging"""
    app = Flask(__name__)
    server = FLServer.get_instance()

    @app.route('/', methods=['GET'])
    def home():
        status = server.get_status()
        return jsonify(status)

    @app.route('/get_model', methods=['GET'])
    def get_model():
        try:
            # Check if training is completed
            if server.training_completed:
                return jsonify({
                    'status': 'completed',
                    'message': 'Training has completed. No more model updates available.',
                    'training_completed': True,
                    'final_round': server.current_round,
                    'visualization_generated': server.visualization_generated
                }), 200

            logger.info("Model weights requested")
            weights = [w.tolist() for w in server.global_model.get_weights()]
            model_size = server.calculate_model_size(server.global_model.get_weights())
            
            response = {
                'weights': weights,
                'round': server.current_round,
                'training_completed': server.training_completed,
                'model_size_mb': model_size
            }
            logger.info(f"Sending model weights (Size: {model_size:.2f} MB)")
            return jsonify(response)
        except Exception as e:
            logger.error(f"Error in get_model: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/update', methods=['POST'])
    def update_model():
        try:
            if server.training_completed:
                return jsonify({
                    'status': 'completed',
                    'message': 'Training has already completed',
                    'final_round': server.current_round,
                    'visualization_generated': server.visualization_generated
                })

            client_update = request.get_json()
            updated, result = server.update_global_model(client_update)

            response = {
                'status': 'success',
                'updated': updated,
                'result': result,
                'current_round': server.current_round,
                'total_rounds': server.total_rounds,
                'active_clients': len(server.active_clients),
                'current_clients': len(server.client_updates),
                'required_clients': server.min_clients
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in update_model: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @app.route('/status', methods=['GET'])
    def get_status():
        return jsonify(server.get_status())

    return app

# Initialize Flask app and server
app = create_app()
fl_server = FLServer.get_instance()

if __name__ == '__main__':
    logger.info(f"Starting FL Server on port {FL_CONFIG.get('PORT', 5001)}")
    app.run(host='0.0.0.0', port=FL_CONFIG.get('PORT', 5001), debug=True)