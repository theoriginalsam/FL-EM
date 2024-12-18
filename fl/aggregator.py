import numpy as np
from typing import List, Dict, Any
import tensorflow as tf

class ModelAggregator:
    """Implements different federated aggregation strategies"""
    
    @staticmethod
    def fedavg(weights_list: List[List[np.ndarray]], 
               contributions: List[float] = None) -> List[np.ndarray]:
        """
        Standard FedAvg aggregation
        Args:
            weights_list: List of model weights from each client
            contributions: Optional contribution factors for weighted averaging
        """
        if contributions is None:
            contributions = [1/len(weights_list)] * len(weights_list)
            
        # Normalize contributions
        contributions = np.array(contributions) / np.sum(contributions)
        
        # Weighted average of weights
        avg_weights = []
        for weights_per_layer in zip(*weights_list):
            layer_weights = np.average(weights_per_layer, axis=0,
                                     weights=contributions)
            avg_weights.append(layer_weights)
            
        return avg_weights
    
    @staticmethod
    def fedavg_with_quality(weights_list: List[List[np.ndarray]], 
                           metrics: List[Dict[str, float]]) -> List[np.ndarray]:
        """
        FedAvg weighted by client performance
        Args:
            weights_list: List of model weights from each client
            metrics: Performance metrics for each client
        """
        # Use IoU scores as quality measures
        quality_scores = [m['iou'] for m in metrics]
        contributions = np.array(quality_scores) / np.sum(quality_scores)
        
        return ModelAggregator.fedavg(weights_list, contributions)
    
    @staticmethod
    def fedavg_with_size(weights_list: List[List[np.ndarray]], 
                        data_sizes: List[int]) -> List[np.ndarray]:
        """
        FedAvg weighted by client dataset sizes
        Args:
            weights_list: List of model weights from each client
            data_sizes: Number of samples for each client
        """
        contributions = np.array(data_sizes) / np.sum(data_sizes)
        return ModelAggregator.fedavg(weights_list, contributions)
    
    @staticmethod
    def trimmed_mean(weights_list: List[List[np.ndarray]], 
                     trim_ratio: float = 0.1) -> List[np.ndarray]:
        """
        Trimmed mean aggregation (robust to outliers)
        Args:
            weights_list: List of model weights from each client
            trim_ratio: Ratio of clients to trim from each end
        """
        n_clients = len(weights_list)
        n_trim = int(n_clients * trim_ratio)
        
        avg_weights = []
        for weights_per_layer in zip(*weights_list):
            # Convert to numpy array for easier manipulation
            weights_array = np.array(weights_per_layer)
            
            # Sort along client dimension
            sorted_idx = np.argsort(weights_array, axis=0)
            
            # Trim highest and lowest values
            trimmed = weights_array[sorted_idx[n_trim:-n_trim]]
            
            # Take mean of remaining values
            layer_weights = np.mean(trimmed, axis=0)
            avg_weights.append(layer_weights)
            
        return avg_weights
    
    @staticmethod
    def median(weights_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        """
        Median aggregation (most robust to outliers)
        Args:
            weights_list: List of model weights from each client
        """
        avg_weights = []
        for weights_per_layer in zip(*weights_list):
            layer_weights = np.median(weights_per_layer, axis=0)
            avg_weights.append(layer_weights)
            
        return avg_weights

class AggregationStrategy:
    """Handles selection and application of aggregation strategies"""
    
    def __init__(self, strategy='fedavg'):
        self.aggregator = ModelAggregator()
        self.strategy = strategy
        
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Aggregate client updates using selected strategy
        Args:
            client_updates: List of client updates containing weights and metrics
        """
        weights_list = [update['weights'] for update in client_updates]
        
        if self.strategy == 'fedavg':
            return self.aggregator.fedavg(weights_list)
            
        elif self.strategy == 'fedavg_quality':
            metrics = [update['metrics'] for update in client_updates]
            return self.aggregator.fedavg_with_quality(weights_list, metrics)
            
        elif self.strategy == 'fedavg_size':
            data_sizes = [update['data_size'] for update in client_updates]
            return self.aggregator.fedavg_with_size(weights_list, data_sizes)
            
        elif self.strategy == 'trimmed_mean':
            return self.aggregator.trimmed_mean(weights_list)
            
        elif self.strategy == 'median':
            return self.aggregator.median(weights_list)
            
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.strategy}")

# Example usage in FL server
if __name__ == "__main__":
    # Initialize aggregator with strategy
    aggregator = AggregationStrategy(strategy='fedavg_quality')
    
    # Example client updates
    client_updates = [
        {
            'client_id': 1,
            'weights': [...],  # model weights
            'metrics': {'iou': 0.8},
            'data_size': 1000
        },
        {
            'client_id': 2,
            'weights': [...],  # model weights
            'metrics': {'iou': 0.7},
            'data_size': 1200
        }
    ]
    
    # Aggregate updates
    aggregated_weights = aggregator.aggregate(client_updates)