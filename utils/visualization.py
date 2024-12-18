
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from datetime import datetime

class FLVisualizer:
    def __init__(self, save_dir='results/visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_training_metrics(self, history, client_id=None, save=True):
        """Plot training metrics (loss, accuracy, etc.)"""
        metrics = ['loss', 'binary_accuracy', 'auc']
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        for idx, metric in enumerate(metrics):
            axes[idx].plot(history[metric], label=f'Training {metric}')
            if f'val_{metric}' in history:
                axes[idx].plot(history[f'val_{metric}'], label=f'Validation {metric}')
            axes[idx].set_title(f'{metric.replace("_", " ").title()}')
            axes[idx].set_xlabel('Epoch')
            axes[idx].legend()
            
        plt.tight_layout()
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'training_metrics_{client_id}_{timestamp}.png' if client_id else f'training_metrics_{timestamp}.png'
            plt.savefig(os.path.join(self.save_dir, filename))
        plt.show()

    def plot_fl_metrics(self, metrics_history, save=True):
        """Plot federated learning metrics across rounds"""
        rounds = range(len(metrics_history))
        metrics = ['iou', 'f1', 'precision', 'recall']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            values = [np.mean([m[metric] for m in round_metrics['metrics']]) 
                     for round_metrics in metrics_history]
            stds = [np.std([m[metric] for m in round_metrics['metrics']]) 
                   for round_metrics in metrics_history]
            
            axes[idx].plot(rounds, values, 'b-', label=f'Mean {metric}')
            axes[idx].fill_between(rounds, 
                                 np.array(values) - np.array(stds),
                                 np.array(values) + np.array(stds),
                                 alpha=0.3)
            axes[idx].set_title(f'{metric.upper()} over Rounds')
            axes[idx].set_xlabel('Round')
            axes[idx].set_ylabel(metric)
            axes[idx].grid(True)
            
        plt.tight_layout()
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(os.path.join(self.save_dir, f'fl_metrics_{timestamp}.png'))
        plt.show()

    def plot_client_comparison(self, client_metrics, save=True):
        """Plot comparison of client performances"""
        metrics = ['iou', 'f1', 'precision', 'recall']
        client_ids = list(client_metrics.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            values = [client_metrics[client_id][metric] for client_id in client_ids]
            axes[idx].bar(client_ids, values)
            axes[idx].set_title(f'{metric.upper()} by Client')
            axes[idx].set_xlabel('Client ID')
            axes[idx].set_ylabel(metric)
            
        plt.tight_layout()
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(os.path.join(self.save_dir, f'client_comparison_{timestamp}.png'))
        plt.show()

    def plot_communication_costs(self, cost_history, save=True):
        """Plot communication costs over rounds"""
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(cost_history)), cost_history, 'g-')
        plt.title('Communication Cost over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Communication Cost (MB)')
        plt.grid(True)
        
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(os.path.join(self.save_dir, f'communication_costs_{timestamp}.png'))
        plt.show()

    def plot_aggregation_times(self, time_history, save=True):
        """Plot aggregation times over rounds"""
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(time_history)), time_history, 'r-')
        plt.title('Aggregation Time over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(os.path.join(self.save_dir, f'aggregation_times_{timestamp}.png'))
        plt.show()

    def plot_confusion_matrices(self, confusion_matrices, save=True):
        """Plot confusion matrices for all clients"""
        n_clients = len(confusion_matrices)
        fig, axes = plt.subplots(1, n_clients, figsize=(5*n_clients, 5))
        if n_clients == 1:
            axes = [axes]
            
        for idx, (client_id, cm) in enumerate(confusion_matrices.items()):
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx])
            axes[idx].set_title(f'Client {client_id}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
            
        plt.tight_layout()
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(os.path.join(self.save_dir, f'confusion_matrices_{timestamp}.png'))
        plt.show()

def create_visualization_report(fl_metrics, client_metrics, output_dir='results/reports'):
    """Create a comprehensive visualization report"""
    os.makedirs(output_dir, exist_ok=True)
    visualizer = FLVisualizer(save_dir=output_dir)
    
    # Plot all metrics
    visualizer.plot_fl_metrics(fl_metrics['metrics_history'])
    visualizer.plot_client_comparison(client_metrics)
    visualizer.plot_communication_costs(fl_metrics['communication_costs'])
    visualizer.plot_aggregation_times(fl_metrics['aggregation_times'])
    
    # Save metrics to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(output_dir, f'metrics_report_{timestamp}.json'), 'w') as f:
        json.dump({
            'fl_metrics': fl_metrics,
            'client_metrics': client_metrics
        }, f, indent=2)
