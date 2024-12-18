from flask import Flask, render_template, jsonify
import os
import json
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fl_dashboard')

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from fl.server import FLServer
from config.settings import FL_CONFIG

# Initialize Flask app
app = Flask(__name__)

# Get FL server instance
fl_server = FLServer.get_instance()

# Define results directory
RESULTS_DIR = os.path.join(project_root, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.route('/')
def dashboard():
    """Render main dashboard"""
    try:
        return render_template('dashboard.html')
    except Exception as e:
        logger.error(f"Error rendering dashboard: {str(e)}")
        return jsonify({'error': 'Failed to load dashboard'}), 500

@app.route('/api/status')
def get_status():
    """Get overall FL training status"""
    try:
        status = fl_server.get_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({'error': 'Failed to get status'}), 500

@app.route('/api/metrics')
def get_metrics():
    """Get training metrics"""
    try:
        status = fl_server.get_status()
        return jsonify({
            'current_round': status['current_round'],
            'total_rounds': status['total_rounds'],
            'metrics_history': status.get('metrics_history', []),
            'training_stats': status.get('training_stats', {}),
            'performance_metrics': status.get('performance_metrics', {})
        })
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return jsonify({'error': 'Failed to get metrics'}), 500

@app.route('/api/clients')
def get_clients():
    """Get active clients information"""
    try:
        return jsonify({
            'active_clients': list(fl_server.active_clients),
            'updates_pending': len(fl_server.client_updates),
            'required_clients': fl_server.min_clients,
            'training_completed': fl_server.training_completed
        })
    except Exception as e:
        logger.error(f"Error getting client info: {str(e)}")
        return jsonify({'error': 'Failed to get client information'}), 500

@app.route('/api/regional_metrics/<region_id>')
def get_regional_metrics(region_id):
    """Get metrics for specific region"""
    try:
        history_path = Path(RESULTS_DIR) / 'clients' / f'client_{region_id}' / 'training_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            return jsonify(history)
        else:
            logger.warning(f"No history found for region {region_id}")
            return jsonify({'error': 'Region history not found'}), 404
    except Exception as e:
        logger.error(f"Error getting regional metrics: {str(e)}")
        return jsonify({'error': 'Failed to get regional metrics'}), 500

@app.route('/api/model_performance')
def get_model_performance():
    """Get current model performance metrics"""
    try:
        if fl_server.metrics_history:
            latest_metrics = fl_server.metrics_history[-1]['metrics']
            aggregated_metrics = {
                'iou': sum(m['iou'] for m in latest_metrics) / len(latest_metrics),
                'f1': sum(m['f1'] for m in latest_metrics) / len(latest_metrics),
                'precision': sum(m['precision'] for m in latest_metrics) / len(latest_metrics),
                'recall': sum(m['recall'] for m in latest_metrics) / len(latest_metrics)
            }
            return jsonify(aggregated_metrics)
        return jsonify({'message': 'No metrics available yet'}), 200
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        return jsonify({'error': 'Failed to get model performance'}), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

def start_dashboard():
    """Start the dashboard server"""
    try:
        port = FL_CONFIG.get('WEB_PORT', 5002)
        logger.info(f"Starting FL Dashboard on port {port}")
        app.run(debug=True, port=port)
    except Exception as e:
        logger.error(f"Failed to start dashboard: {str(e)}")
        raise

if __name__ == '__main__':
    start_dashboard()