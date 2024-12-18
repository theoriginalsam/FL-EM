import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

class ForestChangeMetrics:
    """Metrics for forest change detection"""
    
    @staticmethod
    def calculate_iou(y_true, y_pred, threshold=0.5):
        """Calculate Intersection over Union"""
        y_pred = tf.cast(y_pred > threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        
        return intersection / (union + tf.keras.backend.epsilon())
    
    @staticmethod
    def calculate_change_accuracy(y_true, y_pred, threshold=0.5):
        """Calculate accuracy of detected changes"""
        y_pred = tf.cast(y_pred > threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        correct_pixels = tf.reduce_sum(tf.cast(y_true == y_pred, tf.float32))
        total_pixels = tf.cast(tf.size(y_true), tf.float32)
        
        return correct_pixels / total_pixels
    
    @staticmethod
    def calculate_f1_score(y_true, y_pred, threshold=0.5):
        """Calculate F1 score"""
        y_pred_binary = (y_pred > threshold).astype(np.float32)
        y_true = y_true.astype(np.float32)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true.flatten(),
            y_pred_binary.flatten(),
            average='binary'
        )
        
        return precision, recall, f1
    
    @staticmethod
    def calculate_deforestation_rate(y_true_t1, y_true_t2):
        """Calculate deforestation rate between two time periods"""
        forest_t1 = tf.reduce_sum(y_true_t1)
        forest_t2 = tf.reduce_sum(y_true_t2)
        
        deforestation = tf.maximum(0.0, forest_t1 - forest_t2)
        rate = deforestation / (forest_t1 + tf.keras.backend.epsilon())
        
        return rate
    
    @staticmethod
    def calculate_metrics_dict(y_true, y_pred, threshold=0.5):
        """Calculate all metrics and return as dictionary"""
        precision, recall, f1 = ForestChangeMetrics.calculate_f1_score(
            y_true, y_pred, threshold
        )
        
        return {
            'iou': ForestChangeMetrics.calculate_iou(y_true, y_pred, threshold).numpy(),
            'change_accuracy': ForestChangeMetrics.calculate_change_accuracy(y_true, y_pred, threshold).numpy(),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

def weighted_binary_crossentropy(class_weights):
    """Custom weighted loss function"""
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        weights = tf.where(y_true == 1.0, 
                         tf.cast(class_weights[1], tf.float32),
                         tf.cast(class_weights[0], tf.float32))
        
        bce = -(y_true * tf.math.log(y_pred) + 
                (1 - y_true) * tf.math.log(1 - y_pred))
        
        return tf.reduce_mean(weights * bce)
    return loss