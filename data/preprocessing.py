import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from config.settings import MODEL_CONFIG

class DataAugmentation:
    """Data augmentation techniques for satellite imagery."""
    
    @staticmethod
    def random_flip(image):
        """Random horizontal and vertical flips."""
        if np.random.random() > 0.5:
            image = np.fliplr(image)
        if np.random.random() > 0.5:
            image = np.flipud(image)
        return image
    
    @staticmethod
    def random_rotate(image):
        """Random 90-degree rotations."""
        k = np.random.randint(0, 4)  # 0, 1, 2, or 3 times 90 degrees
        return np.rot90(image, k)
    
    @staticmethod
    def adjust_brightness(image, factor_range=(0.8, 1.2)):
        """Adjust brightness while preserving valid pixel range."""
        factor = np.random.uniform(*factor_range)
        return np.clip(image * factor, 0, 1)
    
    @staticmethod
    def add_noise(image, std=0.01):
        """Add random noise to image."""
        noise = np.random.normal(0, std, image.shape)
        return np.clip(image + noise, 0, 1)

class PatchExtractor:
    """Extract patches from satellite images."""
    
    def __init__(self, patch_size=MODEL_CONFIG['PATCH_SIZE']):
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
    
    def is_valid_location(self, i, j, image_shape):
        """Check if location is valid for patch extraction."""
        return (i >= self.half_patch and 
                i < image_shape[0] - self.half_patch and
                j >= self.half_patch and 
                j < image_shape[1] - self.half_patch)
    
    def extract_patch(self, image, center_i, center_j):
        """Extract patch centered at given location."""
        i_start = center_i - self.half_patch
        i_end = center_i + self.half_patch
        j_start = center_j - self.half_patch
        j_end = center_j + self.half_patch
        
        return image[i_start:i_end, j_start:j_end]

class BalancedPatchGenerator:
    """Generate balanced patches for training."""
    
    def __init__(self, patch_size=MODEL_CONFIG['PATCH_SIZE'], 
                 samples_per_class=1000, augment=True):
        self.patch_extractor = PatchExtractor(patch_size)
        self.samples_per_class = samples_per_class
        self.augment = augment
        self.augmenter = DataAugmentation()
    
    def create_balanced_patches(self, all_images, all_masks):
        """Create balanced patches from images and masks."""
        patches_X = []
        patches_y = []
        
        years = sorted(all_images.keys())
        for i in range(len(years)-1):
            year1, year2 = years[i], years[i+1]
            
            if year1 in all_images and year2 in all_images:
                images1 = all_images[year1][0]
                images2 = all_images[year2][0]
                mask = all_masks[year2]
                
                # Get locations for each class
                forest_locs = np.where(mask > 0)
                non_forest_locs = np.where(mask == 0)
                
                # Sample both classes
                for locations, label in [(forest_locs, 1), (non_forest_locs, 0)]:
                    self._sample_patches(
                        images1, images2, mask, locations, 
                        patches_X, patches_y
                    )
        
        return self._finalize_patches(patches_X, patches_y)
    
    def _sample_patches(self, images1, images2, mask, locations, patches_X, patches_y):
        """Sample patches for a specific class."""
        for _ in range(self.samples_per_class):
            idx = np.random.randint(0, len(locations[0]))
            i_loc, j_loc = locations[0][idx], locations[1][idx]
            
            if self.patch_extractor.is_valid_location(i_loc, j_loc, mask.shape):
                # Extract patches
                patch1 = self.patch_extractor.extract_patch(images1, i_loc, j_loc)
                patch2 = self.patch_extractor.extract_patch(images2, i_loc, j_loc)
                mask_patch = self.patch_extractor.extract_patch(mask, i_loc, j_loc)
                
                # Apply augmentation if enabled
                if self.augment:
                    patch1 = self.augmenter.random_flip(patch1)
                    patch2 = self.augmenter.random_flip(patch2)
                    patch1 = self.augmenter.adjust_brightness(patch1)
                    patch2 = self.augmenter.adjust_brightness(patch2)
                
                # Combine patches
                combined_patch = np.concatenate([patch1, patch2], axis=-1)
                patches_X.append(combined_patch)
                patches_y.append(mask_patch)
    
    def _finalize_patches(self, patches_X, patches_y):
        """Convert lists to arrays and normalize."""
        X = np.array(patches_X)
        y = np.expand_dims(np.array(patches_y), axis=-1)
        
        # Normalize features
        X = self._normalize_features(X)
        
        print(f"Created patches shapes - X: {X.shape}, y: {y.shape}")
        print(f"Class balance - Forest: {np.mean(y == 1):.3f}, "
              f"Non-forest: {np.mean(y == 0):.3f}")
        
        return X, y
    
    @staticmethod
    def _normalize_features(X):
        """Normalize features independently."""
        n_features = X.shape[-1]
        X_normalized = np.zeros_like(X)
        
        for i in range(n_features):
            feature = X[..., i]
            min_val = np.min(feature)
            max_val = np.max(feature)
            if max_val > min_val:
                X_normalized[..., i] = (feature - min_val) / (max_val - min_val)
            else:
                X_normalized[..., i] = feature
        
        return X_normalized

class DataGenerator(Sequence):
    """Data generator for training."""
    
    def __init__(self, X, y, batch_size=32, augment=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.augmenter = DataAugmentation()
        self.indexes = np.arange(len(self.X))
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        """Get batch at position idx."""
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.X))
        
        # Get batch indexes
        batch_indexes = self.indexes[start_idx:end_idx]
        
        # Get batch data
        batch_X = self.X[batch_indexes]
        batch_y = self.y[batch_indexes]
        
        # Apply augmentation if enabled
        if self.augment:
            batch_X = np.array([
                self._augment_sample(x) for x in batch_X
            ])
        
        return batch_X, batch_y
    
    def _augment_sample(self, x):
        """Apply augmentation to a single sample."""
        x = self.augmenter.random_flip(x)
        x = self.augmenter.random_rotate(x)
        x = self.augmenter.adjust_brightness(x)
        return x
    
    def on_epoch_end(self):
        """Shuffle indexes after each epoch."""
        np.random.shuffle(self.indexes)

# Example usage
if __name__ == "__main__":
    from data_loader import get_regional_data
    from config.settings import REGIONS
    
    # Load data
    region_path = REGIONS['region_2']
    all_images, all_masks = get_regional_data(region_path)
    
    # Create patches
    generator = BalancedPatchGenerator()
    X, y = generator.create_balanced_patches(all_images, all_masks)
    
    # Create data generator
    training_generator = DataGenerator(X, y)
    print("Data preprocessing complete!")