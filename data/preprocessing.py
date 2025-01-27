import numpy as np
import cv2
from config.settings import MODEL_CONFIG

def create_balanced_patches_with_oversampling(all_images, all_masks, 
                                            minority_samples=1000, 
                                            majority_samples=1000,
                                            patch_size=64):
    """Create balanced patches with controlled sampling for both classes"""
    patches_X = []
    patches_y = []
    
    years = sorted(all_images.keys())
    for i in range(len(years)-1):
        year1, year2 = years[i], years[i+1]
        
        if year1 in all_images and year2 in all_images and \
           year1 in all_masks and year2 in all_masks:
            
            images1 = all_images[year1][0]
            images2 = all_images[year2][0]
            mask = all_masks[year2]  # Use second year as target
            
            # Find locations for each class
            forest_locs = np.where(mask > 0)
            non_forest_locs = np.where(mask == 0)
            
            half_patch = patch_size // 2
            
            # Sample minority class (forest)
            if len(forest_locs[0]) > 0:
                for _ in range(minority_samples):
                    idx = np.random.randint(0, len(forest_locs[0]))
                    i_loc, j_loc = forest_locs[0][idx], forest_locs[1][idx]
                    
                    if all(coord >= half_patch and coord < dim - half_patch 
                          for coord, dim in zip([i_loc, j_loc], mask.shape)):
                        patch1 = images1[i_loc-half_patch:i_loc+half_patch,
                                      j_loc-half_patch:j_loc+half_patch]
                        patch2 = images2[i_loc-half_patch:i_loc+half_patch,
                                      j_loc-half_patch:j_loc+half_patch]
                        mask_patch = mask[i_loc-half_patch:i_loc+half_patch,
                                        j_loc-half_patch:j_loc+half_patch]
                        
                        combined_patch = np.concatenate([patch1, patch2], axis=-1)
                        patches_X.append(combined_patch)
                        patches_y.append(mask_patch)
            
            # Sample majority class (non-forest)
            if len(non_forest_locs[0]) > 0:
                for _ in range(majority_samples):
                    idx = np.random.randint(0, len(non_forest_locs[0]))
                    i_loc, j_loc = non_forest_locs[0][idx], non_forest_locs[1][idx]
                    
                    if all(coord >= half_patch and coord < dim - half_patch 
                          for coord, dim in zip([i_loc, j_loc], mask.shape)):
                        patch1 = images1[i_loc-half_patch:i_loc+half_patch,
                                      j_loc-half_patch:j_loc+half_patch]
                        patch2 = images2[i_loc-half_patch:i_loc+half_patch,
                                      j_loc-half_patch:j_loc+half_patch]
                        mask_patch = mask[i_loc-half_patch:i_loc+half_patch,
                                        j_loc-half_patch:j_loc+half_patch]
                        
                        combined_patch = np.concatenate([patch1, patch2], axis=-1)
                        patches_X.append(combined_patch)
                        patches_y.append(mask_patch)
    
    X = np.array(patches_X)
    y = np.expand_dims(np.array(patches_y), axis=-1)
    
    print(f"Created patches shapes - X: {X.shape}, y: {y.shape}")
    print(f"Class balance - Forest: {np.mean(y == 1):.3f}, Non-forest: {np.mean(y == 0):.3f}")
    return X, y