import os
import cv2
import numpy as np
from glob import glob
from config.settings import MODEL_CONFIG, MASK_PATHS, REGIONS

class BandLoader:
    """Class to handle loading of satellite image bands"""
    
    def __init__(self, img_height=MODEL_CONFIG['IMG_HEIGHT'], 
                 img_width=MODEL_CONFIG['IMG_WIDTH']):
        self.img_height = img_height
        self.img_width = img_width
    
    def load_band_image(self, band_path):
        """Load a single band image"""
        band_img = cv2.imread(band_path, cv2.IMREAD_UNCHANGED)
        if band_img is None:
            raise ValueError(f"Failed to load band from {band_path}")
        return cv2.resize(band_img, (self.img_width, self.img_height))
    
    def load_bands(self, year_path):
        images = []
        band_files = {
            'B02': glob(f"{year_path}/**/*_B02_10m.jp2", recursive=True),
            'B03': glob(f"{year_path}/**/*_B03_10m.jp2", recursive=True),
            'B04': glob(f"{year_path}/**/*_B04_10m.jp2", recursive=True),
            'B08': glob(f"{year_path}/**/*_B08_10m.jp2", recursive=True)
        }
        
        # Process each set of bands
        for b2, b3, b4, b8 in zip(band_files['B02'], band_files['B03'], 
                                 band_files['B04'], band_files['B08']):
            try:
                # Load bands
                bands = [
                    self.load_band_image(band_path) 
                    for band_path in [b2, b3, b4, b8]
                ]
                
                # Process bands
                processed_bands = []
                for band in bands:
                    band = band.astype(np.float32)
                    band = (band - np.min(band)) / (np.max(band) - np.min(band))
                    processed_bands.append(band)
                
                # Stack bands
                multi_band_img = np.stack(processed_bands, axis=-1)
                images.append(multi_band_img)
                
            except Exception as e:
                print(f"Error processing bands: {str(e)}")
        
        return images

class MaskLoader:
    """Class to handle loading of mask images"""
    
    def __init__(self, img_height=MODEL_CONFIG['IMG_HEIGHT'], 
                 img_width=MODEL_CONFIG['IMG_WIDTH']):
        self.img_height = img_height
        self.img_width = img_width
    
    def load_masks(self, mask_paths, region_id):
        """Load all masks for specific region"""
        masks = []
        region_mask_paths = mask_paths[f'region_{region_id}']
        print(f"\nLoading masks for region {region_id}")
        
        for year in range(2015,2024):
            if year in region_mask_paths:
                try:
                    mask_path = region_mask_paths[year]
                    print(f"Loading mask from: {mask_path}")
                    
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        print(f"Successfully loaded mask for year {year}")
                        mask = cv2.resize(mask, (self.img_width, self.img_height))
                        mask = (mask > 0).astype(np.uint8)
                        masks.append(mask)
                        print(f"Processed mask for year {year}: shape={mask.shape}, unique values={np.unique(mask)}")
                    else:
                        print(f"Error: Unable to load mask for year {year} from {mask_path}")
                except Exception as e:
                    print(f"Error processing mask for year {year}: {str(e)}")
            else:
                print(f"No mask path found for year {year}")
        
        if masks:
            print(f"Successfully loaded {len(masks)} masks for region {region_id}")
            return np.array(masks)
        else:
            print(f"Warning: No masks were loaded for region {region_id}")
            return None
class RegionalDataLoader:
    """Main class to load all data for a region"""
    
    def __init__(self, region_id):
        """Initialize with region ID instead of path"""
        self.region_id = region_id
        self.region_path = REGIONS[f'region_{region_id}']
        self.band_loader = BandLoader()
        self.mask_loader = MaskLoader()
    
    def load_all_data(self):
        """Load all data for the region"""
        all_images = {}
        all_masks = {}
        
        # Get years from region-specific mask paths
        region_mask_paths = MASK_PATHS[f'region_{self.region_id}']
        years = sorted([year for year in range(2015, 2024) 
                       if year in region_mask_paths])
        
        print(f"Processing years for region {self.region_id}: {years}")
        
        # Load images for each year
        for year in years:
            year_path = os.path.join(self.region_path, f"Year_{year}")
            if os.path.exists(year_path):
                print(f"Loading data for year {year}...")
                year_images = self.band_loader.load_bands(year_path)
                if year_images:
                    all_images[year] = year_images
                    print(f"Loaded {len(year_images)} images for year {year}")
            else:
                print(f"Warning: Path not found: {year_path}")
        
        # Load masks for this region
        masks = self.mask_loader.load_masks(MASK_PATHS, self.region_id)
        if masks is not None:
            for idx, year in enumerate(years):
                all_masks[year] = masks[idx]
        
        return all_images, all_masks

def get_regional_data(region_id):
    """Utility function to load data for a region"""
    if isinstance(region_id, str) and region_id.startswith('region_'):
        region_id = int(region_id.split('_')[1])
    loader = RegionalDataLoader(region_id)
    return loader.load_all_data()

# Example usage:
if __name__ == "__main__":
    # Test loading for both regions
    for region_id in [1, 2]:
        print(f"\nLoading data for Region {region_id}")
        all_images, all_masks = get_regional_data(region_id)
        print(f"Region {region_id} data loading complete!")
        print(f"Years with images: {sorted(all_images.keys())}")
        print(f"Years with masks: {sorted(all_masks.keys())}")