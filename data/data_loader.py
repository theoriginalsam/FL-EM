import os
import cv2
import numpy as np
from glob import glob
from config.settings import MODEL_CONFIG, MASK_PATHS

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
        for band_name, paths in band_files.items():
            print(f"Band {band_name}: {paths}")
        
        """ if not all(band_files.values()):
            print("Error: Some bands are missing!")
            return images

        
        # Check if we found all bands
        if not all(band_files.values()):
            print("Warning: Not all band files found")
            return images
         """
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
                    # Convert to float32
                    band = band.astype(np.float32)
                    # Normalize
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
    
    def load_masks(self, mask_paths):
        """Load all masks"""
        masks = []
        
        for year in range(2015, 2017):
            if year in mask_paths:
                try:
                    # Load mask
                    mask = cv2.imread(mask_paths[year], cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        # Resize and binarize
                        mask = cv2.resize(mask, (self.img_width, self.img_height))
                        mask = (mask > 0).astype(np.uint8)
                        masks.append(mask)
                    else:
                        print(f"Error: Unable to load mask for year {year}")
                except Exception as e:
                    print(f"Error processing mask for year {year}: {str(e)}")
        
        return np.array(masks) if masks else None

class RegionalDataLoader:
    """Main class to load all data for a region"""
    
    def __init__(self, region_path):
        self.region_path = region_path
        
        self.band_loader = BandLoader()
        self.mask_loader = MaskLoader()
    
    def load_all_data(self):
        """Load all data for the region"""
        # Initialize storage
        all_images = {}
        all_masks = {}
        print(self.region_path)
        # Get years from mask paths
        years = sorted([year for year in range(2015, 2017) 
                       if year in MASK_PATHS])
        print(f"Processing years: {years}")
        
        # Load images for each year
        for year in years:
            year_path = os.path.join(self.region_path, f"Year_{year}")
            print(year_path)
            #print("/Users/samir/Desktop/MTSU/Research/FL-CS6600/forest_change_fl/data/region_2")
            if os.path.exists("/Users/samir/Desktop/MTSU/MTSU-3rd Sem/Datas/Project_Dataset/" + year_path):
                year_path = os.path.join("/Users/samir/Desktop/MTSU/MTSU-3rd Sem/Datas/Project_Dataset", year_path)
                print(f"Loading data for year {year}...")
                year_images = self.band_loader.load_bands(year_path)
                #print(year_path)
                if year_images:
                    all_images[year] = year_images
                    print(f"Loaded {len(year_images)} images for year {year}")
        
        # Load masks
        masks = self.mask_loader.load_masks(MASK_PATHS)
        if masks is not None:
            for idx, year in enumerate(years):
                all_masks[year] = masks[idx]
        
        return all_images, all_masks

def get_regional_data(region_path):
    """Utility function to load data for a region"""
    loader = RegionalDataLoader(region_path)
    return loader.load_all_data()

# Example usage:
if __name__ == "__main__":
    from config.settings import REGIONS
    
    region_path = REGIONS['region_2']
    print(f"Loading data from {region_path}")
    
    all_images, all_masks = get_regional_data(region_path)
    print("\nData loading complete!")
    print(f"Years with images: {sorted(all_images.keys())}")
    print(f"Years with masks: {sorted(all_masks.keys())}")