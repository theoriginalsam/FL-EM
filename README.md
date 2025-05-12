# Forest Change Detection using Federated Learning

This project implements a Federated Learning (FL) system for forest change detection, allowing multiple clients to train models collaboratively while preserving data privacy.

## Project Structure

```
forest_change_fl/
├── config/             # Configuration files
├── data/              # Dataset directory
├── fl/                # Federated Learning implementation
├── models/            # Model architectures
├── utils/             # Utility functions
├── web/               # Web interface
├── logs/              # Training logs
├── checkpoints/       # Model checkpoints
└── results/           # Results and visualizations
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd forest_change_fl
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Install the project in development mode:
```bash
pip install -e .
```

## Running the Project

### Starting the Server

The FL server can be started using:
```bash
python run_server.py [--port PORT] [--host HOST]
```

Default configuration:
- Host: localhost
- Port: 5000

### Running Clients

1. Individual client:
```bash
python run_client.py
```

2. Multiple clients using tmux (Linux/Mac):
```bash
./run_all_clients_tmux.sh
```

## Project Components

- **Server**: Coordinates the federated learning process
- **Clients**: Train models on local data
- **Web Interface**: Visualize training progress and results
- **Models**: Deep learning architectures for forest change detection

## Dependencies

Main dependencies include:
- TensorFlow 2.13.0
- Flask 2.3.3
- NumPy 1.24.3
- Pandas 2.0.3
- scikit-learn 1.3.0
- OpenCV 4.8.0
- Plotly 5.16.1

## Data

### Dataset Regions
The project uses Hansen dataset (2023 Ground Truth) from Copernicus Sentinel-2 L2A 10M resolution data. The following regions are included:

1. **Region 1 - Amazon**
   - Sentinel-2 Tile: T22MCV
   - Time Period: 2015-2023
   - Ground Truth: Forest vs Non-forest classification

2. **Region 2 - West Africa**
   - Sentinel-2 Tile: T23MPQK
   - Time Period: 2015-2023

3. **Region 3 - Congo Basin**
   - Sentinel-2 Tile:T33NYA
   - Time Period: 2015-2023

4. **Region 4 - East Africa**
   - Sentinel-2 Tile: T36KZF
   - Time Period: 2015-2023

5. **Region 5 - Greater Mekong**
   - Sentinel-2 Tile: T48PXA
   - Time Period: 2015-2023

6. **Region 6 - Central Atlantic Forest**
   - Sentinel-2 Tile: T20JPT
   - Time Period: 2015-2023

### Downloading the Data

1. **Access Copernicus Open Access Hub**
   - Visit [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
   - Create an account if you don't have one

2. **Download Sentinel-2 Data**
   - Search for Sentinel-2 L2A products
   - Use the six-digit tile codes mentioned above to locate specific regions
   - Download data for the period 2015-2023
   - Ensure to download L2A (Level-2A) products with 10m resolution

3. **Data Organization**
   - Place downloaded data in the `data/` directory
   - Organize by region using the following structure:
   ```
   data/
   ├── region1_amazon/
   ├── region2_west_africa/
   ├── region3_congo_basin/
   ├── region4_east_africa/
   ├── region5_greater_mekong/
   └── region6_atlantic_forest/
   ```

4. **Ground Truth Data**
   - For Region 1 (Amazon), ground truth data is available through Google Earth Engine
   - Use the same region boundaries to extract forest vs non-forest classifications
   - Process and align ground truth data with Sentinel-2 imagery

Note: The six-digit codes (e.g., T22MCV) represent specific regions in the Copernicus Sentinel-2 grid system. These codes help in locating and downloading the correct data for each region of interest.

### Ground Truth Data Processing

The ground truth data is processed using Google Earth Engine (GEE) to generate forest vs non-forest classifications for each region. Follow these steps to generate the ground truth data:

1. **Access Google Earth Engine**
   - Go to [Google Earth Engine Code Editor](https://code.earthengine.google.com/)
   - Sign in with your Google account
   - Create a new script

2. **Process Ground Truth Data**
   - Copy the following code template to your GEE script:
   ```javascript
   // Define the Hansen dataset base URL
   var baseDatasetUrl = 'UMD/hansen/global_forest_change_';

   // Define the years and versions
   var yearVersions = {
     2015: 'v1_3',
     2016: 'v1_4',
     2017: 'v1_5',
     2018: 'v1_6',
     2019: 'v1_7',
     2020: 'v1_8',
     2021: 'v1_9',
     2022: 'v1_10',
     2023: 'v1_11'
   };

   // Define your region of interest (ROI) using coordinates from Sentinel-2 data
   var roi = ee.Geometry.Polygon([
     // Add coordinates for your specific region here
     // Example format:
     // [longitude, latitude],
     // [longitude, latitude],
     // ...
   ]);

   // Process data for each year
   for (var year in yearVersions) {
     var dataset = ee.Image(baseDatasetUrl + year + '_' + yearVersions[year]);
     var treeCover2000 = dataset.select('treecover2000');
     var loss = dataset.select('loss');
     
     // Generate forest mask (>30% cover and no loss)
     var forestMask = treeCover2000.gt(30).and(loss.not());
     var clippedForestData = forestMask.clip(roi);

     // Export to Google Drive
     Export.image.toDrive({
       image: clippedForestData,
       description: 'Forest_vs_NoForest_' + year,
       scale: 30,
       region: roi,
       maxPixels: 1e9
     });
   }
   ```

3. **Region-Specific Coordinates**
   For each region, use the following coordinates from the Sentinel-2 data:

   - **Region 1 - Amazon (T22MCV)**
     ```javascript
     var roi = ee.Geometry.Polygon([
       [-62.01370349592315, -24.410215071618904],
       [-60.931158404443025, -24.399291802536993],
       [-60.914570422645404, -25.39026508821524],
       [-62.0057899874797, -25.401692598229527],
       [-62.01370349592315, -24.410215071618904]
     ]);
     ```

   - **Region 2 - West Africa (T23MPQK)**
     [Add coordinates from Sentinel-2 data]

   - **Region 3 - Congo Basin (T33NYA)**
     [Add coordinates from Sentinel-2 data]

   - **Region 4 - East Africa (T36KZF)**
     [Add coordinates from Sentinel-2 data]

   - **Region 5 - Greater Mekong (T48PXA)**
     [Add coordinates from Sentinel-2 data]

   - **Region 6 - Central Atlantic Forest (T20JPT)**
     [Add coordinates from Sentinel-2 data]

4. **Export and Download**
   - Run the script in GEE
   - Each year's data will be exported to your Google Drive
   - Download the exported files and place them in the corresponding region folder in your project's `data/` directory

5. **Data Organization**
   After downloading, organize the ground truth data as follows:
   ```
   data/
   ├── region1_amazon/
   │   ├── sentinel2_data/
   │   └── ground_truth/
   │       ├── Forest_vs_NoForest_2015.tif
   │       ├── Forest_vs_NoForest_2016.tif
   │       └── ...
   ├── region2_west_africa/
   │   └── ...
   └── ...
   ```

Note: The ground truth data is generated at 30m resolution using the Hansen Global Forest Change dataset. The forest classification threshold is set at 30% tree cover, and areas with forest loss are excluded from the forest class.

## Results

Training results, model checkpoints, and visualizations are stored in:
- `checkpoints/`: Model weights and training states
- `results/`: Analysis and visualization outputs
- `logs/`: Training logs and metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Contact

spoudel04@gmaiL.com