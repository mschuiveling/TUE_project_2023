import os
import openslide
from PIL import Image
import json

# Set the folder path
wsi_path = "/mnt/d/TIL_Melanoma_train_database/NDPI_files_UMCU_Ruben&GEOJSON_Tile_selection/rerun_tile_selection"
output_folder_path = "/mnt/d/TIL_Melanoma_train_database/NDPI_files_UMCU_Ruben&GEOJSON_Tile_selection/tiles_1024"

# Create the output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Function to extract and save tiles
def extract_and_save_tile(slide, x, y, level, height, width, output_folder, prefix):
    tile = slide.read_region((x, y), level, (width, height))
    tile_file_name = f"{prefix}_{x}_{y}.png"
    tile_file_path = os.path.join(output_folder, tile_file_name)
    tile.convert("RGB").save(tile_file_path)

# Iterate over the files in the folder
for file_name in os.listdir(wsi_path):
    # Check if the file is a geojson file
    if file_name.endswith(".geojson"):
        # Open geojson
        annotations = os.path.join(wsi_path, file_name)
        with open(annotations) as f:
            data = json.load(f)

        # Open WSI
        WSI_file = annotations.replace('.geojson', '.ndpi')
        slide = openslide.OpenSlide(WSI_file)

        # Set the level at which to read the region (0 = highest resolution level)
        level = 0

        # Extract the first x and y coordinates from the first polygon
        coordinates_1 = data['features'][0]['geometry']['coordinates'][0]
        x1 = coordinates_1[0][0]
        y1 = coordinates_1[0][1]

        # Extract the second x and y coordinates from the second polygon
        coordinates_2 = data['features'][1]['geometry']['coordinates'][0]
        x2 = coordinates_2[0][0]
        y2 = coordinates_2[0][1]

        # Set the height and width for the region
        height = 1024
        width = 1024

        # Extract and save the required tiles
        extract_and_save_tile(slide, x1, y1, level, height, width, output_folder_path, f"{os.path.splitext(file_name)[0]}_1_1")
        extract_and_save_tile(slide, x1 + 1024, y1, level, height, width, output_folder_path, f"{os.path.splitext(file_name)[0]}_1_2")
        extract_and_save_tile(slide, x1, y1 + 1024, level, height, width, output_folder_path, f"{os.path.splitext(file_name)[0]}_1_3")
        extract_and_save_tile(slide, x1 + 1024, y1 + 1024, level, height, width, output_folder_path, f"{os.path.splitext(file_name)[0]}_1_4")

        extract_and_save_tile(slide, x2, y2, level, height, width, output_folder_path, f"{os.path.splitext(file_name)[0]}_2_1")
        extract_and_save_tile(slide, x2 + 1024, y2, level, height, width, output_folder_path, f"{os.path.splitext(file_name)[0]}_2_2")
        extract_and_save_tile(slide, x2, y2 + 1024, level, height, width, output_folder_path, f"{os.path.splitext(file_name)[0]}_2_3")
        extract_and_save_tile(slide, x2 + 1024, y2 + 1024, level, height, width, output_folder_path, f"{os.path.splitext(file_name)[0]}_2_4")

        print(WSI_file, "Tiles have been formed and saved")
