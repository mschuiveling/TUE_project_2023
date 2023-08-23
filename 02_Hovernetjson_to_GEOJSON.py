import glob 
import os
import json
from json import load, dump
import geojson 

annotations_path = "/mnt/d/TIL_Melanoma_train_database/cell_segmentation/tiles_1024/json"


# Define the classification dictionary for cell types
classification_dict_1 = {"name": "Tumor", "color": [200, 0, 0]}
classification_dict_2 = {"name": "Immune cells", "color": [160, 90, 160]}
classification_dict_3 = {"name": "Other", "color": [250, 200, 0]}
classification_dict_4 = {"name": "Necrosis", "color": [50, 50, 50]}
classification_dict_5 = {"name": "Other", "color": [250, 200, 0]}
classification_dict_6 = {"name": "Other", "color": [250, 200, 0]}


# Iterate over the JSON files in the folder
for file_name in glob.glob(os.path.join(annotations_path, "*.json")):

    # Load the JSON data from file
    with open(file_name) as f:
        data = json.load(f)

        # Transform the JSON to a GeoJSON
        geojson_data = geojson.FeatureCollection([
            geojson.Feature(
                geometry=geojson.Polygon([data['nuc'][key]['contour']]),
                properties={
                    'type': data['nuc'][key]['type'],
                    'type_prob': data['nuc'][key]['type_prob']
                }
            ) for key in data['nuc']
        ])

        # Close polygon of GEOJSON to make it readable in QUpath and change annotations 
        # Iterate through each feature
        for feature in geojson_data['features']:
            # Get the last coordinate of the polygon and append it to the coordinates array
            start_point = feature['geometry']['coordinates'][0][0]
            feature['geometry']['coordinates'][0].append(start_point),

            if feature['properties']['type'] == 1:
                feature['properties']['classification'] = classification_dict_1
                del feature['properties']['type']

            elif feature['properties']['type'] == 2:
                feature['properties']['classification'] = classification_dict_2
                del feature['properties']['type']

            elif feature['properties']['type'] == 3:
                feature['properties']['classification'] = classification_dict_3
                del feature['properties']['type']

            elif feature['properties']['type'] == 4:
                feature['properties']['classification'] = classification_dict_4
                del feature['properties']['type']

            elif feature['properties']['type'] == 5:
                feature['properties']['classification'] = classification_dict_5
                del feature['properties']['type'] 
    
        # Write the GeoJSON data to file
        with open (file_name.replace('.json', ' Hovernet_Qupath.geojson'), 'w') as write_file:
            json.dump(geojson_data, write_file)

        print (file_name, "has been converted to GEOJSON")