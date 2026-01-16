This script (gmapID_retriever_GoogleAPI.py) is designed to enrich your dataset by fetching official Google Maps data for a specific list of restaurants. It performs the following actions:
1.	Filters your raw dataset to only include relevant restaurants.
2.	Identifies the unique Google Place ID for each restaurant.
3.	Fetches Details: GPS coordinates, official address, and editorial summaries.
4.	Downloads Photos: Saves up to 3 high-quality images per restaurant.
5.	Budget Protection: Monitors spending in real-time to prevent exceeding Google Cloud billing limits.
2. Google Cloud API Key
You must have a valid Google Cloud API Key with the Places API (New or Legacy) enabled.
•	Cost Warning: Google Places API is not free. This script uses a budget limit, but you must have a billing account attached to your API key. With a student account you can get up to ~$200 of credits.
4. Input Files Structure
Ensure these files are in the same directory as the script (or update the paths in the config):
1.	item_metadata.csv: Must contain at least the columns gmap_id, name, and address.
2.	restaurant_item_map_rest_20.json: A dictionary used as a filter. The script extracts values from this JSON to decide which gmap_ids are relevant.
5. How to Run
Run the script from your terminal:
Bash
python gmapID_retriever_GoogleAPI.py

 
6. Outputs
A. JSON Data (resolved_restaurants.json)
A structured file containing the enriched data. Example entry:
JSON
"gmap_id_123": {
  "place_id": "ChIJ...",
  "name": "Le Diplomate",
  "address": "1601 14th St NW, Washington, DC...",
  "description": "Bustling French brasserie...",
  "location": { "lat": 38.91, "lng": -77.03 },
  "photos": ["photos/ChIJ..._photo_0.jpg"]
}
B. Photo Directory (photos/)
Contains JPG images renamed by Place ID to avoid conflicts.
•	[PlaceID]_photo_0.jpg
•	[PlaceID]_photo_1.jpg



