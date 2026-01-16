import json
import pandas as pd
import requests
import os
import time

# ======================================
# CONFIGURATION
# ======================================

API_KEY = "API KEY"

CSV_PATH = "item_metadata.csv"
JSON_FILTER_PATH = "restaurant_item_map_rest_20.json"

OUTPUT_JSON = "resolved_restaurants.json"
PHOTO_DIR = "photos"
os.makedirs(PHOTO_DIR, exist_ok=True)

MAX_PHOTOS_PER_PLACE = 3
PHOTO_WIDTH = 1024

MAX_BUDGET_EUR = 200  # your limit
EUR_TO_USD = 1.09  # conversion
MAX_BUDGET_USD = MAX_BUDGET_EUR * EUR_TO_USD

# Google official prices (per request):
COST_FINDPLACE = 0.017  # $17 / 1000 requests
COST_DETAILS = 0.017
COST_PHOTO = 0.007

SLEEP_TIME = 0.15  # avoid rate limits
spent_usd = 0.0


# ======================================
# BUDGET CONTROL
# ======================================

def charge(cost, label):
    """
    Deduct cost from budget. If over budget, stop the run safely.
    """
    global spent_usd

    if spent_usd + cost > MAX_BUDGET_USD:
        print(f"\n### STOP — Budget would exceed limit during {label} ###")
        print(f"Spent: ${spent_usd:.2f} / ${MAX_BUDGET_USD:.2f}")
        return False

    spent_usd += cost
    return True


# ======================================
# GOOGLE PLACES HELPERS
# ======================================

def find_place_from_text(name, address):
    """Resolve restaurant using name + address."""
    if not charge(COST_FINDPLACE, "FindPlaceFromText"):
        return None

    query = f"{name} {address}".strip()

    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": query,
        "inputtype": "textquery",
        "fields": "place_id,name,formatted_address",
        "key": API_KEY
    }

    data = requests.get(url, params=params).json()

    if data.get("status") != "OK":
        return None

    candidates = data.get("candidates", [])
    if not candidates:
        return None

    return candidates[0]  # Best matched place


def fetch_place_details(place_id):
    """Fetch metadata + summary + photo references."""

    if not charge(COST_DETAILS, "PlaceDetails"):
        return None

    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,formatted_address,geometry,editorial_summary,photos",
        "key": API_KEY
    }

    data = requests.get(url, params=params).json()

    if data.get("status") != "OK":
        return None

    return data.get("result")


def download_photo(photo_ref, place_id, idx):
    """Download Google Place photo."""

    if not charge(COST_PHOTO, "PlacePhoto"):
        return None

    url = (
        "https://maps.googleapis.com/maps/api/place/photo"
        f"?maxwidth={PHOTO_WIDTH}&photoreference={photo_ref}&key={API_KEY}"
    )

    r = requests.get(url)
    if r.status_code != 200:
        return None

    filepath = os.path.join(PHOTO_DIR, f"{place_id}_photo_{idx}.jpg")
    with open(filepath, "wb") as f:
        f.write(r.content)

    return filepath


# ======================================
# MAIN LOGIC
# ======================================

def main():
    global spent_usd

    # Load full metadata CSV
    df = pd.read_csv(CSV_PATH)

    # Load JSON filter (only restaurants you actually need)
    with open(JSON_FILTER_PATH, "r", encoding="utf-8") as f:
        filter_ids = set(json.load(f).values())

    print(f"JSON filter contains {len(filter_ids)} restaurant IDs.")

    # Filter CSV to only needed restaurants
    df = df[df["gmap_id"].isin(filter_ids)]
    print(f"Filtered down to {len(df)} rows.")

    # Cache-safe loading
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} cached entries.")
    else:
        results = {}

    # Loop through restaurants
    for idx, row in df.iterrows():

        gmap_id = row["gmap_id"]
        name = str(row["name"])
        address = str(row["address"])

        print(f"\n[{idx}] {name}")
        print(f"Budget: ${spent_usd:.2f} / ${MAX_BUDGET_USD:.2f}")

        # Skip cached
        if gmap_id in results:
            print("  Already resolved — skipping.")
            continue

        # -------------------------------
        # 1. FIND PLACE USING TEXT QUERY
        # -------------------------------
        place = find_place_from_text(name, address)
        if not place:
            print("  Could not resolve with FindPlace — skipping.")
            continue

        place_id = place["place_id"]
        print("  Found Place ID:", place_id)

        # -------------------------------
        # 2. FETCH DETAILS
        # -------------------------------
        details = fetch_place_details(place_id)
        if not details:
            print("  Details lookup failed — skipping.")
            continue

        # -------------------------------
        # 3. FETCH PHOTOS
        # -------------------------------
        photos = []
        if "photos" in details:
            refs = details["photos"][:MAX_PHOTOS_PER_PLACE]
            for i, ph in enumerate(refs):
                p = download_photo(ph["photo_reference"], place_id, i)
                if p:
                    photos.append(p)

        # -------------------------------
        # 4. SAVE RESULT
        # -------------------------------
        results[gmap_id] = {
            "place_id": place_id,
            "name": details.get("name"),
            "address": details.get("formatted_address"),
            "description": details.get("editorial_summary", {}).get("overview"),
            "location": details.get("geometry", {}).get("location"),
            "photos": photos
        }

        # Save partial cache every iteration
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Respect rate limits
        time.sleep(SLEEP_TIME)

        if spent_usd >= MAX_BUDGET_USD:
            print("\n### Budget limit reached — stopping early ###")
            break

    print("\nDONE!")
    print(f"Total spent: ${spent_usd:.2f}")
    print(f"Saved {len(results)} results to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
