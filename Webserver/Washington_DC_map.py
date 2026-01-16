import folium
import pandas as pd

def create_user_dc_recommendation_map(
    user_id,
    topn_csv_path,
    metadata_csv_path,
    output_path
):
    """
    Generate a user-specific DC map with recommended restaurants.
    """

    # Load data
    topn_df = pd.read_csv(topn_csv_path)
    meta_df = pd.read_csv(metadata_csv_path)

    # Select user row
    user_row = topn_df[topn_df["user_id"] == user_id]
    if user_row.empty:
        raise ValueError(f"User {user_id} not found in Top-N file.")

    # Extract gmap IDs (rec_1 ... rec_10)
    gmap_ids = (
        user_row
        .iloc[0]
        .drop("user_id")
        .dropna()
        .astype(str)
        .tolist()
    )

    # Filter metadata
    rec_meta = meta_df[meta_df["gmap_id"].isin(gmap_ids)]

    # Create map centered on DC
    dc_center = (38.9072, -77.0369)
    m = folium.Map(location=dc_center, zoom_start=12, tiles="OpenStreetMap")

    # Add markers
    for _, row in rec_meta.iterrows():
        popup_text = f"""
        <b>{row['name']}</b><br>
        {row.get('address', '')}<br>
        average rating: {row['avg_rating']}
        """

        folium.Marker(
            location=(row["latitude"], row["longitude"]),
            popup=popup_text,
            icon=folium.Icon(color="blue", icon="cutlery", prefix="fa")
        ).add_to(m)

    # Save map
    m.save(output_path)
    return m