import folium
import os

def create_dc_map_from_df(df, output_dir):
    dc_center = (38.9072, -77.0369)

    m = folium.Map(
        location=dc_center,
        zoom_start=12,
        tiles="OpenStreetMap"
    )

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=(row["latitude"], row["longitude"]),
            radius=3,
            color="blue",
            fill=True,
            fill_opacity=0.6,
            popup=row["name"]
        ).add_to(m)
    map_file_path = os.path.join(output_dir, "dc_map.html")
    m.save(map_file_path)
    return m




