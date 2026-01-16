import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import random
from scipy import sparse
from scipy.sparse.linalg import inv as sparse_inv
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# --- 1. PAGE CONFIGURATION (Must be first) ---
st.set_page_config(
    page_title="DC Restaurant Recommender",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --- 2. OPTIMIZATIONS & UI CLEANUP ---
def hide_developer_tools():
    """
    Injects CSS to hide Streamlit native UI elements and applies custom App CSS.
    """
    combined_style = """
        <style>
        /* --- HIDE STREAMLIT UI ELEMENTS --- */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }

        /* --- APP CUSTOM CSS --- */
        /* CONNECTIONS GRID BUTTONS */
        div.stButton > button {
            height: 120px;
            width: 100%;
            border-radius: 4px;
            font-weight: 700;
            font-size: 14px;
            text-transform: uppercase;
            border: none;
            margin: 0px;
            transition: all 0.1s;
        }

        div.stButton > button:first-child {
            background-color: #efefe6; 
            color: #000;
        }
        div.stButton > button:hover {
            background-color: #dcdcd5;
            border: 1px solid #999;
        }

        /* RATING CARDS (Cold Start) */
        .rating-container {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #f0f0f0;
            margin-bottom: 10px;
        }

        div.row-widget.stRadio > div {
            justify-content: center;
        }

        /* Recommendation Card Styling (Text part) */
        .rec-card-text {
            background-color: #f8f9fa;
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #ff4b4b;
        }

        /* Description Text Styling */
        .rec-description {
            margin: 8px 0 0 0;
            font-size: 0.95em;
            line-height: 1.4;
            color: #333;
        }

        /* Debug Info Styling */
        .debug-info {
            font-size: 0.8em;
            color: #d63384;
            font-family: monospace;
            margin-left: 5px;
            background-color: #fce4ec;
            padding: 2px 4px;
            border-radius: 4px;
        }

        /* Image Container Styling */
        .img-grid-container {
            margin-bottom: 15px;
        }

        /* Sampling Type Badges */
        .badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: bold;
            text-transform: uppercase;
            margin-bottom: 5px;
            display: inline-block;
        }
        .badge-affirming { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .badge-controversial { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        </style>
    """
    st.markdown(combined_style, unsafe_allow_html=True)


# --- 3. CONSTANTS ---
DATA_DIR = 'Data'
BLOCKED_REST_FILE = 'blocked_restaurants.json'
BLOCKED_CAT_FILE = 'blocked_categories.json'


# --- 4. CORE ENGINE & LOGIC ---
class HybridRecommender:
    def __init__(self, Vt_tilde, L_s):
        if Vt_tilde.shape[0] < Vt_tilde.shape[1]:
            self.Vt = Vt_tilde.T
        else:
            self.Vt = Vt_tilde

        self.V_r = L_s @ self.Vt
        try:
            L_s_T_inv = sparse_inv(L_s.T)
            self.V_l = L_s_T_inv @ self.Vt
        except:
            self.V_l = self.Vt

    def get_scored_indices(self, user_vector):
        if sparse.issparse(user_vector):
            p = user_vector.toarray().flatten()
        else:
            p = np.array(user_vector).flatten()

        user_latent = p @ self.V_r
        scores = user_latent @ self.V_l.T
        return scores

    def recommend(self, user_history_vector, top_k=10):
        scores = self.get_scored_indices(user_history_vector)

        if sparse.issparse(user_history_vector):
            p = user_history_vector.toarray().flatten()
        else:
            p = np.array(user_history_vector).flatten()

        seen_indices = np.where(p != 0)[0]
        scores[seen_indices] = -np.inf

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return top_indices, scores


def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    if norm == 0: return vec
    return vec / norm


def load_blocked_config():
    """Loads blocked restaurants and categories from JSON files."""
    blocked_rest = {}
    blocked_cats = []

    if os.path.exists(BLOCKED_REST_FILE):
        with open(BLOCKED_REST_FILE, 'r') as f:
            try:
                blocked_rest = json.load(f)
            except:
                pass

    if os.path.exists(BLOCKED_CAT_FILE):
        with open(BLOCKED_CAT_FILE, 'r') as f:
            try:
                blocked_cats = json.load(f)
            except:
                pass

    return blocked_rest, set(blocked_cats)


# Cached to prevent lag on reload
@st.cache_resource
def load_data():
    path_R = os.path.join(DATA_DIR, 'R_retrain_rest_20.npz')
    path_Ls = os.path.join(DATA_DIR, 'Ls.npz')
    path_Vt = os.path.join(DATA_DIR, 'Vt_tilde.npy')
    path_personas = os.path.join(DATA_DIR, 'persona_vectors_lift.npy')
    path_map = os.path.join(DATA_DIR, 'restaurant_item_map_rest_20.json')
    path_meta = os.path.join(DATA_DIR, 'meta-District_of_Columbia.json')
    path_photos = os.path.join(DATA_DIR, 'resolved_restaurants.json')
    path_cats = os.path.join(DATA_DIR, 'categorized_restaurants.csv')

    if not os.path.exists(path_R):
        # We return Nones or handle error gracefully in main, but here we stop
        return None

    R_matrix = sparse.load_npz(path_R)
    Ls_matrix = sparse.load_npz(path_Ls)
    Vt_matrix = np.load(path_Vt)
    persona_vectors = np.load(path_personas, allow_pickle=True).item()

    with open(path_map, 'r') as f:
        item_map = json.load(f)

    # Load Photo Map
    photo_map = {}
    if os.path.exists(path_photos):
        with open(path_photos, 'r') as f:
            photo_map = json.load(f)

    meta_list = []
    with open(path_meta, 'r') as f:
        try:
            meta_raw = json.load(f)
            meta_list = meta_raw if isinstance(meta_raw, list) else [meta_raw]
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                if line.strip(): meta_list.append(json.loads(line))

    id_key = 'gmap_id' if 'gmap_id' in meta_list[0] else 'id'
    meta_lookup = {item[id_key]: item for item in meta_list if id_key in item}

    # --- Process Categories ---
    cat_map = {}
    all_categories_set = set(persona_vectors.keys())

    if os.path.exists(path_cats):
        df_c = pd.read_csv(path_cats)
        all_categories_set.update(df_c.columns)

        for col in df_c.columns:
            ids = df_c[col].dropna().astype(str).tolist()
            for g_id in ids:
                if g_id not in cat_map:
                    cat_map[g_id] = []
                cat_map[g_id].append(col)

    # Internal Review Counts
    item_counts = np.diff(R_matrix.tocsc().indptr)
    popularity = item_counts

    aligned_data = []
    for i in range(R_matrix.shape[1]):
        real_id = item_map.get(str(i))
        info = meta_lookup.get(real_id, {})

        description_text = info.get('description')
        if description_text is None:
            description_text = ""

        internal_count = int(item_counts[i])
        tags = cat_map.get(real_id, [])

        meta_cats = info.get('category', [])
        if isinstance(meta_cats, str): meta_cats = [meta_cats]
        clean_meta_cats = [str(c).strip() for c in meta_cats if c]
        tags.extend(clean_meta_cats)
        all_categories_set.update(clean_meta_cats)

        lat = info.get('latitude')
        lon = info.get('longitude')

        aligned_data.append({
            'Matrix_Index': i,
            'Gmap_ID': real_id,
            'Name': info.get('name', f'Unknown {i}'),
            'Category': info.get('category', ['N/A'])[0] if isinstance(info.get('category'), list) else 'N/A',
            'Filter_Tags': list(set(tags)),
            'Rating': info.get('avg_rating', 0.0),
            'Num_Reviews': internal_count,
            'Address': info.get('address', ''),
            'Description': description_text,
            'Latitude': float(lat) if lat else None,
            'Longitude': float(lon) if lon else None
        })
    df_items = pd.DataFrame(aligned_data)
    filter_categories = sorted(list(all_categories_set))
    engine = HybridRecommender(Vt_matrix, Ls_matrix)

    return df_items, R_matrix, engine, persona_vectors, popularity, photo_map, filter_categories


def generate_calibration_batch(user_vec, engine, df_items, popularity, exclude_indices=[]):
    scores = engine.get_scored_indices(user_vec)

    if exclude_indices:
        scores[exclude_indices] = -np.inf

    pop_threshold = np.percentile(popularity, 20)
    valid_mask = (popularity > pop_threshold) & (scores > -999999)
    valid_indices = np.where(valid_mask)[0]

    blocked_rest_ids = st.session_state.get('blocked_restaurants', {}).keys()
    final_valid_indices = []
    for idx in valid_indices:
        gmap_id = df_items.at[idx, 'Gmap_ID']
        if gmap_id not in blocked_rest_ids:
            final_valid_indices.append(idx)

    valid_indices = np.array(final_valid_indices)

    if len(valid_indices) < 20:
        valid_indices = np.where(scores > -999999)[0]

    valid_scores = scores[valid_indices]
    sorted_indices_local = np.argsort(valid_scores)
    sorted_global_indices = valid_indices[sorted_indices_local]

    pool_affirming = sorted_global_indices[-20:]
    pool_controversial = sorted_global_indices[:20]

    k_affirming = min(6, len(pool_affirming))
    affirming_selection = np.random.choice(pool_affirming, k_affirming, replace=False)

    k_controversial = min(4, len(pool_controversial))
    controversial_selection = np.random.choice(pool_controversial, k_controversial, replace=False)

    batch = []
    for idx in affirming_selection:
        batch.append({'index': idx, 'type': 'Affirming'})

    for idx in controversial_selection:
        batch.append({'index': idx, 'type': 'Controversial'})

    random.shuffle(batch)
    return batch


def display_restaurant_photos(gmap_id, photo_map):
    if gmap_id in photo_map and 'photos' in photo_map[gmap_id]:
        raw_paths = photo_map[gmap_id]['photos']
        full_paths = []
        for p in raw_paths:
            clean_path = p.replace('\\', '/')
            full_p = os.path.join(DATA_DIR, clean_path)
            if os.path.exists(full_p):
                full_paths.append(full_p)

        images_to_show = full_paths[:3]
        if images_to_show:
            cols = st.columns(len(images_to_show))
            for i, img_path in enumerate(images_to_show):
                with cols[i]:
                    st.image(img_path, width="stretch")
            return True
    return False


# --- 5. APP INITIALIZATION ---
def init_session_state():
    if 'app_initialized' not in st.session_state:
        st.session_state['app_initialized'] = True
        st.session_state.view_state = 'selection'
        st.session_state.selected_categories = set()
        st.session_state.user_vector = None
        st.session_state.calibration_items = []
        st.session_state.history_indices = []
        st.session_state.pinned_coords = None


def main():
    # 1. UI Styling (Hide Dev Tools)
    hide_developer_tools()

    # 2. State Management
    init_session_state()

    # 3. Load Data (Cached)
    data_bundle = load_data()
    if data_bundle is None:
        st.error(f"Data files not found in {DATA_DIR}")
        st.stop()

    df_items, R_matrix, engine, persona_vectors, popularity, photo_map, filter_categories = data_bundle

    # Load Block Lists (We keep logic to load them, but remove UI to edit them)
    blocked_rest, blocked_cats = load_blocked_config()
    st.session_state.blocked_restaurants = blocked_rest
    st.session_state.blocked_categories = blocked_cats

    # 4. SIDEBAR (Cleaned)
    st.sidebar.title("Mode Selection")
    app_mode = st.sidebar.radio("Select User Mode:", ["I am a new user", "Already existing database user"])

    # --- OPTION A: EXISTING USER DATABASE ---
    if app_mode == "Already existing database user":
        st.sidebar.markdown("---")
        num_users = R_matrix.shape[0]
        st.sidebar.write(f"**Dataset Stats:** Users: {num_users}, Restaurants: {df_items.shape[0]}")

        selected_user_idx = st.sidebar.number_input("Enter your user ID", min_value=0, max_value=num_users - 1, value=0)
        user_history_p = R_matrix[selected_user_idx]

        st.title("🏛️ DC Restaurant Recommender")
        st.caption(f"Viewing Existing User ID: {selected_user_idx}")

        col_left, col_right = st.columns([1, 1.5])

        with col_left:
            st.subheader("📜 User Interaction History")
            history_indices = user_history_p.indices
            if len(history_indices) == 0:
                st.warning("This user has no history.")
            else:
                st.write(f"User has visited **{len(history_indices)}** places.")
                hist_df = df_items.iloc[history_indices]
                for _, row in hist_df.iterrows():
                    with st.expander(f"✅ {row['Name']} (🛠️ {row['Num_Reviews']} reviews)"):
                        st.caption(f"{row['Category']} • {row['Rating']}⭐")
                        if row['Description']: st.caption(row['Description'])
                        st.caption(row['Address'])
                        display_restaurant_photos(row['Gmap_ID'], photo_map)

        with col_right:
            st.subheader("🔮 Recommendations")
            if st.button("Generate Recommendations", type="primary"):
                with st.spinner("Calculating..."):
                    rec_indices, scores = engine.recommend(user_history_p, top_k=20)

                    count = 0
                    for idx in rec_indices:
                        row = df_items.iloc[idx]

                        # BLOCK CHECK
                        if row['Gmap_ID'] in st.session_state.blocked_restaurants:
                            continue

                        if count >= 5: break
                        count += 1

                        desc_html = f'<p class="rec-description">{row["Description"]}</p>' if row['Description'] else ''

                        st.markdown(f"""
                        <div class="rec-card-text">
                            <h4 style="margin:0;">{row['Name']} <span class="debug-info">🛠️ {row['Num_Reviews']} reviews</span></h4>
                            <p style="margin:0; color: #666;">{row['Category']} • {row['Rating']} ⭐</p>
                            {desc_html}
                        </div>
                        """, unsafe_allow_html=True)

                        display_restaurant_photos(row['Gmap_ID'], photo_map)
                        st.markdown("---")

    # --- OPTION B: COLD START SIMULATION ---
    else:
        st.title("🧩 Develop a taste profile as a new user")

        # STEP 1: CATEGORY GRID
        if st.session_state.view_state == 'selection':
            st.subheader("Select your favorite categories of restaurants so that we know which foods you like!")
            st.caption("Pick as many as you like.")

            rng = np.random.RandomState(42)
            all_cats = sorted(list(persona_vectors.keys()))
            grid_cats = rng.choice(all_cats, 12, replace=False)

            for r in range(3):
                cols = st.columns(4)
                for c in range(4):
                    idx = r * 4 + c
                    cat_name = grid_cats[idx]
                    is_selected = cat_name in st.session_state.selected_categories
                    btn_label = f"✅\n{cat_name}" if is_selected else cat_name
                    btn_type = "primary" if is_selected else "secondary"

                    if cols[c].button(btn_label, key=f"cat_{cat_name}", type=btn_type, width="stretch"):
                        if is_selected:
                            st.session_state.selected_categories.remove(cat_name)
                        else:
                            st.session_state.selected_categories.add(cat_name)
                        st.rerun()

            st.markdown("---")
            if st.button("Continue ➡️", type="primary", width="stretch"):
                if not st.session_state.selected_categories:
                    st.error("Please select at least one category.")
                else:
                    u_vec = np.zeros(len(df_items))
                    for c in st.session_state.selected_categories:
                        u_vec += persona_vectors[c]
                    st.session_state.user_vector = normalize_vector(u_vec)
                    batch = generate_calibration_batch(st.session_state.user_vector, engine, df_items, popularity)
                    st.session_state.calibration_items = batch
                    st.session_state.view_state = 'calibration'
                    st.rerun()

        # STEP 2: BLIND CALIBRATION
        elif st.session_state.view_state == 'calibration':
            st.subheader("To help us develop your profile rate these restaurants ")
            st.write("Some of these restaurants affirm your profile and some are controversial to your profile.")

            with st.form("calib_form"):
                ratings = {}
                for item in st.session_state.calibration_items:
                    idx = item['index']
                    item_type = item['type']
                    row = df_items.iloc[idx]
                    with st.container():
                        badge_class = f"badge-{item_type.lower()}"
                        st.markdown(f"""<span class="badge {badge_class}">{item_type}</span>""", unsafe_allow_html=True)
                        st.markdown(
                            f"**{row['Name']}** ({row['Category']}) <span class='debug-info'>🛠️ {row['Num_Reviews']} reviews</span>",
                            unsafe_allow_html=True)
                        if row['Description']: st.caption(row['Description'])
                        display_restaurant_photos(row['Gmap_ID'], photo_map)
                        ratings[idx] = st.radio(f"Rate {row['Name']}", ["Like", "Neutral", "Dislike"], index=1,
                                                horizontal=True, key=f"rate_{idx}", label_visibility="collapsed")
                        st.markdown("---")

                if st.form_submit_button("Get Recommendations 🚀", type="primary", width="stretch"):
                    curr_vec = st.session_state.user_vector
                    for idx, rate in ratings.items():
                        if rate == "Like":
                            curr_vec[idx] += 0.5
                        elif rate == "Dislike":
                            curr_vec[idx] -= 0.5
                        st.session_state.history_indices.append(idx)
                    st.session_state.user_vector = normalize_vector(curr_vec)
                    st.session_state.view_state = 'results'
                    st.rerun()

        # STEP 3: RESULTS & LOOP
        elif st.session_state.view_state == 'results':
            st.subheader("🎯 These are our best matches for you")

            scores = engine.get_scored_indices(st.session_state.user_vector)
            scores[st.session_state.history_indices] = -np.inf

            # --- FILTERS (Combined: Category & Location) ---
            with st.expander("Filter Options (Category & Location)"):
                c1, c2 = st.columns(2)
                with c1:
                    # Category Filter - FILTER OUT BLOCKED CATEGORIES
                    allowed_cats = [c for c in filter_categories if c not in st.session_state.blocked_categories]
                    selected_filter_cats = []
                    if allowed_cats:
                        selected_filter_cats = st.multiselect("Filter by Category", options=allowed_cats, default=[])

                with c2:
                    # Location/Distance Filter
                    st.caption("Click map to pin location")
                    m_pin = folium.Map(location=[38.9072, -77.0369], zoom_start=11)

                    if st.session_state.pinned_coords:
                        folium.Marker(st.session_state.pinned_coords,
                                      icon=folium.Icon(color="red", icon="map-pin", prefix="fa")).add_to(m_pin)

                    pin_data = st_folium(m_pin, height=200, width="100%", key="pin_map")

                    if pin_data and pin_data.get("last_clicked"):
                        new_pin = [pin_data["last_clicked"]["lat"], pin_data["last_clicked"]["lng"]]
                        if new_pin != st.session_state.pinned_coords:
                            st.session_state.pinned_coords = new_pin
                            st.rerun()

                    radius_km = st.slider("Max Distance (km)", 0.5, 10.0, 3.0)

            # 1. Apply Category Filter
            if selected_filter_cats:
                selected_set = set(selected_filter_cats)

                def has_match(item_tags):
                    return not selected_set.isdisjoint(item_tags)

                cat_mask = df_items['Filter_Tags'].apply(has_match)
                scores[~cat_mask] = -np.inf

            # 2. Apply Location Filter
            user_lat, user_lon = None, None
            if st.session_state.pinned_coords:
                user_lat, user_lon = st.session_state.pinned_coords
                st.success(f"📍 Filter active: Pinned Location")

                candidate_indices = np.where(scores > -999999)[0]
                for idx in candidate_indices:
                    r_lat = df_items.at[idx, 'Latitude']
                    r_lon = df_items.at[idx, 'Longitude']

                    if pd.isna(r_lat) or pd.isna(r_lon):
                        scores[idx] = -np.inf
                        continue

                    dist = geodesic((user_lat, user_lon), (r_lat, r_lon)).km
                    if dist > radius_km:
                        scores[idx] = -np.inf

            # Select Top (allowing for blocking)
            top_indices = np.argsort(scores)[-30:][::-1]

            if scores[top_indices[0]] == -np.inf:
                st.warning("No restaurants found with those filters.")
            else:
                view_option = st.radio("View Mode", ["📋 List View", "🗺️ Map View"], horizontal=True)
                st.markdown("---")

                # FILTER BLOCKED RESTAURANTS
                valid_top_indices = []
                for idx in top_indices:
                    if scores[idx] == -np.inf: continue
                    row = df_items.iloc[idx]
                    if row['Gmap_ID'] not in st.session_state.blocked_restaurants:
                        valid_top_indices.append(idx)
                    if len(valid_top_indices) >= 10: break

                if view_option == "📋 List View":
                    for idx in valid_top_indices:
                        row = df_items.iloc[idx]

                        dist_str = ""
                        if user_lat and pd.notna(row['Latitude']):
                            d = geodesic((user_lat, user_lon), (row['Latitude'], row['Longitude'])).km
                            dist_str = f" • 📏 {d:.1f} km"

                        desc_html = f'<p class="rec-description">{row["Description"]}</p>' if row['Description'] else ''

                        st.markdown(f"""
                        <div class="rec-card-text">
                            <h4 style="margin:0;">{row['Name']} <span class="debug-info">🛠️ {row['Num_Reviews']} reviews</span></h4>
                            <p style="margin:0; color: #666;">{row['Category']}{dist_str}</p>
                            {desc_html}
                        </div>
                        """, unsafe_allow_html=True)

                        display_restaurant_photos(row['Gmap_ID'], photo_map)
                        st.markdown("---")

                elif view_option == "🗺️ Map View":
                    start_loc = [user_lat, user_lon] if user_lat else [38.9072, -77.0369]
                    m = folium.Map(location=start_loc, zoom_start=13)

                    if user_lat:
                        folium.Marker([user_lat, user_lon], popup="You",
                                      icon=folium.Icon(color="blue", icon="user", prefix="fa")).add_to(m)
                        folium.Circle(radius=radius_km * 1000, location=[user_lat, user_lon], color="blue", fill=True,
                                      fill_opacity=0.1).add_to(m)

                    for idx in valid_top_indices:
                        row = df_items.iloc[idx]
                        if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                            popup_text = f"<b>{row['Name']}</b><br>{row['Category']}"
                            folium.Marker(
                                [row['Latitude'], row['Longitude']],
                                popup=popup_text,
                                tooltip=f"{row['Name']}",
                                icon=folium.Icon(color="red", icon="cutlery", prefix="fa")
                            ).add_to(m)

                    st_folium(m, width=700, height=500)

            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Disagree with your recommendations? Click here to further develop your taste"):
                    new_batch = generate_calibration_batch(
                        st.session_state.user_vector,
                        engine, df_items,
                        popularity=popularity,
                        exclude_indices=st.session_state.history_indices
                    )
                    st.session_state.calibration_items = new_batch
                    st.session_state.view_state = 'calibration'
                    st.rerun()
            with c2:
                if st.button("Start over from scratch"):
                    st.session_state.view_state = 'selection'
                    st.session_state.selected_categories = set()
                    st.session_state.history_indices = []
                    st.session_state.pinned_coords = None
                    st.rerun()


if __name__ == "__main__":
    main()