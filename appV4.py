import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
import plotly.express as px  # <--- NOUVEL IMPORT NECESSAIRE

# --- 1. Définition des Modèles et de leurs Localisations ---
MODEL_REGISTRY = [
    {
        "name": "Utrecht",
        "path": "modele_lightGBM.pkl",
        "latitude": 51.9701,
        "longitude": 5.3217,
        "location_info": "Modèle d'Utrecht (Pays-Bas)",
    },
    {
        "name": "Lisbon1",
        "path": "modele_lightGBM_Lisbon1.pkl",
        "latitude": 38.728,
        "longitude": -9.138,
        "location_info": "Modèle de Lisbonne (Portugal)",
    },
    {
        "name": "Faro",
        "path": "modele_lightGBM_Faro.pkl",
        "latitude": 37.031,
        "longitude": -7.893,
        "location_info": "Modèle de Faro (Portugal)",
    },
    {
        "name": "Braga",
        "path": "modele_lightGBM_Braga.pkl",
        "latitude": 41.493,
        "longitude": -8.496,
        "location_info": "Modèle de Braga (Portugal)",
    },
    {
        "name": "Setubal",
        "path": "modele_lightGBM_Setubal.pkl",
        "latitude": 38.577,
        "longitude": -8.872,
        "location_info": "Modèle de Setubal (Portugal)",
    },
    {
        "name": "Alice Springs",
        "path": "modele_lightGBM_AliceSprings.pkl",
        "latitude": -23.7002104,
        "longitude": 133.8806114,
        "location_info": "Modèle de Alice Springs (Australie)",
    },
]


# --- 2. Fonction pour la Distance Géographique (Haversine) ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


# --- 3. Fonction pour trouver le Modèle le Plus Proche ---
def find_closest_model(user_latitude, user_longitude):
    min_distance = float('inf')
    closest_model = None
    for model_data in MODEL_REGISTRY:
        lat = model_data['latitude']
        lon = model_data['longitude']
        distance = haversine(user_latitude, user_longitude, lat, lon)
        if distance < min_distance:
            min_distance = distance
            closest_model = model_data
    return closest_model, min_distance


# --- 4. Fonction de Chargement du Modèle ---
@st.cache_resource
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Erreur : Fichier modèle '{path}' introuvable. Assurez-vous qu'il existe.")
        return None


# --- 5. FONCTION DE GÉOCODAGE (Nominatim) ---
def geocode_address(address):
    headers = {'User-Agent': 'PV_Prediction_App_Streamlit/1.0'}
    params = {
        'q': address,
        'format': 'json',
        'limit': 1
    }
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

    try:
        response = requests.get(NOMINATIM_URL, params=params, headers=headers)
        response.raise_for_status()
        results = response.json()

        if results:
            lat = float(results[0]['lat'])
            lon = float(results[0]['lon'])
            display_name = results[0]['display_name']
            return lat, lon, display_name
        else:
            return None, None, None

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API de Géocodage : {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue lors du géocodage : {e}")
        return None, None, None


# Configuration application
st.set_page_config(
    page_title="Prédiction de Production PV",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Fonction de récupération des données météo (inchangée)
def fetch_weather_data(latitude, longitude, tilt, azimuth, days=7):
    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "cloud_cover",
        "global_tilted_irradiance"
    ]

    API_URL = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(hourly_vars),
        "timezone": "auto",
        "forecast_days": days,
        "tilt": tilt,
        "azimuth": azimuth,
        "models": "best_match"
    }

    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if 'hourly' not in data:
            st.warning("Aucune donnée horaire ('hourly') trouvée dans la réponse de l'API.")
            return None
        df = pd.DataFrame(data['hourly'])
        df = df.rename(columns={
            'temperature_2m': 'temperature_2m_(°C)',
            'relative_humidity_2m': 'relative_humidity_2m_(%)',
            'wind_speed_10m': 'wind_speed_10m_(km/h)',
            'cloud_cover': 'cloud_cover_(%)',
            'global_tilted_irradiance': 'global_tilted_irradiance_(W/m²)'
        })
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API Open-Meteo : {e}")
        return None
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue : {e}")
        return None


# Interface de l'application
st.title("Système de Prédiction de Production PV")
st.markdown("Le modèle de prédiction utilisé est un modèle de type LightGBM")
st.markdown("Les données météos proviennent de **open-meteo.com**")

# Présentation des variables utilisées dans le modèle
col_meteo, col_temporelle = st.columns(2)
with col_meteo:
    st.markdown("### Variables Météo")
    st.markdown("- **Température (°C)**")
    st.markdown("- **Humidité Relative (%)**")
    st.markdown("- **Vitesse du vent à 10m (km/h)**")
    st.markdown("- **Couverture nuageuse (%)**")
    st.markdown("- **Irradiation global orientée (W/m$^2$)**")

with col_temporelle:
    st.markdown("### Variables Temporelles")
    st.markdown("- **Mois**")
    st.markdown("- **Jour de l'année**")
    st.markdown("- **Heure**")

st.markdown("---")

# Interface utilisateur pour la localisation
st.header("Localisation du panneau PV et Sélection de Modèle")

# --- INTERFACE DE RECHERCHE D'ADRESSE AVEC ZOOM FIXE GLOBAL ---
col_map, col_input = st.columns([3, 1])

# Initialisation des variables de session (zoom fixé à 3 pour vue planisphère)
if 'latitude' not in st.session_state:
    st.session_state.latitude = 51.9701
if 'longitude' not in st.session_state:
    st.session_state.longitude = 5.3217
if 'current_location_name' not in st.session_state:
    st.session_state.current_location_name = "Utrecht, Pays-Bas (par défaut)"
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 3  # Zoom initial fixe pour la vue globale

# 1. Barre d'adresse (dans la colonne de droite)
with col_input:
    st.subheader("Recherche d'Adresse")
    address_input = st.text_input(
        "Entrez une adresse, une ville ou un lieu :",
        placeholder="Ex: Sydney, Rome, ou Setubal",
        key="address_search"
    )
    search_button = st.button("Rechercher la Localisation", use_container_width=True)

    # Logique de géocodage
    if search_button and address_input:
        with st.spinner(f"Recherche des coordonnées pour '{address_input}'..."):
            new_lat, new_lon, new_name = geocode_address(address_input)

        if new_lat is not None and new_lon is not None:
            st.session_state.latitude = new_lat
            st.session_state.longitude = new_lon
            st.session_state.current_location_name = new_name
            # Le zoom reste fixe à 3
            st.success(f"Localisation trouvée : **{new_name}**.")
        else:
            st.error("Adresse non trouvée. Veuillez réessayer avec plus de détails (ex: rue, ville, pays).")

    st.markdown("---")
    st.caption("Coordonnées Actuelles :")
    st.caption(f"**{st.session_state.current_location_name}**")
    st.caption(f"Lat: {st.session_state.latitude:.4f} | Long: {st.session_state.longitude:.4f}")

# 2. Préparation des données pour la carte (dans la colonne de gauche)
# Point utilisateur marqué distinctement
user_point = pd.DataFrame({
    'lat': [st.session_state.latitude],
    'lon': [st.session_state.longitude],
    'type': ['📍 Votre Localisation']
})

model_points = pd.DataFrame([
    {'lat': m['latitude'], 'lon': m['longitude'], 'type': m['name']}
    for m in MODEL_REGISTRY
])

# Fusionner les deux DataFrames.
map_data = pd.concat([user_point, model_points])

with col_map:
    st.subheader("Visualisation de l'Emplacement")

    # --- NOUVEAU BLOC AVEC PLOTLY EXPRESS ---
    # 1. Définir le mapping des couleurs
    color_map = {
        # J'utilise 'green' pour le point choisi, vous pouvez changer pour 'blue' ou 'teal' si vous préférez.
        '📍 Votre Localisation': 'green',
        # Orange pour tous les autres points
        **{name: 'orange' for name in model_points['type'].unique()}
    }

    # 2. Créer la figure Plotly
    fig = px.scatter_mapbox(
        map_data,
        lat="lat",
        lon="lon",
        # Utiliser la colonne 'type' pour déterminer la couleur et le groupe
        color="type",
        color_discrete_map=color_map,  # Appliquer le mapping de couleur
        zoom=st.session_state.map_zoom,
        height=400,
        mapbox_style="open-street-map",  # Style de carte de base
        # Définir la taille des marqueurs (plus grand pour le point utilisateur)
        size=map_data['type'].apply(lambda x: 15 if x == '📍 Votre Localisation' else 10),
        size_max=15,
        hover_name='type'
    )

    # 3. Mettre à jour la mise en page
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend_title_text='Légende'
    )

    # 4. Afficher la carte dans Streamlit
    st.plotly_chart(fig, use_container_width=True)
    # --- FIN DU NOUVEAU BLOC ---

    st.caption("📍 (Vert) : Votre localisation recherchée. Les autres points (Orange) sont les modèles disponibles.")
# --- FIN DE L'INTERFACE AVEC ZOOM FIXE GLOBAL ---

# Affichage du modèle sélectionné
closest_model_info, distance = find_closest_model(st.session_state.latitude, st.session_state.longitude)

st.info(
    f"**Modèle Sélectionné:** **{closest_model_info['name']}**.\n\n"
    f"Ce modèle est le plus proche géographiquement (à **{distance:,.0f} km**) "
    f"de votre localisation ({closest_model_info['latitude']:.4f}, {closest_model_info['longitude']:.4f})."
)

# Inputs du Système PV (Orientation & Azimuth)
st.markdown("---")
st.subheader("Orientation du panneau PV")

tilt = st.number_input(
    "Inclinaison [°]",
    min_value=0.0, max_value=90.0, value=35.0, format="%.1f",
    help="Angle du panneau par rapport à l'horizontale (0°=plat, 90°=vertical)."
)

azimuth = st.number_input(
    "Azimut (Orientation) [°]",
    min_value=-180.0, max_value=180.0, value=0.0, format="%.1f",
    help="Orientation des panneaux: 0°=Sud, 90°=Ouest, -90°=Est, ±180°=Nord (selon la convention Open-Meteo)."
)

st.markdown("---")
forecast_days = st.slider("Jours de Prévision", 1, 16, 7)

predict_button = st.button("Lancer la Prédiction", type="primary")

# Application du modèle aux données
if predict_button:
    pv_model = load_model(closest_model_info['path'])
else:
    pv_model = None

if pv_model and predict_button:
    st.header("Résultats de la Prédiction")
    st.caption(f"**Modèle utilisé pour cette prédiction:** {closest_model_info['location_info']}")

    # Récupération des données
    with st.spinner(
            f"Récupération des prévisions météo sur {forecast_days} jours pour ({st.session_state.latitude}, {st.session_state.longitude})..."):
        raw_df = fetch_weather_data(st.session_state.latitude, st.session_state.longitude, tilt, azimuth, forecast_days)

    if raw_df is not None:

        # Préparation des données (conversion des dates)
        raw_df['time'] = pd.to_datetime(raw_df['time'])
        df_processed = raw_df.copy()

        df_processed['hour'] = df_processed['time'].dt.hour
        df_processed['month'] = df_processed['time'].dt.month
        df_processed['day_of_year'] = df_processed['time'].dt.dayofyear

        FEATURE_NAMES = [
            'temperature_2m_(°C)', 'relative_humidity_2m_(%)',
            'global_tilted_irradiance_(W/m²)', 'wind_speed_10m_(km/h)',
            'cloud_cover_(%)', 'hour', 'month', 'day_of_year'
        ]

        X = df_processed[FEATURE_NAMES]

        st.subheader("Aperçu des Données Météo Récupérées")

        METEO_VARS_FOR_PLOTTING = [
            'global_tilted_irradiance_(W/m²)',
            'temperature_2m_(°C)',
            'cloud_cover_(%)',
            'relative_humidity_2m_(%)',
            'wind_speed_10m_(km/h)',
        ]

        df_meteo = df_processed.set_index('time')[METEO_VARS_FOR_PLOTTING]

        tab_gti, tab_temp, tab_cloud = st.tabs(
            ["Irradiation (GTI)", "Température & Humidité", "Couverture Nuageuse & Vent"])

        with tab_gti:
            st.markdown("##### Irradiation Globale Inclinée (GTI) sur le panneau")
            st.line_chart(df_meteo[['global_tilted_irradiance_(W/m²)']], use_container_width=True)

        with tab_temp:
            st.markdown("##### Température et Humidité à 2 mètres")
            st.line_chart(df_meteo[['temperature_2m_(°C)', 'relative_humidity_2m_(%)']], use_container_width=True)

        with tab_cloud:
            st.markdown("##### Couverture Nuageuse et Vitesse du Vent")
            st.area_chart(df_meteo[['cloud_cover_(%)']], use_container_width=True)
            st.line_chart(df_meteo[['wind_speed_10m_(km/h)']], use_container_width=True)

        # Faire la Prédiction
        with st.spinner("Calcul des prédictions de production PV..."):
            predictions = pv_model.predict(X)
            predictions[predictions < 0] = 0
            predictions[predictions > 1] = 1
            df_processed['Production_PV_kW'] = predictions

            IRRADIANCE_COLUMN = 'global_tilted_irradiance_(W/m²)'
            irradiance_nulle_mask = X[IRRADIANCE_COLUMN] == 0
            predictions[irradiance_nulle_mask.values] = 0

        # Affichage des Résultats
        total_production = df_processed['Production_PV_kW'].sum()

        st.metric(
            label=f"Production Totale Prévue sur {forecast_days} jours",
            value=f"{total_production:,.2f} kWh/kWc".replace(",", " ")
        )

        daily_production = df_processed.set_index('time').resample('D')['Production_PV_kW'].sum()
        if not daily_production.empty:
            st.subheader("Répartition Journalière (kWh/kWc)")
            st.dataframe(daily_production.to_frame(name='kWh/kWc par jour').style.format("{:,.2f}"))

        st.subheader("Prévision Horaire de Production PV (kW/kwc)")
        df_chart = df_processed.set_index('time')[['Production_PV_kW']]

        st.line_chart(df_chart, use_container_width=True)

        st.caption(
            f"Prévision pour Lat: {st.session_state.latitude}, Long: {st.session_state.longitude}, Inclinaison: {tilt}°, Azimut: {azimuth}°.")

    else:
        st.warning("Impossible de procéder à la prédiction sans données météo valides.")