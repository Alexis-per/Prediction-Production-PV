import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# NOUVELLE LIBRAIRIE REQUISE
# Assurez-vous d'avoir installé cette librairie: pip install streamlit-folium
from streamlit_folium import st_folium
import folium

# --- 1. Définition des Modèles et de leurs Localisations ---
MODEL_REGISTRY = [
    # Utrecht
    {
        "name": "Utrecht",
        "path": "modele_lightGBM.pkl",
        "latitude": 51.9701,
        "longitude": 5.3217,
        "location_info": "Modèle d'Utrecht (Pays-Bas)",
    },
    # Lisbon1
    {
        "name": "Lisbon1",
        "path": "modele_lightGBM_Lisbon1.pkl",
        "latitude": 38.728,
        "longitude": -9.138,
        "location_info": "Modèle de Lisbonne (Portugal)",
    },
    # Faro
    {
        "name": "Faro",
        "path": "modele_lightGBM_Faro.pkl",
        "latitude": 37.031,
        "longitude": -7.893,
        "location_info": "Modèle de Faro (Portugal)",
    },
    # Braga
    {
        "name": "Braga",
        "path": "modele_lightGBM_Braga.pkl",
        "latitude": 41.493,
        "longitude": -8.496,
        "location_info": "Modèle de Braga (Portugal)",
    },
    # Setubal
    {
        "name": "Setubal",
        "path": "modele_lightGBM_Setubal.pkl",
        "latitude": 38.577,
        "longitude": -8.872,
        "location_info": "Modèle de Setubal (Portugal)",
    },
]


# --- 2. Fonction pour la Distance Géographique (Haversine) ---

def haversine(lat1, lon1, lat2, lon2):
    """Calcule la distance Haversine en km."""
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
    """Trouve le modèle le plus proche géographiquement."""
    min_distance = float('inf')
    closest_model = None

    for model_data in MODEL_REGISTRY:
        distance = haversine(user_latitude, user_longitude, model_data['latitude'], model_data['longitude'])
        if distance < min_distance:
            min_distance = distance
            closest_model = model_data

    return closest_model, min_distance


# --- 4. Modification de la Fonction de Chargement du Modèle ---

@st.cache_resource
def load_model(path):
    """Charge le modèle LightGBM pré-entraîné à partir du chemin spécifié."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Erreur : Fichier modèle '{path}' introuvable. Assurez-vous qu'il existe.")
        return None


# Configuration application
st.set_page_config(
    page_title="Prédiction de Production PV",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Utilsiation de l'API de open-meteo.com pour obtenir les données prévisionnelles
def fetch_weather_data(latitude, longitude, tilt, azimuth, days=7):
    """Récupère les prévisions météorologiques horaires."""
    hourly_vars = [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "cloud_cover", "global_tilted_irradiance"
    ]
    API_URL = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude, "longitude": longitude,
        "hourly": ",".join(hourly_vars), "timezone": "auto",
        "forecast_days": days, "tilt": tilt, "azimuth": azimuth,
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
st.markdown(
    "Les données météos (historiques et de prévisions qui sont utilisées sur l'appli) proviennent de **open-meteo.com**")

# Présentation des variables
col_meteo, col_temporelle = st.columns(2)
with col_meteo:
    st.markdown("### Variables Météo")
    st.markdown(
        "- **Température (°C)**, **Humidité Relative (%)**, **Vitesse du vent à 10m (km/h)**, **Couverture nuageuse (%)**, **Irradiation global orientée (W/m$^2$)**")
with col_temporelle:
    st.markdown("### Variables Temporelles")
    st.markdown("- **Mois**, **Jour de l'année**, **Heure**")

st.markdown("---")
st.header("Localisation du panneau PV et Sélection de Modèle")

# Initialisation des variables de session pour la persistance des inputs
if 'latitude' not in st.session_state:
    st.session_state.latitude = 51.9701  # Coordonnées par défaut (Utrecht)
if 'longitude' not in st.session_state:
    st.session_state.longitude = 5.3217

# --- NOUVEAUTÉ : Carte interactive et inputs ---
col_map, col_input = st.columns([3, 1])

with col_input:
    st.subheader("Saisie Manuelle")

    # Les entrées manuelles sont liées directement aux variables de session
    new_lat = st.number_input(
        "Latitude (Lat)",
        min_value=-90.0, max_value=90.0,
        value=st.session_state.latitude,
        format="%.4f",
        key="manual_lat"
    )
    new_lon = st.number_input(
        "Longitude (Long)",
        min_value=-180.0, max_value=180.0,
        value=st.session_state.longitude,
        format="%.4f",
        key="manual_lon"
    )

    # Mise à jour des variables de session après l'input manuel
    st.session_state.latitude = new_lat
    st.session_state.longitude = new_lon

    st.markdown(
        """
        *Conseil : Modifiez ces valeurs, ou cliquez 
        sur la carte à gauche pour choisir un point.*
        """
    )

with col_map:
    st.subheader("Sélection sur la Carte")

    # 1. Création de la carte Folium centrée sur le point actuel de l'utilisateur
    m = folium.Map(
        location=[st.session_state.latitude, st.session_state.longitude],
        zoom_start=7,
        tiles="cartodbpositron"  # Un fond de carte léger
    )

    # 2. Ajout des marqueurs pour les emplacements des modèles
    for model in MODEL_REGISTRY:
        folium.Marker(
            [model['latitude'], model['longitude']],
            tooltip=model['name'],
            icon=folium.Icon(color='blue', icon='solar-panel', prefix='fa')  # Icone de panneau solaire
        ).add_to(m)

    # 3. Ajout du marqueur de l'utilisateur (Rouge)
    folium.Marker(
        [st.session_state.latitude, st.session_state.longitude],
        tooltip="Votre Emplacement",
        icon=folium.Icon(color='red', icon='home', prefix='fa')
    ).add_to(m)

    # 4. Ajout du plugin pour cliquer et dessiner (ce qui permet de récupérer les coordonnées)
    # L'outil "Draw" est nécessaire pour l'interactivité. On n'utilise que le marqueur ('marker')
    draw = folium.plugins.Draw(
        export=False,
        position='topleft',
        draw_options={
            'polyline': False,
            'polygon': False,
            'circle': False,
            'circlemarker': False,
            'rectangle': False,
        },
        edit_options={'edit': False, 'remove': True}
    )
    draw.add_to(m)

    # 5. Rendu de la carte et récupération de l'état
    # `st_folium` retourne un dictionnaire contenant les coordonnées des dernières interactions
    map_data = st_folium(m, width=700, height=450, key="folium_map", return_on_hover=False)

# --- Logique de mise à jour des coordonnées à partir du clic (Draw) ---
# Vérifier si l'utilisateur a cliqué sur le bouton de marqueur et placé un point
if map_data and map_data.get("last_active_drawing"):
    drawing_type = map_data["last_active_drawing"].get("geometry", {}).get("type")

    # Si c'est un point (marker) qui a été dessiné/placé
    if drawing_type == "Point":
        coords = map_data["last_active_drawing"]["geometry"]["coordinates"]

        # Les coordonnées folium sont [longitude, latitude], nous devons les inverser
        new_lon_from_map = coords[0]
        new_lat_from_map = coords[1]

        # Mettre à jour les variables de session, ce qui rafraîchira le st.number_input
        st.session_state.latitude = new_lat_from_map
        st.session_state.longitude = new_lon_from_map
        st.rerun()  # Rafraîchir l'application pour que les number_input soient mis à jour
# --- FIN Logique de mise à jour ---

# Affichage du modèle sélectionné (utilise les coordonnées de session)
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

# Charger le modèle UNIQUEMENT si l'utilisateur clique sur le bouton,
# en utilisant le chemin du modèle le plus proche trouvé.
if predict_button:
    pv_model = load_model(closest_model_info['path'])
else:
    pv_model = None

if pv_model and predict_button:
    st.header("Résultats de la Prédiction")

    # Afficher l'information sur le modèle utilisé dans la section résultat
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

        # S'assurer que l'ordre et le nom des colonnes correspondent à l'entraînement du modèle
        FEATURE_NAMES = [
            'temperature_2m_(°C)', 'relative_humidity_2m_(%)',
            'global_tilted_irradiance_(W/m²)', 'wind_speed_10m_(km/h)',
            'cloud_cover_(%)', 'hour', 'month', 'day_of_year'
        ]

        # Filtrer et réorganiser les colonnes
        X = df_processed[FEATURE_NAMES]

        st.subheader("Aperçu des Données Météo Récupérées")

        # Créer un DataFrame avec uniquement les variables météo pertinentes pour le plot des variables météos
        METEO_VARS_FOR_PLOTTING = [
            'global_tilted_irradiance_(W/m²)',
            'temperature_2m_(°C)',
            'cloud_cover_(%)',
            'relative_humidity_2m_(%)',
            'wind_speed_10m_(km/h)',
        ]

        df_meteo = df_processed.set_index('time')[METEO_VARS_FOR_PLOTTING]

        # Utilisation de st.tabs pour organiser l'affichage des graphiques
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

            # Application du modèle
            predictions = pv_model.predict(X)

            # S'assurer que la production est positive (physiquement impossible d'être négative et =< 1)
            predictions[predictions < 0] = 0
            predictions[predictions > 1] = 1

            # Ajouter les prédictions au DataFrame
            df_processed['Production_PV_kW'] = predictions

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

        # Affichage graphique de la production prévue
        st.subheader("Prévision Horaire de Production PV (kW/kwc)")
        df_chart = df_processed.set_index('time')[['Production_PV_kW']]

        st.line_chart(df_chart, use_container_width=True)

        st.caption(
            f"Prévision pour Lat: {st.session_state.latitude}, Long: {st.session_state.longitude}, Inclinaison: {tilt}°, Azimut: {azimuth}°.")

    else:
        st.warning("Impossible de procéder à la prédiction sans données météo valides.")