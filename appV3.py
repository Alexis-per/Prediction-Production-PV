import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# NOUVELLES LIBRAIRIES NÉCESSAIRES POUR LA CARTE INTERACTIVE ET LES ICÔNES
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import json  # Nécessaire pour analyser le résultat du géocodage

# --- CONFIGURATION DE L'ICÔNE D'IMAGE ---
MODEL_ICON_URL = "istockphoto-1455686956-612x612.jpg"
ICON_SIZE = (30, 30)

try:
    CUSTOM_SOLAR_ICON = folium.CustomIcon(
        icon_image=MODEL_ICON_URL,
        icon_size=ICON_SIZE,
        icon_anchor=(ICON_SIZE[0] // 2, ICON_SIZE[1])
    )
except FileNotFoundError:
    st.warning(f"ATTENTION : Le fichier icône '{MODEL_ICON_URL}' est introuvable. Utilisé l'icône par défaut.")
    CUSTOM_SOLAR_ICON = folium.Icon(color='blue', icon='solar-panel', prefix='fa')

# --- 1. Définition des Modèles et de leurs Localisations (INCHANGÉ) ---
MODEL_REGISTRY = [
    {"name": "Utrecht", "path": "modele_lightGBM.pkl", "latitude": 51.9701, "longitude": 5.3217,
     "location_info": "Modèle d'Utrecht (Pays-Bas)"},
    {"name": "Lisbon1", "path": "modele_lightGBM_Lisbon1.pkl", "latitude": 38.728, "longitude": -9.138,
     "location_info": "Modèle de Lisbonne (Portugal)"},
    {"name": "Faro", "path": "modele_lightGBM_Faro.pkl", "latitude": 37.031, "longitude": -7.893,
     "location_info": "Modèle de Faro (Portugal)"},
    {"name": "Braga", "path": "modele_lightGBM_Braga.pkl", "latitude": 41.493, "longitude": -8.496,
     "location_info": "Modèle de Braga (Portugal)"},
    {"name": "Setubal", "path": "modele_lightGBM_Setubal.pkl", "latitude": 38.577, "longitude": -8.872,
     "location_info": "Modèle de Setubal (Portugal)"},
]


# --- 2. Fonctions de Calcul (INCHANGÉES) ---

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


@st.cache_resource
def load_model(path):
    """Charge le modèle LightGBM pré-entraîné."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Erreur : Fichier modèle '{path}' introuvable. Assurez-vous qu'il existe.")
        return None


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
            'temperature_2m': 'temperature_2m_(°C)', 'relative_humidity_2m': 'relative_humidity_2m_(%)',
            'wind_speed_10m': 'wind_speed_10m_(km/h)', 'cloud_cover': 'cloud_cover_(%)',
            'global_tilted_irradiance': 'global_tilted_irradiance_(W/m²)'
        })
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API Open-Meteo : {e}")
        return None
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue : {e}")
        return None


# --- 3. NOUVELLE FONCTION DE GÉOCODAGE ---

def geocode_address(address):
    """Convertit une adresse textuelle en coordonnées (latitude, longitude) en utilisant Nominatim."""
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": 1
    }
    try:
        headers = {'User-Agent': 'PV_Prediction_App/1.0'}
        response = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data:
            lat = float(data[0].get('lat'))
            lon = float(data[0].get('lon'))
            display_name = data[0].get('display_name', address)
            return lat, lon, display_name
        else:
            return None, None, "Adresse non trouvée."

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API de géocodage (Nominatim) : {e}")
        return None, None, "Erreur de connexion."
    except Exception as e:
        st.error(f"Erreur inattendue lors du géocodage : {e}")
        return None, None, "Erreur inconnue."


# --- 4. Configuration et Initialisation des États ---

st.set_page_config(
    page_title="Prédiction de Production PV",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation des variables de session pour la persistance
if 'latitude' not in st.session_state:
    st.session_state.latitude = 51.9701  # Coordonnées par défaut (Utrecht)
if 'longitude' not in st.session_state:
    st.session_state.longitude = 5.3217
if 'current_address' not in st.session_state:
    st.session_state.current_address = "Utrecht, Netherlands"

# --- 5. Interface Principale ---
st.title("Système de Prédiction de Production PV")
st.markdown("Le modèle de prédiction utilisé est un modèle de type LightGBM")
st.markdown("Les données météos proviennent de **open-meteo.com**")

# Présentation des variables (INCHANGÉ)
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

# --- Section Adresse, Coordonnées et Carte interactive ---
col_search, col_coord_map = st.columns([1, 3])

# Bloc de Recherche et Coordonnées
with col_search:
    st.subheader("1. Recherche par Adresse")

    address_input = st.text_input(
        "Adresse / Lieu",
        value=st.session_state.current_address,
        key="address_input_widget"
    )

    geocode_button = st.button("Chercher l'Adresse 🔎", type="secondary")

    # Logique de géocodage
    if geocode_button:
        with st.spinner(f"Recherche de '{address_input}'..."):
            new_lat, new_lon, found_address = geocode_address(address_input)

            if new_lat is not None and new_lon is not None:
                # Mise à jour des coordonnées et de l'adresse dans l'état de session
                st.session_state.latitude = new_lat
                st.session_state.longitude = new_lon
                st.session_state.current_address = found_address
                st.success(f"Adresse trouvée: **{found_address}**")
                # Pas de st.rerun ici, la carte sera mise à jour au prochain rafraîchissement
            else:
                st.error(f"Impossible de trouver les coordonnées pour : {address_input}")

    st.markdown("---")
    st.subheader("2. Coordonnées Actuelles")

    # Affichage des coordonnées (lecture seule)
    st.metric(
        label="Latitude",
        value=f"{st.session_state.latitude:.4f}"
    )
    st.metric(
        label="Longitude",
        value=f"{st.session_state.longitude:.4f}"
    )
    st.markdown(
        """
        *Les coordonnées peuvent être modifiées via la recherche 
        d'adresse ou en cliquant sur la carte ci-contre.*
        """
    )

# Bloc Carte Interactive
with col_coord_map:
    st.subheader("3. Visualisation & Sélection sur la Carte")

    # 1. Création de la carte Folium centrée sur le point actuel de l'utilisateur
    m = folium.Map(
        location=[st.session_state.latitude, st.session_state.longitude],
        zoom_start=7,
        tiles="cartodbpositron"
    )

    # 2. Ajout des marqueurs pour les emplacements des modèles
    for model in MODEL_REGISTRY:
        folium.Marker(
            [model['latitude'], model['longitude']],
            tooltip=f"{model['name']} (Modèle disponible)",
            icon=CUSTOM_SOLAR_ICON
        ).add_to(m)

    # 3. Ajout du marqueur de l'utilisateur (Point Rouge)
    folium.CircleMarker(
        [st.session_state.latitude, st.session_state.longitude],
        radius=8,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=1.0,
        tooltip=st.session_state.current_address
    ).add_to(m)

    # 4. Ajout du plugin pour cliquer et dessiner (pour la mise à jour des coordonnées)
    draw = Draw(
        export=False,
        position='topleft',
        draw_options={
            # Désactiver toutes les formes sauf le marqueur
            'polyline': False, 'polygon': False, 'circle': False,
            'circlemarker': False, 'rectangle': False,
            # Configurer le marqueur
            'marker': {'icon': folium.Icon(color='red', icon='map-pin', prefix='fa')}
        },
        edit_options={'edit': False, 'remove': False}  # Empêche l'édition/suppression des marqueurs existants
    )
    draw.add_to(m)

    # 5. Rendu de la carte et récupération de l'état
    map_data = st_folium(m, width=None, height=450, key="folium_map", return_on_hover=False)

    st.caption(
        f"🔴 : Votre emplacement.  : Emplacements des modèles disponibles. Utilisez l'icône de punaise (top-left) pour placer un nouveau point.")

# --- Logique de mise à jour des coordonnées à partir du clic (Draw) ---
if map_data and map_data.get("last_active_drawing"):
    drawing_type = map_data["last_active_drawing"].get("geometry", {}).get("type")

    if drawing_type == "Point":
        coords = map_data["last_active_drawing"]["geometry"]["coordinates"]

        # Folium retourne [longitude, latitude], nous devons les inverser
        new_lon_from_map = coords[0]
        new_lat_from_map = coords[1]

        # Pour le point cliqué sur la carte, on ne connaît pas l'adresse immédiatement.
        # Vous pourriez faire un géocodage inverse ici si vous le souhaitez,
        # mais pour simplifier, nous mettons à jour les coordonnées et laissons l'adresse
        # telle quelle (ou la réinitialiser si vous préférez).

        # Mettre à jour les variables de session, ce qui rafraîchira l'interface
        st.session_state.latitude = new_lat_from_map
        st.session_state.longitude = new_lon_from_map
        st.session_state.current_address = f"Coords cliquées ({new_lat_from_map:.4f}, {new_lon_from_map:.4f})"
        st.rerun()

# --- Suite de l'Interface (Modèle le Plus Proche) ---

# Affichage du modèle sélectionné (utilise les coordonnées de session)
closest_model_info, distance = find_closest_model(st.session_state.latitude, st.session_state.longitude)

st.info(
    f"**Modèle Sélectionné:** **{closest_model_info['name']}**.\n\n"
    f"Ce modèle est le plus proche géographiquement (à **{distance:,.0f} km**) "
    f"de votre localisation ({closest_model_info['latitude']:.4f}, {closest_model_info['longitude']:.4f})."
)

# Inputs du Système PV (Orientation & Azimuth) (INCHANGÉ)
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

# --- 6. Logique de Prédiction (INCHANGÉE) ---

if predict_button:
    pv_model = load_model(closest_model_info['path'])
else:
    pv_model = None

if pv_model and predict_button:
    st.header("Résultats de la Prédiction")
    st.caption(f"**Modèle utilisé pour cette prédiction:** {closest_model_info['location_info']}")

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
            'global_tilted_irradiance_(W/m²)', 'temperature_2m_(°C)',
            'cloud_cover_(%)', 'relative_humidity_2m_(%)', 'wind_speed_10m_(km/h)',
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

        with st.spinner("Calcul des prédictions de production PV..."):
            predictions = pv_model.predict(X)
            predictions[predictions < 0] = 0
            predictions[predictions > 1] = 1
            df_processed['Production_PV_kW'] = predictions

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
            f"Prévision pour {st.session_state.current_address}, Inclinaison: {tilt}°, Azimut: {azimuth}°.")
    else:
        st.warning("Impossible de procéder à la prédiction sans données météo valides.")