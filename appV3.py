import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# --- 1. Définition des Modèles et de leurs Localisations ---
# Ajoutez ici tous vos modèles (Utrecht, Paris, Tokyo, etc.)
MODEL_REGISTRY = [
    {
        "name": "Utrecht (NL)",
        "path": "modele_lightGBM.pkl",  # Le chemin de votre modèle actuel
        "latitude": 52.0907,
        "longitude": 5.1214,
        "location_info": "Modèle d'Utrecht (Pays-Bas)",
    },
    {
        "name": "Paris (FR)",
        "path": "modele_paris_lightGBM.pkl",  # Exemple: ce fichier doit exister !
        "latitude": 48.8566,
        "longitude": 2.3522,
        "location_info": "Modèle de Paris (France)",
    },
    # Ajoutez d'autres modèles ici si vous les avez:
    # {
    #     "name": "Localité X",
    #     "path": "modele_X.pkl",
    #     "latitude": X.XXX,
    #     "longitude": Y.YYY,
    #     "location_info": "Modèle de Localité X",
    # },
]


# --- 2. Fonction pour la Distance Géographique (Haversine) ---

def haversine(lat1, lon1, lat2, lon2):
    """
    Calcule la distance entre deux points (lat, lon) sur une sphère (Terre).
    Utilise la formule de Haversine. Le résultat est en kilomètres (approx).
    """
    R = 6371  # Rayon de la Terre en km

    # Conversion des degrés en radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Différences
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Formule de Haversine
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance


# --- 3. Fonction pour trouver le Modèle le Plus Proche ---

def find_closest_model(user_latitude, user_longitude):
    """
    Trouve le modèle dans le registre qui est géographiquement le plus proche
    des coordonnées fournies.
    """
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


# --- 4. Fonction pour le Géocodage (Adresse -> Lat/Lon) ---

def geocode_address(address):
    """
    Convertit une adresse textuelle en coordonnées (latitude, longitude)
    en utilisant l'API Nominatim (OpenStreetMap).
    """
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": 1
    }

    try:
        # Définir un agent utilisateur pour être poli avec l'API OSM
        headers = {'User-Agent': 'PV_Prediction_App/1.0'}
        response = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data:
            # On prend le premier résultat
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


# --- 5. Fonction de Chargement du Modèle ---

@st.cache_resource
def load_model(path):
    """Charge le modèle LightGBM pré-entraîné à partir du chemin spécifié."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Erreur : Fichier modèle '{path}' introuvable. Assurez-vous qu'il existe.")
        return None


# --- 6. Fonction de Récupération des Données Météo ---

# Utilsiation de l'API de open-meteo.com pour obtenir les données prévisionnelles
def fetch_weather_data(latitude, longitude, tilt, azimuth, days=7):
    """
    Récupère les prévisions météorologiques horaires incluant l'irradiation globale inclinée (GTI).
    """

    # Variables météorologiques requises par le modèle
    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "cloud_cover",
        "global_tilted_irradiance"  # Le plus important pour le PV
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

        # Créer le DataFrame à partir des données horaires
        df = pd.DataFrame(data['hourly'])

        # Renommer les colonnes pour la compatibilité avec le modèle
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


# --- 7. Configuration de l'Application Streamlit ---

st.set_page_config(
    page_title="Prédiction de Production PV",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration de l'état de session par défaut (pour la barre d'adresse)
if 'latitude' not in st.session_state:
    st.session_state.latitude = 48.8584  # Default: Paris (Tour Eiffel)
if 'longitude' not in st.session_state:
    st.session_state.longitude = 2.2945
if 'current_address' not in st.session_state:
    st.session_state.current_address = "Tour Eiffel, Paris, France"

# Interface de l'application
st.title("Système de Prédiction de Production PV")
st.markdown("Le modèle de prédiction utilisé est un modèle de type LightGBM.")
st.markdown("Les données météos proviennent de **open-meteo.com**.")

# Présentation des variables utilisées dans le modèle
col_meteo, col_temporelle = st.columns(2)
# Variables météos
with col_meteo:
    st.markdown("### Variables Météo")
    st.markdown("- **Température (°C)**")
    st.markdown("- **Humidité Relative (%)**")
    st.markdown("- **Vitesse du vent à 10m (km/h)**")
    st.markdown("- **Couverture nuageuse (%)**")
    st.markdown("- **Irradiation global orientée (W/m$^2$)**")
# Variables temporelles
with col_temporelle:
    st.markdown("### Variables Temporelles")
    st.markdown("- **Mois**")
    st.markdown("- **Jour de l'année**")
    st.markdown("- **Heure**")

st.markdown("---")

# --- 8. Section Localisation (Adresse) ---
st.header("Localisation du panneau PV et Sélection de Modèle")

address_input = st.text_input(
    "Rechercher une Adresse / Lieu (Ex: 'Amsterdam', '30 St Mary Axe, London')",
    value=st.session_state.current_address,
    key="address_input_widget"
)

# Le bouton est important pour déclencher le géocodage explicitement
geocode_button = st.button("Chercher les Coordonnées", type="secondary")

# Logique de géocodage
if geocode_button or (address_input != st.session_state.current_address and address_input):
    # L'utilisateur a cliqué OU l'utilisateur a modifié l'adresse et le champ n'est pas vide
    with st.spinner(f"Recherche de '{address_input}'..."):
        new_lat, new_lon, found_address = geocode_address(address_input)

        if new_lat is not None and new_lon is not None:
            # Mise à jour des coordonnées et de l'adresse dans l'état de session
            st.session_state.latitude = new_lat
            st.session_state.longitude = new_lon
            st.session_state.current_address = found_address
            st.success(f"Adresse trouvée: **{found_address}**")

        elif address_input:
            st.error(f"Impossible de trouver les coordonnées pour : {address_input}")

# Utilisation des coordonnées stockées dans session_state
latitude = st.session_state.latitude
longitude = st.session_state.longitude

# Affichage des coordonnées trouvées et de la carte
st.markdown(f"**Coordonnées Actuelles :** Lat **{latitude:.4f}**, Long **{longitude:.4f}**")
st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}), zoom=10)

# Sélection du Modèle le plus proche
closest_model_info, distance = find_closest_model(latitude, longitude)

st.info(
    f"ℹ️ **Modèle Sélectionné:** **{closest_model_info['name']}**.\n\n"
    f"Ce modèle est le plus proche géographiquement (à **{distance:,.0f} km**) "
    f"de votre localisation ({closest_model_info['latitude']:.4f}, {closest_model_info['longitude']:.4f})."
)

# --- 9. Inputs du Système PV et Lancement ---

st.markdown("---")
st.subheader("Orientation du panneau PV")

col_tilt, col_azimuth = st.columns(2)

with col_tilt:
    tilt = st.number_input(
        "Inclinaison [°]",
        min_value=0.0, max_value=90.0, value=35.0, format="%.1f",
        help="Angle du panneau par rapport à l'horizontale (0°=plat, 90°=vertical)."
    )

with col_azimuth:
    azimuth = st.number_input(
        "Azimut (Orientation) [°]",
        min_value=-180.0, max_value=180.0, value=0.0, format="%.1f",
        help="Orientation des panneaux: 0°=Sud, 90°=Ouest, -90°=Est, ±180°=Nord (selon la convention Open-Meteo)."
    )

st.markdown("---")
forecast_days = st.slider("Jours de Prévision", 1, 16, 7)

predict_button = st.button("Lancer la Prédiction", type="primary")

# --- 10. Logique de Prédiction ---

pv_model = None
if predict_button:
    # Charger le modèle UNIQUEMENT si l'utilisateur clique sur le bouton
    pv_model = load_model(closest_model_info['path'])

if pv_model and predict_button:
    st.header("Résultats de la Prédiction")

    st.caption(f"**Modèle utilisé:** {closest_model_info['location_info']}")

    # Récupération des données
    with st.spinner(f"Récupération des prévisions météo sur {forecast_days} jours pour ({latitude}, {longitude})..."):
        raw_df = fetch_weather_data(latitude, longitude, tilt, azimuth, forecast_days)

    if raw_df is not None:

        # Préparation des données pour le modèle
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

        # Filtrer et réorganiser les colonnes
        X = df_processed[FEATURE_NAMES]

        # --- Visualisation des Données Météo ---
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

        # --- Faire la Prédiction ---
        with st.spinner("Calcul des prédictions de production PV..."):

            predictions = pv_model.predict(X)

            # S'assurer que la production est positive et plafonnée à 1 kW/kWc
            predictions[predictions < 0] = 0
            predictions[predictions > 1] = 1

            df_processed['Production_PV_kW'] = predictions

        # --- Affichage des Résultats ---
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

        st.caption(f"Prévision pour {st.session_state.current_address}, Inclinaison: {tilt}°, Azimut: {azimuth}°.")

    else:
        st.warning("Impossible de procéder à la prédiction sans données météo valides.")