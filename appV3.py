import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# --- 1. Définition des Modèles et de leurs Localisations ---
# Ajoutez ici tous vos modèles (Utrecht, Lisbon, etc.)
MODEL_REGISTRY = [
    # Utrecht
    {
        "name": "Utrecht",
        "path": "modele_lightGBM.pkl",  # Le chemin de votre modèle actuel
        "latitude": 51.9701,
        "longitude": 5.3217,
        "location_info": "Modèle d'Utrecht (Pays-Bas)",
    },
    # Lisbon1
    {
        "name": "Lisbon1",
        "path": "modele_lightGBM_Lisbon1.pkl",  # Exemple: ce fichier doit exister !
        "latitude": 38.728,
        "longitude": -9.138,
        "location_info": "Modèle de Lisbonne (Portugal)",
    },
    # Faro
    {
        "name": "Faro",
        "path": "modele_lightGBM_Faro.pkl",  # Exemple: ce fichier doit exister !
        "latitude": 37.031,
        "longitude": -7.893,
        "location_info": "Modèle de Faro (Portugal)",
    },
    # Braga
    {
        "name": "Braga",
        "path": "modele_lightGBM_Braga.pkl",  # Exemple: ce fichier doit exister !
        "latitude": 41.493,
        "longitude": -8.496,
        "location_info": "Modèle de Braga (Portugal)",
    },
    # Setubal
    {
        "name": "Setubal",
        "path": "modele_lightGBM_Setubal.pkl",  # Exemple: ce fichier doit exister !
        "latitude": 38.577,
        "longitude": -8.872,
        "location_info": "Modèle de Setubal (Portugal)",
    },
]


# --- 2. Fonction pour la Distance Géographique (Haversine) ---

def haversine(lat1, lon1, lat2, lon2):
    """
    Calcule la distance entre deux points (lat, lon) sur une sphère (Terre).
    Utilise la formule de Haversine. Le résultat est en kilomètres (approx).
    """
    # Rayon de la Terre en km
    R = 6371

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
    """
    Récupère les prévisions météorologiques horaires incluant l'irradiation globale inclinée (GTI).
    (Fonction inchangée)
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
        response.raise_for_status()  # Lève une exception pour les codes d'état 4xx ou 5xx
        data = response.json()

        if 'hourly' not in data:
            st.warning("Aucune donnée horaire ('hourly') trouvée dans la réponse de l'API.")
            return None

        # Créer le DataFrame à partir des données horaires
        df = pd.DataFrame(data['hourly'])

        # Renommer les colonnes pour une meilleure lisibilité (et pour la compatibilité avec le modèle à l'étape suivante)
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
# Titre en haut du site
st.title("Système de Prédiction de Production PV")

# Explication de l'application
st.markdown("Le modèle de prédiction utilisé est un modèle de type LightGBM")
st.markdown(
    "Les données météos (historiques et de prévisions qui sont utilisées sur l'appli) proviennent de **open-meteo.com**")

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

# Interface utilisateur pour la localisation
st.header("Localisation du panneau PV et Sélection de Modèle")

# --- NOUVEAUTÉ : Affichage de la carte et des inputs ---
col_map, col_input = st.columns([3, 1])

default_lat = 40.0  # Centre de la carte initiale (région Portugal/NL)
default_lon = 0.0

# Initialisation des variables de session pour la persistance des inputs
if 'latitude' not in st.session_state:
    st.session_state.latitude = 51.9701  # Coordonnées par défaut (Utrecht)
if 'longitude' not in st.session_state:
    st.session_state.longitude = 5.3217

# 1. Inputs manuels (dans la colonne de droite)
with col_input:
    st.subheader("Saisie Manuelle")
    st.session_state.latitude = st.number_input(
        "Latitude (Lat)",
        min_value=-90.0, max_value=90.0,
        value=st.session_state.latitude,
        format="%.4f",
        key="manual_lat"
    )
    st.session_state.longitude = st.number_input(
        "Longitude (Long)",
        min_value=-180.0, max_value=180.0,
        value=st.session_state.longitude,
        format="%.4f",
        key="manual_lon"
    )

    st.markdown(
        """
        *Conseil : Utilisez ces champs pour ajuster 
        précisément votre position, en vous aidant 
        de la carte à gauche.*
        """
    )

# 2. Préparation des données pour la carte (dans la colonne de gauche)
# Créer le DataFrame du point utilisateur
user_point = pd.DataFrame({
    'lat': [st.session_state.latitude],
    'lon': [st.session_state.longitude],
    'type': ['Votre Emplacement']
})

# Créer le DataFrame des emplacements des modèles
model_points = pd.DataFrame([
    {'lat': m['latitude'], 'lon': m['longitude'], 'type': m['name']}
    for m in MODEL_REGISTRY
])

# Fusionner les deux DataFrames pour l'affichage de la carte
# Attention: Streamlit map utilise 'lat' et 'lon' par défaut, mais les couleurs ne sont pas modifiables facilement
map_data = pd.concat([user_point, model_points])

with col_map:
    st.subheader("Visualisation de l'Emplacement")

    # Affichage de la carte
    st.map(
        map_data,
        latitude='lat',
        longitude='lon',
        zoom=7,  # Zoom par défaut pour l'Europe de l'Ouest
        use_container_width=True
    )
    st.caption("🔴 : Votre emplacement. Les autres points sont les modèles disponibles.")
# --- FIN NOUVEAUTÉ ---

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