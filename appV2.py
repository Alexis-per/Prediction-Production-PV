import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np
import altair as alt
from datetime import datetime

# Configuration application
st.set_page_config(
    page_title="Prédiction de Production PV",
    layout="wide",
    initial_sidebar_state="expanded"
)

# On charge le modèle de prédiction
MODEL_PATH = "modele_lightGBM.pkl"


@st.cache_resource
def load_model(path):
    """Charge le modèle LightGBM pré-entraîné."""
    try:
        model = joblib.load(path)
        st.success("Modèle de prédiction chargé avec succès.")
        return model
    except FileNotFoundError:
        st.error(f"Erreur : Fichier modèle '{path}' introuvable. Assurez-vous qu'il est dans le même répertoire.")
        return None


pv_model = load_model(MODEL_PATH)


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
        # Utiliser un modèle précis pour l'Europe (si applicable) ou GFS (Global)
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
st.markdown("Il a été conçu à partir de données de production PV à Utrecht (Pays-Bas)")
st.markdown("Les données proviennent de (mettre lien ressource)")
st.markdown("Les données météos (historiques et de prévisions qui sont utilisées sur l'appli) proviennent de **open-meteo.com**")
st.markdown("Le modèle prend à la fois des variables météos :")
st.markdown("- **Température (°C)**")
st.markdown("- **Humidité Relative (%)**")
st.markdown("- **Vitesse du vent à 10m (km/h)**")
st.markdown("- **Couverture nuageuse (%)**")
st.markdown("- **Irradiation global orientée (W/m$^2$)**")
st.markdown("Et des variables temporelles :")
st.markdown("- **Mois**")
st.markdown("- **Jour de l'année**")
st.markdown("- **Heure**")
st.markdown("---")

# Interface utilisateur
st.header("Localisation du panneau PV")

# Inputs de Localisation
latitude = st.number_input("Latitude (Lat)", min_value=-90.0, max_value=90.0, value=48.8566, format="%.4f")
longitude = st.number_input("Longitude (Long)", min_value=-180.0, max_value=180.0, value=2.3522, format="%.4f")

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

if pv_model and predict_button:
    st.header("Résultats de la Prédiction")

    # Récupération des données
    with st.spinner(f"Récupération des prévisions météo sur {forecast_days} jours pour ({latitude}, {longitude})..."):
        raw_df = fetch_weather_data(latitude, longitude, tilt, azimuth, forecast_days)

    if raw_df is not None:

        # Préparation des données (conversion des dates)

        # Convertir la colonne 'time' en datetime et extraire les features temporelles
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
        # Nous allons exclure 'hour', 'month', 'day_of_year' qui sont pour le modèle
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
            # Notez que st.area_chart est souvent visuellement agréable pour la couverture nuageuse
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

        # Affichage de la production totale sur la période considérée
        total_production = df_processed['Production_PV_kW'].sum()

        st.write(f"Production Totale Prévue sur {forecast_days} jours")
        production_value = f"{total_production:,.2f}".replace(",", " ")

        st.markdown(f"**##{production_value} kWh/kWc**")

        daily_production = df_processed.set_index('time').resample('D')['Production_PV_kW'].sum()
        if not daily_production.empty:
                st.subheader("Répartition Journalière (kWh/kWc)")
                st.dataframe(daily_production.to_frame(name='kWh/kWc par jour').style.format("{:,.2f}"))

        # 4.4.2 Affichage graphique (Série Temporelle)
        st.subheader("Prévision Horaire de Production PV (kW/kwc)")
        df_chart = df_processed.set_index('time')[['Production_PV_kW']]

        st.line_chart(df_chart, use_container_width=True)

        st.caption(f"Prévision pour Lat: {latitude}, Long: {longitude}, Inclinaison: {tilt}°, Azimut: {azimuth}°.")

    else:
        st.warning("Impossible de procéder à la prédiction sans données météo valides.")