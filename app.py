import streamlit as st
import pandas as pd
import pickle
import joblib
import altair as alt

# Création de l'interface principale

st.title('Modèle de prédiction de production PV')
st.write("Ce modèle a été établit à partir de données de production PV obtenues pour un site d'Utrecht (Pays-Bas)")
st.write("Les données proviennent de XXXX")
st.write("Le modèle utilisé dans cette application est un modèle de type **LightGBM** prenant en données d'entrées : ")
st.markdown("- **Température (°C)**")
st.markdown("- **Humidité Relative (%)**")
st.markdown("- **Irradiance (W/m$^2$)**")


# --- 1. Chargement du Modèle ---
try:
    # Charger la pipeline de Scikit-learn
    pipeline = joblib.load('modele_lineaire.pkl')
    st.sidebar.success("✅ Modèle 'modele_lineaire.pkl' chargé avec succès.")

    st.info(
        "**NOTE IMPORTANTE :** La prédiction est post-traitée (clipping) pour rester dans la plage **[0, 1]** "
        "comme spécifié par votre configuration de modèle. La sortie représente donc une **production normalisée** "
        "(ratio sur la capacité maximale installée)."
    )
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle : {e}")
    st.stop()

# --- 2. Importation du Fichier de Données ---
st.header('📥 Importez vos données de prédiction')
uploaded_file = st.file_uploader(
    "Choisissez un fichier CSV ou Excel contenant les données météorologiques à prédire",
    type=['csv', 'xlsx']
)

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)

        st.success("Fichier chargé avec succès. Aperçu des 5 premières lignes :")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"❌ Erreur de lecture du fichier : {e}")
        uploaded_file = None
else:
    st.warning("Veuillez importer un fichier pour continuer.")

# --- 3. Mappage des Colonnes et Exécution du Modèle ---

if df is not None:
    st.header('🔗 Mappage des colonnes d\'entrée')

    # Liste des colonnes disponibles
    column_list = df.columns.tolist()

    st.subheader("Variables Temporelles (Nécessaires pour le modèle)")
    datetime_col = st.selectbox(
        "**Colonne Date/Heure** (doit contenir la date et l'heure pour l'extraction)",
        column_list
    )

    st.subheader("Variables Météorologiques")
    col1, col2, col3 = st.columns(3)

    with col1:
        temp_col = st.selectbox(
            "Colonne pour la **Température (°C)**",
            column_list
        )
    with col2:
        humidity_col = st.selectbox(
            "Colonne pour l'**Humidité Relative (%)**",
            column_list
        )
    with col3:
        irradiance_col = st.selectbox(
            "Colonne pour l'**Irradiance (W/m$^2$)**",
            column_list
        )

    st.markdown("---")

    # Bouton d'exécution
    if st.button("🚀 Appliquer le Modèle et Prédire"):

        try:
            # 1. Préparation des données: Extraction des variables temporelles
            df['datetime_parsed'] = pd.to_datetime(df[datetime_col])

            # Créer un DataFrame avec les 6 noms de colonnes EXACTEMENT attendus
            data_to_predict = pd.DataFrame()

            # a) Variables Météo
            data_to_predict['temperature_2m (°C)'] = df[temp_col]
            data_to_predict['relative_humidity_2m (%)'] = df[humidity_col]
            data_to_predict['global_tilted_irradiance (W/m²)'] = df[irradiance_col]

            # b) Variables Temporelles
            data_to_predict['hour'] = df['datetime_parsed'].dt.hour
            data_to_predict['month'] = df['datetime_parsed'].dt.month
            data_to_predict['day_of_year'] = df['datetime_parsed'].dt.dayofyear

            # 2. Exécution de la prédiction
            predictions = pipeline.predict(data_to_predict)

            # --- APPLICATION DU CLIPPING DEMANDÉ PAR L'UTILISATEUR ---
            # Clipping la prédiction entre 0 et 1 (min=0 et max=1)
            predictions = pd.Series(predictions).clip(lower=0, upper=1).values

            # 3. Affichage des résultats
            PREDICTION_COL_NAME = 'Prédiction PV Normalisée (0-1)'
            df[PREDICTION_COL_NAME] = predictions

            st.success("✅ Prédictions terminées !")

            # --- 4. Création du Graphique ---
            st.subheader("Graphique de la Production PV Prédite (Normalisée)")

            # Création du graphique interactif Altair
            chart = alt.Chart(df).mark_line().encode(
                x=alt.X('datetime_parsed', title=datetime_col),
                y=alt.Y(PREDICTION_COL_NAME, title='Ratio de Production (0-1)'),
                tooltip=[datetime_col, PREDICTION_COL_NAME]
            ).properties(
                title='Ratio de Production PV Prédit au fil du temps'
            ).interactive()

            st.altair_chart(chart, use_container_width=True)

            # --- 5. Affichage des Données et Téléchargement ---
            st.subheader("Données avec Prédictions")

            # On retire la colonne parsée avant l'affichage final et le téléchargement
            df_final = df.drop(columns=['datetime_parsed'])
            st.dataframe(df_final)

            # Option de téléchargement
            st.download_button(
                label="Télécharger les résultats (CSV)",
                data=df_final.to_csv(index=False).encode('utf-8'),
                file_name='predictions_pv_normalisees.csv',
                mime='text/csv',
            )

        except KeyError as ke:
            st.error(
                f"❌ Erreur : Colonne '{ke}' introuvable. Assurez-vous que toutes les colonnes sont correctement sélectionnées.")
        except AttributeError as ae:
            st.error(
                f"❌ Erreur de format de date/heure : Impossible de convertir la colonne '{datetime_col}' en format Date/Heure valide. Détails : {ae}")
        except Exception as e:
            st.error(f"❌ Une erreur inattendue est survenue lors de la prédiction : {e}")

