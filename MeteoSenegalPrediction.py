import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import datetime

# Charger les données
df = pd.read_csv('df_final.csv')

# Convertir la colonne 'Date' en type datetime
df['Date'] = pd.to_datetime(df['Date'])

# Configuration de l'interface utilisateur
st.title("Prévisions Météorologiques")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
nav_option = st.sidebar.radio("Choisissez une option", ("Accueil", "Prédictions", "Exploration des données", "Comparaison historique"))

# Accueil
if nav_option == "Accueil":
    st.header("Bienvenue sur l'application de Prévisions Météorologiques")
    st.write("""
        Cette application vous permet de :
        - Explorer les données météorologiques historiques de différentes stations.
        - Faire des prédictions météorologiques pour des périodes futures.
        - Comparer les prévisions avec des données historiques pour une même période.
        Utilisez la barre latérale pour naviguer entre les différentes options.
    """)

# Prédictions
if nav_option == "Prédictions":
    st.header("Prédictions Météorologiques")
    
    # Sélection de la station
    station = st.sidebar.selectbox("Sélectionnez une station", df['Name_station'].unique())
    
    # Sélection de la date de départ
    start_date = st.sidebar.date_input("Date de début", datetime.date.today())
    
    # Sélection du nombre de mois de prédiction
    months_to_predict = st.sidebar.slider("Nombre de mois à prédire", 1, 12)
    
    # Bouton pour lancer la prédiction
    if st.sidebar.button("Prédire"):
        model = tf.keras.models.load_model(f'model_{station}.keras')
        
        # Prétraitement des données
        df_station = df[df['Name_station'] == station].copy()
        df_station['Date'] = pd.to_datetime(df_station['Date'])
        df_station.set_index('Date', inplace=True)
        df_station = df_station[['EVAPO', 'FF', 'INS', 'RR', 'TT, moy. des max', 'TT, moy. des min', 'UMAX', 'UMIN']]
        
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_station), columns=df_station.columns, index=df_station.index)
        
        # Générer les séquences à partir de la dernière date historique
        SEQ_LENGTH = 12
        last_sequence = df_scaled[-SEQ_LENGTH:].values
        predictions = []
        
        # Prédictions depuis la dernière date historique jusqu'à la date de début souhaitée
        historical_future_dates = pd.date_range(start=df_station.index[-1] + pd.DateOffset(months=1), end=start_date, freq='MS')
        
        for _ in range(len(historical_future_dates)):
            prediction = model.predict(last_sequence[np.newaxis, :, :])[0]
            predictions.append(prediction)
            last_sequence = np.append(last_sequence[1:], [prediction], axis=0)
        
        # Prédictions pour la période sélectionnée par l'utilisateur
        future_dates = pd.date_range(start=start_date, periods=months_to_predict, freq='MS')
        for _ in range(len(future_dates)):
            prediction = model.predict(last_sequence[np.newaxis, :, :])[0]
            predictions.append(prediction)
            last_sequence = np.append(last_sequence[1:], [prediction], axis=0)
        
        # Mise à l'échelle inverse et filtrage des résultats pour n'afficher que la période souhaitée
        predictions = scaler.inverse_transform(predictions[-months_to_predict:])
        predictions_df = pd.DataFrame(predictions, index=future_dates, columns=[
            'Évaporation (mm)', 'Vitesse du vent (m/s)', 'Insolation (h)', 'Précipitations (mm)', 
            'Temp. moy. max (°C)', 'Temp. moy. min (°C)', 'Humidité max (%)', 'Humidité min (%)'])
        
        # Affichage des résultats
        st.subheader(f"Prévisions pour {station}")
        st.write(f"Date de début: {start_date}")
        st.write(f"Nombre de mois prédit: {months_to_predict}")
        
        st.write(predictions_df)
        
        # Graphiques
        st.subheader("Graphiques des prévisions")
        fig = px.line(predictions_df, x=predictions_df.index, y=predictions_df.columns)
        st.plotly_chart(fig)
        
        # Exportation des données en CSV
        csv = predictions_df.to_csv().encode('utf-8')
        st.download_button(
            label="Télécharger les prédictions en CSV",
            data=csv,
            file_name=f'predictions_{station}.csv',
            mime='text/csv',
        )
        
        # Résumé textuel des prévisions
        st.subheader("Résumé des prévisions")
        st.write(f"Pour la station {station}, les prévisions indiquent une tendance de {months_to_predict} mois. "
                 "Les variations les plus marquantes concernent la température moyenne maximale, l'humidité maximale, "
                 "et la vitesse du vent. Ces indicateurs peuvent être essentiels pour l'agriculture, la gestion des ressources en eau, "
                 "et les préparations en cas de conditions météorologiques extrêmes.")

# Exploration des données
if nav_option == "Exploration des données":
    st.header("Exploration des Données Météorologiques")
    
    station = st.selectbox("Sélectionnez une station", df['Name_station'].unique())
    df_station = df[df['Name_station'] == station]
    
    # Sélection de la variable à explorer
    variable = st.selectbox("Sélectionnez une variable à explorer", [col for col in df_station.columns if col != 'Name_station'])
    
    # Affichage des statistiques descriptives
    st.subheader(f"Statistiques descriptives pour {variable}")
    st.write(df_station[variable].describe())
    
    # Affichage d'un graphique
    st.subheader(f"Évolution de {variable} au fil du temps")
    fig = px.line(df_station, x='Date', y=variable, title=f"{variable} au fil du temps pour {station}")
    st.plotly_chart(fig)
    
    # Affichage de la matrice de corrélation
    st.subheader("Matrice de Corrélation")
    corr_matrix = df_station.drop(columns=['Name_station']).corr()
    st.write(corr_matrix)
    fig_corr = px.imshow(corr_matrix, text_auto=True, title="Matrice de Corrélation")
    st.plotly_chart(fig_corr)
    
    # Option d'exportation
    if st.button("Exporter les données explorées"):
        df_station.to_csv(f'data_exploration_{station}.csv')
        st.success("Les données ont été exportées avec succès.")

# Comparaison historique
if nav_option == "Comparaison historique":
    st.header("Comparaison avec les données historiques")
    
    # Sélection de la station
    station = st.sidebar.selectbox("Sélectionnez une station", df['Name_station'].unique())
    
    # Sélection des variables à comparer
    var_options = st.multiselect("Sélectionnez les variables météorologiques à comparer", df.columns[3:])
    
    # Sélection des années de comparaison
    pred_year = st.sidebar.selectbox("Sélectionnez l'année pour les prévisions", range(2024, 2045))
    hist_year = st.sidebar.selectbox("Sélectionnez l'année historique", df['Date'].dt.year.unique())
    
    # Affichage des graphiques
    if st.sidebar.button("Comparer"):
        model = tf.keras.models.load_model(f'model_{station}.keras')
        
        # Données historiques
        df_hist = df[(df['Name_station'] == station) & (df['Date'].dt.year == hist_year)][['Date'] + var_options]
        
        # Données prévues
        start_date = datetime.date(pred_year, 1, 1)
        df_station = df[df['Name_station'] == station].copy()
        df_station['Date'] = pd.to_datetime(df_station['Date'])
        df_station.set_index('Date', inplace=True)
        df_station = df_station[['EVAPO', 'FF', 'INS', 'RR', 'TT, moy. des max', 'TT, moy. des min', 'UMAX', 'UMIN']]
        
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_station), columns=df_station.columns, index=df_station.index)
        
        SEQ_LENGTH = 12
        last_sequence = df_scaled[-SEQ_LENGTH:].values
        predictions = []
        future_dates = pd.date_range(start=start_date, periods=12, freq='MS')
        
        for _ in range(len(future_dates)):
            prediction = model.predict(last_sequence[np.newaxis, :, :])[0]
            predictions.append(prediction)
            last_sequence = np.append(last_sequence[1:], [prediction], axis=0)
        
        predictions = scaler.inverse_transform(predictions)
        df_pred = pd.DataFrame(predictions, index=future_dates, columns=df_station.columns)[var_options]
        
        # Graphiques
        st.subheader(f"Comparaison pour la station {station}")
        st.write(f"Année historique : {hist_year} | Année de prédiction : {pred_year}")
        
        st.subheader("Données historiques")
        fig_hist = px.line(df_hist, x='Date', y=var_options)
        st.plotly_chart(fig_hist)
        
        st.subheader("Données prévues")
        fig_pred = px.line(df_pred, x=df_pred.index, y=var_options)
        st.plotly_chart(fig_pred)
        
        # Exportation des graphiques
        csv_hist = df_hist.to_csv().encode('utf-8')
        csv_pred = df_pred.to_csv().encode('utf-8')
        
        st.download_button(
            label="Télécharger les données historiques en CSV",
            data=csv_hist,
            file_name=f'historique_{station}_{hist_year}.csv',
            mime='text/csv',
        )
        
        st.download_button(
            label="Télécharger les données prévues en CSV",
            data=csv_pred,
            file_name=f'previsions_{station}_{pred_year}.csv',
            mime='text/csv',
        )
