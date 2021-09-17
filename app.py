# Import des librairies
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.figure_factory as ff

# Import des données
df_model = pd.read_csv('csv/data_model_sampled.csv',index_col = 'SK_ID_CURR')

# Modification des données en chaîne de charactères pour la transposition
df = df_model.astype(np.dtype(str))

# Chargement du modèle
model = joblib.load('reg_log.sav')

# Instanciation des différentes section
header = st.container()
result_ml = st.container()
info_client = st.container()
info_comp = st.container()
feature_imp = st.container()
filtered_dataset = st.container()

# Section 1 de la sidebar
st.sidebar.header('Sélection du numéro client')
id_client = st.sidebar.selectbox('Identifiant client', df_model.index)

# Affichage du titre et des données relatives au client sélectionné
with header :
    st.title('''Tableau de bord d'allocation de crédit client''' )
df_client = df.loc[df.index == int(id_client)].transpose()
df_client.columns = ['Informations clients']

with info_client :
    st.header('''Informations personnelles''')
    st.write(df_client)

# Préparation des graphiques comparatifs
amt_inc_total = np.log(df_model.loc[df_model.index == int(id_client), 'AMT_INCOME_TOTAL'].values[0])
x_a = [np.log(df_model['AMT_INCOME_TOTAL'])]
fig_a = ff.create_distplot(x_a,['AMT_INCOME_TOTAL'], bin_size=0.3)
fig_a.add_vline(x=amt_inc_total, annotation_text=' Vous êtes ici')


x_b = [np.log(df_model['AMT_CREDIT'])]
var = np.log(df_model.loc[df_model.index == int(id_client), 'AMT_CREDIT'].values[0])
fig_b = ff.create_distplot(x_b,['AMT_CREDIT'], bin_size=0.3)
fig_b.add_vline(x=var, annotation_text=' Vous êtes ici')


x_c = [np.log(df_model['SUM(previous.AMT_APPLICATION)']+1)]
var = np.log(df_model.loc[df_model.index == int(id_client), 'SUM(previous.AMT_APPLICATION)'].values[0])
fig_c = ff.create_distplot(x_c,['SUM(previous.AMT_APPLICATION)'], bin_size=0.3)
fig_c.add_vline(x=var, annotation_text=' Vous êtes ici')

# Visualisation des graphiques
with info_comp :
    st.header('''Informations comparatives''')
    st.subheader('Vos revenus')
    st.plotly_chart(fig_a, use_container_width=True)
    st.subheader('Montant du crédit')
    st.plotly_chart(fig_b, use_container_width=True)
    st.subheader('Somme des crédits demandés lors des précédentes demandes')
    st.plotly_chart(fig_c, use_container_width=True)

# Résultat de la modélisation    
with result_ml :
    st.header('''Résultat de la demande de crédit''')
    per_pos = model.predict_proba(df_model.loc[df_model.index == int(id_client)])[0][1]
    if per_pos < 0.59 :
        st.markdown("<p style=color:Green;font-weight:bold> Votre crédit est accepté</p>" , unsafe_allow_html=True)
    else :
        st.markdown("<p style=color:Red;font-weight:bold> Votre crédit est refusé</p>" , unsafe_allow_html=True)
    st.write('Votre crédit est refusé si ce score est supérieur à 0.59 : {}'.format(round(per_pos,3)))

# Les variables les plus importantes dans la modélisation
with feature_imp :
    st.header('''Variables les plus importantes dans le calcul de l'allocation''')
    var_pos = ['AMT_CREDIT',
       'SUM(previous.AMT_CREDIT)', 'SKEW(previous.CNT_PAYMENT)',
       'STD(previous.RATE_DOWN_PAYMENT)']

    var_neg = ['SUM(previous.AMT_APPLICATION)', 'SUM(previous.AMT_GOODS_PRICE)',
        'SKEW(previous.AMT_APPLICATION)', 'SKEW(previous.SELLERPLACE_AREA)']
    
    df_feat_imp = df 
    st.subheader('Plus ces valeurs sont hautes, plus vous avez de chances que votre crédit soit **refusé**')
    df_pos = df_feat_imp.loc[df_feat_imp.index == int(id_client)][var_pos].transpose()
    st.write(df_pos)
    st.subheader('Plus ces valeurs sont hautes, plus vous avez de chances que votre crédit soit **accepté**')
    df_neg = df_feat_imp.loc[df_feat_imp.index == int(id_client)][var_neg].transpose()
    st.write(df_neg)

# Outil permettant de sélectionner des variables et visualiser les données par rapport à plusieurs clients
with filtered_dataset :
    st.title('''Explorateur du jeu de données''')
    is_check = st.checkbox("Affichage des données")
    st.sidebar.header('Explorateur du jeu de données')
    columns_display = st.sidebar.multiselect("Étape n°1 : Sélectionner les variables à afficher", df_model.columns)
    columns_display = list(columns_display)

    columns_filter = st.sidebar.multiselect("Étape n°2 : Sélectionner les variables à filtrer", columns_display)
    columns_filter = list(columns_filter)
    
    min_max = st.checkbox("Ordre croissant")

    if is_check & min_max :
        st.write(df_model[columns_display].sort_values(by=columns_filter, ascending=True))

    elif is_check :
        st.write(df_model[columns_display].sort_values(by=columns_filter, ascending=False))
