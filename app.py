import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
<<<<<<< HEAD
import plotly.figure_factory as ff

df_model = pd.read_csv('csv/data_model_sampled.csv')

=======


df_model = pd.read_csv('csv/data_model_sampled.csv')

df_model.drop('Unnamed: 0', axis=1, inplace=True)

>>>>>>> ca1d78941d9e8d8487144c692660597e95816bee
df = df_model.astype(np.dtype(str))

model = joblib.load('reg_log.sav')

header = st.container()
result_ml = st.container()
info_client = st.container()
info_comp = st.container()
feature_imp = st.container()

id_client = st.sidebar.text_input('Identifiant client', value="174545",max_chars=6)

with header :
    st.title('''Tableau de bord d'allocation de crédit client''' )

df_client = df.loc[df.SK_ID_CURR == id_client].transpose()
df_client.columns = ['Informations clients']


with info_client :
    st.header(''' informations personnelles''')
    st.write(df_client)

<<<<<<< HEAD
amt_inc_total = np.log(df_model.loc[df_model.SK_ID_CURR == int(id_client), 'AMT_INCOME_TOTAL'].values[0])
x_a = [np.log(df_model['AMT_INCOME_TOTAL'])]
fig_a = ff.create_distplot(x_a,['AMT_INCOME_TOTAL'], bin_size=0.3)
fig_a.add_vline(x=amt_inc_total, annotation_text='Vous etes ici')


x_b = [np.log(df_model['AMT_CREDIT'])]
var = np.log(df_model.loc[df_model.SK_ID_CURR == int(id_client), 'AMT_CREDIT'].values[0])
fig_b = ff.create_distplot(x_b,['AMT_CREDIT'], bin_size=0.3)
fig_b.add_vline(x=var, annotation_text='Vous etes ici')


x_c = [np.log(df_model['SUM(previous.AMT_APPLICATION)']+1)]
var = np.log(df_model.loc[df_model.SK_ID_CURR == int(id_client), 'SUM(previous.AMT_APPLICATION)'].values[0])
fig_c = ff.create_distplot(x_c,['SUM(previous.AMT_APPLICATION)'], bin_size=0.3)
fig_c.add_vline(x=var, annotation_text='Vous etes ici')

=======
fig, ax = plt.subplots()
sns.histplot(np.log(df_model['AMT_INCOME_TOTAL']), bins=30, ax=ax)
amt_inc_total = np.log(df_model.loc[df_model.SK_ID_CURR == int(id_client), 'AMT_INCOME_TOTAL'].values[0])
ax.axvline(x=amt_inc_total, color='red')

fig1, ax1 = plt.subplots()
sns.histplot(np.log(df_model['AMT_CREDIT']), bins=30, ax=ax1)
var = np.log(df_model.loc[df_model.SK_ID_CURR == int(id_client), 'AMT_CREDIT'].values[0])
ax1.axvline(x=var, color='red')

fig2, ax2 = plt.subplots()
sns.histplot(np.log(df_model['SUM(previous.AMT_APPLICATION)']), bins=30, ax=ax2)
var = np.log(df_model.loc[df_model.SK_ID_CURR == int(id_client), 'SUM(previous.AMT_APPLICATION)'].values[0])
ax2.axvline(x=var, color='red')
>>>>>>> ca1d78941d9e8d8487144c692660597e95816bee


with info_comp :
    st.header(''' informations comparatives''')
<<<<<<< HEAD
    st.plotly_chart(fig_a, use_container_width=True)
    st.plotly_chart(fig_b, use_container_width=True)
    st.plotly_chart(fig_c, use_container_width=True)
    
=======
    st.pyplot(fig)
    st.pyplot(fig1)
    st.pyplot(fig2)

>>>>>>> ca1d78941d9e8d8487144c692660597e95816bee
with result_ml :
    st.header('''Pourcentage de chance d'acceptation du crédit''')
    st.write(model.predict_proba(df_model.loc[df_model.SK_ID_CURR == int(id_client)])[0][1])

with feature_imp :
    st.title('''Variables les plus importantes''')
    var_pos = ['AMT_CREDIT', 'ORGANIZATION_TYPE',
       'SUM(previous.AMT_CREDIT)', 'SKEW(previous.CNT_PAYMENT)',
       'STD(previous.RATE_DOWN_PAYMENT)']

    var_neg = ['SUM(previous.AMT_APPLICATION)', 'SUM(previous.AMT_GOODS_PRICE)',
       'ORGANIZATION_TYPE',
       'SKEW(previous.AMT_APPLICATION)', 'SKEW(previous.SELLERPLACE_AREA)']
    
    df_feat_imp = df 

    df_pos = df_feat_imp.loc[df_feat_imp.SK_ID_CURR == id_client][var_pos].transpose()
    df_neg = df_feat_imp.loc[df_feat_imp.SK_ID_CURR == id_client][var_neg].transpose()
    st.write(df_pos)
    st.write(df_neg)
