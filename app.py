import pandas as pd
import numpy as np
import shap
import streamlit as st
import matplotlib.pyplot as plt
import pickle as p
st.title("C'est ma première application de prédiction de défaut de paiement des clients d'une banque")

LABELS = ["PAID", "DEFALUT"]
model = p.load(open("model.pkl", 'rb'))
df=pd.read_csv("scoring.csv")
data = df.copy()
st.dataframe(data)
#st.write('first row original data')
#st.write(data.head())
st.sidebar.header("Veuillez choisir le client : ")
st.sidebar.write('Nombre de clients disponibles :',data.shape[0])
 # ID Client sur liste déroulante
liste_ids_filtre = data['SK_ID_CURR'].unique().tolist()
id_code = st.sidebar.selectbox('IDs clients disponibles :', liste_ids_filtre)
if (id_code in data['SK_ID_CURR'].values) == True:
    df_code = data[data['SK_ID_CURR'] == id_code]
    index = df_code.index.tolist()[0]
    
st.write(df_code.head())
result = model.predict(df_code.drop('SK_ID_CURR', axis=1))
if result[0] == 1:
    st.write(f"Le client {id_code} a fait un défaut de paiement")
else:
    st.write(f"Le client {id_code} a peut être remboursé son prêt")

explainer = shap.LinearExplainer(model, data.drop('SK_ID_CURR',axis=1), feature_dependence="independent")
shap_values = explainer.shap_values(data.drop('SK_ID_CURR',axis=1))

if st.checkbox('Interprétatbilté globale'):
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, data.drop('SK_ID_CURR',axis=1), plot_type="bar", class_names=model.classes_)
        st.pyplot(fig)
    # display data
if st.checkbox('Interprétatbilté globale individuelle'):
        st.header("Données du client contribuant à la décision :")
        st.set_option('deprecation.showPyplotGlobalUse', False)        
        st.pyplot(shap.waterfall_plot(shap.Explanation(values=shap_values[index,:],
                                                     base_values=explainer.expected_value,
                                                     data=data.drop('SK_ID_CURR',axis=1).iloc[index,:],
                                                     feature_names=data.drop('SK_ID_CURR',axis=1).columns.tolist()),max_display=20))