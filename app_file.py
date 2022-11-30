# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 18:19:27 2022

@author: asus
"""




import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import joblib
import plotly.express as px

#%%

uploaded_file = st.file_uploader("Choose the file in requisite format")
if uploaded_file is not None:
    dataframe = pd.read_excel(uploaded_file)

else:
    st.warning('Please Upload File')


model = joblib.load("Model.sav")
heatmap_data=pd.read_excel("Heatmap Data.xlsx")

fig=px.imshow(heatmap_data.corr())
fig.show()
st.plotly_chart(fig, use_container_width=True)


#%%


feature_imp=pd.read_excel("Feature Importances.xlsx")
feature_imp.set_index('Feature',inplace=True)


st.subheader("The feature importances of top 20 features are represented below:")
st.bar_chart(feature_imp,use_container_width=True)

inf=dataframe[['School name','Block','District']]



X=dataframe[['Gen_Studen', 'x_girls', 'Boundary_w', 'Per_m_Lit', 'ST_Student',
       'x_boys', 'Tot_Teachers', 'Qualified_T', 'PTR', 'SC_Student',
       'OBC_Studen', 'Repeater_x_g', 'Repeater_x_b', 'No_of_Girls',
       'Regular_Te', 'No_of_Boys', 'Area', 'No_of_Classrooms', 'PER_P_06',
       'Per_Illit']]
y_preds=model.predict(X)

predictions=pd.DataFrame(y_preds,columns=['Predictions'])

dataframe['Predictions']=predictions
st.subheader(" ")

st.subheader("The input provided and pass percentage prediction is denoted in the table below:")
st.write(dataframe)


st.caption("The pass percentage prediction represented in Chart:")
st.bar_chart(predictions[['Predictions']])

st.header(" ")
from PIL import Image
image = Image.open('deepspatial.jpg')
image_1=image.resize((180,30))
st.image(image_1)














