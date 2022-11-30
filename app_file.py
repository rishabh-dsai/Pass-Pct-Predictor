# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 20:22:12 2022

@author: asus
"""

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

feature_imp=pd.read_excel("Feature importances.xlsx")
feature_imp.set_index('Feature',inplace=True)


#%%

# Displaying the Feature Importances & Heatmap

with st.expander("Feature Importance",expanded=True):
    st.subheader("The feature importances of top 10 features are represented below:")
    st.bar_chart(feature_imp,use_container_width=True)


with st.expander("Correaltion Heatmap"):
    st.subheader("A correlation heatmap to show the relationship between features. \
                 More importantly between the Pass Percentage & other features.")
    fig=px.imshow(heatmap_data.corr())
#    fig.show()
    st.plotly_chart(fig, use_container_width=True)

#%%

# Predictions on Data

X=dataframe[['Gen_Studen', 'x_girls', 'Boundary_w', 'Per_m_Lit', 'PTR', 'x_boys',
       'Tot_Teachers', 'OBC_Studen', 'Qualified_T', 'ST_Student']]
y_preds=model.predict(X)

#predictions=pd.DataFrame(y_preds,columns=['Predictions'],index=dataframe['School name'])

dataframe['Predicted pass percentage (%)']=y_preds
st.subheader(" ")


#%%

# Option to type the school name to get exact preds:

tab_school,tab_block,tab_district=st.tabs(['School Specific','Block Specific','District Overview'])    

with tab_school:
    
    school=st.text_input("Please type the name of School",value="Dummy_11")
    st.caption("School "+school+" metrics and Predicted Pass Percentage:")
    st.write(dataframe[dataframe['School Name']==school].set_index("School Name"))
    
with tab_block:
    
    block=st.selectbox("Please select a Block",\
                       ('Pynursla', 'Mylliem', 'Mawphlang', 'Mawknyrew', 'Mawsynram'))
    block_df=dataframe[dataframe['Block']==block]
    st.write("Number of Schools in the block: ",block_df['School Name'].nunique())
    sub_df=block_df[['School Name', 'Block', 'District','Predicted pass percentage (%)']]
    st.write(sub_df)
    bar_ch=px.bar(sub_df[['School Name','Predicted pass percentage (%)']].set_index("School Name"),\
                  color_continuous_scale="ylgnbu")

    st.plotly_chart(bar_ch,use_container_width=True)





#%%

# Displaying Disclaimer & Header

st.header(" ")
from PIL import Image
image = Image.open('deepspatial.jpg')
image_1=image.resize((180,30))
st.image(image_1)



