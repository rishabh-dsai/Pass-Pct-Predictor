

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
trend=pd.read_excel("trend.xlsx")
trend['PP_2021']=trend['PP_2021'].apply(lambda z:np.round(z,2)*100)
trend['PP_2020']=trend['PP_2020'].apply(lambda z:np.round(z,2)*100)
trend['PP_2019']=trend['PP_2019'].apply(lambda z:np.round(z,2)*100)
trend['PP_2018']=trend['PP_2018'].apply(lambda z:np.round(z,2)*100)
trend['PP_2017']=trend['PP_2017'].apply(lambda z:np.round(z,2)*100)

#%%

# Displaying the Feature Importances & Heatmap

with st.expander("Feature Importance",expanded=True):
    st.subheader("The feature importances of top 10 features are represented below:")
    st.bar_chart(feature_imp,use_container_width=True)


with st.expander("Correlation Heatmap"):
    st.write("A correlation heatmap to show the relationship between features. \
                 More importantly between the Pass Percentage & other features.")
    fig=px.imshow(heatmap_data.corr(),color_continuous_scale="viridis")
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
    st.write("School "+school+" metrics and Predicted Pass Percentage 2022:")
    school_df=dataframe[dataframe['School Name']==school].set_index("School Name")
    st.write(school_df)
    line_ch_sch=px.bar(x=list(school_df.columns),y=school_df.iloc[0,0])
    st.plotly_chart(line_ch_sch,use_container_width=True)
    
with tab_block:
    
    block=st.selectbox("Please select a Block",\
                       ('Pynursla', 'Mylliem', 'Mawphlang', 'Mawknyrew', 'Mawsynram'))
    block_df=dataframe[dataframe['Block']==block]
    st.write("Number of Schools in the block: ",block_df['School Name'].nunique())
    sub_df=block_df[['School Name', 'Block', 'District','Predicted pass percentage (%)']]
    st.write(sub_df.set_index("School Name"))
    st.write(" ")
    st.write("The average predicted pass percentage for the block: ",sub_df['Predicted pass percentage (%)'].mean(),"%")

    bar_ch=px.bar(sub_df[['School Name','Predicted pass percentage (%)']].set_index("School Name"),\
                  y='Predicted pass percentage (%)')

    st.plotly_chart(bar_ch,use_container_width=True)


with tab_district:
    st.write("Number of Schools in the district: ",dataframe['School Name'].nunique())
    dis_df=dataframe[['School Name', 'Block', 'District','Predicted pass percentage (%)']]
    st.write(dis_df.set_index("School Name"))
    st.write(" ")
    st.write("The average predicted pass percentage for the district: ",dis_df['Predicted pass percentage (%)'].mean(),"%")

    bar_ch_2=px.bar(dis_df[['School Name','Predicted pass percentage (%)']].set_index("School Name"),\
                  y='Predicted pass percentage (%)')

    st.plotly_chart(bar_ch_2,use_container_width=True)
    



#%%

# Displaying Disclaimer & Header

st.header(" ")
from PIL import Image
image = Image.open('deepspatial.jpg')
image_1=image.resize((180,30))
st.image(image_1)



