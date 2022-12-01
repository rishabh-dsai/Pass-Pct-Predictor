

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
    
    st.write("Impact of features on Pass Percentage:")
    base_df=pd.DataFrame([heatmap_data.iloc[5,:]],columns=heatmap_data.columns)
    base_df.drop(columns='Pass_Perce',inplace=True)
    st.write(base_df)
    feature=st.selectbox("Select a Feature to visualize its impact.",('Gen_Studen', 'x_girls', 'Boundary_w',\
                  'Per_m_Lit', 'PTR', 'x_boys','Tot_Teachers', 'OBC_Studen', 'Qualified_T', \
                  'ST_Student'))
    new_val=st.number_input("Enter the value to change to")
    new_df=base_df.copy()
    new_df[feature]=new_val
    old_pred=model.predict(base_df[['Gen_Studen', 'x_girls', 'Boundary_w', 'Per_m_Lit', 'PTR', 'x_boys',
       'Tot_Teachers', 'OBC_Studen', 'Qualified_T', 'ST_Student']])
    new_pred=model.predict(new_df[['Gen_Studen', 'x_girls', 'Boundary_w', 'Per_m_Lit', 'PTR', 'x_boys',
       'Tot_Teachers', 'OBC_Studen', 'Qualified_T', 'ST_Student']])
    base_df['Predicted Pass Percentage']=old_pred
    new_df['Predicted Pass Percentage']=new_pred
    show_df=pd.concat([base_df,new_df])
    show_df.index=['Old','New']
    line_ch_sch=px.line(x=show_df.index,y=show_df['Predicted Pass Percentage'],color=show_df.index\
                        ,markers=True,title="Impact of Features")
    st.plotly_chart(line_ch_sch,use_container_width=True)
    
    
    

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
    st.write(dataframe[dataframe['School Name']==school].set_index("School Name"))
    school_df=trend[trend['School Name']==school].set_index("School Name").drop(columns=['Block','District'])
    school_df=school_df[['PP_2017','PP_2018','PP_2019','PP_2020','PP_2021']]
    ch_df=pd.DataFrame(zip(school_df.columns,school_df.iloc[0,:]),columns=["Year","Pass Pct"])
    ch_df['Year']=ch_df['Year'].apply(lambda z:z[-4:])
    line_ch_sch=px.line(ch_df,x='Year',y='Pass Pct',markers=True,\
                        title="Trend of Pass Percentage for the School")
    st.plotly_chart(line_ch_sch,use_container_width=True)
    
with tab_block:
    
    block=st.selectbox("Please select a Block",\
                       ('Pynursla', 'Mylliem', 'Mawphlang', 'Mawknyrew', 'Mawsynram'))
    block_df=dataframe[dataframe['Block']==block]
    st.write("Number of Schools in the block: ",block_df['School Name'].nunique())
    sub_df=block_df[['School Name', 'Block', 'District','Predicted pass percentage (%)']]
    st.write(sub_df.set_index("School Name"))
    st.write(" ")
    st.write("The average predicted pass percentage for the block: ",np.round(sub_df['Predicted pass percentage (%)'].mean(),2),"%")

    bar_ch=px.bar(sub_df[['School Name','Predicted pass percentage (%)']].set_index("School Name"),\
                  y='Predicted pass percentage (%)')

    st.plotly_chart(bar_ch,use_container_width=True)
    
    block_trend_df=trend[trend['Block']==block]
    block_trend_df=block_trend_df.set_index("School Name").drop(columns=['Block','District'])
    block_trend_df=block_trend_df[['PP_2017','PP_2018','PP_2019','PP_2020','PP_2021']]
    block_trend_df.columns=[z[-4:] for z in block_trend_df.columns]
    chk=block_trend_df.T
    chk['Block Average']=[block_trend_df['2017'].mean(),block_trend_df['2018'].mean(),\
                          block_trend_df['2019'].mean(),block_trend_df['2020'].mean(),\
                          block_trend_df['2021'].mean()]
    line_ch_sch=px.line(chk,markers=True,title="Trend of Pass Percentage in the Block")
    st.plotly_chart(line_ch_sch,use_container_width=True)   


with tab_district:
    st.write("Number of Schools in the district: ",dataframe['School Name'].nunique())
    dis_df=dataframe[['School Name', 'Block', 'District','Predicted pass percentage (%)']]
    st.write(dis_df.set_index("School Name"))
    st.write(" ")
    st.write("The average predicted pass percentage for the district: ",np.round(dis_df['Predicted pass percentage (%)'].mean(),2),"%")

    bar_ch_2=px.bar(dis_df[['School Name','Predicted pass percentage (%)']].set_index("School Name"),\
                  y='Predicted pass percentage (%)')

    st.plotly_chart(bar_ch_2,use_container_width=True)
    
    district_trend_df=trend.drop(columns=['School Name','District']).groupby("Block").mean()
    district_trend_df=district_trend_df[['PP_2017','PP_2018','PP_2019','PP_2020','PP_2021']]
    district_trend_df.columns=[z[-4:] for z in district_trend_df.columns]
    dt_df=district_trend_df.T
    dt_df['District Average']=[district_trend_df['2017'].mean(),district_trend_df['2018'].mean(),\
                               district_trend_df['2019'].mean(),district_trend_df['2020'].mean(),\
                                   district_trend_df['2021'].mean()]
    line_ch_sch=px.line(dt_df,markers=True,title="Trend of Pass Percentage in the District")
    st.plotly_chart(line_ch_sch,use_container_width=True)       



#%%

# Displaying Disclaimer & Header

st.header(" ")
from PIL import Image
image = Image.open('deepspatial.jpg')
image_1=image.resize((180,30))
st.image(image_1)



