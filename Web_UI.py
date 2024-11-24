# Creating Web UI for Mini-Hack-HR-Analystics


# Load the necessary libaraires
# If streamlit is not installed install streamlit
# pip install streamlit
import streamlit as st
import numpy as np
import pandas as pd
import joblib

#Title 
st.title('Promotion prediction App')


#Reading the train.csv 
df=pd.read_csv('train.csv')

# create input elements

#categorical inputs
department = st.selectbox("Department",pd.unique(df['department']))
region = st.selectbox("Region",pd.unique(df['region']))
education = st.selectbox("Education",pd.unique(df['education']))
gender = st.selectbox("Gender",pd.unique(df['gender']))
recruitment_channel = st.selectbox("Recruitment channel",pd.unique(df['recruitment_channel']))

#non-categorical inputs
no_of_trainings = st.selectbox("No of Trainings",pd.unique(df['no_of_trainings']))
age = st.number_input("Age",min_value=16,max_value=70,step=1)
previous_year_rating = st.selectbox("Previous Year Rating",pd.unique(df['previous_year_rating']))
length_of_service = st.number_input("Length of Service",min_value=0,max_value=45,step=1)
KPIs_met_80 = st.selectbox("KPIs_met >80%",pd.unique(df['KPIs_met >80%']))
awards_won =  st.selectbox("Awards won",pd.unique(df['awards_won?']))
avg_training_score =  st.number_input("Avg Training Score",min_value=0,max_value=100,step=1)

#map the user inputs to respective columns of the data format
inputs = {
    'department':department, 
    'region':region, 
    'education':education, 
    'gender':gender,
    'recruitment_channel':recruitment_channel, 
    'no_of_trainings':no_of_trainings, 
    'age':age, 
    'previous_year_rating':previous_year_rating,
    'length_of_service':length_of_service, 
    'KPIs_met >80%':KPIs_met_80, 
    'awards_won?':awards_won,
    'avg_training_score':avg_training_score
}


# loading ML-Model from the pickel-file
model = joblib.load('promote_pipeline_model.pkl')

# Action for submit button
if st.button ('Predict'):
    X_input = pd.DataFrame(inputs,index=[0])
    prediction = model.predict(X_input)
    if prediction == 0:
        val = 'NO'
        st.write(f"Is Employee eligible for Promotion **{val}**")
    else:
        val = 'YES'
        st.write(f"Is Employee eligible for Promotion **{val}**")
        













'''
Columns:
'employee_id', 'department', 'region', 'education', 'gender',
       'recruitment_channel', 'no_of_trainings', 'age', 'previous_year_rating',
       'length_of_service', 'KPIs_met >80%', 'awards_won?',
       'avg_training_score'

Streamlit run command:
streamlit run Web_UI.py
'''