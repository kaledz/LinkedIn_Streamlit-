import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and prepare data
@st.cache_data
def load_data():
    # Load your dataset here
    ss = pd.read_csv('social_media_usage.csv')
    return ss

# Train model
def train_model(ss):
    X = ss.drop('sm_li', axis=1)
    y = ss['sm_li']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LogisticRegression(class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    return lr

# Streamlit interface
st.title('LinkedIn User Predictor')
st.write('Enter user characteristics to predict LinkedIn usage')

# User inputs
income = st.slider('Income Level (1-9)', 1, 9, 5)
education = st.slider('Education Level (1-8)', 1, 8, 4)
parent = st.selectbox('Parent Status', ['Not a Parent', 'Parent'], index=0)
married = st.selectbox('Marital Status', ['Not Married', 'Married'], index=0)
female = st.selectbox('Gender', ['Male', 'Female'], index=0)
age = st.number_input('Age', min_value=18, max_value=98, value=30)

# Convert inputs
parent = 1 if parent == 'Parent' else 0
married = 1 if married == 'Married' else 0
female = 1 if female == 'Female' else 0

# Make prediction
if st.button('Predict'):
    # Load data and train model
    ss = load_data()
    model = train_model(ss)
    
    # Create input array
    input_data = pd.DataFrame([[income, education, parent, married, female, age]], 
                             columns=['income', 'education', 'parent', 'married', 'female', 'age'])
    
    # Get prediction and probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    # Display results
    st.write('Prediction:', 'LinkedIn User' if prediction == 1 else 'Non-LinkedIn User')
    st.write(f'Probability of being a LinkedIn user: {probability:.2%}')
