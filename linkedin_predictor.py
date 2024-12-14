import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Set page title and description
st.title('LinkedIn User Predictor')
st.write('Enter characteristics to predict LinkedIn usage probability')

# Load and prepare data
@st.cache_data
def load_data():
    ss = pd.read_csv('social_media_usage.csv')
    
    # Create binary LinkedIn usage indicator
    def clean_sm(x):
        return np.where(x == 1, 1, 0)
    
    ss['sm_li'] = clean_sm(ss['web1h'])
    
    # Handle missing values
    ss.loc[ss['income'] > 9, 'income'] = np.nan
    ss.loc[ss['educ2'] > 8, 'educ2'] = np.nan
    ss.loc[ss['age'] > 98, 'age'] = np.nan
    
    # Select and rename columns
    ss = ss[['sm_li', 'income', 'educ2', 'parent', 'marital', 'gender', 'age']]
    ss = ss.rename(columns={
        'educ2': 'education',
        'marital': 'married',
        'gender': 'female'
    })
    
    # Drop missing values
    ss = ss.dropna()
    return ss

# Train model
def train_model(data):
    X = data.drop('sm_li', axis=1)
    y = data['sm_li']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LogisticRegression(class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    return lr

# User input features
income = st.slider('Income Level', 1, 9, 5, 
                  help='1 (lowest) to 9 (highest)')
education = st.slider('Education Level', 1, 8, 4, 
                     help='1 (lowest) to 8 (highest)')
parent = st.selectbox('Parent Status', ['Not a Parent', 'Parent'], index=0)
married = st.selectbox('Marital Status', ['Not Married', 'Married'], index=0)
female = st.selectbox('Gender', ['Male', 'Female'], index=0)
age = st.number_input('Age', min_value=18, max_value=98, value=30)

# Convert categorical inputs to numeric
parent = 1 if parent == 'Parent' else 0
married = 1 if married == 'Married' else 0
female = 1 if female == 'Female' else 0

# Make prediction when button is clicked
if st.button('Predict LinkedIn Usage'):
    # Load data and train model
    data = load_data()
    model = train_model(data)
    
    # Prepare input data
    input_data = pd.DataFrame({
        'income': [income],
        'education': [education],
        'parent': [parent],
        'married': [married],
        'female': [female],
        'age': [age]
    })
    
    # Get prediction and probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    # Display results with formatting
    st.write('---')
    st.write('### Results')
    if prediction == 1:
        st.success('Prediction: LinkedIn User')
    else:
        st.error('Prediction: Non-LinkedIn User')
    
    st.write(f'Probability of being a LinkedIn user: **{probability:.1%}**')
    
    # Add interpretation
    st.write('### Interpretation')
    if probability > 0.75:
        st.write('This profile shows very strong indicators of LinkedIn usage.')
    elif probability > 0.5:
        st.write('This profile shows moderate likelihood of LinkedIn usage.')
    else:
        st.write('This profile shows lower likelihood of LinkedIn usage.')

# Add footer with information
st.write('---')
st.write('Model based on social media usage patterns and demographic characteristics.')
