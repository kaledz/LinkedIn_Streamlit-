# 📊 LinkedIn Streamlit Predictor

This project is a lightweight **Streamlit web application** that uses a dataset on social media usage to **predict LinkedIn user likelihood** based on user attributes.

---

## 🚀 Features

- 🧠 Machine learning predictor built with Python
- 🖥️ Interactive front-end using Streamlit
- 📂 Loads and processes `social_media_usage.csv`
- 🔍 User input fields for prediction
- 📈 Real-time display of prediction results (likely LinkedIn user or not)

---

## 📂 Files in This Repo

| File / Folder           | Description |
|-------------------------|-------------|
| `linkedin_predictor.py` | Main Streamlit app |
| `social_media_usage.csv`| Cleaned dataset used for model prediction |
| `requirements.txt`      | Dependencies for running the app |
| `.devcontainer/`        | (Optional) VSCode Dev Container setup for development |

---

## ▶️ How to Run

1. **Clone the repo**
```bash
git clone https://github.com/kaledz/LinkedIn_Streamlit.git
cd LinkedIn_Streamlit
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app

bash
Copy
Edit
streamlit run linkedin_predictor.py
Open in your browser at http://localhost:8501

📊 Dataset
The dataset includes anonymized survey data on various social media platforms.
It contains features such as:

Age

Education level

Income

Parental status

And whether or not the respondent uses LinkedIn

Used to train a basic classification model for predicting usage likelihood.

🛠 Tech Stack
Python

Pandas

Streamlit

Scikit-learn (or similar for modeling)

✅ To-Do / Improvements
Add model training and evaluation notebook

Deploy to Streamlit Cloud or HuggingFace Spaces

Add visuals or charts of model accuracy

Add README badge (e.g., Streamlit deployed link)

🧑‍💻 Author
Created by Kris Lederer
Feel free to fork, contribute, or reach out!

📜 License
Open-source under the MIT License.
