## Movie Recommendation and Success Prediction System

A web-based machine learning project that recommends movies based on content similarity and predicts movie success. Built with Python, scikit-learn, Pandas, and Streamlit, and deployed on Streamlit Cloud for easy access.

## Features

📌 Movie recommendation engine based on cosine similarity of movie features.

📌 Success prediction model using machine learning.

📌 Clean, interactive Streamlit web app.

📌 Uses preprocessed .pkl files for fast loading.

📌 Deployed live via Streamlit Cloud.

## Tech Stack

Frontend & Deployment: Streamlit

Machine Learning: scikit-learn, numpy, pandas

Visualization: matplotlib, seaborn

## Project Structure

movie-recommendation-system/
│── app.py                # Streamlit web app
│── main.py               # ML pipeline / training code
│── movies.pkl            # Preprocessed movies dataset
│── movies_dict.pkl       # Dictionary of movies
│── similarity.pkl        # Similarity matrix
│── requirements.txt      # Dependencies

│── screenshots/ msr2, msr3, msr4, msr5

# Project screenshots
│── msr2, msr3, msr4, msr5           # Project documentation

## How to Run Locally

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py

## This project demonstrates:

> Data preprocessing & feature engineering

> Machine learning model development

> Web app deployment (Streamlit Cloud)

> Working with large real-world datasets
