# 🎬 Movie Recommendation and Success Prediction System

A machine learning web application that recommends similar movies using content-based filtering and predicts movie success using machine learning. Built with Python, Scikit-learn, Pandas, and Streamlit, and deployed on Streamlit Community Cloud.

## 🚀 Live Demo

**Try the application here:** [Movie Intelligence System](https://movie-recommendation-success-predictor-system.streamlit.app/)

## ✨ Features

- 🎥 **Movie Recommendation:** Recommends five similar movies based on content similarity.
- 📊 **Hit / Flop Prediction:** Predicts movie success using a machine learning classifier.
- 🖼️ **Movie Posters:** Displays movie posters alongside recommendations using TMDB.
- 🖥️ **Interactive Web Interface:** Simple and user-friendly Streamlit application.
- ☁️ **Cloud Deployment:** Accessible online through Streamlit Community Cloud.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Frontend & Web Framework:** Streamlit
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Recommendation System:** CountVectorizer, Cosine Similarity
- **Success Prediction:** TF-IDF, Random Forest Classifier
- **Visualization:** Matplotlib, Seaborn
- **Movie Data & Posters:** TMDB API

## 📂 Project Structure

```text
Movie-Recommendation-System/
├── app.py
├── movies.pkl
├── movies_dict.pkl
├── requirements.txt
├── contents/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── MovieRSRP.ipynb
├── README.md
├── .gitignore
└── screenshots/
    ├── msr2.png
    ├── msr3.png
    ├── msr4.png
    └── msr5.png
```

## ⚙️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/anushab-2512/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the TMDB API key (optional)

To display movie posters, configure your TMDB API key using an environment variable or Streamlit secrets. Never commit your API key to GitHub.

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

## 🎯 Key Concepts Demonstrated

- Data preprocessing and feature engineering
- Content-based movie recommendation
- Text vectorization and cosine similarity
- Machine learning model development
- Interactive web application development
- Cloud deployment using Streamlit Community Cloud
- Integration with an external API

## 👩‍💻 Author

**Anusha B.**

- GitHub: [anushab-2512](https://github.com/anushab-2512)
- Live Project: [Movie Intelligence System](https://movie-recommendation-success-predictor-system.streamlit.app/)
