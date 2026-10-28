import os
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Intelligence",
    page_icon="🎬",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MOVIES_DICT_PATH = BASE_DIR / "movies_dict.pkl"
NEW_DF_PATH = BASE_DIR / "movies.pkl"
SIMILARITY_PATH = BASE_DIR / "similarity.pkl"
TMDB_CSV_PATH = BASE_DIR / "contents" / "tmdb_5000_movies.csv"

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
session = requests.Session()
retry_strategy = Retry(
    total=3,
    connect=3,
    read=2,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def load_pickle(path):
    """Load a pickle file safely."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path.name}")

    with open(path, "rb") as file:
        return pickle.load(file)


def normalize_id_column(data):
    """Support datasets using either id or movie_id."""
    data = data.copy()

    if "id" not in data.columns and "movie_id" in data.columns:
        data = data.rename(columns={"movie_id": "id"})

    return data


def clean_tags(series):
    """Convert missing tag values to empty strings."""
    return series.fillna("").astype(str)



@st.cache_data(ttl=3600, show_spinner=False)
def fetch_poster(movie_id):
    """Fetch a movie poster from TMDB."""
    if not TMDB_API_KEY:
        return None

    try:
        response = session.get(
        f"https://api.themoviedb.org/3/movie/{int(movie_id)}",
        params={
            "api_key": TMDB_API_KEY,
            "language": "en-US"
        },
        timeout=20
        )
        response.raise_for_status()

        poster_path = response.json().get("poster_path")

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except (requests.RequestException, ValueError, TypeError) as error:
        print(f"Poster request failed for movie {movie_id}: {error}")

    return None



# --------------------------------------------------
# 3. LOAD THE SAVED DATA
# --------------------------------------------------

@st.cache_data(show_spinner="Loading movie data...")
def load_data():
    movies_data = load_pickle(MOVIES_DICT_PATH)
    similarity_data = load_pickle(SIMILARITY_PATH)
    processed_data = load_pickle(NEW_DF_PATH)

    # movies_dict.pkl may contain a dictionary of columns.
    movies = (
        movies_data.copy()
        if isinstance(movies_data, pd.DataFrame)
        else pd.DataFrame(movies_data)
    )

    new_df = (
        processed_data.copy()
        if isinstance(processed_data, pd.DataFrame)
        else pd.DataFrame(processed_data)
    )

    movies = normalize_id_column(movies)
    new_df = normalize_id_column(new_df)

    required_movie_columns = {"id", "title"}
    required_processed_columns = {"id", "title", "tags"}

    if not required_movie_columns.issubset(movies.columns):
        raise ValueError(
            "movies_dict.pkl must contain id (or movie_id) and title. "
            f"Found: {movies.columns.tolist()}"
        )

    if not required_processed_columns.issubset(new_df.columns):
        raise ValueError(
            "movies.pkl must contain id (or movie_id), title and tags. "
            f"Found: {new_df.columns.tolist()}"
        )

    similarity = np.asarray(similarity_data)

    if similarity.shape != (len(movies), len(movies)):
        raise ValueError(
            "The similarity matrix does not match movies_dict.pkl. "
            f"Movies: {len(movies)}, similarity shape: {similarity.shape}. "
            "Regenerate the movie data and similarity.pkl together, "
            "in the same order."
        )

    # Avoid missing titles and invalid movie IDs.
    movies = movies.dropna(subset=["id", "title"]).reset_index(drop=True)
    new_df = new_df.dropna(subset=["id", "title"]).copy()
    new_df["tags"] = clean_tags(new_df["tags"])
    new_df = new_df.reset_index(drop=True)

    # Dropping rows from movies would invalidate similarity indexes.
    # Check the matrix against the original movie order before using it.
    return movies, new_df, similarity


# --------------------------------------------------
# 4. LOAD TMDB RATINGS, BUDGET AND REVENUE
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def load_tmdb_data():
    if not TMDB_CSV_PATH.exists():
        raise FileNotFoundError(
            "Cannot find contents/tmdb_5000_movies.csv. "
            "Check your project folder."
        )

    tmdb = pd.read_csv(TMDB_CSV_PATH)

    required = {"id", "title", "budget", "revenue", "vote_average"}
    missing = required - set(tmdb.columns)

    if missing:
        raise ValueError(f"TMDB CSV is missing columns: {sorted(missing)}")

    tmdb = tmdb[[
        "id", "title", "budget", "revenue", "vote_average"
    ]].copy()

    for column in ["budget", "revenue", "vote_average"]:
        tmdb[column] = pd.to_numeric(tmdb[column], errors="coerce")

    tmdb[["budget", "revenue", "vote_average"]] = (
        tmdb[["budget", "revenue", "vote_average"]].fillna(0)
    )

    # Label 1: revenue exceeded production budget.
    # Label 0: revenue did not exceed production budget,
    # or the production budget was unavailable/zero.
    tmdb["label"] = (
        (tmdb["budget"] > 0) &
        (tmdb["revenue"] > tmdb["budget"])
    ).astype(int)

    return tmdb


# --------------------------------------------------
# 5. TRAIN THE HIT/FLOP MODEL
# --------------------------------------------------

@st.cache_resource(show_spinner="Training the movie success model...")
def train_model(tags_tuple, labels_tuple):
    tags = pd.Series(tags_tuple, dtype="string").fillna("").tolist()
    labels = np.asarray(labels_tuple, dtype=int)

    if len(tags) != len(labels):
        raise ValueError("Tags and labels have different lengths.")

    if len(set(labels.tolist())) < 2:
        raise ValueError(
            "The dataset must contain both hit and flop examples."
        )

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    # Split raw text before fitting TF-IDF to avoid data leakage.
    indices = np.arange(len(tags))

    class_counts = np.bincount(labels, minlength=2)
    can_stratify = class_counts.min() >= 2

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=labels if can_stratify else None
    )

    train_text = [tags[i] for i in train_idx]
    test_text = [tags[i] for i in test_idx]

    X_train = tfidf.fit_transform(train_text)
    X_test = tfidf.transform(test_text)

    y_train = labels[train_idx]
    y_test = labels[test_idx]

    if len(set(y_train.tolist())) < 2:
        raise ValueError(
            "The training split contains only one class. "
            "Check the hit/flop labels."
        )

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    clf.fit(X_train, y_train)

    test_predictions = clf.predict(X_test)

    evaluation = {
        "accuracy": accuracy_score(y_test, test_predictions),
        "report": classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            target_names=["Flop", "Hit"],
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, test_predictions, labels=[0, 1]
        )
    }

    # Refit TF-IDF and the model on all labeled examples for deployment.
    X_all = tfidf.fit_transform(tags)

    final_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    final_model.fit(X_all, labels)

    return tfidf, final_model, evaluation


# --------------------------------------------------
# 6. INITIALIZE THE APP
# --------------------------------------------------

try:
    movies, new_df, similarity = load_data()
    tmdb = load_tmdb_data()

except (FileNotFoundError, ValueError, OSError, pickle.UnpicklingError) as error:
    st.error(f"Could not load project data: {error}")
    st.info(
        "Check that your pickle files and TMDB CSV exist, "
        "and that the pickle files were generated together."
    )
    st.stop()

# Merge using movie IDs, not titles.
df = new_df.merge(
    tmdb[["id", "budget", "revenue", "label", "vote_average"]],
    on="id",
    how="inner",
    validate="many_to_one"
)

df["tags"] = clean_tags(df["tags"])
df = df.reset_index(drop=True)

if df.empty:
    st.error(
        "No matching movie IDs were found between movies.pkl "
        "and the TMDB CSV. Regenerate movies.pkl from the same dataset."
    )
    st.stop()

try:
    tfidf, model, evaluation = train_model(
        tuple(df["tags"].tolist()),
        tuple(df["label"].astype(int).tolist())
    )
except ValueError as error:
    st.error(f"Could not train the prediction model: {error}")
    st.stop()

# --------------------------------------------------
# 7. MOVIE RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend(movie_title):
    matching_indices = np.flatnonzero(
        movies["title"].astype(str).to_numpy() == movie_title
    )

    if len(matching_indices) == 0:
        return []

    index = int(matching_indices[0])

    scores = similarity[index]

    # Sort similarity scores descending; exclude the selected movie.
    ranked_indices = np.argsort(scores)[::-1]
    ranked_indices = [
        int(i) for i in ranked_indices
        if i != index
    ][:5]

    results = []

    for movie_index in ranked_indices:
        movie = movies.iloc[movie_index]

        results.append({
            "title": str(movie["title"]),
            "id": movie["id"]
        })

    return results


# --------------------------------------------------
# 8. CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
.main-title {
    font-size: 44px;
    font-weight: 800;
    color: #FF4B4B;
    text-align: center;
    padding: 12px 0;
    letter-spacing: 1px;
}
.block-container {
    padding-top: 1.2rem;
}
.poster-title {
    text-align: center;
    font-weight: 600;
    font-size: 15px;
    margin-top: 8px;
    overflow-wrap: anywhere;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 9. APP HEADER
# --------------------------------------------------

st.markdown(
    "<div class='main-title'>🎬 Movie Intelligence System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "##### Discover similar movies and predict revenue-based success."
)

if not TMDB_API_KEY:
    st.warning(
        "TMDB_API_KEY is not configured. Recommendations will work, "
        "but poster images may be unavailable."
    )

tab1, tab2 = st.tabs([
    "🎞️ Movie Recommender",
    "📈 Hit / Flop Predictor"
])

# --------------------------------------------------
# 10. TAB ONE: MOVIE RECOMMENDER
# --------------------------------------------------

with tab1:
    st.subheader("🔍 Discover Movies Like Your Favorite")

    movie_titles = sorted(
        movies["title"].dropna().astype(str).unique().tolist()
    )

    selected_movie = st.selectbox(
        "🎥 Choose a movie:",
        movie_titles,
        key="recommend_movie"
    )

    if st.button("🎯 Get Recommendations", key="recommend_button"):
        recommendations = recommend(selected_movie)

        if not recommendations:
            st.warning("No recommendations found for this movie.")
        else:
            st.markdown("### 🍿 Top Recommended Movies")

            columns = st.columns(len(recommendations))

            for column, item in zip(columns, recommendations):
                with column:
                    poster = fetch_poster(item["id"])

                    if poster:
                        st.image(poster, width="stretch")
                    else:
                        st.info("Poster unavailable")

                    st.markdown(
                        f"<div class='poster-title'>{item['title']}</div>",
                        unsafe_allow_html=True
                    )

# --------------------------------------------------
# 11. TAB TWO: HIT / FLOP PREDICTOR
# --------------------------------------------------

with tab2:
    st.subheader("🎯 Predict Revenue-Based Movie Success")

    st.caption(
        "Hit = revenue exceeded the recorded production budget. "
        "This is not a guarantee of overall profitability."
    )

    selected_title = st.selectbox(
        "📽️ Select a movie to classify:",
        sorted(df["title"].dropna().astype(str).unique().tolist()),
        key="predict_movie"
    )

    selected_rows = df[df["title"].astype(str) == selected_title]

    if not selected_rows.empty:
        movie_row = selected_rows.iloc[0]
        tag_text = str(movie_row["tags"])

        vector = tfidf.transform([tag_text])
        prediction = int(model.predict(vector)[0])
        probabilities = model.predict_proba(vector)[0]

        predicted_class_index = list(model.classes_).index(prediction)
        confidence = float(probabilities[predicted_class_index])

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 🔎 Prediction Result")

            if prediction == 1:
                st.success("🎯 Predicted outcome: HIT")
            else:
                st.error("🎯 Predicted outcome: FLOP")

            st.metric("Model confidence", f"{confidence * 100:.2f}%")
            st.metric("⭐ TMDB rating", f"{movie_row['vote_average']:.1f}/10")
            st.metric("💵 Production budget", f"${movie_row['budget']:,.0f}")
            st.metric("🎟️ Recorded revenue", f"${movie_row['revenue']:,.0f}")

        with col2:
            st.markdown("### 🧠 Text Features Used")

            st.text_area(
                "Movie tags",
                value=tag_text,
                height=220,
                disabled=True
            )

        st.markdown("---")
        st.markdown("### 📊 Model Evaluation")

        st.metric(
            "Held-out test accuracy",
            f"{evaluation['accuracy'] * 100:.2f}%"
        )

        with st.expander("View classification report"):
            st.text(evaluation["report"])

        with st.expander("View confusion matrix"):
            fig, ax = plt.subplots(figsize=(5, 4))

            sns.heatmap(
                evaluation["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="Oranges",
                xticklabels=["Flop", "Hit"],
                yticklabels=["Flop", "Hit"],
                ax=ax
            )

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Hit vs Flop")
            st.pyplot(fig)
            plt.close(fig)

st.markdown("---")
st.caption(
    "Movie Intelligence System | Recommendations use the saved similarity "
    "matrix; success predictions use TF-IDF text features and a "
    "Random Forest classifier."
)














# import streamlit as st
# import pandas as pd
# import pickle
# import requests
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier

# # --- Config (must be first!) ---
# st.set_page_config(page_title="🎬 Movie Intelligence", layout="wide")

# # --- Load Data ---
# movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
# movies = pd.DataFrame(movies_dict)
# similarity = pickle.load(open('similarity.pkl', 'rb'))
# new_df = pickle.load(open('movies.pkl', 'rb'))
# tmdb = pd.read_csv('contents/tmdb_5000_movies.csv')[['id', 'title', 'budget', 'revenue', 'vote_average']]

# # --- Labeling Based on Profit ---
# tmdb['label'] = tmdb.apply(
#     lambda row: 1 if row['revenue'] > row['budget'] and row['budget'] > 0 else 0, axis=1
# )

# # --- Merge DataFrames ---
# df = new_df.merge(tmdb[['id', 'budget', 'revenue', 'label']], on='id')

# # --- TF-IDF Processing ---
# tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
# X = tfidf.fit_transform(df['tags']).toarray()
# y = df['label']

# # --- Model Training ---
# @st.cache_resource
# def get_model():
#     clf = RandomForestClassifier()
#     clf.fit(X, y)
#     return clf

# model = get_model()

# # --- Helper Functions ---
# def fetch_poster(movie_id):
#     url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=834f4f7604f1e118e0e25466da9ae622"
#     data = requests.get(url).json()
#     path = data.get("poster_path")
#     return f"https://image.tmdb.org/t/p/w500{path}" if path else "https://via.placeholder.com/500x750?text=No+Image"

# def recommend(movie):
#     index = movies[movies['title'] == movie].index[0]
#     distances = similarity[index]
#     movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
#     names, posters = [], []
#     for i in movie_list:
#         mid = movies.iloc[i[0]].id
#         names.append(movies.iloc[i[0]].title)
#         posters.append(fetch_poster(mid))
#     return names, posters

# # --- Custom Styling ---
# st.markdown("""
#     <style>
#     .main-title {
#         font-size: 48px;
#         font-weight: bold;
#         color: #FF4B4B;
#         text-align: center;
#         padding: 10px;
#         letter-spacing: 1px;
#     }
#     .block-container { padding-top: 1rem; }
#     .poster-title { text-align: center; font-weight: bold; font-size: 16px; margin-top: 10px; }
#     </style>
# """, unsafe_allow_html=True)

# # --- App Title ---
# st.markdown("<div class='main-title'>🎬 Movie Intelligence System</div>", unsafe_allow_html=True)
# st.markdown("##### Discover Similar Movies & Predict Success Based on Real-world Revenue!")

# tab1, tab2 = st.tabs(["🎞️ Recommender", "📈 Success Predictor"])

# # --- Tab 1: Recommender ---
# # --- Tab 1: Recommender ---
# with tab1:
#     st.subheader("🔍 Discover Movies Like Your Favorite")
#     movie_name = st.selectbox("🎥 Choose a movie:", movies['title'].values)

#     if st.button("🎯 Get Recommendations"):
#         names, posters = recommend(movie_name)
#         st.markdown("### 🍿 Top 5 Recommended Movies")
#         cols = st.columns(5)
#         for i in range(5):
#             with cols[i]:
#                 st.image(posters[i], use_container_width=True)  # ✅ Fixed warning here
#                 st.markdown(f"<div class='poster-title'>{names[i]}</div>", unsafe_allow_html=True)


# # --- Tab 2: Success Classifier ---
# with tab2:
#     st.subheader("🎯 Predict if the Movie Was a Hit or Flop")

#     selected = st.selectbox("📽️ Select a movie to classify:", df['title'].values)
#     tag_text = df[df['title'] == selected]['tags'].values[0]
#     vector = tfidf.transform([tag_text])
#     prediction = model.predict(vector)[0]
#     probability = model.predict_proba(vector)[0][prediction]

#     movie_id = df[df['title'] == selected]['id'].values[0]
#     movie_info = tmdb[tmdb['id'] == movie_id]

#     col1, col2 = st.columns([1, 2])
#     with col1:
#         st.markdown("### 🔎 Prediction Result")
#         st.markdown(f"**🎯 Outcome:** {'✅ Hit' if prediction == 1 else '❌ Flop'}")
#         st.metric("📊 Confidence", f"{probability*100:.2f}%")
#         st.metric("💵 Budget", f"${int(movie_info['budget'].values[0]):,}")
#         st.metric("🎟️ Revenue", f"${int(movie_info['revenue'].values[0]):,}")

#     with col2:
#         st.markdown("### 🧠 Features Used")
#         st.markdown(
#             f"<div style='padding:10px; background-color:#f9f9f9; border-radius:6px; "
#             f"font-family:monospace; font-size:14px; color:#444;'>{tag_text}</div>",
#             unsafe_allow_html=True
#         )

#     st.markdown("---")
#     st.info("✅ This prediction is based on TF-IDF of plot, cast, and genre, and trained using actual profit-based success.")

