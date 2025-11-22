import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# ================================
# Page Config & Style
# ================================
st.set_page_config(
    page_title="Spotify Recommender",
    page_icon="headphones",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Spotify Dark Theme
st.markdown("""
<style>
    .main {background-color: #121212; color: white;}
    .stApp {background-color: #121212;}
    h1, h2, h3 {color: #1DB954;}
    .song-card {
        background: #181818;
        padding: 16px;
        border-radius: 12px;
        margin: 12px 0;
        border-left: 5px solid #1DB954;
        transition: all 0.3s;
    }
    .song-card:hover {background: #282828; transform: translateY(-3px);}
    .score {background:#1DB954; color:black; padding:4px 12px; border-radius:20px; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# ================================
# Load Pre-trained Models
# ================================
@st.cache_resource
def load_system():
    df = pd.read_pickle(r"C:\Users\Lenovo\OneDrive\سطح المكتب\spotify-recommender\pretrained\df.pkl")
    with open(r"C:\Users\Lenovo\OneDrive\سطح المكتب\spotify-recommender\pretrained\similarity_matrix.pkl", "rb") as f:
        sim_matrix = pickle.load(f)
    return df, sim_matrix

df, similarity_matrix = load_system()
st.success("System loaded successfully!")

# ================================
# UI
# ================================
st.markdown("<h1 style='text-align:center;'>Spotify Song Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#b3b3b3;'>Type a song name and get AI-powered recommendations</p>", unsafe_allow_html=True)

search = st.text_input("", placeholder="Examples: Shape of You, Believer, Perfect, Blinding Lights...", label_visibility="collapsed")

if search:
    matches = df[df['name'].str.contains(search, case=False, na=False, regex=False)]
    
    if matches.empty:
        st.warning(f"No song found with the name: **{search}**")
        st.info("Try: Shape of You, Perfect, Believer, Despacito, Someone You Loved")
    else:
        # Pick the most popular match
        song = matches.loc[matches['popularity'].idxmax()]
        idx = song.name

        st.markdown(f"""
        <div style='text-align:center; padding:25px; background:#181818; border-radius:15px; margin:20px 0;'>
            <h2>Selected Song:</h2>
            <h3>{song['name']}</h3>
            <p style='color:#1DB954; font-size:1.3em;'>{song['artists']}</p>
            <p><small>{song['genre']} • Popularity: {int(song['popularity'])}</small></p>
        </div>
        """, unsafe_allow_html=True)

        # Top 10 recommendations
        scores = similarity_matrix[idx]
        top_idx = np.argsort(scores)[::-1][1:10+1]

        st.markdown("<h2 style='text-align:center; color:#1DB954;'>Recommended for You</h2>", unsafe_allow_html=True)

        cols = st.columns([1, 1])
        for i, rec_idx in enumerate(top_idx):
            rec = df.iloc[rec_idx]
            match_score = int(scores[rec_idx] * 100)
            with cols[i % 2]:
                st.markdown(f"""
                <div class="song-card">
                    <strong>{i+1}. {rec['name']}</strong><br>
                    <small style="color:#b3b3b3;">{rec['artists']}</small> • 
                    <small style="color:#666;">{rec['genre']}</small>
                    <div style="float:right;">
                        <span class="score">Match {match_score}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

st.markdown("<br><hr><p style='text-align:center; color:#666;'>Powered by Deep Autoencoder + TF-IDF • Made by Omar</p>", unsafe_allow_html=True)
