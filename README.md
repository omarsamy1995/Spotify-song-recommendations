# 🎵 Spotify Song Recommender

An AI-powered music recommendation system built using **TF-IDF**, **SVD dimensionality reduction**, and a **Deep Autoencoder**.  
The Streamlit interface provides fast, accurate song recommendations using a precomputed similarity matrix.

---

## 🚀 Features

- Search for any song by name  
- Automatically selects the best matching track  
- Provides **Top 10 AI-generated recommendations**  
- Modern Spotify-style dark UI  
- Instant inference using a precomputed similarity matrix  

**Powered by:**

- TF-IDF (text features)  
- Numerical audio features  
- TruncatedSVD  
- Deep Autoencoder  
- Cosine similarity  

---

## 🧠 Model Pipeline

1. Load and prepare dataset  
2. Extract TF-IDF and numerical features  
3. Reduce dimensionality with TruncatedSVD  
4. Train a deep Autoencoder  
5. Extract latent embeddings  
6. Compute cosine similarity matrix  
7. Save the following in `app/pretrained/`:
   - `df.pkl`
   - `similarity_matrix.pkl`
   - `autoencoder_encoder.keras`  

> These files are used directly by Streamlit for fast recommendations.

---

## 🏃 Running the App

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt


## Run Streamlit app

streamlit run app/app.py
```
## 📦 Required Model Files
Make sure the following files exist in the app/pretrained/ directory:

File	Description
df.pkl	Clean processed dataset
similarity_matrix.pkl	Cosine similarity matrix
autoencoder_encoder.keras	Trained encoder model
## 🛠 Technologies Used

Python

Streamlit

NumPy / Pandas

Scikit-Learn

TensorFlow / Keras

Cosine similarity
## 👤 Author

Developed by Omar.

