import streamlit as st
import librosa
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("model.pkl")
scaler_X = joblib.load("scaler_X.pkl")

st.title("🎙️ Identifikasi Suara 'Buka' dan 'Tutup'")
st.write("Unggah file audio (.wav atau .mp3) untuk dikenali oleh model.")

uploaded_file = st.file_uploader("Pilih file audio", type=["wav", "mp3"])

def extract_features(y, sr):
    return np.array([
        np.mean(y),
        np.std(y),
        np.min(y),
        np.max(y),
        np.median(y),
        np.mean(librosa.feature.rms(y=y)),
        np.mean(librosa.feature.zero_crossing_rate(y))
    ])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    y, sr = librosa.load(uploaded_file, sr=None)
    features = extract_features(y, sr).reshape(1, -1)
    features_scaled = scaler_X.transform(features)

    prediction = model.predict(features_scaled)[0]

    st.success(f"Hasil Prediksi: **{prediction.upper()}**")
else:
    st.info("Silakan unggah file audio untuk memulai prediksi.")
