import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ----------------------------------
# Layout halaman
# ----------------------------------
st.set_page_config(page_title="Prediksi NO₂", layout="wide")
st.title("🌫️ Analisis Konsentrasi NO₂ per Hari")
st.markdown("""
Prediksi konsentrasi NO₂ berdasarkan data historis menggunakan model **Random Forest**.  
Pilih tanggal mulai dan jumlah hari prediksi, lalu tekan tombol **🏍️ Jalankan Prediksi**.
""", unsafe_allow_html=True)

# ----------------------------------
# Load Dataset asli
# ----------------------------------
try:
    data = pd.read_csv("no2_awal_perjam.csv", parse_dates=['Waktu'])
except Exception as e:
    st.error(f"Error membaca CSV: {e}")
    st.stop()

# Rename kolom agar konsisten
data.rename(columns={'Waktu': 'tanggal', 'Konsentrasi_NO2 (mol/m²)': 'no2'}, inplace=True)

# ----------------------------------
# Resample per hari (jika data per jam)
# ----------------------------------
data = data.set_index('tanggal').resample('D').mean().reset_index()

# Filter data 1 Agustus - 1 Oktober
start_filter = pd.to_datetime('2025-08-01')
end_filter = pd.to_datetime('2025-10-01')
data = data[(data['tanggal'] >= start_filter) & (data['tanggal'] <= end_filter)].copy()

# ----------------------------------
# Preprocessing untuk lag
# ----------------------------------
data['no2_lag1'] = data['no2'].shift(1)
data['no2_lag2'] = data['no2'].shift(2)
data['no2_lag3'] = data['no2'].shift(3)
data = data.dropna()

X = data[['no2_lag1', 'no2_lag2', 'no2_lag3']]
y = data['no2']

train_size = int(len(data) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ----------------------------------
# Model Training
# ----------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------------
# Input tanggal & jumlah hari prediksi
# ----------------------------------
st.subheader("🗓️ Pilih Periode Prediksi")
min_pred_date = pd.to_datetime('2025-08-01')
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Tanggal Mulai Prediksi", min_value=min_pred_date, value=min_pred_date)
with col2:
    n_days = st.number_input("Jumlah Hari Prediksi", min_value=1, max_value=30, value=7)

# ----------------------------------
# Tombol Jalankan Prediksi Hijau
# ----------------------------------
run_pred = st.button("🏍️ Jalankan Prediksi")

if run_pred:
    last_data = X.iloc[-1:].copy()
    mean_no2 = data['no2'].mean()
    std_no2 = data['no2'].std()

    predictions = []
    for _ in range(n_days):
        pred = model.predict(last_data)[0]
        # Tentukan kategori berdasarkan mean & std
        if pred < mean_no2:
            kategori = "Baik"
        elif pred < mean_no2 + std_no2:
            kategori = "Sedang"
        else:
            kategori = "Buruk"
        predictions.append((pred, kategori))
        # Update lag per kolom agar konsisten
        last_data['no2_lag3'] = last_data['no2_lag2']
        last_data['no2_lag2'] = last_data['no2_lag1']
        last_data['no2_lag1'] = pred

    # Buat DataFrame hasil prediksi
    future_dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    forecast_df = pd.DataFrame(predictions, columns=['Prediksi_NO2', 'Kategori'])
    forecast_df['Tanggal'] = future_dates
    forecast_df = forecast_df[['Tanggal', 'Prediksi_NO2', 'Kategori']]
    forecast_df['Tanggal'] = forecast_df['Tanggal'].dt.date  # format tanpa jam

    # ----------------------------------
    # Tampilkan tabel hasil prediksi
    # ----------------------------------
    st.subheader("📅 Hasil Prediksi Konsentrasi NO₂")
    st.dataframe(forecast_df)

    # ----------------------------------
    # Grafik hasil prediksi
    # ----------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data['tanggal'], data['no2'], label="Data Aktual", marker='o')
    ax.plot(forecast_df['Tanggal'], forecast_df['Prediksi_NO2'], label="Prediksi", marker='x', color='red')
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Konsentrasi NO₂ (mol/m²)")
    ax.legend()
    ax.set_title("Prediksi Konsentrasi NO₂ per Hari")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ----------------------------------
    # Tombol Unduh CSV Hijau
    # ----------------------------------
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.markdown(
        f"""
        <a href="data:file/csv;base64,{csv.decode().encode('base64').decode()}" download="prediksi_no2.csv">
            <button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px;">
                💾 Unduh Hasil Prediksi CSV
            </button>
        </a>
        """, unsafe_allow_html=True
    )
