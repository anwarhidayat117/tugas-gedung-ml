# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
from datetime import datetime, timedelta
from src.features import create_features

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
  page_title="Prediksi Energi Gedung - Kasus 02",
  page_icon="⚡",
  layout="wide"
)

# 2. Load Model & Metadata 
@st.cache_resource
def load_artifacts():
  model = joblib.load('model.joblib')
  with open('metadata.json', 'r') as f:
    metadata = json.load(f)
  return model, metadata
  
try:
  model, metadata = load_artifacts()
except Exception as e:
  st.error(f"Gagal memuat model/metadata: {e}")
  st.stop()

  # Header Aplikasi
  st.title("⚡ Dashboard Prediksi Konsumsi Listrik Gedung")
  st.subheader("Sistem Peramalan Beban Energi 24 Jam (Hari Esok)")
  st.markdown("---")

  # 3. Sidebar: Form Input Pengelola Gedung
  st.sidebar.header("📋 Parameter Prediksi Besok")
  tanggal_target = st.sidebar.date_input("Pilih Tanggal Prediksi", datetime.now() + timedelta(days=1))
  suhu_rata_rata = st.sidebar.slider("Estimasi Suhu Udara Rata-rata (°C)", 18.0, 35.0, 26.5, 0.5)
  okupansi_persen = st.sidebar.slider("Estimasi Okupansi Gedung (%)", 0, 100, 75, 5)
  hari_libur = st.sidebar.checkbox("Hari Libur Nasional / Tanggal Merah", value=False)

  # Button Trigger Prediksi
  if st.sidebar.button("🚀 Jalankan Prediksi 24 Jam"):
    # Buat Grid Waktu 24 Jam untuk Hari Target
    start_time = datetime.combine(tanggal_target, datetime.min.time())
    waktu_range = [start_time + timedelta(hours=i) for i in range(24)]

    # Simulasi DataFrame Input dengan Data Historis Dummy (atau muat dari data historis Anda) 
    # Dalam sistem nyata, lag\_24 dan lag\_168 diambil dari database 24 jam & 7 hari sebelumnya
    df_future = pd.DataFrame({'waktu': waktu_range})
    df_future['suhu_luar_C'] = suhu_rata_rata + np.sin(np.linspace(0, np.pi, 24)) * 3
    df_future['kelembaban_luar_pct'] = 65.0 df_future['okupansi_orang'] = (okupansi_persen / 100.0) * 500 # Kapasitas gedung 500 orang
    df_future['hari_libur'] = int(hari_libur)

    # Nilai lag historis disimulasikan untuk demo antarmuka
    df_future['konsumsi_kWh'] = 150.0 # Placeholder nilai lag

    # Jalankan Rekayasa Fitur
    df_featured = create_features(df_future)

    # Ambil kolom fitur sesuai metadata
    feature_cols = metadata['features']
    X_input = df_featured[feature_cols].fillna(150.0)

    # Prediksi
    predictions = model.predict(X_input)
    df_future['prediksi_kWh'] = np.maximum(0, predictions)
    
    total_kWh = df_future['prediksi_kWh'].sum()
    peak_kWh = df_future['prediksi_kWh'].max()
    peak_hour = df_future.loc[df_future['prediksi_kWh'].idxmax(), 'waktu'].strftime('%H:00 WIB')

    # Metric Display
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Prediksi Energi Harian", f"{total_kWh:,.2f} kWh")
    col2.metric("Beban Puncak (Peak Demand)", f"{peak_kWh:,.2f} kWh")
    col3.metric("Jam Beban Puncak", peak_hour)
    
    st.markdown("---")

    # Grafik Prediksi 24 Jam
    st.write(f"### Grafik Prediksi Konsumsi Listrik per Jam ({tanggal_target.strftime('%d %B %Y')})")
    fig = px.line(
      df_future,
      x='waktu',
      y='prediksi_kWh',
      labels={'waktu': 'Jam', 'prediksi_kWh': 'Konsumsi Listrik (kWh)'},
      title="Profil Beban Listrik 24 Jam Hari Esok",
      markers=True
    )
    fig.update_traces(line_color='#FF4B4B', line_width=3)
    st.plotly_chart(fig, use_container_width=True)

    # Tabel Detail
    with st.expander("🔍 Lihat Detail Tabel Prediksi 24 Jam"):
      st.dataframe(df_future[['waktu', 'prediksi_kWh', 'suhu_luar_C', 'okupansi_orang']])

# Info Metadata
st.markdown("---")
st.caption(f"📌 Model: {metadata['model_type']} | Evaluasi MAE Validation: {metadata['metrics']['mae_model_val']} kWh | Test MAE: {metadata['metrics']['mae_model_test']} kWh")
