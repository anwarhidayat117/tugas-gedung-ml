import glob
import json
import os
from datetime import datetime, time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Mendukung import baik dari root folder maupun subfolder src/
try:
    from features import create_features, get_feature_columns
except ImportError:
    from src.features import create_features, get_feature_columns

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Prediksi Konsumsi Energi Gedung",
    page_icon="⚡",
    layout="wide",
)


# 2. Helper Pemuatan Model & Metadata
def extract_expected_features(model, metadata: dict | None) -> list | None:
    """Mendeteksi daftar fitur yang diharapkan oleh model secara otomatis."""
    if metadata:
        for key in ["features", "feature_cols", "feature_names", "columns"]:
            if key in metadata and isinstance(metadata[key], list):
                return metadata[key]

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "feature_name_"):
        fn = model.feature_name_
        return list(fn()) if callable(fn) else list(fn)

    if hasattr(model, "feature_name") and callable(model.feature_name):
        return list(model.feature_name())

    return None


@st.cache_resource
def load_local_artifacts():
    """Mencari file model (.joblib/.pkl) dan metadata (.json) di repository."""
    model_candidates = sorted(
        glob.glob("*.joblib")
        + glob.glob("*.pkl")
        + glob.glob("models/*.joblib")
        + glob.glob("models/*.pkl")
    )
    meta_candidates = sorted(
        glob.glob("*.json") + glob.glob("models/*.json")
    )

    model = None
    model_path = None
    for path in model_candidates:
        try:
            model = joblib.load(path)
            model_path = path
            break
        except Exception:
            continue

    metadata = None
    for path in meta_candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            break
        except Exception:
            continue

    return model, model_path, metadata


def align_features(df_feat: pd.DataFrame, expected_cols: list | None) -> pd.DataFrame:
    """Menyelaraskan kolom hasil create_features() dengan kolom yang diharapkan model."""
    if not expected_cols:
        feature_cols = get_feature_columns(df_feat)
        return df_feat[feature_cols]

    aligned = pd.DataFrame(index=df_feat.index)
    for col in expected_cols:
        if col in df_feat.columns:
            aligned[col] = df_feat[col]
        else:
            # Jika model memakai fitur lag/kustom yang belum ada saat prediksi tunggal
            aligned[col] = 0.0
    return aligned


def predict_energy(df_input: pd.DataFrame, model, metadata: dict | None) -> np.ndarray:
    """Menjalankan rekayasa fitur dan prediksi konsumsi energi (kWh)."""
    df_feat = create_features(df_input)

    if model is not None:
        expected_cols = extract_expected_features(model, metadata)
        X = align_features(df_feat, expected_cols)
        preds = model.predict(X)
        return np.clip(preds, a_min=0.0, a_max=None)

    # Estimasi fisik fallback jika file .joblib belum diunggah ke repository
    base_load = 75.0
    occupancy_load = 0.32 * df_feat["okupansi_orang"]
    cooling_load = 3.8 * np.maximum(0.0, df_feat["suhu_luar_C"] - 21.0)
    working_bonus = 22.0 * df_feat["is_working_hour"]
    holiday_discount = np.where(df_feat["hari_libur"] == 1, 0.85, 1.0)
    est = (base_load + occupancy_load + cooling_load + working_bonus) * holiday_discount
    return np.asarray(est, dtype=float)


# 3. Sidebar: Status Model & Upload Artefak Opsional
st.sidebar.header("⚙️ Konfigurasi Model")
uploaded_model = st.sidebar.file_uploader(
    "Unggah Model (.joblib / .pkl) [Opsional]",
    type=["joblib", "pkl"],
    help="Jika file model sudah ada di GitHub, model akan dimuat otomatis.",
)

if uploaded_model is not None:
    model = joblib.load(uploaded_model)
    model_path = uploaded_model.name
    metadata = None
else:
    model, model_path, metadata = load_local_artifacts()

if model is not None:
    st.sidebar.success(f"Model aktif: `{model_path}`")
else:
    st.sidebar.warning(
        "File `.joblib` belum ditemukan di folder utama. "
        "Aplikasi berjalan dalam **Mode Estimasi Simulasi** hingga Anda mengunggah file model."
    )

tarif_kwh = st.sidebar.number_input(
    "Tarif Listrik (Rp / kWh)",
    min_value=100.0,
    max_value=10000.0,
    value=1444.70,
    step=50.0,
)

# 4. Header Utama
st.title("⚡ Sistem Prediksi Konsumsi Energi Gedung")
st.markdown(
    "Aplikasi peramalan beban listrik per jam berdasarkan variabel cuaca, okupansi gedung, dan kalender operasional."
)

tab_simulasi, tab_batch = st.tabs(
    ["🎛️ Simulasi & Prediksi Per Jam", "📂 Prediksi Batch (Upload CSV)"]
)

# ==============================================================================
# TAB 1: SIMULASI & PREDIKSI TUNGGAL + PROFIL HARIAN 24 JAM
# ==============================================================================
with tab_simulasi:
    col_input, col_output = st.columns([1, 1.4], gap="large")

    with col_input:
        st.subheader("Parameter Operasional")
        tanggal = st.date_input("Tanggal Operasional", value=datetime(2026, 5, 15))
        jam_int = st.slider("Jam Operasional (00:00 - 23:00)", min_value=0, max_value=23, value=13)
        suhu = st.slider("Suhu Luar (°C)", min_value=13.0, max_value=36.0, value=27.5, step=0.1)
        kelembaban = st.slider("Kelembaban Luar (%)", min_value=40.0, max_value=100.0, value=76.0, step=1.0)
        okupansi = st.number_input(
            "Okupansi Gedung (Jumlah Orang)",
            min_value=0,
            max_value=1000,
            value=320,
            step=10,
        )
        hari_libur = st.selectbox(
            "Status Hari",
            options=[0, 1],
            format_func=lambda x: "Hari Kerja / Operasional Biasa (0)" if x == 0 else "Hari Libur (1)",
        )

    with col_output:
        st.subheader("Hasil Prediksi Konsumsi Energi")
        waktu_pilihan = datetime.combine(tanggal, time(hour=jam_int))

        df_single = pd.DataFrame(
            [
                {
                    "waktu": waktu_pilihan,
                    "suhu_luar_C": suhu,
                    "kelembaban_luar_pct": kelembaban,
                    "okupansi_orang": okupansi,
                    "hari_libur": hari_libur,
                }
            ]
        )

        pred_kwh = float(predict_energy(df_single, model, metadata)[0])
        estimasi_biaya = pred_kwh * tarif_kwh

        m1, m2 = st.columns(2)
        m1.metric("Prediksi Beban Listrik", f"{pred_kwh:.2f} kWh")
        m2.metric("Estimasi Biaya / Jam", f"Rp {estimasi_biaya:,.0f}")

        # Simulasi profil beban 24 jam pada tanggal yang dipilih
        st.markdown("#### Proyeksi Profil Beban 24 Jam (Tanggal Terpilih)")
        hours = list(range(24))
        sim_rows = []
        for h in hours:
            # Variasi okupansi dan suhu realistis sepanjang hari
            if hari_libur == 1 or waktu_pilihan.weekday() >= 5:
                occ_h = okupansi * (0.25 if 9 <= h <= 16 else 0.05)
            else:
                if 8 <= h <= 17:
                    occ_h = okupansi
                elif h in (7, 18, 19):
                    occ_h = okupansi * 0.35
                else:
                    occ_h = okupansi * 0.05

            suhu_h = suhu - 2.5 * np.cos(2 * np.pi * (h - 14) / 24.0)
            sim_rows.append(
                {
                    "waktu": datetime.combine(tanggal, time(hour=h)),
                    "suhu_luar_C": round(float(suhu_h), 1),
                    "kelembaban_luar_pct": kelembaban,
                    "okupansi_orang": round(float(occ_h)),
                    "hari_libur": hari_libur,
                }
            )

        df_24h = pd.DataFrame(sim_rows)
        df_24h["Prediksi_kWh"] = predict_energy(df_24h, model, metadata)
        df_24h["Jam"] = df_24h["waktu"].dt.strftime("%H:%M")

        fig_24h = px.line(
            df_24h,
            x="Jam",
            y="Prediksi_kWh",
            markers=True,
            labels={"Jam": "Jam", "Prediksi_kWh": "Prediksi Konsumsi (kWh)"},
        )
        fig_24h.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_24h, use_container_width=True)

# ==============================================================================
# TAB 2: PREDIKSI BATCH DARI FILE CSV
# ==============================================================================
with tab_batch:
    st.subheader("Unggah Dataset CSV untuk Prediksi Multi-Jam")
    st.markdown(
        "File CSV harus memiliki kolom: `waktu`, `suhu_luar_C`, `kelembaban_luar_pct`, "
        "`okupansi_orang`, dan `hari_libur` (opsional: `konsumsi_kWh` untuk evaluasi akurasi)."
    )

    uploaded_csv = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_csv is not None:
        try:
            df_csv = pd.read_csv(uploaded_csv)
            required_cols = {"waktu", "suhu_luar_C", "kelembaban_luar_pct", "okupansi_orang", "hari_libur"}
            missing_cols = required_cols - set(df_csv.columns)

            if missing_cols:
                st.error(f"Kolom wajib berikut tidak ditemukan di dalam CSV: {', '.join(sorted(missing_cols))}")
            else:
                df_result = create_features(df_csv)
                df_result["prediksi_konsumsi_kWh"] = predict_energy(df_csv, model, metadata)

                # Jika data aktual tersedia, tampilkan metrik evaluasi MAE & RMSE
                if "konsumsi_kWh" in df_result.columns:
                    valid_eval = df_result.dropna(subset=["konsumsi_kWh", "prediksi_konsumsi_kWh"])
                    # Filter outlier ekstrem sesuai pembersihan notebook (> 0 dan <= 700 kWh)
                    valid_eval = valid_eval[
                        (valid_eval["konsumsi_kWh"] > 0) & (valid_eval["konsumsi_kWh"] <= 700)
                    ]
                    if len(valid_eval) > 0:
                        mae = mean_absolute_error(
                            valid_eval["konsumsi_kWh"], valid_eval["prediksi_konsumsi_kWh"]
                        )
                        rmse = root_mean_squared_error(
                            valid_eval["konsumsi_kWh"], valid_eval["prediksi_konsumsi_kWh"]
                        )
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Jumlah Baris Terproses", f"{len(df_result):,}")
                        c2.metric("MAE (Mean Absolute Error)", f"{mae:.2f} kWh")
                        c3.metric("RMSE (Root Mean Squared Error)", f"{rmse:.2f} kWh")

                # Visualisasi deret waktu (maksimal 500 titik terakhir agar interaktif & ringan)
                plot_df = df_result.tail(500)
                y_cols = ["prediksi_konsumsi_kWh"]
                if "konsumsi_kWh" in plot_df.columns:
                    y_cols.insert(0, "konsumsi_kWh")

                fig_batch = px.line(
                    plot_df,
                    x="waktu",
                    y=y_cols,
                    labels={"waktu": "Waktu", "value": "Konsumsi (kWh)", "variable": "Keterangan"},
                    title="Grafik Perbandingan Konsumsi Energi (500 Jam Terakhir)",
                )
                st.plotly_chart(fig_batch, use_container_width=True)

                # Tabel hasil & tombol unduh
                tampil_cols = [
                    c for c in [
                        "waktu",
                        "suhu_luar_C",
                        "kelembaban_luar_pct",
                        "okupansi_orang",
                        "hari_libur",
                        "konsumsi_kWh",
                        "prediksi_konsumsi_kWh",
                    ]
                    if c in df_result.columns
                ]
                st.dataframe(df_result[tampil_cols].head(100), use_container_width=True)

                csv_bytes = df_result[tampil_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Unduh Hasil Prediksi (.csv)",
                    data=csv_bytes,
                    file_name="hasil_prediksi_energi.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file CSV: {e}")
