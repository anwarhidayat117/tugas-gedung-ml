import numpy as np
import pandas as pd


def clean_timestamp(series: pd.Series) -> pd.Series:
    """Menyeragamkan format waktu (mengganti titik pada jam menjadi titik dua)."""
    cleaned = series.astype(str).str.strip().str.replace(".", ":", regex=False)
    return pd.to_datetime(cleaned, format="mixed", dayfirst=True, errors="coerce")


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Rekayasa fitur untuk prediksi konsumsi energi gedung (kWh).
    Kompatibel untuk prediksi tunggal (1 baris) maupun batch/time-series.
    """
    df = data.copy()

    # 1. Pastikan kolom waktu bertipe datetime dan terurut
    if "waktu" in df.columns:
        df["waktu"] = clean_timestamp(df["waktu"])
        df = df.sort_values("waktu").reset_index(drop=True)
        dt = df["waktu"].dt
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
        dt = df.index
    else:
        raise ValueError("Data harus memiliki kolom 'waktu' atau DatetimeIndex.")

    # 2. Pastikan tipe data numerik pada fitur dasar
    base_num_cols = ["suhu_luar_C", "kelembaban_luar_pct", "okupansi_orang", "hari_libur"]
    for col in base_num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "suhu_luar_C" in df.columns:
        df["suhu_luar_C"] = df["suhu_luar_C"].interpolate().bfill().ffill().fillna(24.0)
    if "kelembaban_luar_pct" in df.columns:
        df["kelembaban_luar_pct"] = df["kelembaban_luar_pct"].interpolate().bfill().ffill().fillna(78.0)
    if "okupansi_orang" in df.columns:
        df["okupansi_orang"] = df["okupansi_orang"].interpolate().bfill().ffill().fillna(0.0).clip(lower=0)
    if "hari_libur" in df.columns:
        df["hari_libur"] = df["hari_libur"].ffill().bfill().fillna(0).round().astype(int)

    # 3. Ekstraksi fitur kalender & waktu
    df["hour"] = dt.hour
    df["dayofweek"] = dt.dayofweek
    df["day"] = dt.day
    df["month"] = dt.month
    df["quarter"] = dt.quarter
    df["dayofyear"] = dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    hari_libur_val = df["hari_libur"] if "hari_libur" in df.columns else 0
    df["is_working_hour"] = (
        (df["hour"] >= 8)
        & (df["hour"] <= 17)
        & (df["is_weekend"] == 0)
        & (hari_libur_val == 0)
    ).astype(int)

    # 4. Fitur siklikal (Cyclical Encoding)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # 5. Fitur interaksi domain energi gedung
    if "suhu_luar_C" in df.columns:
        df["cooling_degree"] = np.maximum(0.0, df["suhu_luar_C"] - 22.0)
    if "suhu_luar_C" in df.columns and "kelembaban_luar_pct" in df.columns:
        df["heat_index_proxy"] = df["suhu_luar_C"] * (df["kelembaban_luar_pct"] / 100.0)
    if "suhu_luar_C" in df.columns and "okupansi_orang" in df.columns:
        df["suhu_x_okupansi"] = df["suhu_luar_C"] * df["okupansi_orang"]
    if "okupansi_orang" in df.columns:
        df["okupansi_per_jam_kerja"] = df["okupansi_orang"] * df["is_working_hour"]

    # 6. Fitur Lag & Rolling (aktif jika data historis konsumsi_kWh tersedia > 1 baris)
    if "konsumsi_kWh" in df.columns and df["konsumsi_kWh"].notna().sum() > 1:
        target_series = pd.to_numeric(df["konsumsi_kWh"], errors="coerce").interpolate().bfill().ffill()
        for lag in [1, 2, 3, 24, 48, 168]:
            df[f"lag_{lag}"] = target_series.shift(lag).bfill().ffill()

        shifted = target_series.shift(1)
        df["rolling_mean_3"] = shifted.rolling(3, min_periods=1).mean().bfill().ffill()
        df["rolling_mean_24"] = shifted.rolling(24, min_periods=1).mean().bfill().ffill()
        df["rolling_std_24"] = shifted.rolling(24, min_periods=1).std().fillna(0.0)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Mengambil daftar kolom fitur numerik (mengecualikan waktu dan target)."""
    excluded = {"waktu", "konsumsi_kWh"}
    return [
        col for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
