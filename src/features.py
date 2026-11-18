\# src/features.py 
import numpy as np 
import pandas as pd 

def create_features(data):
  """
  Fungsi rekayasa fitur konsisten dengan notebook Colab.
  Membuat fitur kalender, siklis jam, lag (&gt;= 24 jam), rolling stats, dan beban termal.
  """
  df_feat = data.copy()

  # Fitur Kalender 
  df_feat['jam'] = df_feat['waktu'].dt.hour
  df_feat['hari'] = df_feat['waktu'].dt.dayofweek
  df_feat['bulan'] = df_feat['waktu'].dt.month
  df_feat['is_weekend'] = df_feat['hari'].isin([5, 6]).astype(int) 

  # Fitur Jam Kerja (08.00 - 17.00 WIB di hari kerja)
  df_feat['is_work_hour'] = ((df_feat['jam'] >= 8) & (df_feat['jam'] <= 17) & (df_feat['is_weekend'] == 0)).astype(int) 

  # Fitur Siklis Waktu (Jam) 
  df_feat['jam_sin'] = np.sin(2 * np.pi * df_feat['jam'] / 24)
  df_feat['jam_cos'] = np.cos(2 * np.pi * df_feat['jam'] / 24)

  # Fitur Lag (Wajib &gt;= 24 Jam)
  df_feat['lag_24'] = df_feat['konsumsi_kWh'].shift(24)
  df_feat['lag_48'] = df_feat['konsumsi_kWh'].shift(48) 
  df_feat['lag_72'] = df_feat['konsumsi_kWh'].shift(72)
  df_feat['lag_168'] = df_feat['konsumsi_kWh'].shift(168)

  # Fitur Statistik Bergerak (Rolling) berdasarkan lag\_24
  df_feat['rolling_mean_24'] = df_feat['lag_24'].rolling(window=24).mean()
  df_feat['rolling_std_24'] = df_feat['lag_24'].rolling(window=24).std()
  df_feat['rolling_min_24'] = df_feat['lag_24'].rolling(window=24).min()
  df_feat['rolling_max_24'] = df_feat['lag_24'].rolling(window=24).max()
  df_feat['rolling_mean_168'] = df_feat['lag_24'].rolling(window=168).mean()

  # Fitur Selisih &amp; Interaksi
  df_feat['selisih_lag24_lag168'] = df_feat['lag_24'] - df_feat['lag_168']
  df_feat['suhu_x_okupansi_lag24'] = df_feat['suhu_luar_C'] * df_feat['okupansi_orang'] 
  
  return df_feat
