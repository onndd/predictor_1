# app.py DOSYASININ TAM İÇERİĞİ

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import warnings
import sqlite3
from datetime import datetime
import sys

# Mantık sınıfımızı yeni dosyasından import ediyoruz
from predictor_logic import JetXPredictor

# Uyarıları gizle
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Buradan sonrası Streamlit arayüz kodudur ---
# Colab'da çalıştırılmaz, sadece `streamlit run app.py` ile çalışır.

if __name__ == '__main__':
    st.set_page_config(
        page_title="JetX Tahmin Uygulaması",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .big-font {font-size:30px !important; font-weight: bold;}
        .result-box {padding: 20px; border-radius: 10px; margin-bottom: 20px;}
        .high-confidence {background-color: rgba(0, 255, 0, 0.2);}
        .medium-confidence {background-color: rgba(255, 255, 0, 0.2);}
        .low-confidence {background-color: rgba(255, 0, 0, 0.2);}
        .uncertain-confidence {background-color: rgba(128, 128, 128, 0.2);}
        .above-threshold {color: green; font-weight: bold;}
        .below-threshold {color: red; font-weight: bold;}
        .uncertain-text {color: #555; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

    # Session state ile uygulama durumunu takip et
    if 'predictor' not in st.session_state:
        st.session_state.predictor = JetXPredictor(db_path="jetx_data.db")
        with st.spinner("Veriler yükleniyor ve modeller hazırlanıyor... Bu işlem biraz sürebilir."):
            df = st.session_state.predictor.load_data(limit=st.session_state.predictor.training_window_size)
            if df is not None and len(df) > 0:
                st.session_state.predictor.initialize_models(df)
            else:
                st.warning("Veritabanında eğitim için yeterli veri bulunamadı. Önce veri ekleyin.")

    if 'last_values' not in st.session_state:
        # Son değerleri göstermek için veritabanından çek
        try:
            conn = sqlite3.connect("jetx_data.db")
            df_last = pd.read_sql_query("SELECT value FROM jetx_results ORDER BY id DESC LIMIT 15", conn)
            conn.close()
            st.session_state.last_values = df_last['value'].tolist()[::-1] # Ters çevirerek doğru sırayı al
        except:
            st.session_state.last_values = []
    
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None

    if 'new_data_count' not in st.session_state:
        st.session_state.new_data_count = 0

    def add_value_and_predict(value):
        if 'last_prediction' in st.session_state and st.session_state.last_prediction:
            with st.spinner("Önceki tahmin sonucu güncelleniyor..."):
                st.session_state.predictor.update_result(
                    st.session_state.last_prediction['prediction_id'], 
                    value
                )

        record_id = st.session_state.predictor.add_new_result(value)
        if record_id:
            st.session_state.last_values.append(value)
            if len(st.session_state.last_values) > 15:
                st.session_state.last_values.pop(0)

            st.session_state.new_data_count += 1

            # Her 15 yeni veriden sonra modelleri yeniden eğit
            if st.session_state.new_data_count >= 15:
                with st.spinner("Modeller güncelleniyor..."):
                    st.session_state.predictor.retrain_models()
                st.session_state.new_data_count = 0

            with st.spinner("Tahmin yapılıyor..."):
                prediction = st.session_state.predictor.predict_next()
                if prediction:
                    st.session_state.last_prediction = prediction
            return True
        else:
            st.error("Değer eklenirken bir hata oluştu.")
            return False

    # --- ARAYÜZ KODLARI ---
    st.markdown('<p class="big-font">JetX Tahmin Uygulaması</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Veri Girişi")
        with st.expander("Tek Değer Girişi", expanded=True):
            with st.form(key="single_value_form"):
                value_input = st.number_input(
                    "JetX Değeri Girin:", min_value=1.0, max_value=5000.0, value=1.5, step=0.01, format="%.2f"
                )
                submit_button = st.form_submit_button("Değeri Ekle ve Tahmin Et")
                if submit_button:
                    if add_value_and_predict(value_input):
                        st.success(f"Değer eklendi: {value_input}")
                        st.rerun()

    with col2:
        st.subheader("Sonraki Tur Tahmini")
        if st.session_state.last_prediction:
            pred = st.session_state.last_prediction
            conf = pred.get('confidence_score', 0.0)
            
            conf_class = "high-confidence" if conf >= 0.7 else ("medium-confidence" if conf >= 0.6 else "low-confidence")
            decision_text = pred.get('decision_text', 'Hata')
            
            if pred.get('above_threshold') is None: # Belirsiz durumu
                conf_class = "uncertain-confidence"
                threshold_class = "uncertain-text"
            else:
                threshold_class = "above-threshold" if pred.get('above_threshold') else "below-threshold"

            st.markdown(f"""
            <div class="result-box {conf_class}">
                <h3 style="text-align: center;" class="{threshold_class}">{decision_text}</h3>
                <p><b>1.5 Üstü Olasılığı:</b> {pred.get('above_threshold_probability', 0.0):.2%}</p>
                <p><b>Güven Skoru:</b> {conf:.2%}</p>
                <p><b>CrashDetector Riski:</b> {pred.get('crash_risk_by_special_model', 0.0):.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Henüz bir tahmin yapılmadı.")

    st.subheader("Geçmiş Veriler")
    if st.session_state.last_values:
        fig, ax = plt.subplots(figsize=(10, 2))
        colors = ['red' if v < 1.5 else 'green' for v in st.session_state.last_values]
        ax.bar(range(len(st.session_state.last_values)), st.session_state.last_values, color=colors)
        ax.axhline(y=1.5, color='grey', linestyle='--', alpha=0.7)
        st.pyplot(fig)
    else:
        st.info("Henüz veri girilmedi.")