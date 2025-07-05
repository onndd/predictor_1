import streamlit as st
import statistics
import math
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import warnings
from datetime import datetime

# Uyarıları gizle
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow uyarılarını gizle

# Enhanced JetX tahmin sınıfını import et
from predictor_logic import JetXPredictor

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="JetX Tahmin Uygulaması",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile bazı stil ayarlamaları
st.markdown("""
<style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .high-confidence {
        background-color: rgba(0, 255, 0, 0.2);
    }
    .medium-confidence {
        background-color: rgba(255, 255, 0, 0.2);
    }
    .low-confidence {
        background-color: rgba(255, 0, 0, 0.2);
    }
    .above-threshold {
        color: green;
        font-weight: bold;
    }
    .below-threshold {
        color: red;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Session state ile uygulama durumunu takip et
if 'predictor' not in st.session_state:
    st.session_state.predictor = JetXPredictor(db_path="jetx_data.db")
    # Modellerin yüklü olup olmadığını kontrol et
    if not st.session_state.predictor.models:
        st.warning("Modeller henüz eğitilmemiş. Sidebar'dan 'Modelleri Yeniden Eğit' butonuna tıklayın.")

if 'last_values' not in st.session_state:
    st.session_state.last_values = []

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# Yeniden eğitim için durum değişkenleri
if 'should_retrain' not in st.session_state:
    st.session_state.should_retrain = False

if 'new_data_count' not in st.session_state:
    st.session_state.new_data_count = 0

# Değer ekleme ve yeniden eğitim fonksiyonu güncellemesi
def add_value_and_predict(value):
    # Eğer önceki bir tahmin varsa, önce onun sonucunu güncelle
    if 'last_prediction' in st.session_state and st.session_state.last_prediction:
        with st.spinner("Önceki tahmin sonucu güncelleniyor..."):
            success = st.session_state.predictor.update_result(
                st.session_state.last_prediction['prediction_id'], 
                value
            )
            if success:
                st.success(f"Önceki tahmin için sonuç güncellendi: {value}")

    # Değeri veritabanına ekle
    record_id = st.session_state.predictor.add_new_result(value)
    if record_id:
        # Son değerleri güncelle
        st.session_state.last_values.append(value)
        if len(st.session_state.last_values) > 15:
            st.session_state.last_values = st.session_state.last_values[-15:]

        # Yeni veri sayısını artır (sadece istatistik için)
        st.session_state.new_data_count += 1

        # Hemen tahmin yap
        with st.spinner("Tahmin yapılıyor..."):
            prediction = st.session_state.predictor.predict_next()
            if prediction:
                st.session_state.last_prediction = prediction

        return True
    else:
        st.error("Değer eklenirken bir hata oluştu.")
        return False

# Başlık
st.markdown('<p class="big-font">JetX Tahmin Uygulaması</p>', unsafe_allow_html=True)
st.markdown("---")

# Ana sayfa düzeni - 2 sütun
col1, col2 = st.columns([3, 2])

with col1:
    # Veri Giriş Bölümü
    st.subheader("Veri Girişi")

    # Tek değer girişi
    with st.expander("Tek Değer Girişi", expanded=True):
        with st.form(key="single_value_form"):
            value_input = st.number_input(
                "JetX Değeri Girin:", 
                min_value=1.0, 
                max_value=3000.0, 
                value=1.5, 
                step=0.01,
                format="%.2f"
            )

            submit_button = st.form_submit_button("Değeri Ekle")

            if submit_button:
                success = add_value_and_predict(value_input)
                if success:
                    st.success(f"Değer eklendi: {value_input}")
                    # Sayfayı yenile
                    st.rerun()

    # Toplu değer girişi
    with st.expander("Toplu Değer Girişi"):
        with st.form(key="bulk_value_form"):
            bulk_text = st.text_area(
                "Her satıra bir değer gelecek şekilde değerleri girin:",
                height=200,
                help="Örnek:\n1.55\n2.89\n1.56"
            )

            submit_bulk_button = st.form_submit_button("Toplu Değerleri Ekle")

            if submit_bulk_button:
                lines = bulk_text.strip().split('\n')
                success_count = 0
                error_count = 0

                progress_bar = st.progress(0)
                for i, line in enumerate(lines):
                    try:
                        value = float(line.strip())
                        if 1.0 <= value <= 3000.0:
                            if add_value_and_predict(value):
                                success_count += 1
                        else:
                            error_count += 1
                    except:
                        error_count += 1

                    # İlerleme çubuğunu güncelle
                    progress_bar.progress((i + 1) / len(lines))

                st.success(f"{success_count} değer başarıyla eklendi. {error_count} değer eklenemedi.")
                
                # Sayfayı yenile
                st.rerun()

    # Son 15 veri
    st.subheader("Son 15 Veri")
    if st.session_state.last_values:
        # Tablo formatında göster
        df_last = pd.DataFrame({
            'Sıra': range(1, len(st.session_state.last_values) + 1),
            'Değer': st.session_state.last_values
        })
        st.dataframe(df_last, width=300)

        # Grafik olarak göster
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(st.session_state.last_values)), st.session_state.last_values, 'bo-')
        ax.axhline(y=1.5, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Sıra')
        ax.set_ylabel('Değer')
        ax.set_title('Son Değerler')
        st.pyplot(fig)
    else:
        st.info("Henüz veri girilmedi.")

with col2:
    # Tahmin Bölümü
    st.subheader("Tahmin Yap")

    if st.button("Yeni Tahmin Yap", key="predict_button"):
        if st.session_state.last_values:
            with st.spinner("Tahmin yapılıyor..."):
                # Tahmin yap
                prediction = st.session_state.predictor.predict_next()
                if prediction:
                    st.session_state.last_prediction = prediction
                else:
                    st.error("Tahmin yapılamadı. Yeterli veri yok.")
        else:
            st.warning("Tahmin için önce veri girmelisiniz.")

    # Tahmin sonuçlarını göster
    if st.session_state.last_prediction:
        prediction = st.session_state.last_prediction

        # Güven skoruna göre stil belirle
        confidence = prediction['confidence_score']
        confidence_class = ""
        if confidence >= 0.7:
            confidence_class = "high-confidence"
        elif confidence >= 0.4:
            confidence_class = "medium-confidence"
        else:
            confidence_class = "low-confidence"
        
        # Detaylı güven analizi
        confidence_analysis = prediction.get('confidence_analysis', {})
        confidence_level = confidence_analysis.get('confidence_level', 'Belirsiz')
        factors = confidence_analysis.get('factors', {})
        recommendations = confidence_analysis.get('recommendations', [])

        # Threshold durumuna göre stil
        threshold_class = "above-threshold" if prediction['above_threshold'] else "below-threshold"
        threshold_text = "1.5 ÜZERİNDE" if prediction['above_threshold'] else "1.5 ALTINDA"

        # Tahmin kutusunu daha belirgin yap
        st.markdown(f"""
        <div class="result-box {confidence_class}" style="padding: 20px; border: 2px solid {'green' if prediction['above_threshold'] else 'red'}; border-radius: 10px;">
            <h2 style="text-align: center;">Tahmin Sonucu</h2>
            <h3 style="text-align: center; margin-bottom: 20px;" class="{threshold_class}">{threshold_text}</h3>
            <p style="font-size: 18px;">Tahmini değer: <b>{prediction['predicted_value']:.2f}</b></p>
            <p style="font-size: 18px;">1.5 üstü olasılığı: <b>{prediction['above_threshold_probability']:.2f}</b></p>
            <p style="font-size: 18px;">Güven skoru: <b>{prediction['confidence_score']:.2f}</b></p>
            <p style="font-size: 16px; margin-top: 15px; color: #666;">📈 Güven seviyesi: <b>{confidence_level}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detaylı güven analizi
        if factors:
            with st.expander("🔍 Detaylı Güven Analizi"):
                st.subheader("📊 Güven Faktörleri")
                
                for factor_name, factor_value in factors.items():
                    factor_display_name = {
                        'model_performance': '🎯 Model Performansı',
                        'data_quality': '📊 Veri Kalitesi',
                        'temporal_consistency': '⏱️ Zamansal Tutarlılık',
                        'market_volatility': '📈 Piyasa Volatilitesi',
                        'prediction_certainty': '🎲 Tahmin Kesinliği',
                        'model_freshness': '🔄 Model Tazeliği',
                        'trend_alignment': '📊 Trend Uyumu'
                    }.get(factor_name, factor_name)
                    
                    st.write(f"**{factor_display_name}**: {factor_value:.1%}")
                    st.progress(factor_value)
                
                if recommendations:
                    st.subheader("💡 Öneriler")
                    for rec in recommendations:
                        st.write(f"• {rec}")



    # Model Performansı
    st.subheader("Model Performansı")
    model_info = st.session_state.predictor.get_model_info()

    if model_info:
        # Model performans tablosu
        model_df = pd.DataFrame([
            {
                'Model': name,
                'Doğruluk': f"{info['accuracy']*100:.1f}%",
                'Ağırlık': f"{info['weight']:.2f}"
            }
            for name, info in model_info.items()
        ])
        st.dataframe(model_df)

        # Doğruluk grafiği
        fig, ax = plt.subplots(figsize=(10, 4))
        models = list(model_info.keys())
        accuracies = [info['accuracy'] for info in model_info.values()]
        weights = [info['weight'] for info in model_info.values()]

        x = range(len(models))
        width = 0.35

        ax.bar([i - width/2 for i in x], accuracies, width, label='Doğruluk')
        ax.bar([i + width/2 for i in x], weights, width, label='Ağırlık')

        ax.set_xlabel('Model')
        ax.set_ylabel('Değer')
        ax.set_title('Model Performansı')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()

        # Şans seviyesi
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

        st.pyplot(fig)
    else:
        st.info("Henüz model performans bilgisi yok.")

# Otomatik yeniden eğitim kaldırıldı - sadece manuel eğitim

# Sidebar - Ek Ayarlar
st.sidebar.title("Ayarlar ve Bilgiler")

# Veritabanı istatistikleri
st.sidebar.subheader("Veritabanı İstatistikleri")
df = st.session_state.predictor.load_data()
if df is not None:
    st.sidebar.info(f"Toplam kayıt sayısı: {len(df)}")
    if len(df) > 0:
        st.sidebar.info(f"Ortalama değer: {df['value'].mean():.2f}")
        st.sidebar.info(f"1.5 üstü oranı: {(df['value'] >= 1.5).mean():.2f}")
else:
    st.sidebar.warning("Veritabanı verisi yüklenemedi.")

# Modelleri yeniden eğit
if st.sidebar.button("Modelleri Yeniden Eğit"):
    with st.spinner("Modeller yeniden eğitiliyor..."):
        success = st.session_state.predictor.retrain_models()
        if success:
            st.sidebar.success("Modeller başarıyla yeniden eğitildi.")
        else:
            st.sidebar.error("Modeller eğitilirken bir hata oluştu.")

# Uygulama bilgileri
st.sidebar.subheader("Hakkında")
st.sidebar.info("Bu uygulama, JetX oyunu sonuçlarını analiz etmek ve tahminler yapmak için geliştirilmiştir. Makine öğrenimi ve zaman serisi analizi teknikleri kullanılmaktadır.")
st.sidebar.warning("Uyarı: Bu uygulama sadece bilimsel amaçlar içindir ve kumar oynamayı teşvik etmez.")

# Son güncelleme zamanı
st.sidebar.text(f"Son güncelleme: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
