import streamlit as st
import statistics
import math
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import warnings
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Uyarıları gizle
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Optimize edilmiş tahmin sınıfını import et
from optimized_predictor import OptimizedJetXPredictor

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="JetX Optimize Tahmin Sistemi",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile gelişmiş stil
st.markdown("""
<style>
    .big-font {
        font-size: 30px !important;
        font-weight: bold;
        color: #1e3a8a;
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        border: 2px solid;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .high-confidence {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(34, 197, 94, 0.05));
        border-color: #22c55e;
    }
    .medium-confidence {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.1), rgba(251, 191, 36, 0.05));
        border-color: #fbbf24;
    }
    .low-confidence {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05));
        border-color: #ef4444;
    }
    .uncertain-confidence {
        background: linear-gradient(135deg, rgba(107, 114, 128, 0.1), rgba(107, 114, 128, 0.05));
        border-color: #6b7280;
    }
    .above-threshold {
        color: #059669;
        font-weight: bold;
        font-size: 24px;
    }
    .below-threshold {
        color: #dc2626;
        font-weight: bold;
        font-size: 24px;
    }
    .uncertain-text {
        color: #6b7280;
        font-weight: bold;
        font-size: 24px;
    }
    .speed-indicator {
        background: linear-gradient(90deg, #10b981, #059669);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .performance-metric {
        background: #f8fafc;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state ile uygulama durumunu takip et
if 'predictor' not in st.session_state:
    with st.spinner("🚀 Optimize edilmiş JetX Tahmin Sistemi başlatılıyor..."):
        st.session_state.predictor = OptimizedJetXPredictor(db_path="jetx_data.db")

if 'last_values' not in st.session_state:
    st.session_state.last_values = []

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Değer ekleme ve tahmin fonksiyonu
def add_value_and_predict(value):
    # Önceki tahmin sonucunu güncelle
    if st.session_state.last_prediction and st.session_state.last_prediction.get('prediction_id'):
        st.session_state.predictor.update_result(
            st.session_state.last_prediction['prediction_id'], 
            value
        )

    # Yeni değeri ekle
    record_id = st.session_state.predictor.add_new_result(value)
    if record_id:
        # Son değerleri güncelle
        st.session_state.last_values.append(value)
        if len(st.session_state.last_values) > 20:
            st.session_state.last_values = st.session_state.last_values[-20:]

        # Hızlı tahmin yap
        with st.spinner("⚡ Hızlı tahmin yapılıyor..."):
            start_time = time.time()
            prediction = st.session_state.predictor.predict_next()
            prediction_time = time.time() - start_time
            
            if prediction:
                prediction['total_prediction_time'] = prediction_time
                st.session_state.last_prediction = prediction
                
                # Prediction history'ye ekle
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now(),
                    'actual_value': value,
                    'prediction': prediction
                })
                
                # Son 50 tahmin geçmişini sakla
                if len(st.session_state.prediction_history) > 50:
                    st.session_state.prediction_history = st.session_state.prediction_history[-50:]

        return True
    else:
        st.error("❌ Değer eklenirken hata oluştu!")
        return False

# Ana başlık
st.markdown('<p class="big-font">🚀 JetX Optimize Tahmin Sistemi</p>', unsafe_allow_html=True)

# Model durumu kontrolü
if st.session_state.predictor.current_models is None:
    st.error("⚠️ Eğitilmiş model bulunamadı!")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎯 Yeni Modelleri Eğit", type="primary"):
            with st.spinner("🔥 Modeller eğitiliyor... Bu işlem birkaç dakika sürebilir."):
                success = st.session_state.predictor.train_new_models(
                    window_size=5000,
                    model_types=['rf', 'gb', 'svm']
                )
                if success:
                    st.success("✅ Modeller başarıyla eğitildi!")
                    st.rerun()
                else:
                    st.error("❌ Model eğitimi başarısız!")
    
    with col2:
        st.info("📝 Model eğitimi için en az 500 veri gereklidir.")
        
    st.stop()

# Model bilgileri
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    model_info = st.session_state.predictor.get_model_info()
    if 'ensemble' in model_info:
        st.metric(
            "🎯 Ensemble Accuracy", 
            f"{model_info['ensemble']['accuracy']:.3f}",
            delta="Optimized"
        )

with col2:
    performance_stats = st.session_state.predictor.get_performance_stats()
    if performance_stats:
        st.metric(
            "⚡ Ortalama Tahmin Süresi", 
            f"{performance_stats['avg_prediction_time_ms']:.1f}ms",
            delta=f"{performance_stats['total_predictions']} tahmin"
        )

with col3:
    if performance_stats:
        predictions_per_sec = 1000 / performance_stats['avg_prediction_time_ms']
        st.metric(
            "🚀 Saniyede Tahmin", 
            f"{predictions_per_sec:.1f}",
            delta="Real-time"
        )

with col4:
    # Retrain durumu
    needs_retrain = st.session_state.predictor.should_retrain()
    retrain_status = "🔄 Gerekli" if needs_retrain else "✅ Güncel"
    st.metric("🔄 Model Durumu", retrain_status)

st.markdown("---")

# Ana içerik
col1, col2 = st.columns([2, 1])

with col1:
    # Veri Giriş Bölümü
    st.subheader("📊 Veri Girişi")

    # Tek değer girişi
    with st.expander("🎯 Tek Değer Girişi", expanded=True):
        with st.form(key="single_value_form"):
            value_input = st.number_input(
                "JetX Değeri:", 
                min_value=1.0, 
                max_value=10000.0, 
                value=1.5, 
                step=0.01,
                format="%.2f"
            )

            submit_button = st.form_submit_button("⚡ Hızlı Tahmin Et", type="primary")

            if submit_button:
                success = add_value_and_predict(value_input)
                if success:
                    st.success(f"✅ Değer eklendi: {value_input}")
                    st.rerun()

    # Toplu veri girişi
    with st.expander("📝 Toplu Veri Girişi"):
        bulk_text = st.text_area(
            "Her satıra bir değer (virgül ile de ayırabilirsiniz):",
            height=150,
            placeholder="1.55\n2.89\n1.56\n\nveya\n\n1.55, 2.89, 1.56"
        )

        if st.button("📊 Toplu Veri Ekle"):
            # Virgül ve satır sonu ile ayrıştır
            import re
            values_text = re.split(r'[,\n]+', bulk_text.strip())
            
            success_count = 0
            error_count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, value_str in enumerate(values_text):
                try:
                    value = float(value_str.strip())
                    if 1.0 <= value <= 10000.0:
                        record_id = st.session_state.predictor.add_new_result(value)
                        if record_id:
                            st.session_state.last_values.append(value)
                            success_count += 1
                    else:
                        error_count += 1
                except:
                    if value_str.strip():  # Boş değilse error say
                        error_count += 1

                # Progress update
                progress = (i + 1) / len(values_text)
                progress_bar.progress(progress)
                status_text.text(f"İşlenen: {i+1}/{len(values_text)}")

            st.success(f"✅ {success_count} değer eklendi, {error_count} hata")
            if success_count > 0:
                st.rerun()

with col2:
    # Tahmin Sonuçları
    st.subheader("🎯 Son Tahmin")

    if st.session_state.last_prediction:
        pred = st.session_state.last_prediction
        confidence = pred.get('confidence_score', 0.0)
        prob = pred.get('above_threshold_probability', 0.5)
        pred_time = pred.get('prediction_time_ms', 0)
        
        # Confidence class belirleme
        if confidence >= 0.8:
            conf_class = "high-confidence"
            conf_icon = "🟢"
        elif confidence >= 0.6:
            conf_class = "medium-confidence"
            conf_icon = "🟡"
        elif confidence >= 0.4:
            conf_class = "low-confidence"
            conf_icon = "🟠"
        else:
            conf_class = "uncertain-confidence"
            conf_icon = "⚪"
        
        # Detaylı güven analizi
        confidence_analysis = pred.get('confidence_analysis', {})
        confidence_level = confidence_analysis.get('confidence_level', 'Belirsiz')
        factors = confidence_analysis.get('factors', {})
        recommendations = confidence_analysis.get('recommendations', [])

        # Decision text ve class
        above_threshold = pred.get('above_threshold')
        if above_threshold is None:
            decision_text = "BELİRSİZ"
            threshold_class = "uncertain-text"
            decision_icon = "❓"
        elif above_threshold:
            decision_text = "1.5 ÜZERİNDE"
            threshold_class = "above-threshold"
            decision_icon = "📈"
        else:
            decision_text = "1.5 ALTINDA"
            threshold_class = "below-threshold"
            decision_icon = "📉"

        # Tahmin kutusu
        st.markdown(f"""
        <div class="result-box {conf_class}">
            <h3 style="text-align: center; margin-bottom: 20px;">
                {decision_icon} <span class="{threshold_class}">{decision_text}</span>
            </h3>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div>
                    <strong>🎯 Olasılık:</strong><br>
                    <span style="font-size: 18px;">{prob:.1%}</span>
                </div>
                <div>
                    <strong>{conf_icon} Güven:</strong><br>
                    <span style="font-size: 18px;">{confidence:.1%}</span>
                </div>
                <div>
                    <strong>⚡ Süre:</strong><br>
                    <span style="font-size: 18px;">{pred_time:.1f}ms</span>
                </div>
                <div>
                    <strong>💡 Cache:</strong><br>
                    <span style="font-size: 18px;">{"✅" if pred.get('cache_hit') else "❌"}</span>
                </div>
            </div>
            
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e0e0e0;">
                <strong>📊 Güven Seviyesi:</strong> {confidence_level}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detaylı güven analizi expander
        if factors:
            with st.expander("🔍 Detaylı Güven Analizi"):
                st.subheader("📈 Güven Faktörleri")
                
                # Faktörleri progress bar ile göster
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
                    
                    st.write(f"**{factor_display_name}**")
                    st.progress(factor_value, text=f"{factor_value:.1%}")
                
                # Öneriler
                if recommendations:
                    st.subheader("💡 Öneriler")
                    for rec in recommendations:
                        st.write(f"• {rec}")
        
        # Güven geçmişi
        if hasattr(st.session_state, 'prediction_history') and st.session_state.prediction_history:
            with st.expander("📈 Güven Geçmişi"):
                recent_predictions = st.session_state.prediction_history[-10:]
                
                confidence_values = []
                timestamps = []
                
                for pred_record in recent_predictions:
                    pred_data = pred_record.get('prediction', {})
                    conf_analysis = pred_data.get('confidence_analysis', {})
                    
                    if conf_analysis:
                        confidence_values.append(conf_analysis.get('total_confidence', 0))
                        timestamps.append(pred_record.get('timestamp', datetime.now()))
                
                if confidence_values:
                    # Güven trend grafiği
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(len(confidence_values))),
                        y=confidence_values,
                        mode='lines+markers',
                        name='Güven Skoru',
                        line=dict(color='blue', width=3),
                        marker=dict(size=8)
                    ))
                    
                    fig.add_hline(
                        y=0.8, 
                        line_dash="dash", 
                        line_color="green",
                        annotation_text="Yüksek Güven (0.8)"
                    )
                    
                    fig.add_hline(
                        y=0.5, 
                        line_dash="dash", 
                        line_color="orange",
                        annotation_text="Orta Güven (0.5)"
                    )
                    
                    fig.update_layout(
                        title="Son 10 Tahmin Güven Skoru Trendi",
                        xaxis_title="Tahmin Sırası",
                        yaxis_title="Güven Skoru",
                        height=300,
                        yaxis=dict(range=[0, 1])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

        # Hızlı tekrar tahmin butonu
        if st.button("🔄 Yeniden Tahmin Et"):
            with st.spinner("⚡ Tahmin yapılıyor..."):
                new_prediction = st.session_state.predictor.predict_next()
                if new_prediction:
                    st.session_state.last_prediction = new_prediction
                    st.rerun()

    else:
        st.info("📝 Henüz tahmin yapılmadı. Veri girin ve tahmin ettirin!")

# Grafik ve analiz bölümü
st.markdown("---")
st.subheader("📈 Veri Analizi")

# Son 20 veri grafiği
if st.session_state.last_values:
    # Plotly ile interaktif grafik
    fig = go.Figure()
    
    values = st.session_state.last_values[-20:]
    colors = ['red' if v < 1.5 else 'green' for v in values]
    
    fig.add_trace(go.Scatter(
        x=list(range(len(values))),
        y=values,
        mode='lines+markers',
        marker=dict(
            color=colors,
            size=8,
            line=dict(width=2, color='white')
        ),
        line=dict(width=3),
        name='JetX Values'
    ))
    
    # Threshold line
    fig.add_hline(
        y=1.5, 
        line_dash="dash", 
        line_color="gray",
        annotation_text="Threshold (1.5)"
    )
    
    fig.update_layout(
        title="Son 20 JetX Değeri",
        xaxis_title="Sıra",
        yaxis_title="Değer",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # İstatistikler
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_val = statistics.mean(values)
        st.metric("📊 Ortalama", f"{avg_val:.2f}")
    
    with col2:
        above_threshold_ratio = sum(1 for v in values if v >= 1.5) / len(values)
        st.metric("📈 1.5+ Oranı", f"{above_threshold_ratio:.1%}")
    
    with col3:
        max_val = max(values)
        st.metric("🔺 Maksimum", f"{max_val:.2f}")
    
    with col4:
        volatility = statistics.stdev(values) if len(values) > 1 else 0.0
        st.metric("📊 Volatilite", f"{volatility:.2f}")

# Sidebar - Gelişmiş Ayarlar
st.sidebar.title("⚙️ Sistem Kontrolü")

# Model performansı
st.sidebar.subheader("📊 Model Performansı")
model_info = st.session_state.predictor.get_model_info()

if model_info:
    for model_name, info in model_info.items():
        if model_name != 'ensemble':
            st.sidebar.text(f"{model_name}: {info['accuracy']:.3f}")
        else:
            st.sidebar.success(f"🎯 Ensemble: {info['accuracy']:.3f}")

# Sistem durumu
st.sidebar.subheader("🔧 Sistem Durumu")
performance_stats = st.session_state.predictor.get_performance_stats()

if performance_stats:
    st.sidebar.metric("Toplam Tahmin", performance_stats['total_predictions'])
    st.sidebar.metric("Ortalama Süre", f"{performance_stats['avg_prediction_time_ms']:.1f}ms")

# Model yönetimi
st.sidebar.subheader("🔄 Model Yönetimi")

if st.sidebar.button("🔄 Otomatik Güncelleme Kontrol"):
    with st.spinner("Model durumu kontrol ediliyor..."):
        success = st.session_state.predictor.auto_retrain_if_needed()
        if success:
            st.sidebar.success("✅ Model güncel!")
        else:
            st.sidebar.warning("⚠️ Güncelleme gerekli!")

if st.sidebar.button("🎯 Elle Model Eğit"):
    with st.spinner("Yeni modeller eğitiliyor..."):
        success = st.session_state.predictor.train_new_models(
            window_size=6000,
            model_types=['rf', 'gb', 'svm']
        )
        if success:
            st.sidebar.success("✅ Modeller güncellendi!")
            st.rerun()

# Benchmark
if st.sidebar.button("⚡ Hız Testi"):
    with st.spinner("Hız testi yapılıyor..."):
        results = st.session_state.predictor.benchmark_prediction_speed(50)
        if results:
            st.sidebar.success(f"⚡ {results['avg_time_ms']:.1f}ms ortalama")
            st.sidebar.info(f"🚀 {results['predictions_per_second']:.1f} tahmin/saniye")

# About
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Hakkında")
st.sidebar.info("""
🚀 **Optimize JetX Tahmin Sistemi**

✅ Hızlı model yükleme
✅ Millisaniye tahmin süresi  
✅ Otomatik model güncelleme
✅ Gelişmiş feature engineering
✅ Real-time performance tracking

⚠️ **Uyarı:** Sadece eğitim amaçlıdır.
""")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6b7280; font-size: 14px;">
    🕒 Son güncellenme: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | 
    🚀 Optimize Sistem v2.0
</div>
""", unsafe_allow_html=True)