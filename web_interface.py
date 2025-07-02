from flask import Flask, render_template, jsonify, request
import json
from datetime import datetime
from prediction_interface import PredictionInterface, GameRecommendationEngine
import threading
import time

app = Flask(__name__)

# Global interface instance
prediction_interface = PredictionInterface()
latest_data = {
    'predictions': [],
    'recommendation': {},
    'last_update': None
}

def background_updater():
    """Background thread to update predictions every 2 minutes"""
    global latest_data
    
    while True:
        try:
            print("🔄 Background update başlıyor...")
            result = prediction_interface.display_prediction_interface()
            
            if result:
                latest_data = {
                    'predictions': result['predictions'],
                    'recommendation': result['recommendation'],
                    'last_update': datetime.now().isoformat()
                }
                print("✅ Background update tamamlandı")
            
        except Exception as e:
            print(f"❌ Background update hatası: {e}")
        
        time.sleep(120)  # 2 dakikada bir güncelle

# Background thread başlat
update_thread = threading.Thread(target=background_updater, daemon=True)
update_thread.start()

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')

@app.route('/api/predictions')
def get_predictions():
    """5 tahmin ve öneri API'si"""
    try:
        # Fresh predictions üret
        result = prediction_interface.display_prediction_interface()
        
        if result:
            return jsonify({
                'success': True,
                'data': {
                    'predictions': result['predictions'],
                    'recommendation': result['recommendation'],
                    'timestamp': datetime.now().isoformat()
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Tahmin üretilemedi'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/latest')
def get_latest():
    """En son data'yı al (cache'den)"""
    global latest_data
    
    if latest_data['predictions']:
        return jsonify({
            'success': True,
            'data': latest_data
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Henüz veri yok'
        })

@app.route('/api/refresh')
def refresh_predictions():
    """Tahminleri yenile"""
    try:
        result = prediction_interface.display_prediction_interface()
        
        if result:
            global latest_data
            latest_data = {
                'predictions': result['predictions'],
                'recommendation': result['recommendation'],
                'last_update': datetime.now().isoformat()
            }
            
            return jsonify({
                'success': True,
                'data': latest_data
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Tahmin üretilemedi'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎮 JetX Tahmin Sistemi</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 30px;
        }
        
        .predictions-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
        }
        
        .predictions-section h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .prediction-item {
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 5px solid;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        
        .prediction-item:hover {
            transform: translateY(-2px);
        }
        
        .prediction-item.high {
            border-left-color: #e74c3c;
        }
        
        .prediction-item.medium {
            border-left-color: #f39c12;
        }
        
        .prediction-item.low {
            border-left-color: #27ae60;
        }
        
        .prediction-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .prediction-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .prediction-time {
            background: #ecf0f1;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
            color: #7f8c8d;
        }
        
        .prediction-details {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .category-badge {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .category-badge.high {
            background: #e74c3c;
            color: white;
        }
        
        .category-badge.medium {
            background: #f39c12;
            color: white;
        }
        
        .category-badge.low {
            background: #27ae60;
            color: white;
        }
        
        .confidence {
            font-size: 0.9em;
            color: #7f8c8d;
        }
        
        .recommendation-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
        }
        
        .recommendation-section h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .recommendation-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .main-recommendation {
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 15px;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .rec-play {
            background: #2ecc71;
            color: white;
        }
        
        .rec-wait {
            background: #f39c12;
            color: white;
        }
        
        .rec-no-play {
            background: #e74c3c;
            color: white;
        }
        
        .risk-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .risk-badge {
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }
        
        .risk-low {
            background: #2ecc71;
            color: white;
        }
        
        .risk-medium {
            background: #f39c12;
            color: white;
        }
        
        .risk-high {
            background: #e74c3c;
            color: white;
        }
        
        .strategies {
            margin-top: 20px;
        }
        
        .strategy-item {
            background: #ecf0f1;
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 8px;
            border-left: 3px solid #3498db;
        }
        
        .controls {
            text-align: center;
            padding: 20px;
            border-top: 1px solid #ecf0f1;
        }
        
        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            font-size: 1em;
            cursor: pointer;
            margin: 0 10px;
            transition: all 0.3s;
        }
        
        .btn:hover {
            background: #2980b9;
            transform: translateY(-2px);
        }
        
        .btn-refresh {
            background: #e74c3c;
        }
        
        .btn-refresh:hover {
            background: #c0392b;
        }
        
        .loading {
            text-align: center;
            padding: 50px;
            color: #7f8c8d;
        }
        
        .loading-spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .last-update {
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 20px;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .prediction-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 JetX Tahmin Sistemi</h1>
            <p>Ultimate AI ile Gelecekteki 5 Tahmin + Oyun Önerisi</p>
        </div>
        
        <div class="main-content">
            <div class="predictions-section">
                <h2>🔮 Gelecekteki 5 Tahmin</h2>
                <div id="predictions-container">
                    <div class="loading">
                        <div class="loading-spinner"></div>
                        <p>Tahminler yükleniyor...</p>
                    </div>
                </div>
            </div>
            
            <div class="recommendation-section">
                <h2>🎯 Oyun Önerisi</h2>
                <div id="recommendation-container">
                    <div class="loading">
                        <div class="loading-spinner"></div>
                        <p>Öneri analizi yapılıyor...</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="loadPredictions()">🔄 Yenile</button>
            <button class="btn btn-refresh" onclick="forceRefresh()">⚡ Zorla Yenile</button>
            <div class="last-update" id="last-update"></div>
        </div>
    </div>

    <script>
        let autoRefreshInterval;
        
        function loadPredictions() {
            fetch('/api/latest')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        displayPredictions(data.data.predictions);
                        displayRecommendation(data.data.recommendation);
                        updateLastUpdateTime(data.data.last_update);
                    } else {
                        console.error('Error:', data.error);
                        forceRefresh();
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    forceRefresh();
                });
        }
        
        function forceRefresh() {
            // Show loading
            document.getElementById('predictions-container').innerHTML = `
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <p>Yeni tahminler üretiliyor...</p>
                </div>
            `;
            
            document.getElementById('recommendation-container').innerHTML = `
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <p>Yeni analiz yapılıyor...</p>
                </div>
            `;
            
            fetch('/api/refresh')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        displayPredictions(data.data.predictions);
                        displayRecommendation(data.data.recommendation);
                        updateLastUpdateTime(data.data.last_update);
                    } else {
                        alert('Hata: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Bağlantı hatası!');
                });
        }
        
        function displayPredictions(predictions) {
            const container = document.getElementById('predictions-container');
            
            const html = predictions.map((pred, index) => {
                const categoryClass = pred.category_prediction.toLowerCase();
                const emoji = {
                    'low': '📉',
                    'medium': '📊',
                    'high': '📈'
                }[categoryClass] || '📊';
                
                const time = new Date(pred.timestamp).toLocaleTimeString('tr-TR', {
                    hour: '2-digit',
                    minute: '2-digit'
                });
                
                return `
                    <div class="prediction-item ${categoryClass}">
                        <div class="prediction-header">
                            <div class="prediction-value">${emoji} ${pred.predicted_value.toFixed(2)}x</div>
                            <div class="prediction-time">${time}</div>
                        </div>
                        <div class="prediction-details">
                            <span class="category-badge ${categoryClass}">${pred.category_prediction}</span>
                            <span class="confidence">Güven: ${(pred.confidence_score * 100).toFixed(0)}%</span>
                        </div>
                    </div>
                `;
            }).join('');
            
            container.innerHTML = html;
        }
        
        function displayRecommendation(recommendation) {
            const container = document.getElementById('recommendation-container');
            
            // Recommendation type için class belirle
            let recClass = 'rec-wait';
            if (recommendation.recommendation.includes('OYNA')) {
                recClass = 'rec-play';
            } else if (recommendation.recommendation.includes('OYNAMA')) {
                recClass = 'rec-no-play';
            }
            
            // Risk level için class belirle
            const riskClass = {
                'DÜŞÜK': 'risk-low',
                'ORTA': 'risk-medium',
                'YÜKSEK': 'risk-high'
            }[recommendation.risk_level] || 'risk-medium';
            
            const html = `
                <div class="recommendation-card">
                    <div class="main-recommendation ${recClass}">
                        ${recommendation.recommendation}
                    </div>
                    
                    <div class="risk-indicator">
                        <span>Risk Seviyesi:</span>
                        <span class="risk-badge ${riskClass}">${recommendation.risk_level}</span>
                        <span>(${recommendation.risk_score.toFixed(0)}/100)</span>
                    </div>
                    
                    <div style="margin-bottom: 15px;">
                        <strong>📊 Sonraki Tahmin:</strong><br>
                        Değer: <strong>${recommendation.next_prediction.value.toFixed(2)}x</strong> | 
                        Kategori: <strong>${recommendation.next_prediction.category}</strong> | 
                        Güven: <strong>${(recommendation.next_prediction.confidence * 100).toFixed(0)}%</strong>
                    </div>
                    
                    <div style="margin-bottom: 15px;">
                        <strong>📝 Analiz Sonuçları:</strong>
                        <ul style="margin-left: 20px; margin-top: 5px;">
                            ${recommendation.reasons.map(reason => `<li>${reason}</li>`).join('')}
                        </ul>
                    </div>
                    
                    <div class="strategies">
                        <strong>🎲 Strateji Önerileri:</strong>
                        ${recommendation.strategy.map(strategy => `
                            <div class="strategy-item">${strategy}</div>
                        `).join('')}
                    </div>
                </div>
            `;
            
            container.innerHTML = html;
        }
        
        function updateLastUpdateTime(timestamp) {
            if (timestamp) {
                const date = new Date(timestamp);
                const timeStr = date.toLocaleString('tr-TR');
                document.getElementById('last-update').textContent = `Son güncelleme: ${timeStr}`;
            }
        }
        
        // Sayfa yüklendiğinde tahminleri al
        document.addEventListener('DOMContentLoaded', function() {
            loadPredictions();
            
            // Her 30 saniyede bir otomatik yenile
            autoRefreshInterval = setInterval(loadPredictions, 30000);
        });
        
        // Sayfa kapanırken interval'ı temizle
        window.addEventListener('beforeunload', function() {
            if (autoRefreshInterval) {
                clearInterval(autoRefreshInterval);
            }
        });
    </script>
</body>
</html>
'''

# Template dosyasını oluştur
import os
templates_dir = 'templates'
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)

with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
    f.write(HTML_TEMPLATE)

if __name__ == '__main__':
    print("🌐 Web arayüzü başlatılıyor...")
    print("📍 Adres: http://localhost:5000")
    print("🔄 Otomatik güncelleme: 30 saniye")
    print("⚡ Background güncelleme: 2 dakika")
    
    app.run(debug=True, host='0.0.0.0', port=5000)