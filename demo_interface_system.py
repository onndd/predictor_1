#!/usr/bin/env python3
"""
JetX Tahmin ve Oyun Önerisi Sistemi - Kapsamlı Demo
===============================================

Bu demo, ilerideki 5 tahmin gösterimi ve oyun önerisi sisteminin
nasıl kullanılacağını gösterir.

Özellikler:
- Gelecekteki 5 tahmin
- Risk analizi ve oyun önerileri
- Strateji tavsiyeleri
- Konsol ve web arayüzü
- Otomatik güncelleme
"""

import sys
import os
from datetime import datetime
import time

# Import our systems
try:
    from prediction_interface import PredictionInterface, GameSession
    from web_interface import app
    import threading
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    print("🔧 Lütfen önce gerekli dosyaların var olduğundan emin olun.")
    sys.exit(1)


def demo_console_interface():
    """
    Konsol arayüzü demosu
    """
    print("\n" + "="*70)
    print("🖥️ KONSOL ARAYÜZÜ DEMOSU")
    print("="*70)
    
    # Interface oluştur
    interface = PredictionInterface()
    
    print("\n1️⃣ Tek tahmin seti ve öneri:")
    print("-" * 50)
    
    try:
        # Tek seferlik tahmin ve öneri göster
        result = interface.display_prediction_interface()
        
        if result:
            print("\n✅ Tahmin başarıyla üretildi!")
            
            # Özet bilgi
            predictions = result['predictions']
            recommendation = result['recommendation']
            
            print(f"\n📊 ÖZET:")
            print(f"   🔮 5 tahmin üretildi")
            print(f"   🎯 Ana öneri: {recommendation['recommendation']}")
            print(f"   ⚠️ Risk seviyesi: {recommendation['risk_level']}")
            print(f"   💪 Güven skoru: {recommendation['confidence']:.2f}")
            
            # Kategori dağılımı
            categories = [p['category_prediction'] for p in predictions]
            print(f"   📈 HIGH: {categories.count('HIGH')}")
            print(f"   📊 MEDIUM: {categories.count('MEDIUM')}")
            print(f"   📉 LOW: {categories.count('LOW')}")
            
        else:
            print("❌ Tahmin üretilemedi!")
            
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n2️⃣ Interaktif konsol modu başlatmak için:")
    print("-" * 50)
    print("   python prediction_interface.py")
    print("   veya")
    print("   from prediction_interface import GameSession")
    print("   session = GameSession()")
    print("   session.start_session()")


def demo_web_interface():
    """
    Web arayüzü demosu
    """
    print("\n" + "="*70)
    print("🌐 WEB ARAYÜZÜ DEMOSU")
    print("="*70)
    
    print("\n🚀 Web arayüzü özellikleri:")
    print("-" * 50)
    print("✅ Modern, responsive tasarım")
    print("✅ Gerçek zamanlı tahmin güncellemeleri")
    print("✅ Interaktif oyun önerileri")
    print("✅ Risk analizi göstergeleri")
    print("✅ Strateji tavsiyeleri")
    print("✅ Otomatik yenileme (30 saniye)")
    print("✅ Background güncelleme (2 dakika)")
    print("✅ Mobil uyumlu")
    
    print("\n🌐 Web arayüzünü başlatmak için:")
    print("-" * 50)
    print("1. Terminal'de çalıştırın:")
    print("   python web_interface.py")
    print("\n2. Tarayıcıda açın:")
    print("   http://localhost:5000")
    
    print("\n🎮 Web arayüzü fonksiyonları:")
    print("-" * 50)
    print("• 🔄 Yenile - Mevcut cache'den veri al")
    print("• ⚡ Zorla Yenile - Yeni tahmin üret")
    print("• 🔄 Otomatik yenileme - Her 30 saniyede")
    print("• 📱 Mobil uyumlu - Tüm cihazlarda çalışır")
    
    # Web sunucusunu başlatmak isteyip istemediğini sor
    choice = input("\n❓ Web arayüzünü şimdi başlatmak ister misiniz? (e/h): ").strip().lower()
    
    if choice in ['e', 'evet', 'yes', 'y']:
        print("\n🚀 Web sunucusu başlatılıyor...")
        print("📍 Adres: http://localhost:5000")
        print("🛑 Durdurmak için Ctrl+C kullanın\n")
        
        try:
            # Web sunucusunu başlat
            app.run(debug=False, host='0.0.0.0', port=5000)
        except KeyboardInterrupt:
            print("\n🛑 Web sunucusu durduruldu.")
        except Exception as e:
            print(f"❌ Web sunucusu hatası: {e}")
    else:
        print("👍 Web arayüzünü daha sonra başlatabilirsiniz:")
        print("   python web_interface.py")


def demo_api_usage():
    """
    API kullanım demosu
    """
    print("\n" + "="*70)
    print("🔌 API KULLANIM DEMOSU")
    print("="*70)
    
    print("\n📡 API Endpoint'leri:")
    print("-" * 50)
    print("GET /api/latest        - En son tahminleri al (cache)")
    print("GET /api/predictions   - Yeni tahmin üret")
    print("GET /api/refresh       - Zorla yenile")
    
    print("\n📝 API Response Formatı:")
    print("-" * 50)
    
    # Örnek API response
    example_response = {
        "success": True,
        "data": {
            "predictions": [
                {
                    "sequence_number": 1,
                    "predicted_value": 2.45,
                    "category_prediction": "MEDIUM",
                    "confidence_score": 0.75,
                    "timestamp": "2024-01-15T10:30:00"
                }
            ],
            "recommendation": {
                "recommendation": "OYNA - Güvenli tahmin",
                "risk_level": "DÜŞÜK",
                "risk_score": 25,
                "confidence": 0.75,
                "reasons": ["✅ Yüksek güven seviyesi"],
                "strategy": ["🎯 Hedef: 2x-4x arası güvenli çıkış"]
            },
            "last_update": "2024-01-15T10:30:00"
        }
    }
    
    import json
    print(json.dumps(example_response, indent=2, ensure_ascii=False))
    
    print("\n💻 Python'da API kullanımı:")
    print("-" * 50)
    print("""
import requests

# En son tahminleri al
response = requests.get('http://localhost:5000/api/latest')
data = response.json()

if data['success']:
    predictions = data['data']['predictions']
    recommendation = data['data']['recommendation']
    
    print(f"Ana öneri: {recommendation['recommendation']}")
    print(f"Risk seviyesi: {recommendation['risk_level']}")
    
    for pred in predictions:
        print(f"Tahmin: {pred['predicted_value']:.2f}x")
""")
    
    print("\n🌐 JavaScript'te API kullanımı:")
    print("-" * 50)
    print("""
fetch('/api/latest')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const predictions = data.data.predictions;
            const recommendation = data.data.recommendation;
            
            console.log('Ana öneri:', recommendation.recommendation);
            console.log('Risk seviyesi:', recommendation.risk_level);
            
            predictions.forEach(pred => {
                console.log(`Tahmin: ${pred.predicted_value.toFixed(2)}x`);
            });
        }
    });
""")


def demo_system_architecture():
    """
    Sistem mimarisi demosu
    """
    print("\n" + "="*70)
    print("🏗️ SİSTEM MİMARİSİ")
    print("="*70)
    
    print("\n📊 Sistem Bileşenleri:")
    print("-" * 50)
    
    components = [
        "🧠 UltimateJetXPredictor - Ana tahmin sistemi",
        "   ├── 🎯 GeneralSystem - Genel tahminler (1.5-10x)",
        "   ├── 📉 LowValueSpecialist - Düşük değer uzmanı (<1.5)",
        "   └── 📈 HighValueSpecialist - Yüksek değer uzmanı (10x+)",
        "",
        "🎮 PredictionInterface - Konsol arayüzü",
        "   ├── 🔮 5 tahmin üretimi",
        "   ├── 🎯 Oyun önerileri",
        "   ├── 📊 Risk analizi",
        "   └── 🎲 Strateji tavsiyeleri",
        "",
        "🌐 Web Interface - Flask uygulaması",
        "   ├── 🖥️ Modern web arayüzü",
        "   ├── 📡 RESTful API",
        "   ├── 🔄 Otomatik güncelleme",
        "   └── 📱 Mobil uyumlu",
        "",
        "⚙️ GameRecommendationEngine - Öneri motoru",
        "   ├── 📊 Risk analizi",
        "   ├── 🎯 Strateji önerileri",
        "   ├── 💡 Akıllı karar verme",
        "   └── 📈 Performans takibi"
    ]
    
    for component in components:
        print(component)
    
    print("\n🔄 Veri Akışı:")
    print("-" * 50)
    print("1. 📊 Geçmiş veriler yüklenir")
    print("2. 🧠 3 uzman sistem tahmin yapar")
    print("3. 🤖 Akıllı füzyon sistemi birleştirir")
    print("4. 🎯 Risk analizi yapılır")
    print("5. 💡 Oyun önerisi üretilir")
    print("6. 🎲 Strateji tavsiyeleri eklenir")
    print("7. 🖥️ Kullanıcıya sunulur")
    
    print("\n⚡ Performans Özellikleri:")
    print("-" * 50)
    print("✅ Çoklu model ensemble")
    print("✅ Uzmanlaşmış alt sistemler")
    print("✅ Güven tabanlı karar verme")
    print("✅ Gerçek zamanlı işleme")
    print("✅ Otomatik model güncellemesi")
    print("✅ Hata toleransı")


def demo_usage_scenarios():
    """
    Kullanım senaryoları demosu
    """
    print("\n" + "="*70)
    print("📋 KULLANIM SENARYOLARI")
    print("="*70)
    
    scenarios = [
        {
            "title": "🎮 Günlük Oyuncu",
            "description": "Her gün birkaç oyun oynayan kullanıcı",
            "usage": [
                "• Web arayüzünü aç",
                "• 5 tahmin ve öneriyi incele",
                "• Risk seviyesine göre karar ver",
                "• Strateji önerilerini takip et"
            ]
        },
        {
            "title": "📊 Profesyonel Analiz",
            "description": "Detaylı analiz yapan ileri seviye kullanıcı",
            "usage": [
                "• API'yi kullanarak veri al",
                "• Konsol arayüzünde detaylı inceleme",
                "• Sürekli monitoring modu",
                "• Özel stratejiler geliştir"
            ]
        },
        {
            "title": "🔄 Otomatik Sistem",
            "description": "Bot veya otomatik karar sistemi",
            "usage": [
                "• API endpoint'lerini kullan",
                "• JSON formatında veri al",
                "• Risk skorlarını değerlendir",
                "• Otomatik karar algoritması"
            ]
        },
        {
            "title": "📱 Mobil Kullanım",
            "description": "Telefon/tablet ile erişim",
            "usage": [
                "• Responsive web arayüzü",
                "• Dokunmatik optimizasyon",
                "• Hızlı karar verme",
                "• Basit arayüz"
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['title']}")
        print(f"📝 {scenario['description']}")
        print("💡 Kullanım:")
        for usage in scenario['usage']:
            print(f"   {usage}")


def main_demo():
    """
    Ana demo fonksiyonu
    """
    print("🚀 JetX TAHMİN VE OYUN ÖNERİSİ SİSTEMİ")
    print("=" * 70)
    print("📅 Tahmin: Gelecekteki 5 sonuç")
    print("🎯 Öneri: Oynayıp oynamama kararı")
    print("💡 Strateji: Detaylı oyun tavsiyeleri")
    print("📊 Risk: Kapsamlı analiz")
    
    while True:
        print("\n" + "="*50)
        print("DEMO MENÜSÜ:")
        print("1. 🖥️ Konsol Arayüzü Demosu")
        print("2. 🌐 Web Arayüzü Demosu")
        print("3. 🔌 API Kullanım Demosu")
        print("4. 🏗️ Sistem Mimarisi")
        print("5. 📋 Kullanım Senaryoları")
        print("6. 🎮 Canlı Oyun Oturumu")
        print("7. 🌐 Web Sunucusu Başlat")
        print("8. ❌ Çıkış")
        
        choice = input("\nSeçiminiz (1-8): ").strip()
        
        try:
            if choice == '1':
                demo_console_interface()
                
            elif choice == '2':
                demo_web_interface()
                
            elif choice == '3':
                demo_api_usage()
                
            elif choice == '4':
                demo_system_architecture()
                
            elif choice == '5':
                demo_usage_scenarios()
                
            elif choice == '6':
                print("\n🎮 Canlı oyun oturumu başlatılıyor...")
                session = GameSession()
                session.start_session()
                
            elif choice == '7':
                print("\n🌐 Web sunucusu başlatılıyor...")
                print("📍 Adres: http://localhost:5000")
                print("🛑 Durdurmak için Ctrl+C kullanın\n")
                app.run(debug=False, host='0.0.0.0', port=5000)
                
            elif choice == '8':
                print("\n👋 Demo sona erdi!")
                break
                
            else:
                print("❌ Geçersiz seçim! Lütfen 1-8 arası bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n🛑 Demo durduruldu.")
            break
        except Exception as e:
            print(f"\n❌ Hata: {e}")
            import traceback
            traceback.print_exc()
            
        input("\n⏸️ Devam etmek için Enter'a basın...")


if __name__ == "__main__":
    try:
        main_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo kapatıldı!")
    except Exception as e:
        print(f"\n❌ Kritik hata: {e}")
        import traceback
        traceback.print_exc()