import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from enhanced_predictor_v3 import UltimateJetXPredictor


class GameRecommendationEngine:
    """
    Oyun önerisi motoru - risk analizi ve stratejik öneriler
    """
    
    def __init__(self):
        self.risk_factors = {
            'consecutive_losses': 0,
            'recent_volatility': 0.0,
            'prediction_confidence': 0.0,
            'pattern_strength': 0.0
        }
        
    def analyze_risk(self, predictions, recent_results=None):
        """
        Risk analizi yap
        """
        if not predictions:
            return {'risk_level': 'HIGH', 'recommendation': 'BEKLE'}
        
        # Confidence analizi
        avg_confidence = np.mean([p.get('confidence_score', 0) for p in predictions])
        
        # Pattern consistency
        categories = [p.get('category_prediction', 'MEDIUM') for p in predictions]
        pattern_consistency = len(set(categories)) / len(categories)  # Düşük = consistent
        
        # Volatility analizi
        values = [p.get('predicted_value', 2.0) for p in predictions]
        volatility = np.std(values) if len(values) > 1 else 0
        
        # Risk skoru hesapla
        risk_score = self._calculate_risk_score(
            avg_confidence, pattern_consistency, volatility, recent_results
        )
        
        return {
            'risk_score': risk_score,
            'risk_level': self._get_risk_level(risk_score),
            'recommendation': self._get_recommendation(risk_score, predictions),
            'confidence': avg_confidence,
            'pattern_consistency': 1 - pattern_consistency,
            'volatility': volatility
        }
    
    def _calculate_risk_score(self, confidence, pattern_consistency, volatility, recent_results):
        """Risk skoru hesapla (0-100, yüksek = riskli)"""
        
        # Base risk from confidence (0-40 points)
        confidence_risk = (1 - confidence) * 40
        
        # Pattern inconsistency risk (0-30 points)
        pattern_risk = pattern_consistency * 30
        
        # Volatility risk (0-20 points)  
        volatility_risk = min(volatility / 10.0, 1.0) * 20
        
        # Recent results risk (0-10 points)
        recent_risk = 0
        if recent_results:
            recent_losses = sum(1 for r in recent_results[-5:] if not r.get('success', True))
            recent_risk = min(recent_losses / 5.0, 1.0) * 10
        
        total_risk = confidence_risk + pattern_risk + volatility_risk + recent_risk
        return min(total_risk, 100)
    
    def _get_risk_level(self, risk_score):
        """Risk seviyesi belirle"""
        if risk_score < 30:
            return 'DÜŞÜK'
        elif risk_score < 60:
            return 'ORTA'
        else:
            return 'YÜKSEK'
    
    def _get_recommendation(self, risk_score, predictions):
        """Oyun önerisi ver"""
        if not predictions:
            return 'BEKLE'
        
        # İlk tahmin analizi
        first_pred = predictions[0]
        category = first_pred.get('category_prediction', 'MEDIUM')
        confidence = first_pred.get('confidence_score', 0)
        
        if risk_score < 25 and confidence > 0.7:
            if category == 'HIGH':
                return 'OYNA - Yüksek değer fırsatı!'
            elif category == 'LOW':
                return 'OYNA - Düşük risk, erken çık!'
            else:
                return 'OYNA - Güvenli tahmin'
        
        elif risk_score < 40 and confidence > 0.6:
            return 'DİKKATLİ OYNA - Düşük miktar'
        
        elif risk_score < 60:
            return 'BEKLE - Belirsizlik var'
        
        else:
            return 'OYNAMA - Yüksek risk!'


class PredictionInterface:
    """
    Kullanıcı arayüzü için tahmin sistemi
    """
    
    def __init__(self):
        self.predictor = UltimateJetXPredictor()
        self.recommendation_engine = GameRecommendationEngine()
        self.prediction_history = []
        
    def generate_next_5_predictions(self, recent_values=None):
        """
        Gelecekteki 5 tahmin üret
        """
        print("🔮 Gelecek 5 Tahmin Üretiliyor...")
        
        if recent_values is None:
            recent_values = self.predictor._load_recent_data(limit=200)
        
        if len(recent_values) < 50:
            return None
        
        predictions = []
        current_sequence = recent_values.copy()
        
        for i in range(5):
            print(f"  -> Tahmin {i+1}/5 hesaplanıyor...")
            
            # Mevcut sequence ile tahmin yap
            result = self.predictor.predict_ultimate(current_sequence)
            
            if result:
                # Tahmin sonucunu temizle ve kaydet
                clean_prediction = {
                    'sequence_number': i + 1,
                    'predicted_value': result['predicted_value'],
                    'category_prediction': result['category_prediction'],
                    'decision_text': result['decision_text'],
                    'confidence_score': result['confidence_score'],
                    'above_threshold_probability': result['above_threshold_probability'],
                    'decision_source': result['decision_source'],
                    'timestamp': datetime.now() + timedelta(minutes=i*2)  # Her 2 dakikada bir
                }
                
                predictions.append(clean_prediction)
                
                # Sequence'i güncelle (tahmin edilen değeri ekle)
                current_sequence.append(result['predicted_value'])
                if len(current_sequence) > 200:  # Sequence boyutunu sınırla
                    current_sequence = current_sequence[-200:]
            else:
                # Hata durumunda varsayılan tahmin
                predictions.append({
                    'sequence_number': i + 1,
                    'predicted_value': np.mean(current_sequence[-10:]),
                    'category_prediction': 'MEDIUM',
                    'decision_text': 'Belirsiz',
                    'confidence_score': 0.3,
                    'above_threshold_probability': 0.5,
                    'decision_source': 'Fallback',
                    'timestamp': datetime.now() + timedelta(minutes=i*2)
                })
        
        return predictions
    
    def get_game_recommendation(self, predictions=None, recent_results=None):
        """
        Oyun önerisi al
        """
        if predictions is None:
            predictions = self.generate_next_5_predictions()
        
        if not predictions:
            return {
                'recommendation': 'BEKLE - Tahmin üretilemedi',
                'risk_level': 'YÜKSEK',
                'reason': 'Sistem hatası'
            }
        
        # Risk analizi
        risk_analysis = self.recommendation_engine.analyze_risk(predictions, recent_results)
        
        # Detaylı öneri oluştur
        detailed_recommendation = self._create_detailed_recommendation(
            predictions, risk_analysis
        )
        
        return detailed_recommendation
    
    def _create_detailed_recommendation(self, predictions, risk_analysis):
        """
        Detaylı oyun önerisi oluştur
        """
        first_pred = predictions[0]
        
        # Ana öneri
        main_recommendation = risk_analysis['recommendation']
        
        # Sebep analizi
        reasons = []
        
        # Confidence analizi
        if risk_analysis['confidence'] > 0.7:
            reasons.append("✅ Yüksek güven seviyesi")
        elif risk_analysis['confidence'] < 0.4:
            reasons.append("❌ Düşük güven seviyesi")
        
        # Pattern analizi
        if risk_analysis['pattern_consistency'] > 0.7:
            reasons.append("✅ Tutarlı pattern")
        elif risk_analysis['pattern_consistency'] < 0.3:
            reasons.append("❌ Belirsiz pattern")
        
        # Volatility analizi
        if risk_analysis['volatility'] < 2.0:
            reasons.append("✅ Düşük volatilite")
        elif risk_analysis['volatility'] > 5.0:
            reasons.append("❌ Yüksek volatilite")
        
        # Kategori analizi
        category = first_pred.get('category_prediction', 'MEDIUM')
        if category == 'HIGH':
            reasons.append("🚀 Yüksek değer potansiyeli")
        elif category == 'LOW':
            reasons.append("⚠️ Düşük değer riski")
        
        # Strateji önerisi
        strategy = self._get_strategy_advice(predictions, risk_analysis)
        
        return {
            'recommendation': main_recommendation,
            'risk_level': risk_analysis['risk_level'],
            'risk_score': risk_analysis['risk_score'],
            'confidence': risk_analysis['confidence'],
            'reasons': reasons,
            'strategy': strategy,
            'next_prediction': {
                'value': first_pred['predicted_value'],
                'category': first_pred['category_prediction'],
                'confidence': first_pred['confidence_score']
            }
        }
    
    def _get_strategy_advice(self, predictions, risk_analysis):
        """
        Strateji tavsiyesi ver
        """
        first_pred = predictions[0]
        category = first_pred.get('category_prediction', 'MEDIUM')
        confidence = first_pred.get('confidence_score', 0)
        
        strategies = []
        
        if category == 'HIGH' and confidence > 0.6:
            strategies.extend([
                "🎯 Hedef: 5x-15x arası çıkış yapın",
                "💰 Risk: Orta miktar yatırım",
                "⏰ Timing: Hızla yükselişi bekleyin"
            ])
            
        elif category == 'LOW' and confidence > 0.6:
            strategies.extend([
                "🎯 Hedef: 1.2x-1.4x erken çıkış",
                "💰 Risk: Düşük miktar, hızlı çıkış",
                "⏰ Timing: İlk 5-10 saniyede çıkın"
            ])
            
        elif category == 'MEDIUM':
            strategies.extend([
                "🎯 Hedef: 2x-4x arası güvenli çıkış",
                "💰 Risk: Normal miktar",
                "⏰ Timing: Trend takip edin"
            ])
        
        # Risk seviyesine göre ek öneriler
        if risk_analysis['risk_level'] == 'YÜKSEK':
            strategies.append("🚨 Bu turda oynamayın veya çok düşük miktar kullanın")
        elif risk_analysis['risk_level'] == 'DÜŞÜK':
            strategies.append("✅ Güvenli oyun fırsatı")
        
        return strategies
    
    def display_prediction_interface(self):
        """
        Tahmin arayüzünü göster
        """
        print("\n" + "="*70)
        print("🎮 JETX TAHMİN VE OYUN ÖNERİSİ SİSTEMİ")
        print("="*70)
        
        # 5 tahmin üret
        predictions = self.generate_next_5_predictions()
        
        if not predictions:
            print("❌ Tahmin üretilemedi!")
            return
        
        # Tahminleri göster
        print("\n🔮 GELECEKTEKİ 5 TAHMİN:")
        print("-" * 70)
        
        for i, pred in enumerate(predictions, 1):
            time_str = pred['timestamp'].strftime("%H:%M")
            category_emoji = {
                'LOW': '📉',
                'MEDIUM': '📊', 
                'HIGH': '📈'
            }.get(pred['category_prediction'], '📊')
            
            print(f"{i}. {time_str} | {category_emoji} {pred['predicted_value']:.2f}x | "
                  f"{pred['category_prediction']} | Güven: {pred['confidence_score']:.2f}")
        
        # Oyun önerisi al
        recommendation = self.get_game_recommendation(predictions)
        
        # Önerileri göster
        print("\n🎯 OYUN ÖNERİSİ:")
        print("-" * 70)
        print(f"📋 Ana Öneri: {recommendation['recommendation']}")
        print(f"⚠️ Risk Seviyesi: {recommendation['risk_level']} ({recommendation['risk_score']:.0f}/100)")
        print(f"💪 Güven Seviyesi: {recommendation['confidence']:.2f}")
        
        print(f"\n📊 Sonraki Tahmin:")
        next_pred = recommendation['next_prediction']
        print(f"   Değer: {next_pred['value']:.2f}x")
        print(f"   Kategori: {next_pred['category']}")
        print(f"   Güven: {next_pred['confidence']:.2f}")
        
        print(f"\n📝 Analiz Sonuçları:")
        for reason in recommendation['reasons']:
            print(f"   {reason}")
        
        print(f"\n🎲 Strateji Önerileri:")
        for strategy in recommendation['strategy']:
            print(f"   {strategy}")
        
        # Özet tablo
        self._display_summary_table(predictions, recommendation)
        
        return {
            'predictions': predictions,
            'recommendation': recommendation
        }
    
    def _display_summary_table(self, predictions, recommendation):
        """
        Özet tablo göster
        """
        print("\n📋 ÖZET TABLO:")
        print("-" * 70)
        
        # Risk analizi
        risk_color = {
            'DÜŞÜK': '🟢',
            'ORTA': '🟡', 
            'YÜKSEK': '🔴'
        }.get(recommendation['risk_level'], '⚪')
        
        print(f"Risk Durumu: {risk_color} {recommendation['risk_level']}")
        
        # Kategori dağılımı
        categories = [p['category_prediction'] for p in predictions]
        low_count = categories.count('LOW')
        medium_count = categories.count('MEDIUM') 
        high_count = categories.count('HIGH')
        
        print(f"Tahmin Dağılımı: 📉 {low_count} | 📊 {medium_count} | 📈 {high_count}")
        
        # Confidence ortalaması
        avg_conf = np.mean([p['confidence_score'] for p in predictions])
        conf_emoji = '🔥' if avg_conf > 0.7 else '⚡' if avg_conf > 0.5 else '❓'
        print(f"Ortalama Güven: {conf_emoji} {avg_conf:.2f}")
        
        # Final tavsiye
        if 'OYNA' in recommendation['recommendation']:
            print(f"\n🎮 SONUÇ: {recommendation['recommendation']}")
        else:
            print(f"\n⏸️ SONUÇ: {recommendation['recommendation']}")
    
    def continuous_monitoring(self, interval_minutes=2):
        """
        Sürekli monitoring modu
        """
        print(f"🔄 Sürekli monitoring başlatıldı (Her {interval_minutes} dakika)")
        
        try:
            while True:
                print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Güncelleme")
                self.display_prediction_interface()
                
                print(f"\n⏳ {interval_minutes} dakika bekleniyor...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring durduruldu.")


class GameSession:
    """
    Oyun oturumu yönetimi
    """
    
    def __init__(self):
        self.interface = PredictionInterface()
        self.session_results = []
        self.start_time = datetime.now()
        
    def start_session(self):
        """Oyun oturumu başlat"""
        print("🎮 Oyun Oturumu Başlatıldı!")
        print(f"🕐 Başlangıç: {self.start_time.strftime('%H:%M:%S')}")
        
        while True:
            print("\n" + "="*50)
            print("MENÜ:")
            print("1. 5 Tahmin Göster + Öneri Al")
            print("2. Sürekli Monitoring")
            print("3. Oturum İstatistikleri")
            print("4. Çıkış")
            
            choice = input("\nSeçiminiz (1-4): ").strip()
            
            if choice == '1':
                result = self.interface.display_prediction_interface()
                if result:
                    self.session_results.append({
                        'timestamp': datetime.now(),
                        'predictions': result['predictions'],
                        'recommendation': result['recommendation']
                    })
                
            elif choice == '2':
                interval = input("Güncelleme aralığı (dakika, varsayılan 2): ").strip()
                interval = int(interval) if interval.isdigit() else 2
                self.interface.continuous_monitoring(interval)
                
            elif choice == '3':
                self.show_session_stats()
                
            elif choice == '4':
                print("👋 Oyun oturumu sona erdi!")
                break
                
            else:
                print("❌ Geçersiz seçim!")
    
    def show_session_stats(self):
        """Oturum istatistiklerini göster"""
        if not self.session_results:
            print("📊 Henüz tahmin yapılmadı.")
            return
        
        print("\n📊 OTURUM İSTATİSTİKLERİ:")
        print("-" * 50)
        
        duration = datetime.now() - self.start_time
        print(f"⏱️ Oturum Süresi: {duration}")
        print(f"🔢 Toplam Tahmin: {len(self.session_results)}")
        
        # Öneri dağılımı
        recommendations = [r['recommendation']['recommendation'] for r in self.session_results]
        play_count = sum(1 for r in recommendations if 'OYNA' in r)
        wait_count = sum(1 for r in recommendations if 'BEKLE' in r or 'OYNAMA' in r)
        
        print(f"🎮 Oyna Önerisi: {play_count}")
        print(f"⏸️ Bekle/Oynama: {wait_count}")
        
        # Risk seviyesi dağılımı
        risk_levels = [r['recommendation']['risk_level'] for r in self.session_results]
        for level in ['DÜŞÜK', 'ORTA', 'YÜKSEK']:
            count = risk_levels.count(level)
            print(f"🎯 {level} Risk: {count}")


def main():
    """Ana fonksiyon"""
    print("🚀 JetX Tahmin ve Oyun Önerisi Sistemi")
    
    try:
        session = GameSession()
        session.start_session()
        
    except Exception as e:
        print(f"❌ Sistem hatası: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()