#!/usr/bin/env python3
"""
Performance Reporting Utility for JetX Prediction System
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any

def explain_test_results(metrics: Dict[str, float]) -> str:
    """
    Test sonuçlarını Türkçe ve anlaşılır şekilde açıklar.
    
    Args:
        metrics: Test metriklerini içeren dictionary
        
    Returns:
        Açıklayıcı metin
    """
    explanation = []
    
    # Header
    explanation.append("=" * 80)
    explanation.append("🎯 TEST SONUÇLARI DETAYLI AÇIKLAMA")
    explanation.append("=" * 80)
    
    # Ana metrikler
    threshold_acc = metrics.get('threshold_accuracy', 0)
    balanced_acc = metrics.get('balanced_accuracy', 0)
    precision = metrics.get('precision', 0)
    recall = metrics.get('recall', 0)
    f1 = metrics.get('f1', 0)
    
    # Confusion matrix değerleri
    tn = metrics.get('true_negative', 0)
    fp = metrics.get('false_positive', 0)
    fn = metrics.get('false_negative', 0)
    tp = metrics.get('true_positive', 0)
    
    total_tests = tn + fp + fn + tp
    
    explanation.append(f"\n📊 GENEL PERFORMANS:")
    explanation.append(f"   • Test edilen örnek sayısı: {int(total_tests)}")
    explanation.append(f"   • Genel doğruluk: %{threshold_acc*100:.1f}")
    explanation.append(f"   • F1-Score: %{f1*100:.1f} (Genel kalite göstergesi)")
    
    # Model davranış analizi
    explanation.append(f"\n🤖 MODEL DAVRANIŞI ANALİZİ:")
    
    if recall > 0.95:
        explanation.append(f"   ⚠️  MODEL ÇOK AGRESİF! Neredeyse her durumda 'OYNA' diyor (%{recall*100:.1f})")
        explanation.append(f"   📈 Bu iyi: Kazanç fırsatlarının çoğunu yakalıyor")
        explanation.append(f"   📉 Bu kötü: Çok fazla yanlış alarm üretiyor")
    elif recall > 0.8:
        explanation.append(f"   ✅ Model kazanç fırsatlarını iyi yakalıyor (%{recall*100:.1f})")
    else:
        explanation.append(f"   ⚠️  Model çok fazla fırsat kaçırıyor (%{recall*100:.1f})")
    
    if balanced_acc < 0.5:
        explanation.append(f"   🚨 ÖNEMLI: Balanced Accuracy (%{balanced_acc*100:.1f}) rastgele tahminden kötü!")
        explanation.append(f"   💡 Model bir sınıfa (muhtemelen 'OYNA') çok yanlı")
    
    # Detaylı sonuç analizi
    explanation.append(f"\n📋 DETAYLI SONUÇ ANALİZİ:")
    explanation.append(f"   ✅ Doğru 'OYNA' tavsiyeleri: {int(tp)} adet")
    explanation.append(f"   ❌ Yanlış 'OYNA' tavsiyeleri: {int(fp)} adet (Para kaybı riski!)")
    explanation.append(f"   ❌ Kaçırılan fırsatlar: {int(fn)} adet (Missed opportunities)")
    explanation.append(f"   ✅ Doğru 'OYNAMA' tavsiyeleri: {int(tn)} adet")
    
    # Risk analizi
    if fp > 0 and tp > 0:
        risk_ratio = fp / (tp + fp)
        explanation.append(f"\n⚖️  RİSK ANALİZİ:")
        explanation.append(f"   • Model 'OYNA' dediğinde %{(1-risk_ratio)*100:.1f} ihtimalle doğru")
        explanation.append(f"   • Model 'OYNA' dediğinde %{risk_ratio*100:.1f} ihtimalle yanlış (RİSK!)")
        
        if risk_ratio > 0.4:
            explanation.append(f"   🚨 UYARI: Çok yüksek risk oranı! Canlı kullanım için tehlikeli")
        elif risk_ratio > 0.2:
            explanation.append(f"   ⚠️  Orta seviye risk. Dikkatli kullanım gerekli")
        else:
            explanation.append(f"   ✅ Kabul edilebilir risk seviyesi")
    
    # Trading perspektifi
    explanation.append(f"\n💰 TRADİNG PERSPEKTİFİ:")
    if tn == 0:
        explanation.append(f"   🚨 KRİTİK: Model hiç 'OYNAMA' tavsiyesi vermiyor!")
        explanation.append(f"   📊 Bu durumda model sadece agresif bir 'her zaman oyna' stratejisi")
        explanation.append(f"   ⚠️  Gerçek bir tahmin sistemi değil, dikkatli olun!")
    else:
        explanation.append(f"   ✅ Model hem 'OYNA' hem 'OYNAMA' tavsiyeleri veriyor")
    
    # MAE ve RMSE açıklaması
    mae = metrics.get('mae', 0)
    rmse = metrics.get('rmse', 0)
    
    explanation.append(f"\n📐 TAHMIN HATASI ANALİZİ:")
    explanation.append(f"   • Ortalama hata (MAE): {mae:.2f} birim")
    explanation.append(f"   • Büyük hatalar (RMSE): {rmse:.2f} birim")
    
    if rmse > mae * 2:
        explanation.append(f"   ⚠️  RMSE çok yüksek: Model bazen çok büyük hatalar yapıyor")
    else:
        explanation.append(f"   ✅ Hatalar genel olarak tutarlı")
    
    # Öneri bölümü
    explanation.append(f"\n💡 ÖNERİLER:")
    
    if balanced_acc < 0.5:
        explanation.append(f"   1. Model çok yanlı - class balancing gerekli")
        explanation.append(f"   2. Threshold değerini ayarlayın (şu an 1.5)")
        explanation.append(f"   3. Loss function'ı false positive'leri cezalandıracak şekilde düzenleyin")
    
    if recall > 0.95 and precision < 0.8:
        explanation.append(f"   4. Model çok agresif - daha muhafazakar olmalı")
        explanation.append(f"   5. Precision/Recall dengesini ayarlayın")
    
    if tn == 0:
        explanation.append(f"   6. 🚨 EN ÖNEMLİSİ: Model 'OYNAMA' öğrenemiyor - ciddi problem!")
        explanation.append(f"   7. Veri dengesizliği var - negatif örnekleri artırın")
    
    explanation.append(f"\n⭐ GENEL DEĞERLENDİRME:")
    if f1 > 0.8 and balanced_acc > 0.6:
        explanation.append(f"   ✅ Model genel olarak iyi performans gösteriyor")
    elif f1 > 0.6 and balanced_acc > 0.5:
        explanation.append(f"   🟡 Model orta seviye - iyileştirilebilir")
    else:
        explanation.append(f"   🔴 Model performansı yetersiz - ciddi revizyon gerekli")
        explanation.append(f"   📝 Özellikle canlı trading için henüz kullanıma hazır değil")
    
    explanation.append("=" * 80)
    
    return "\n".join(explanation)

def generate_text_report(results: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Generates a text-based summary report from training results.

    Args:
        results: A dictionary containing results for each model.

    Returns:
        A formatted string with the summary report.
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("📊 JETX MODEL TRAINING - FINAL PERFORMANCE REPORT")
    report_lines.append(f"Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)

    if not results:
        report_lines.append("Eğitim sonucu bulunamadı.")
        return "\n".join(report_lines)

    # Find the best overall model
    best_overall_model = None
    best_f1 = -1.0

    for model_name, model_results in results.items():
        if not model_results:
            continue
        
        # En iyi döngüyü F1 skoruna göre bul
        best_cycle = max(model_results, key=lambda x: x['performance'].get('f1', 0))
        if best_cycle['performance'].get('f1', 0) > best_f1:
            best_f1 = best_cycle['performance']['f1']
            best_overall_model = {
                'name': model_name,
                'f1': best_f1,
                'recall': best_cycle['performance'].get('recall', 0),
                'precision': best_cycle['performance'].get('precision', 0),
                'mae': best_cycle['performance'].get('mae', 0)
            }

    if best_overall_model:
        report_lines.append("\n🏆 EN İYİ MODEL (F1-Skoruna Göre)")
        report_lines.append(f"   - Model: {best_overall_model['name']}")
        report_lines.append(f"   - F1-Skoru: {best_overall_model['f1']:.4f}")
        report_lines.append(f"   - Recall (Duyarlılık): {best_overall_model['recall']:.4f} (Crash'leri yakalama oranı)")
        report_lines.append(f"   - Precision (Kesinlik): {best_overall_model['precision']:.4f}")
        report_lines.append(f"   - Ortalama Mutlak Hata (MAE): {best_overall_model['mae']:.4f}")

    report_lines.append("\n" + "-" * 60)
    report_lines.append("MODEL DETAYLARI")
    report_lines.append("-" * 60)

    for model_name, model_results in results.items():
        if not model_results:
            report_lines.append(f"\n--- {model_name} ---")
            report_lines.append("  Başarılı döngü bulunamadı.")
            continue

        best_cycle = max(model_results, key=lambda x: x['performance'].get('f1', 0))
        avg_f1 = np.mean([r['performance'].get('f1', 0) for r in model_results])
        avg_recall = np.mean([r['performance'].get('recall', 0) for r in model_results])
        avg_mae = np.mean([r['performance'].get('mae', 0) for r in model_results])

        report_lines.append(f"\n--- {model_name} ---")
        report_lines.append(f"  - Toplam Döngü: {len(model_results)}")
        report_lines.append(f"  - Ortalama F1-Skoru: {avg_f1:.4f}")
        report_lines.append(f"  - Ortalama Recall: {avg_recall:.4f}")
        report_lines.append(f"  - Ortalama MAE: {avg_mae:.4f}")
        report_lines.append(f"  - En İyi Döngü (Cycle {best_cycle['cycle']}):")
        report_lines.append(f"    - F1-Skoru: {best_cycle['performance'].get('f1', 0):.4f}")
        report_lines.append(f"    - Recall: {best_cycle['performance'].get('recall', 0):.4f}")
        report_lines.append(f"    - MAE: {best_cycle['performance'].get('mae', 0):.4f}")
        report_lines.append(f"    - Model Yolu: {os.path.basename(best_cycle['model_path'])}")

    return "\n".join(report_lines)

def generate_performance_plot(results: Dict[str, List[Dict[str, Any]]], save_path: str) -> bool:
    """
    Generates and saves a bar plot comparing model performances.

    Args:
        results: A dictionary containing results for each model.
        save_path: Path to save the plot image.

    Returns:
        True if the plot was saved successfully, False otherwise.
    """
    if not results:
        return False

    performance_data = []
    for model_name, model_results in results.items():
        if model_results:
            best_cycle = max(model_results, key=lambda x: x['performance'].get('f1', 0))
            performance_data.append({
                'Model': model_name,
                'MAE': best_cycle['performance'].get('mae', 0),
                'F1-Score': best_cycle['performance'].get('f1', 0),
                'Recall': best_cycle['performance'].get('recall', 0)
            })

    if not performance_data:
        return False

    df = pd.DataFrame(performance_data)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Model Performans Karşılaştırması (En İyi Döngüler)', fontsize=16, weight='bold')

    # MAE Plot (lower is better)
    sns.barplot(x='Model', y='MAE', data=df.sort_values('MAE', ascending=True), ax=axes[0], palette='viridis')
    axes[0].set_title('Ortalama Mutlak Hata (MAE) - Düşük olan daha iyi')
    axes[0].set_ylabel('MAE')
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.4f')

    # F1-Score Plot (higher is better)
    sns.barplot(x='Model', y='F1-Score', data=df.sort_values('F1-Score', ascending=False), ax=axes[1], palette='plasma')
    axes[1].set_title('F1-Skoru - Yüksek olan daha iyi (Recall ve Precision Dengesi)')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_ylim(0, max(1.0, df['F1-Score'].max() * 1.1))
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.4f')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    try:
        plt.savefig(save_path)
        plt.close(fig)
        print(f"✅ Performans grafiği kaydedildi: {save_path}")
        return True
    except Exception as e:
        print(f"❌ Grafik kaydedilemedi: {e}")
        return False
