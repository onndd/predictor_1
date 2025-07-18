#!/usr/bin/env python3
"""
Performance Reporting Utility for JetX Prediction System
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any

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