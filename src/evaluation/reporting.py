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
    best_mae = float('inf')

    for model_name, model_results in results.items():
        if not model_results:
            continue
        
        best_cycle = min(model_results, key=lambda x: x['performance']['mae'])
        if best_cycle['performance']['mae'] < best_mae:
            best_mae = best_cycle['performance']['mae']
            best_overall_model = {
                'name': model_name,
                'mae': best_mae,
                'accuracy': best_cycle['performance']['accuracy']
            }

    if best_overall_model:
        report_lines.append("\n🏆 EN İYİ MODEL")
        report_lines.append(f"   - Model: {best_overall_model['name']}")
        report_lines.append(f"   - Ortalama Mutlak Hata (MAE): {best_overall_model['mae']:.4f}")
        report_lines.append(f"   - Doğruluk (Accuracy): {best_overall_model['accuracy']:.4f}")

    report_lines.append("\n" + "-" * 60)
    report_lines.append("MODEL DETAYLARI")
    report_lines.append("-" * 60)

    for model_name, model_results in results.items():
        if not model_results:
            report_lines.append(f"\n--- {model_name} ---")
            report_lines.append("  Başarılı döngü bulunamadı.")
            continue

        best_cycle = min(model_results, key=lambda x: x['performance']['mae'])
        avg_mae = sum(r['performance']['mae'] for r in model_results) / len(model_results)
        avg_acc = sum(r['performance']['accuracy'] for r in model_results) / len(model_results)

        report_lines.append(f"\n--- {model_name} ---")
        report_lines.append(f"  - Toplam Döngü: {len(model_results)}")
        report_lines.append(f"  - Ortalama MAE: {avg_mae:.4f}")
        report_lines.append(f"  - Ortalama Doğruluk: {avg_acc:.4f}")
        report_lines.append(f"  - En İyi Döngü (Cycle {best_cycle['cycle']}):")
        report_lines.append(f"    - MAE: {best_cycle['performance']['mae']:.4f}")
        report_lines.append(f"    - Doğruluk: {best_cycle['performance']['accuracy']:.4f}")
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
            best_cycle = min(model_results, key=lambda x: x['performance']['mae'])
            performance_data.append({
                'Model': model_name,
                'MAE': best_cycle['performance']['mae'],
                'Accuracy': best_cycle['performance']['accuracy']
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

    # Accuracy Plot (higher is better)
    sns.barplot(x='Model', y='Accuracy', data=df.sort_values('Accuracy', ascending=False), ax=axes[1], palette='plasma')
    axes[1].set_title('Doğruluk (Accuracy) - Yüksek olan daha iyi')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, max(1.0, df['Accuracy'].max() * 1.1))
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