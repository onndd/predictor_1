import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from .metrics import calculate_threshold_metrics, calculate_regression_metrics
import warnings
from typing import List
from src.config.settings import CONFIG

class BackTester:
    def __init__(self, predictor, data, test_size=0.2, threshold=1.5, investment=1.0):
        """
        Gelişmiş Geri Test sistemi
        
        Args:
            predictor: Tahmin modeli
            data: Test verisi
            test_size: Test seti oranı
            threshold: Eşik değeri
            investment: Her bir işlem için yatırım miktarı
        """
        self.predictor = predictor
        self.data = data
        self.test_size = test_size
        self.threshold = threshold
        self.investment = investment
        
        # Sonuçlar
        self.predictions = []
        self.actuals = []
        self.confidences = []
        self.metrics = None
        
        # Finansal Sonuçlar
        self.cumulative_return: List[float] = [0.0]
        self.drawdowns = []
        self.financial_metrics = {}

    def run(self, window_size=None, step_size=None, show_progress=True, strategy_threshold=None):
        """
        Geri test yapar ve finansal stratejiyi simüle eder.
        
        Args:
            window_size: Pencere boyutu
            step_size: Adım boyutu
            show_progress: İlerleme göster
            strategy_threshold: Pozisyon almak için gereken tahmin olasılığı eşiği
            
        Returns:
            dict: Standart ve finansal metrikler
        """
        backtest_config = CONFIG.get('backtesting', {})
        window_size = window_size or backtest_config.get('window_size', 100)
        step_size = step_size or backtest_config.get('step_size', 1)
        strategy_threshold = strategy_threshold or backtest_config.get('strategy_threshold', 0.7)
        test_size = self.test_size or backtest_config.get('test_size', 0.2)

        n = len(self.data)
        train_size = int((1 - test_size) * n)
        train_data = self.data[:train_size]
        test_data = self.data[train_size:]
        
        if len(test_data) == 0:
            print("Test veri seti boş!")
            return None
            
        if hasattr(self.predictor, 'fit'):
            self.predictor.fit(train_data)
        
        self.predictions, self.actuals, self.confidences, self.cumulative_return = [], [], [], [0.0]
        
        iterator = range(window_size, len(test_data), step_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Advanced Backtest")
            
        for i in iterator:
            history = test_data[i-window_size:i]
            actual = test_data[i]
            
            try:
                result = self.predictor.predict_next_value(history)
                
                if isinstance(result, tuple) and len(result) == 3:
                    pred_value, above_prob, confidence = result
                elif isinstance(result, (int, float)):
                     pred_value, above_prob, confidence = result, (1.0 if result >= self.threshold else 0.0), 0.5
                else:
                    continue # Skip if result is not in expected format

                self.predictions.append(above_prob)
                self.actuals.append(actual)
                self.confidences.append(confidence)
                
                # --- Finansal Strateji Simülasyonu ---
                # Strateji: Eğer modelin çökme olasılığı (tahmin < 1.5) %70'den fazlaysa,
                # yani `above_prob` < 0.3 ise, pozisyon al.
                # Gerçekte çöküş olmazsa (actual >= 1.5) yatırımı kaybederiz.
                # Gerçekte çöküş olursa (actual < 1.5) yatırımı kazanırız (örneğin 2x).
                # Bu örnek bir "short" stratejisidir.
                
                current_return = self.cumulative_return[-1]
                
                # Basit bir "short" stratejisi
                if (1 - above_prob) >= strategy_threshold: # Çökme olasılığı yüksekse
                    if actual < self.threshold: # Başarılı short
                        current_return += self.investment * (self.threshold - 1) # Kazanç
                    else: # Başarısız short
                        current_return -= self.investment # Kayıp
                
                self.cumulative_return.append(float(current_return))

            except Exception as e:
                print(f"Tahmin hatası (i={i}): {e}")
        
        self._calculate_metrics()
        if self.metrics and self.financial_metrics:
            return {**self.metrics, **self.financial_metrics}
        return self.metrics or self.financial_metrics or {}

    def _calculate_metrics(self):
        """Hesaplamaları merkezi bir fonksiyonda topla."""
        if not self.predictions:
            print("Hiç geçerli tahmin yok!")
            self.metrics = {}
            self.financial_metrics = {}
            return

        # Standart Metrikler
        self.metrics = calculate_threshold_metrics(self.actuals,
                                                 [1.0 if p >= 0.5 else 0.0 for p in self.predictions],
                                                 threshold=self.threshold)
        
        # Finansal Metrikler
        returns = pd.Series(self.cumulative_return).pct_change().fillna(0)
        
        # Max Drawdown
        cumulative = pd.Series(self.cumulative_return)
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak
        self.drawdowns = drawdown.tolist()
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (risk-free rate = 0)
        sharpe_ratio = returns.mean() / returns.std() if returns.std() != 0 else 0
        
        self.financial_metrics = {
            'total_return': self.cumulative_return[-1],
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

    def plot_results(self, figsize=(15, 12)):
        """Sonuçları ve finansal performansı görselleştirir."""
        if not self.predictions or not self.actuals:
            print("Gösterilecek sonuç yok!")
            return None
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            fig, axes = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1, 2]})
            
            # 1. Grafik: Tahmin vs Gerçek
            ax1 = axes[0]
            actuals_binary = [1.0 if a >= self.threshold else 0.0 for a in self.actuals]
            ax1.plot(self.predictions, 'b.-', alpha=0.6, label='Prediction (Prob > Threshold)')
            ax1.plot(actuals_binary, 'ro', alpha=0.4, label='Actual (Binary)')
            ax1.axhline(y=0.5, color='g', linestyle='--', alpha=0.7, label='Decision Boundary (0.5)')
            ax1.set_title('Backtest: Prediction vs Actual')
            ax1.set_ylabel('Probability / Binary')
            ax1.legend()

            # 2. Grafik: Metrikler
            ax2 = axes[1]
            if self.metrics:
                metrics_to_plot = {k: v for k, v in self.metrics.items() if isinstance(v, (int, float))}
                names = list(metrics_to_plot.keys())
            values = list(metrics_to_plot.values())
            bars = ax2.bar(names, values, alpha=0.8)
            ax2.set_title('Performance Metrics')
            ax2.set_ylim(0, 1.1)
            for bar in bars:
                yval = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom')

            # 3. Grafik: Finansal Performans
            ax3 = axes[2]
            ax3.plot(self.cumulative_return, label='Cumulative Return', color='green')
            ax3.set_title('Financial Performance')
            ax3.set_xlabel('Number of Trades')
            ax3.set_ylabel('Cumulative Return ($)')
            
            # Drawdown'ı doldur
            ax3_twin = ax3.twinx()
            ax3_twin.fill_between(range(len(self.drawdowns)), self.drawdowns, 0, color='red', alpha=0.3, label='Drawdown')
            ax3_twin.set_ylabel('Drawdown', color='red')
            
            lines, labels = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            ax3_twin.legend(lines + lines2, labels + labels2, loc='upper left')
            
            plt.tight_layout()
            return fig
