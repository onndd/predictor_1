# Enhanced JetX Prediction System

A comprehensive time series forecasting system for JetX game predictions, featuring advanced deep learning models and statistical ensemble methods.

## 🚀 Features

### New Deep Learning Models
- **N-BEATS**: Neural Basis Expansion Analysis for Time Series
- **TFT**: Temporal Fusion Transformer with interpretable attention
- **Informer**: Long sequence time series forecasting with efficient attention
- **Autoformer**: Auto-correlation based time series forecasting
- **Pathformer**: Path-based attention for temporal modeling

### Enhanced Features
- **Manual Control**: Heavy models are not automatically trained - you control when to train them
- **Lazy Loading**: Models are loaded only when needed
- **Ensemble Predictions**: Combines predictions from multiple models
- **Performance Tracking**: Monitor model performance and training history
- **Modular Architecture**: Clean, organized code structure

## 📁 Project Structure

```
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   ├── deep_learning/        # Deep learning models
│   │   │   ├── n_beats/         # N-BEATS implementation
│   │   │   ├── tft/             # TFT implementation
│   │   │   ├── informer/        # Informer implementation
│   │   │   ├── autoformer/      # Autoformer implementation
│   │   │   └── pathformer/      # Pathformer implementation
│   │   ├── statistical/         # Statistical models
│   │   ├── ensemble/            # Ensemble methods
│   │   └── advanced_model_manager.py  # Model management
│   ├── data_processing/         # Data handling
│   ├── feature_engineering/     # Feature extraction
│   ├── evaluation/              # Model evaluation
│   ├── config/                  # Configuration files
│   └── main_app.py              # Main application
├── docs/                        # Documentation
├── tests/                       # Test files
├── trained_models/              # Saved models
├── requirements_enhanced.txt    # Dependencies
└── README.md                    # This file
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd jetx-prediction-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_enhanced.txt
   ```

3. **Run the application**
   ```bash
   streamlit run src/main_app.py
   ```

## 🎯 Usage

### Starting the Application

1. **Launch the app**:
   ```bash
   streamlit run src/main_app.py
   ```

2. **Initialize the system**:
   - Click "Initialize App (Light Models Only)" for quick start
   - Click "Initialize App (All Models)" to include deep learning models

### Adding Data

1. Enter JetX values in the "Add New Data" section
2. Values are automatically saved to the database
3. Visualize recent data in the charts

### Training Models

#### Light Models (Automatic)
- Statistical models are trained automatically when you initialize the app
- No manual intervention required

#### Heavy Models (Manual)
- Deep learning models require manual training
- Use the sidebar controls to train specific models:
  - Click "Train N_BEATS" to train the N-BEATS model
  - Click "Train TFT" to train the Temporal Fusion Transformer
  - Click "Train INFORMER" to train the Informer model
  - Click "Train AUTOFORMER" to train the Autoformer model
  - Click "Train PATHFORMER" to train the Pathformer model

### Making Predictions

1. **Ensure models are trained** (check the Model Status in sidebar)
2. **Click "Make Prediction"** to get ensemble predictions
3. **View results**:
   - Ensemble prediction with confidence score
   - Individual model predictions
   - Above/below 1.5 threshold probability

### Model Management

#### Loading Pre-trained Models
- Click "Load All Saved Models" to load previously trained models
- Models are automatically saved after training

#### Retraining Models
- Click "Retrain All Models" to update all models with new data
- Individual model retraining is also available

## 🔧 Configuration

### Model Settings
Edit `src/config/settings.py` to modify:
- Model parameters (sequence length, hidden size, etc.)
- Training settings (epochs, batch size, etc.)
- Prediction thresholds
- Application behavior

### Key Settings
```python
# Auto-train heavy models (default: False)
APP_CONFIG['auto_train_heavy_models'] = False

# Training epochs for heavy models
TRAINING_CONFIG['default_epochs'] = 100

# Minimum data points for training
TRAINING_CONFIG['min_data_points'] = 500
```

## 📊 Model Performance

### Monitoring
- View training history plots for each model
- Track validation loss over epochs
- Compare model performance

### Best Models
- System automatically ranks models by performance
- Top 3 models are displayed
- Ensemble weights can be adjusted

## 🚨 Important Notes

### Heavy Models
- **N-BEATS, TFT, Informer, Autoformer, Pathformer** are computationally intensive
- Training can take several minutes to hours depending on data size
- GPU acceleration is recommended for faster training
- Models are saved automatically after training

### Data Requirements
- Minimum 500 data points for heavy model training
- More data = better model performance
- Regular retraining recommended with new data

### Memory Usage
- Heavy models require significant RAM
- Consider using smaller model configurations for limited resources
- Models can be trained separately to manage memory

## 🔍 Troubleshooting

### Common Issues

1. **"Models not available" error**
   - Ensure all dependencies are installed
   - Check PyTorch installation

2. **Training fails**
   - Verify sufficient data points (minimum 500)
   - Check available memory
   - Try reducing model complexity

3. **Prediction fails**
   - Ensure models are trained
   - Check data format
   - Verify sequence length matches model configuration

### Performance Tips

1. **Use GPU if available**
   - Uncomment GPU-specific PyTorch installation in requirements
   - Training will be significantly faster

2. **Adjust model complexity**
   - Reduce hidden sizes for faster training
   - Use fewer layers for memory efficiency

3. **Batch training**
   - Train models one at a time
   - Monitor system resources

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- N-BEATS paper: "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"
- TFT paper: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
- Informer paper: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
- Autoformer paper: "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub