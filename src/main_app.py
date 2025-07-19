"""
Enhanced JetX Prediction Application
Using Advanced Model Manager with Deep Learning Models
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import warnings
import sqlite3
from datetime import datetime
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our modules
from src.models.advanced_model_manager import AdvancedModelManager
from src.models.rolling_model_manager import RollingModelManager
from src.data_processing.loader import load_data_from_sqlite, save_result_to_sqlite
from src.explainability.lime_explainer import LimeExplainer
from src.explainability.attention_visualizer import plot_attention_heatmap
from src.explainability.explanation_reporting import generate_lime_summary
from src.feature_engineering.unified_extractor import UnifiedFeatureExtractor

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class EnhancedJetXApp:
    """
    Enhanced JetX Application with Advanced Model Management
    """
    
    def __init__(self):
        self.db_path = "data/jetx_data.db"
        self.models_dir = "trained_models"
        
        # Initialize model manager
        self.model_manager = AdvancedModelManager(self.models_dir, self.db_path)
        
        # Initialize rolling model manager
        self.rolling_manager = RollingModelManager(self.models_dir, self.db_path)
        
        # App state
        self.is_initialized = False
        self.current_data = []
        self.rolling_models_loaded = False
        self.lime_explainer = None
        self.last_prediction_features = None
        
    def initialize_app(self, auto_train_heavy=False):
        """Initialize the application"""
        if self.is_initialized:
            return
        
        # Load data
        data = self.load_data()
        if data is None or len(data) == 0:
            st.error("No data available. Please add some data first.")
            return
        
        self.current_data = data
        
        # Initialize models
        with st.spinner("Initializing models..."):
            self.model_manager.initialize_models(data, auto_train_heavy=auto_train_heavy)
        
        # Initialize LIME Explainer
        self._initialize_lime_explainer(data)

        self.is_initialized = True
    
    def load_data(self, limit=None):
        """Load data from database with SQL injection protection"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if limit:
                # Use parameterized query to prevent SQL injection
                cursor.execute("SELECT value FROM jetx_results ORDER BY id DESC LIMIT ?", (limit,))
            else:
                cursor.execute("SELECT value FROM jetx_results ORDER BY id")
            
            rows = cursor.fetchall()
            conn.close()
            
            if rows:
                values = [row[0] for row in rows]
                return values[::-1]  # Reverse to get chronological order
            return []
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return []
    
    def add_new_value(self, value):
        """Add new value to database"""
        try:
            save_result_to_sqlite(value, self.db_path) # Corrected order
            self.current_data.append(value)
            return True
        except Exception as e:
            st.error(f"Error adding value: {e}")
            return False
    
    def make_prediction(self, sequence_length=200):
        """Make prediction using ensemble of models"""
        if len(self.current_data) < sequence_length:
            st.warning(f"Need at least {sequence_length} data points for prediction")
            return None
        
        # Get recent sequence
        recent_sequence = self.current_data[-sequence_length:]
        
        # Make ensemble prediction
        result = self.model_manager.ensemble_predict(recent_sequence)
        
        # Store features for LIME
        if self.model_manager.feature_extractor and self.model_manager.feature_extractor.is_fitted:
            self.last_prediction_features = self.model_manager.feature_extractor.transform(recent_sequence)[-1]

        return result
    
    def make_optimized_prediction(self, sequence_length=200):
        """Make prediction using optimized ensemble"""
        if len(self.current_data) < sequence_length:
            st.warning(f"Need at least {sequence_length} data points for prediction")
            return None
        
        # Get recent sequence
        recent_sequence = self.current_data[-sequence_length:]
        
        # Make optimized ensemble prediction
        result = self.model_manager.predict_with_ensemble(recent_sequence, use_optimized=True)
        
        # Store features for LIME
        if self.model_manager.feature_extractor and self.model_manager.feature_extractor.is_fitted:
            self.last_prediction_features = self.model_manager.feature_extractor.transform(recent_sequence)[-1]

        return result

    def _initialize_lime_explainer(self, data):
        """Initializes the LIME explainer if possible."""
        try:
            extractor = self.model_manager.get_feature_extractor()
            if extractor and extractor.is_fitted:
                st.session_state.feature_names = extractor.get_feature_names()
                # Use a sample of the data for LIME background
                background_data = extractor.transform(data)
                
                self.lime_explainer = LimeExplainer(
                    training_data=background_data,
                    feature_names=st.session_state.feature_names,
                    class_names=['Below 1.5', 'Above 1.5'],
                    mode='classification'
                )
                st.success("✅ LIME Explainer initialized.")
        except Exception as e:
            st.warning(f"Could not initialize LIME Explainer: {e}")

    def get_lime_explanation(self):
        """Generates and returns a LIME explanation for the last prediction."""
        if not self.lime_explainer or self.last_prediction_features is None:
            st.error("LIME Explainer is not ready or no prediction has been made.")
            return None

        # The prediction function for LIME needs to accept a numpy array of features
        def predict_fn_for_lime(numpy_features):
            # This is a simplified prediction function for the light-model ensemble
            # It assumes the ensemble model can predict from features directly.
            # This part needs a proper implementation based on how the ensemble model works.
            # For now, we'll mock a prediction based on a few features.
            
            # Find indices for key features
            try:
                mean_idx = st.session_state.feature_names.index('stat_rolling_mean_10')
                std_idx = st.session_state.feature_names.index('stat_rolling_std_10')
            except (ValueError, AttributeError):
                # Fallback if feature names are not as expected
                mean_idx, std_idx = 0, 1

            probabilities = []
            for row in numpy_features:
                # A simple heuristic for demonstration
                prob_above = 0.5 + (row[mean_idx] - 1.5) * 0.1 - row[std_idx] * 0.2
                prob_above = np.clip(prob_above, 0, 1)
                probabilities.append([1 - prob_above, prob_above])
            return np.array(probabilities)

        with st.spinner("Generating LIME explanation..."):
            explanation = self.lime_explainer.explain_instance(
                self.last_prediction_features,
                predict_fn_for_lime,
                num_features=10
            )
        return explanation

    def train_heavy_model(self, model_name, epochs=100):
        """Train a specific heavy model"""
        if len(self.current_data) < 500:
            st.warning("Need at least 500 data points for training")
            return False
        
        with st.spinner(f"Training {model_name}..."):
            try:
                history = self.model_manager.train_heavy_model(model_name, self.current_data, epochs)
                st.success(f"{model_name} training completed!")
                return True
            except Exception as e:
                st.error(f"Training failed: {e}")
                return False

def main():
    """Main application function"""
    st.set_page_config(
        page_title="Enhanced JetX Prediction System",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .big-font {font-size:30px !important; font-weight: bold;}
        .result-box {padding: 20px; border-radius: 10px; margin-bottom: 20px;}
        .high-confidence {background-color: rgba(0, 255, 0, 0.2);}
        .medium-confidence {background-color: rgba(255, 255, 0, 0.2);}
        .low-confidence {background-color: rgba(255, 0, 0, 0.2);}
        .uncertain-confidence {background-color: rgba(128, 128, 128, 0.2);}
        .above-threshold {color: green; font-weight: bold;}
        .below-threshold {color: red; font-weight: bold;}
        .uncertain-text {color: #555; font-weight: bold;}
        .model-status {padding: 10px; border-radius: 5px; margin: 5px 0;}
        .model-ready {background-color: rgba(0, 255, 0, 0.1);}
        .model-not-ready {background-color: rgba(255, 0, 0, 0.1);}
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize app
    if 'app' not in st.session_state:
        st.session_state.app = EnhancedJetXApp()
    
    app = st.session_state.app
    
    # Header
    st.markdown('<p class="big-font">Enhanced JetX Prediction System</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Check for dependency errors and display them prominently
    dependency_error = app.model_manager.get_dependency_error()
    if dependency_error:
        st.error(f"""
        **Critical Dependency Error:** Deep learning models could not be loaded.
        The system will run in a limited mode. Please fix the following error:
        
        `{dependency_error}`
        
        **Suggested Solution:** Run the command `pip install -r requirements_enhanced.txt`.
        """)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Model Management")
        
        # Model status
        st.subheader("Model Status")
        if app.is_initialized:
            status = app.model_manager.get_model_status()
            for model_name, info in status.items():
                status_class = "model-ready" if info['loaded'] and info['trained'] else "model-not-ready"
                st.markdown(f"""
                <div class="model-status {status_class}">
                    <strong>{model_name}</strong><br>
                    Loaded: {'✓' if info['loaded'] else '✗'}<br>
                    Trained: {'✓' if info['trained'] else '✗'}<br>
                    Type: {'Heavy' if info['is_heavy'] else 'Light'}
                </div>
                """, unsafe_allow_html=True)
        
        # Model training controls
        st.subheader("Model Training")
        
        if st.button("Initialize App (Light Models Only)"):
            app.initialize_app(auto_train_heavy=False)
            st.success("App initialized with light models!")
        
        if st.button("Initialize App (All Models)"):
            app.initialize_app(auto_train_heavy=True)
            st.success("App initialized with all models!")
        
        # Heavy model training
        st.subheader("Train Heavy Models")
        heavy_models = ['n_beats', 'tft', 'informer', 'autoformer', 'pathformer']
        
        for model_name in heavy_models:
            if st.button(f"Train {model_name.upper()}"):
                if app.is_initialized:
                    app.train_heavy_model(model_name)
                else:
                    st.warning("Please initialize the app first")
        
        # Load pre-trained models
        st.subheader("Load Pre-trained Models")
        if st.button("Load All Saved Models"):
            app.model_manager.load_all_models()
            st.success("Models loaded!")
        
        # Retrain models
        st.subheader("Retrain Models")
        if st.button("Retrain All Models"):
            if app.is_initialized:
                app.model_manager.retrain_models(app.current_data)
                st.success("Models retrained!")
            else:
                st.warning("Please initialize the app first")
        
        # Rolling Window Models Section
        st.markdown("---")
        st.header("🔄 Rolling Window Models")
        st.caption("Models trained with Colab Rolling Training System")
        
        # Load rolling models
        if st.button("🔍 Scan for Rolling Models"):
            with st.spinner("Scanning for rolling models..."):
                summary = app.rolling_manager.get_rolling_training_summary()
                available = app.rolling_manager.get_available_rolling_models()
                
                if available:
                    st.success(f"Found {len(available)} model types!")
                    for model_type, models in available.items():
                        st.write(f"• {model_type}: {len(models)} cycles")
                else:
                    st.info("No rolling models found. Train models using Colab interface first.")
        
        # Load best rolling models
        if st.button("⚡ Load Best Rolling Models"):
            with st.spinner("Loading best rolling models..."):
                available = app.rolling_manager.get_available_rolling_models()
                loaded_count = 0
                
                for model_type in available.keys():
                    best_model = app.rolling_manager.get_best_rolling_model(model_type)
                    if best_model:
                        model = app.rolling_manager.load_rolling_model_for_prediction(best_model)
                        if model:
                            loaded_count += 1
                
                if loaded_count > 0:
                    app.rolling_models_loaded = True
                    st.success(f"Loaded {loaded_count} rolling models!")
                else:
                    st.warning("No rolling models could be loaded.")
        
        # Show rolling model status
        if app.rolling_models_loaded:
            st.subheader("Rolling Model Status")
            for model_type, model_data in app.rolling_manager.rolling_models.items():
                performance = model_data['info']['performance']
                st.markdown(f"""
                <div class="model-status model-ready">
                    <strong>🔄 {model_type}</strong><br>
                    MAE: {performance.get('mae', 'N/A'):.4f}<br>
                    Accuracy: {performance.get('accuracy', 'N/A'):.2%}<br>
                    Cycle: {model_data['info']['cycle']}
                </div>
                """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Data Management")
        
        # Data input
        with st.expander("Add New Data", expanded=True):
            with st.form(key="data_input_form"):
                value_input = st.number_input(
                    "JetX Value:", min_value=1.0, max_value=5000.0, value=1.5, step=0.01, format="%.2f"
                )
                submit_button = st.form_submit_button("Add Value")
                
                if submit_button:
                    if app.add_new_value(value_input):
                        st.success(f"Value {value_input} added successfully!")
                        st.rerun()
        
        # Data visualization
        st.subheader("Recent Data")
        if app.current_data:
            # Show last 50 values
            recent_data = app.current_data[-50:]
            fig, ax = plt.subplots(figsize=(12, 4))
            
            colors = ['red' if v < 1.5 else 'green' for v in recent_data]
            ax.bar(range(len(recent_data)), recent_data, color=colors, alpha=0.7)
            ax.axhline(y=1.5, color='grey', linestyle='--', alpha=0.7, label='Threshold (1.5)')
            ax.set_xlabel('Recent Values')
            ax.set_ylabel('JetX Value')
            ax.set_title('Recent JetX Values')
            ax.legend()
            
            st.pyplot(fig)
            
            # Statistics
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Total Values", len(app.current_data))
            with col_stats2:
                above_threshold = sum(1 for v in recent_data if v >= 1.5)
                st.metric("Above 1.5", f"{above_threshold}/{len(recent_data)}")
            with col_stats3:
                avg_value = np.mean(recent_data)
                st.metric("Average", f"{avg_value:.2f}")
        else:
            st.info("No data available. Please add some values first.")
    
    with col2:
        st.subheader("Prediction")
        
        if app.is_initialized:
            # Prediction buttons
            col_pred1, col_pred2 = st.columns(2)
            
            with col_pred1:
                # Regular prediction
                if st.button("🔮 Standard Prediction"):
                    with st.spinner("Making prediction..."):
                        result = app.make_prediction()
                    
                    if result and result['ensemble_prediction'] is not None:
                        prediction = result['ensemble_prediction']
                        confidence = result['confidence']
                        model_count = result['model_count']
                        
                        # Determine confidence class
                        if confidence >= 0.7:
                            conf_class = "high-confidence"
                        elif confidence >= 0.5:
                            conf_class = "medium-confidence"
                        else:
                            conf_class = "low-confidence"
                        
                        # Determine threshold class
                        above_threshold = prediction >= 1.5
                        threshold_class = "above-threshold" if above_threshold else "below-threshold"
                        decision_text = "ABOVE 1.5" if above_threshold else "BELOW 1.5"
                        
                        # Display result
                        st.markdown(f"""
                        <div class="result-box {conf_class}">
                            <h3 style="text-align: center;" class="{threshold_class}">{decision_text}</h3>
                            <p><b>Predicted Value:</b> {prediction:.3f}</p>
                            <p><b>Confidence:</b> {confidence:.2%}</p>
                            <p><b>Models Used:</b> {model_count}</p>
                            <p><b>Above 1.5 Probability:</b> {max(0, min(1, (prediction - 1.0) / 2.0)):.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show individual model predictions
                        if result['predictions']:
                            st.subheader("Individual Model Predictions")
                            for model_name, pred in result['predictions'].items():
                                st.write(f"{model_name.upper()}: {pred:.3f}")

                        # --- Explainability Section ---
                        with st.expander("🔬 Explain Prediction"):
                            # LIME Explanation Button
                            if st.button("Explain with LIME", key="explain_lime"):
                                explanation = app.get_lime_explanation()
                                if explanation:
                                    st.subheader("Local Explanation (LIME)")
                                    summary = generate_lime_summary(explanation, num_features=5)
                                    if summary:
                                        for line in summary:
                                            st.markdown(line, unsafe_allow_html=True)
                                    st.components.v1.html(explanation.as_html(), height=800, scrolling=True)
                            
                            # Attention Visualization Button
                            if 'tft' in app.model_manager.models: # Check if TFT model exists
                                if st.button("Visualize TFT Attention", key="explain_tft_attn"):
                                    with st.spinner("Generating Attention Heatmap..."):
                                        sequence_length = app.model_manager.models['tft'].sequence_length
                                        recent_sequence = app.current_data[-sequence_length:]
                                        attn_result = app.model_manager.predict_with_attention('tft', recent_sequence)
                                        if attn_result and attn_result.get('attention_weights'):
                                            st.subheader("TFT Attention Map")
                                            fig = plot_attention_heatmap(
                                                attention_weights=attn_result['attention_weights'],
                                                input_sequence=recent_sequence
                                            )
                                            if fig:
                                                st.pyplot(fig)
                                            else:
                                                st.warning("Could not generate attention plot.")
                                        else:
                                            st.error("Failed to retrieve attention weights for TFT.")
                            
                            # SHAP Plot Display
                            st.subheader("Global Feature Importance (SHAP)")
                            reports_dir = "reports"
                            shap_plots = [f for f in os.listdir(reports_dir) if f.startswith('shap_summary') and f.endswith('.png')]
                            if shap_plots:
                                selected_shap_plot = st.selectbox("Select a SHAP plot to view:", shap_plots)
                                if selected_shap_plot:
                                    st.image(os.path.join(reports_dir, selected_shap_plot))
                            else:
                                st.info("No SHAP plots found. Train a model to generate them.")

                    else:
                        st.warning("Prediction failed. Make sure models are trained.")
            
            with col_pred2:
                # Rolling window prediction
                if st.button("🔄 Rolling Window Prediction"):
                    if app.rolling_models_loaded:
                        with st.spinner("Making rolling window prediction..."):
                            result = app.rolling_manager.get_ensemble_rolling_prediction(app.current_data)
                        
                        if result and result['ensemble_value'] is not None:
                            prediction = result['ensemble_value']
                            confidence = result['ensemble_confidence']
                            model_count = result['model_count']
                            
                            # Determine confidence class
                            if confidence >= 0.7:
                                conf_class = "high-confidence"
                            elif confidence >= 0.5:
                                conf_class = "medium-confidence"
                            else:
                                conf_class = "low-confidence"
                            
                            # Determine threshold class
                            above_threshold = prediction >= 1.5
                            threshold_class = "above-threshold" if above_threshold else "below-threshold"
                            decision_text = "ABOVE 1.5" if above_threshold else "BELOW 1.5"
                            
                            # Display result
                            st.markdown(f"""
                            <div class="result-box {conf_class}">
                                <h3 style="text-align: center;" class="{threshold_class}">🔄 {decision_text}</h3>
                                <p><b>Rolling Predicted Value:</b> {prediction:.3f}</p>
                                <p><b>Rolling Confidence:</b> {confidence:.2%}</p>
                                <p><b>Rolling Models Used:</b> {model_count}</p>
                                <p><b>Above 1.5 Probability:</b> {max(0, min(1, (prediction - 1.0) / 2.0)):.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show rolling individual model predictions
                            if result['individual_predictions']:
                                st.subheader("Rolling Model Predictions")
                                for model_type, pred in result['individual_predictions'].items():
                                    weight = result['weights'][model_type]
                                    st.write(f"🔄 {model_type}: {pred['value']:.3f} (weight: {weight:.2f})")
                        else:
                            st.warning("Rolling prediction failed.")
                    else:
                        st.warning("Load rolling models first!")
        else:
            st.info("Please initialize the app first to make predictions.")
        
        # Show rolling training summary if available
        if app.rolling_models_loaded:
            st.markdown("---")
            st.subheader("🔄 Rolling Training Summary")
            summary = app.rolling_manager.get_rolling_training_summary()
            
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            with col_sum1:
                st.metric("Training Sessions", summary['total_sessions'])
            with col_sum2:
                st.metric("Model Types", len(summary['model_types']))
            with col_sum3:
                best_mae = min([perf['mae'] for perf in summary['best_performances'].values()]) if summary['best_performances'] else 0
                st.metric("Best MAE", f"{best_mae:.4f}")
            
            # Best performances
            if summary['best_performances']:
                st.write("**Best Performance by Model Type:**")
                for model_type, perf in summary['best_performances'].items():
                    st.write(f"• {model_type}: MAE={perf['mae']:.4f}, Accuracy={perf['accuracy']:.2%}")
    
    # Model performance section
    if app.is_initialized:
        st.subheader("Model Performance")
        
        # Show training histories
        for model_name, performance in app.model_manager.model_performances.items():
            if performance and 'val_losses' in performance:
                with st.expander(f"{model_name.upper()} Training History"):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(performance['train_losses'], label='Training Loss')
                    ax.plot(performance['val_losses'], label='Validation Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title(f'{model_name.upper()} Training History')
                    ax.legend()
                    st.pyplot(fig)
        
        # Best models
        best_models = app.model_manager.get_best_models(top_k=3)
        if best_models:
            st.subheader("Best Performing Models")
            for i, model_name in enumerate(best_models, 1):
                st.write(f"{i}. {model_name.upper()}")

if __name__ == "__main__":
    main()
