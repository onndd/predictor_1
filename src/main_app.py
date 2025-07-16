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
from src.data_processing.loader import load_data_from_sqlite, save_result_to_sqlite

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
        
        # App state
        self.is_initialized = False
        self.current_data = []
        
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
        
        self.is_initialized = True
    
    def load_data(self, limit=None):
        """Load data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            if limit:
                query = f"SELECT value FROM jetx_results ORDER BY id DESC LIMIT {limit}"
            else:
                query = "SELECT value FROM jetx_results ORDER BY id"
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) > 0:
                return df['value'].tolist()[::-1]  # Reverse to get chronological order
            return []
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return []
    
    def add_new_value(self, value):
        """Add new value to database"""
        try:
            save_result_to_sqlite(self.db_path, value)
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
        
        return result
    
    def make_optimized_prediction(self, sequence_length=200):
        """Make prediction using optimized ensemble"""
        if len(self.current_data) < sequence_length:
            st.warning(f"Need at least {sequence_length} data points for prediction")
            return None
        
        # Get recent sequence
        recent_sequence = self.current_data[-sequence_length:]
        
        # Make optimized ensemble prediction
        result = self.model_manager.predict_with_optimized_ensemble(recent_sequence)
        
        return result
    
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
            # Make prediction
            if st.button("Make Prediction"):
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
                else:
                    st.warning("Prediction failed. Make sure models are trained.")
        else:
            st.info("Please initialize the app first to make predictions.")
    
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
