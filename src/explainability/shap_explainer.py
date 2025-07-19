import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, List, Dict, Optional

class ShapExplainer:
    """
    A wrapper for SHAP (SHapley Additive exPlanations) to explain model predictions.
    """
    def __init__(self, model: Any, background_data: np.ndarray, feature_names: List[str]):
        """
        Initializes the SHAP explainer.

        Args:
            model: The trained model object. It must have a `predict_proba` or `predict` method.
            background_data (np.ndarray): A sample of data (e.g., training data) to use as a background
                                          for computing SHAP values.
            feature_names (List[str]): A list of names for the features in the background_data.
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self.explainer = self._create_explainer()
        self.shap_values = self.explainer.shap_values(self.background_data)

    def _create_explainer(self) -> shap.Explainer:
        """Creates a SHAP explainer based on the model type."""
        # We use KernelExplainer as a model-agnostic approach
        # It requires a function that takes a numpy array and returns predictions.
        def predict_fn(x):
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(x)
            elif hasattr(self.model, 'predict'):
                return self.model.predict(x)
            else:
                raise TypeError("Model must have a 'predict' or 'predict_proba' method.")

        return shap.KernelExplainer(predict_fn, self.background_data)

    def plot_summary(self, save_path: Optional[str] = None):
        """
        Generates and optionally saves a SHAP summary plot.
        This plot shows the importance of each feature.
        """
        shap.summary_plot(self.shap_values, self.background_data, feature_names=self.feature_names, show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def explain_single_prediction(self, data_instance: np.ndarray, save_path: Optional[str] = None):
        """
        Generates and optionally saves a SHAP force plot for a single prediction.
        
        Args:
            data_instance (np.ndarray): A single data instance to explain.
            save_path (Optional[str]): Path to save the plot.
        """
        shap_values_instance = self.explainer.shap_values(data_instance)
        
        # For classification, shap_values can be a list of arrays (one for each class)
        # We'll use the values for the "positive" class (class 1)
        if isinstance(shap_values_instance, list):
            shap_values_instance = shap_values_instance[1]

        shap.force_plot(
            self.explainer.expected_value[1], 
            shap_values_instance, 
            data_instance, 
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()