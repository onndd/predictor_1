import lime
import lime.lime_tabular
import numpy as np
from typing import Any, List, Optional

class LimeExplainer:
    """
    A wrapper for LIME (Local Interpretable Model-agnostic Explanations)
    to explain individual predictions.
    """
    def __init__(self, training_data: np.ndarray, feature_names: List[str], class_names: List[str], mode: str = 'classification'):
        """
        Initializes the LIME tabular explainer.

        Args:
            training_data (np.ndarray): A numpy array of the training data.
            feature_names (List[str]): List of feature names.
            class_names (List[str]): List of class names (e.g., ['Below Threshold', 'Above Threshold']).
            mode (str): 'classification' or 'regression'.
        """
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=class_names,
            mode=mode,
            discretize_continuous=True
        )

    def explain_instance(self, data_instance: np.ndarray, predict_fn: Any, num_features: int = 5) -> Optional[lime.explanation.Explanation]:
        """
        Explains a single prediction instance.

        Args:
            data_instance (np.ndarray): The instance to explain.
            predict_fn: The prediction function of the model. For classification,
                        it should return probabilities for each class.
            num_features (int): The number of features to include in the explanation.

        Returns:
            An explanation object, or None if an error occurs.
        """
        try:
            explanation = self.explainer.explain_instance(
                data_row=data_instance,
                predict_fn=predict_fn,
                num_features=num_features
            )
            return explanation
        except Exception as e:
            print(f"❌ LIME explanation failed: {e}")
            return None