from typing import List, Tuple, Optional
import lime

def generate_lime_summary(explanation: lime.explanation.Explanation, num_features: int = 5) -> Optional[List[str]]:
    """
    Generates a human-readable summary from a LIME explanation object.

    Args:
        explanation: The LIME explanation object.
        num_features: The number of top features to report.

    Returns:
        A list of strings summarizing the feature contributions, or None.
    """
    if not explanation:
        return None

    try:
        # Get the feature contributions for the predicted class
        explanation_list = explanation.as_list()
        
        summary = []
        summary.append("**Key Factors for this Prediction:**")

        top_features = explanation_list[:num_features]

        for feature, weight in top_features:
            if weight > 0:
                # This feature pushed the prediction towards the predicted class
                summary.append(f"✅ **Supports Prediction:** `{feature}`")
            else:
                # This feature pushed the prediction away from the predicted class
                summary.append(f"❌ **Opposes Prediction:** `{feature}`")
        
        return summary

    except Exception as e:
        print(f"❌ Failed to generate LIME summary: {e}")
        return None