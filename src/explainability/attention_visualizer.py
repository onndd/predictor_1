import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from typing import List, Optional

def plot_attention_heatmap(
    attention_weights: List[torch.Tensor], 
    input_sequence: List[float], 
    layer: int = 0, 
    head: int = 0, 
    figsize: tuple = (15, 5)
):
    """
    Plots a heatmap of attention weights for a specific layer and head.

    Args:
        attention_weights: List of attention tensors from the model. 
                           Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
        input_sequence: The original input sequence for labeling the axes.
        layer: The attention layer to visualize.
        head: The attention head to visualize.
        figsize: The size of the figure.
    """
    if not attention_weights:
        print("⚠️ Attention weights are not available.")
        return
    
    try:
        # Detach and move to CPU
        weights = attention_weights[layer].detach().cpu().numpy()
        
        # Select the specific head and remove batch dimension
        head_weights = weights[0, head, :, :]
        
        fig, ax = plt.subplots(figsize=figsize)
        # Convert rounded numbers to strings for labels
        labels = [str(round(x, 2)) for x in input_sequence]
        
        sns.heatmap(
            head_weights,
            xticklabels=labels,
            yticklabels=labels,
            cmap='viridis',
            cbar_kws={'label': 'Attention Score'},
            ax=ax
        )
        ax.set_xlabel("Keys (Input Sequence Steps)")
        ax.set_ylabel("Queries (Input Sequence Steps)")
        ax.set_title(f"Attention Heatmap (Layer {layer}, Head {head})")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        
        return fig

    except IndexError:
        print(f"❌ Error: Invalid layer ({layer}) or head ({head}) index.")
        print(f"   Available layers: {len(attention_weights)}")
        if attention_weights:
            print(f"   Available heads: {attention_weights[0].shape[1]}")
        return None
    except Exception as e:
        print(f"❌ An error occurred during heatmap plotting: {e}")
        return None