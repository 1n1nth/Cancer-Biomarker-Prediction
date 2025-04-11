import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model import GeneSelectorNN

def get_top_genes(model, X_val, gene_names, q=20):
    """
    Legacy function kept for backward compatibility.
    Now uses gradient-based approach for gene importance calculation.
    """
    return get_top_genes_gradient_based(model, X_val, gene_names, q)

def get_top_genes_gradient_based(model, X_val, gene_names, q=20):
    """
    Calculate gene importance using gradient-based approach.
    For each sample, computes gradients of outputs with respect to inputs.
    Returns top q genes based on their importance scores.
    """
    model.eval()
    all_top_genes = []
    gene_importance_scores = []
    
    # Process in small batches to avoid memory issues
    batch_size = 10
    num_samples = X_val.shape[0]
    
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_samples = X_val[batch_start:batch_end]
        
        for i in range(len(batch_samples)):
            sample = batch_samples[i:i+1].clone().detach().requires_grad_(True)
            
            # Forward pass
            try:
                output = model(sample)
                
                # Initialize importance scores
                importance = torch.zeros(sample.shape[1], device=sample.device)
                
                # Compute gradients for each output class
                for j in range(output.shape[1]):
                    if sample.grad is not None:
                        sample.grad.zero_()
                    
                    # Compute gradient of the output with respect to the input
                    output[0, j].backward(retain_graph=(j < output.shape[1]-1))
                    
                    if sample.grad is not None:
                        # Add the absolute gradient values to the importance scores
                        importance += sample.grad[0].abs()
                
                # Store the importance scores
                gene_importance_scores.append(importance.detach().cpu())
                
                # Get top genes based on importance
                top_genes_indices = torch.argsort(importance, descending=True)[:q].cpu().numpy()
                top_genes = gene_names[top_genes_indices]
                all_top_genes.append(top_genes.tolist())
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    
    # If no valid samples were processed, return empty list
    if len(gene_importance_scores) == 0:
        print("Warning: No valid samples were processed for gene importance calculation")
        return []
    
    # Create DataFrame for top genes
    output_df = pd.DataFrame(all_top_genes, columns=[f"Gene_{i+1}" for i in range(q)])
    
    # ✅ Ensure "results" folder exists
    results_path = r"./results"
    os.makedirs(results_path, exist_ok=True)
    
    # ✅ Save top genes for each sample
    output_path = f"{results_path}/top_{q}_genes_per_sample.csv"
    output_df.to_csv(output_path, index=False)
    
    # ✅ Save importance scores for all genes (averaged across samples)
    avg_importance = torch.stack(gene_importance_scores).mean(dim=0)
    importance_df = pd.DataFrame({
        'Gene': gene_names,
        'Importance': avg_importance.numpy()
    }).sort_values('Importance', ascending=False)
    
    importance_path = f"{results_path}/gene_importance_scores.csv"
    importance_df.to_csv(importance_path, index=False)
    
    # ✅ Visualize top 50 genes by importance
    plt.figure(figsize=(12, 8))
    top_50_genes = importance_df.head(50)
    sns.barplot(x='Importance', y='Gene', data=top_50_genes)
    plt.title('Top 50 Genes by Gradient-Based Importance')
    plt.tight_layout()
    plt.savefig(f"{results_path}/top_50_genes_importance.png")
    plt.close()
    
    print(f"✅ Top {q} Genes Saved for Each Test Sample to {output_path}")
    print(f"✅ Overall Gene Importance Scores Saved to {importance_path}")
    print(f"✅ Top 50 Genes Visualization Saved")
    
    return all_top_genes

# ✅ Save predictions for test samples
def save_predictions(model, X_val, prediction_cols, output_file=r"./results/predictions.csv"):
    with torch.no_grad():
        try:
            predictions = model(X_val).numpy()  # Mutation probabilities
        except Exception as e:
            print(f"Error generating predictions: {e}")
            return
            
    # Ensure we have the correct number of column names
    if len(prediction_cols) != predictions.shape[1]:
        prediction_cols = [f"Class_{i}" for i in range(predictions.shape[1])]
        print(f"Warning: Column count mismatch. Using generic column names.")

    pred_df = pd.DataFrame(predictions, columns=prediction_cols)
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pred_df.to_csv(output_file, index=False)

    print(f"✅ Test set predictions saved to {output_file}")

# ✅ Draw Neural Network Structure for Each Trial
def draw_nn(hidden_dim1, hidden_dim2, trial_number):
    """
    Draws and saves a simple neural network structure for visualization.
    """
    layers = ["Input Layer (2048)"] + \
             [f"Hidden Layer 1 ({hidden_dim1})"] + \
             [f"Hidden Layer 2 ({hidden_dim2})"] + \
             ["Output Layer (Mutation Targets)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, len(layers))

    for i, layer in enumerate(layers):
        ax.text(0, len(layers) - i - 1, layer, ha='center', fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.5))

    ax.axis("off")

    results_path = r"C:\Users\anant\OneDrive\Documents\code\miniproject\results"
    os.makedirs(results_path, exist_ok=True)
    plt.savefig(f"{results_path}/nn_structure_trial_{trial_number}.png")
    plt.close()

# ✅ Analyze Best Trial (Hyperparameter Insights)
def analyze_best_trial(study):
    """
    Generates visualizations for the best trial in the Optuna study.
    """
    best_trial = study.best_trial
    best_params = best_trial.params

    # Extract data for visualization
    trials = study.trials_dataframe()
    trials = trials.dropna()  # Remove unfinished trials

    results_path = r"C:\Users\anant\OneDrive\Documents\code\miniproject\results"
    os.makedirs(results_path, exist_ok=True)

    # ✅ Bar chart for best hyperparameters
    plt.figure(figsize=(8, 5))
    params = list(best_params.keys())
    values = [best_params[k] for k in params]
    
    # Handle different scales for parameters
    if 'lr' in best_params:
        lr_idx = params.index('lr')
        values[lr_idx] = values[lr_idx] * 1000  # Scale up learning rate for visibility
        
    sns.barplot(x=params, y=values)
    plt.title("Best Trial Hyperparameters")
    plt.ylabel("Value")
    plt.savefig(f"{results_path}/best_trial_hyperparams.png")
    plt.close()

    print(f"✅ Saved Best Trial Hyperparameter Analysis: {results_path}/best_trial_hyperparams.png")

    # ✅ Scatter plot of Learning Rate vs. Final Loss
    if 'params_lr' in trials.columns and trials['params_lr'].nunique() > 1:
        plt.figure(figsize=(8, 5))
        valid_trials = trials[np.isfinite(trials['value'])]
        sns.scatterplot(x=valid_trials["params_lr"], y=valid_trials["value"])
        plt.xlabel("Learning Rate")
        plt.ylabel("Final Loss")
        plt.title("Learning Rate vs. Loss")
        plt.xscale('log')  # Use log scale for learning rate
        plt.savefig(f"{results_path}/lr_vs_loss.png")
        plt.close()

        print(f"✅ Saved Learning Rate vs. Loss Plot: {results_path}/lr_vs_loss.png")

def load_model_and_predict(input_sample, model_path, hyperparams_path, gene_names, q=20, num_classes=13):
    """
    Loads the trained model and predicts the top 20 important genes for a single test sample.
    
    Parameters:
    - input_sample: Tensor of shape (1, 2048) representing a single RNA sample.
    - model_path: Path to the saved model.
    - hyperparams_path: Path to the saved best hyperparameters.
    - gene_names: List of all gene names.
    - q: Number of top genes to return.
    - num_classes: Number of output classes in the model.

    Returns:
    - List of top q gene names for the given input sample.
    """
    # ✅ Load Best Hyperparameters
    best_params = torch.load(hyperparams_path, weights_only=True)
    
    # ✅ Ensure Correct Model Structure
    input_dim = input_sample.shape[1]  # Should be 2048
    model = GeneSelectorNN(input_dim=input_dim, num_classes=num_classes, 
                           hidden_dim1=best_params["hidden_dim1"], 
                           hidden_dim2=best_params["hidden_dim2"],
                           dropout=best_params.get("dropout", 0.3))
    
    # ✅ Load Trained Weights
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Set model to evaluation mode

    # Enable gradient calculation
    input_sample.requires_grad_(True)
    
    # Forward pass
    output = model(input_sample)
    
    # Calculate importance using gradients
    importance = torch.zeros(input_sample.shape[1])
    
    # For each output class
    for j in range(output.shape[1]):
        model.zero_grad()
        if input_sample.grad is not None:
            input_sample.grad.zero_()
            
        output[0, j].backward(retain_graph=(j < output.shape[1]-1))
        
        if input_sample.grad is not None:
            importance += input_sample.grad[0].abs()
    
    # Get top genes
    top_genes_indices = torch.argsort(importance, descending=True)[:q]
    top_genes = [gene_names[i] for i in top_genes_indices.numpy()]

    return top_genes,top_genes_indices
