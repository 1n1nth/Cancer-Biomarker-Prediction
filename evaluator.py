import torch
import torch.optim as optim
import pandas as pd
import optuna
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score

import utils
from model import GeneSelectorNN
from utils import draw_nn

# ✅ Xavier Initialization Function (unchanged)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# ✅ Modified: Load Data with Cross-Validation Folds and Held-out Test Set
def load_data(holdout_percentage=0.1):
    in_dirc = r"./data"
    GeneExp = pd.read_csv(f"{in_dirc}/RNA.csv").iloc[:, :2048]  # First 2048 genes
    TargPaths = pd.read_csv(f"{in_dirc}/MUTATION.csv")
    
    # Make sure datasets have the same number of samples
    min_samples = min(len(GeneExp), len(TargPaths))
    GeneExp = GeneExp.iloc[:min_samples, :]
    TargPaths = TargPaths.iloc[:min_samples, :]
    
    # Identify the fold columns and mutation columns
    fold_columns = [col for col in TargPaths.columns if col.startswith('FOLD_')]
    mutation_cols = [col for col in TargPaths.columns if col not in fold_columns and 
                     col in ['CD53', 'CD58', 'CD2', 'CD1D', 'CD1A', 'CD1C', 'CD1B', 'CD1E', 
                            'CD84', 'CD48', 'CD8A', 'CD8B', 'CTLA4']]
    
    # Convert boolean columns to int and handle any NaN values
    TargPaths = TargPaths.astype({col: 'int64' for col in TargPaths.select_dtypes('bool').columns})
    TargPaths = TargPaths.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Use FOLD_0 as the fold indicator
    fold_col = 'FOLD_0'
    fold_indices = TargPaths[fold_col].values
    print(f"Using existing fold column: {fold_col}")
    
    
    Y_data = TargPaths[mutation_cols]
    
    # info for debugging
    print(f"Number of mutation columns: {len(mutation_cols)}")
    print(f"Mutation columns: {mutation_cols}")
    
    # Create tensors from all data
    X_all = torch.tensor(GeneExp.values, dtype=torch.float32)
    Y_all = torch.tensor(Y_data.values, dtype=torch.float32)
    
    # Extract the final portion as a completely held-out test set
    holdout_size = int(min_samples * holdout_percentage)
    
    # Use the last holdout_size samples as the held-out test set
    X_test = X_all[-holdout_size:]
    Y_test = Y_all[-holdout_size:]
    
    # Use the remaining samples for cross-validation
    X_cv = X_all[:-holdout_size]
    Y_cv = Y_all[:-holdout_size]
    fold_indices_cv = fold_indices[:-holdout_size]
    
    return X_cv, Y_cv, fold_indices_cv, X_test, Y_test, GeneExp.columns, len(mutation_cols)

# Train and validate a model using cross-validation folds
def train_and_validate_model(model, X_cv, Y_cv, fold_indices, current_fold, learning_rate, num_epochs=200):
    # Prepare train and validation data for this fold
    train_indices = fold_indices != current_fold
    val_indices = fold_indices == current_fold
    
    # Ensure we have data for this fold
    if not torch.any(torch.tensor(val_indices)):
        # If no validation data for this fold, return a high error
        return float('inf'), [], []
    
    X_train = X_cv[train_indices]
    Y_train = Y_cv[train_indices]
    X_val = X_cv[val_indices]
    Y_val = Y_cv[val_indices]
    
    # Reset model weights
    model.apply(init_weights)
    
    # Set up optimizer and loss function
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, Y_val)
            val_losses.append(val_loss.item())
            
            # Check for NaN loss - early stopping if needed
            if torch.isnan(val_loss):
                return float('inf'), train_losses, val_losses
    
    # Final validation metrics
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        final_val_loss = criterion(val_outputs, Y_val).item()
        
        # Check for NaN in final loss
        if np.isnan(final_val_loss):
            return float('inf'), train_losses, val_losses
    
    return final_val_loss, train_losses, val_losses

# Cross-Validation for Hyperparameter Optimization
def objective(trial, X_cv, Y_cv, fold_indices, unique_folds, output_dim):
    input_dim = X_cv.shape[1]
    
    # Suggest hyperparameters with lower initial values
    hidden_dim1 = trial.suggest_int("hidden_dim1", 32, 256)
    hidden_dim2 = trial.suggest_int("hidden_dim2", 16, 128)
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
    
    # Create model with correct output dimension
    model = GeneSelectorNN(input_dim, output_dim, hidden_dim1, hidden_dim2, dropout_rate)
    
    # Cross-validation across all folds
    fold_val_losses = []
    for fold in unique_folds:
        fold_loss, _, _ = train_and_validate_model(
            model, X_cv, Y_cv, fold_indices, fold, learning_rate, num_epochs=50
        )
        if np.isfinite(fold_loss):  # Only add if not NaN or inf
            fold_val_losses.append(fold_loss)
    
    # Average validation loss across valid folds
    if len(fold_val_losses) == 0:
        return float('inf')  # Return high value if all folds failed
    
    avg_val_loss = sum(fold_val_losses) / len(fold_val_losses)
    
    
    draw_nn(hidden_dim1, hidden_dim2, trial.number)
    
    return avg_val_loss

def main(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    n_iter = 100  
    q = 20  
    
    # Load data with cross-validation folds and held-out test set
    X_cv, Y_cv, fold_indices, X_test, Y_test, gene_names, output_dim = load_data(holdout_percentage=0.1)
    input_dim = X_cv.shape[1]
    
    # Get unique fold indices
    unique_folds = np.unique(fold_indices)
    
    # Creating the Optuna study for hyperparameter optimization
    study = optuna.create_study(direction='minimize')
    
    # Run optimization with cross-validation
    study.optimize(lambda trial: objective(trial, X_cv, Y_cv, fold_indices, unique_folds, output_dim), n_trials=n_iter)
    
    best_params = study.best_params
    print(f"✅ Best hyperparameters: {best_params}")
    
    # Train final model on all CV data with best hyperparameters
    best_model = GeneSelectorNN(input_dim, output_dim, best_params["hidden_dim1"], best_params["hidden_dim2"], best_params["dropout"])
    best_model.apply(init_weights)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(best_model.parameters(), lr=best_params["lr"])
    
    # Train on all cross-validation data with a more robust approach
    try:
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = best_model(X_cv)
            loss = criterion(outputs, Y_cv)
            loss.backward()
            optimizer.step()
            
            
            if epoch % 9 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"Error during final training: {e}")
    
    # Create results directory if it doesn't exist
    results_path = r"./results"
    os.makedirs(results_path, exist_ok=True)
    
    # Save the best model
    best_model_path = f"{results_path}/best_model.pt"
    torch.save(best_model.state_dict(), best_model_path)
    print(f"✅ Best Model Saved to {best_model_path}")
    
    # Save best hyperparameters
    best_hyperparams_path = f"{results_path}/best_hyperparams.pth"
    torch.save(best_params, best_hyperparams_path)
    print(f"✅ Best Hyperparameters Saved to {best_hyperparams_path}")
    
    # Evaluate on held-out test set
    evaluate_on_test_set(best_model, X_test, Y_test, results_path)
    
    # Get gene importance using gradient-based method on test set
    from utils import get_top_genes_gradient_based
    get_top_genes_gradient_based(best_model, X_test, gene_names, q=20)

# ✅ New function: Evaluate model on held-out test set
def evaluate_on_test_set(model, X_test, Y_test, results_path):
    model.eval()
    
    with torch.no_grad():
        y_pred = model(X_test).numpy()
    
    # Convert probabilities to binary predictions for F1 score calculation
    y_pred_binary = (y_pred > 0.5).astype(int)
    y_true = Y_test.numpy()
    
    # Calculate metrics for each class
    results = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    auc_scores = []
    
    for i in range(Y_test.shape[1]):
        try:
            if len(set(y_true[:, i])) > 1:  # Only evaluate if class has both positive and negative examples
                class_f1 = f1_score(y_true[:, i], y_pred_binary[:, i])
                class_precision = precision_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
                class_recall = recall_score(y_true[:, i], y_pred_binary[:, i])
                class_auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                
                f1_scores.append(class_f1)
                precision_scores.append(class_precision)
                recall_scores.append(class_recall)
                auc_scores.append(class_auc)
                
                results.append({
                    'Class': i,
                    'F1': class_f1,
                    'Precision': class_precision,
                    'Recall': class_recall,
                    'AUC-ROC': class_auc
                })
        except Exception as e:
            print(f"Error calculating metrics for class {i}: {e}")
    
    # Calculate macro averages
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    avg_auc = sum(auc_scores) / len(auc_scores) if auc_scores else 0
    
    # Calculate micro F1 (treat entire prediction as one)
    micro_f1 = f1_score(y_true.ravel(), y_pred_binary.ravel())
    
    print("✅ Test Set Evaluation Results:")
    print(f"✅ Average F1 Score (Macro): {avg_f1:.4f}")
    print(f"✅ Micro F1 Score: {micro_f1:.4f}")
    print(f"✅ Average Precision (Macro): {avg_precision:.4f}")
    print(f"✅ Average Recall (Macro): {avg_recall:.4f}")
    print(f"✅ Average AUC-ROC: {avg_auc:.4f}")
    
    # Save metrics to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{results_path}/test_class_metrics.csv", index=False)
    print(f"✅ Per-class test metrics saved to {results_path}/test_class_metrics.csv")
    
    # Create summary metrics DataFrame
    summary_metrics = pd.DataFrame({
        'Metric': ['F1 (Macro)', 'F1 (Micro)', 'Precision (Macro)', 'Recall (Macro)', 'AUC-ROC'],
        'Value': [avg_f1, micro_f1, avg_precision, avg_recall, avg_auc]
    })
    summary_metrics.to_csv(f"{results_path}/test_summary_metrics.csv", index=False)
    print(f"✅ Summary test metrics saved to {results_path}/test_summary_metrics.csv")
    
    # Visualize metrics
    plt.figure(figsize=(10, 6))
    metrics = ['F1', 'Precision', 'Recall', 'AUC-ROC']
    values = [avg_f1, avg_precision, avg_recall, avg_auc]
    
    sns.barplot(x=metrics, y=values)
    plt.title('Model Performance Metrics on Test Set')
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig(f"{results_path}/test_model_metrics.png")
    plt.close()
    print(f"✅ Test model metrics visualization saved to {results_path}/test_model_metrics.png")
