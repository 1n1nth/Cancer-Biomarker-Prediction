import torch
import pandas as pd
from utils import load_model_and_predict
import os
import matplotlib.pyplot as plt
import numpy as np

# File paths
model_path = r"./results/best_model.pt"
hyperparams_path = r"./results/best_hyperparams.pth"
file_path = r"./data/RNA.csv"
mutation_path = r"./data/MUTATION.csv"

# Load the RNA data
df = pd.read_csv(file_path)

# Load the mutation data to get the columns
mutation_df = pd.read_csv(mutation_path)
fold_columns = [col for col in mutation_df.columns if col.startswith('FOLD_')]
mutation_cols = [col for col in mutation_df.columns if col not in fold_columns and 
                 col in ['CD53', 'CD58', 'CD2', 'CD1D', 'CD1A', 'CD1C', 'CD1B', 'CD1E', 
                        'CD84', 'CD48', 'CD8A', 'CD8B', 'CTLA4']]

# Get the number of output classes
num_classes = len(mutation_cols)
#print(f"Number of mutation target columns: {num_classes}")
#print(f"Mutation columns: {mutation_cols}")

# Select a sample row for prediction
row_index = 7010
row_values = df.iloc[row_index, :2048].tolist()

# Convert to tensor
input_sample = torch.tensor([row_values], dtype=torch.float32) 

# Get gene names from RNA data
gene_names = pd.read_csv(file_path).columns[:2048]

# Get top genes for the sample
top_genes,top_genes_indices = load_model_and_predict(input_sample, model_path, hyperparams_path, gene_names, num_classes=num_classes)

top_genes_indices=top_genes_indices.numpy()/1000
top_genes = np.array(top_genes)
sorted_indices = np.argsort(top_genes_indices)  # Ascending order

# Reorder both arrays using sorted indices
top_genes_sorted = top_genes[sorted_indices]
top_genes_indices_sorted = top_genes_indices[sorted_indices]

top_genes_sorted=top_genes_sorted[::-1]
top_genes_indices_sorted=top_genes_indices_sorted[::-1]
      

def create_excel_style_gene_table(genes, values, save_path, img_size=(4, 10), font_size=12):
    # Convert data to a Pandas DataFrame
    df = pd.DataFrame({'Gene': genes, 'Importance Score ': values})

    fig, ax = plt.subplots(figsize=img_size)
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.auto_set_column_width([0, 1])  
    table.scale(1.2, 1.5) 

    os.makedirs(save_path, exist_ok=True)

    output_path = os.path.join(save_path, "gene_table.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"Image saved at: {output_path}")

output_path=r"./results"   
create_excel_style_gene_table(top_genes_sorted,top_genes_indices_sorted,output_path,(4,10),12)

    
