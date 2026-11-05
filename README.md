# D-MELT: Dirichlet process Mixture-guided Embedding Learning via Two-stage training for scRNA-seq Clustering 🧊




## 🧊 Model overview

D-MELT is a two-stage deep clustering framework for scRNA-seq data that combines contrastive representation learning with a Dirichlet process mixture model (DPMM). It learns cell embeddings while accounting for similarities between cells and infers the number of clusters from the learned representations.

1. **Warm-up stage.** Two augmented views of each cell are processed by a MoCo-based encoder with instance and cluster projection heads. Instance-level contrastive learning accounts for potential false negatives, while cluster-level learning encourages consistent assignments across views.
2. **Adaptive clustering stage.** The pretrained encoder and instance head are transferred to the second stage. A DPMM estimates mixture components in the embedding space, and component-level learning is combined with instance-level contrastive learning to refine the representations and clustering structure.

<img width="3542" height="1878" alt="Fig" src="https://github.com/user-attachments/assets/c30ce74e-cb3c-4e5a-83ca-0b698e3ad00e" />






## Data format and output


* Applying to new data, Data must include at least the following two components: a raw gene–cell expression matrix and cell type annotations.
* Unless unavoidable circumstances arise, the **h5ad** format is recommended.
* During training, the model uses the **cell_type** column in **obs** from the h5ad file. The resulting embeddings are stored in **result["features"]**, which can then be saved in **.h5ad** for further downstream task.




## Training D-MELT
We strongly recommend using the `.h5ad` file format for training D-MELT. The `.h5` and `.mat` formats are also supported. To run D-MELT follow these steps:
1. Prepare Datasets File:<br>
  Save your dataset file and change the `path` at `main.py` to your Data directory
2. Configure the Dataset Type:<br>
  In `main.py`, add the dataset name(without extension) to the appropriate list:
    * For H5-format, add to the `h5_datasets` list.
    * For mat-format, add to the `mat_datasets` list.
    * For H5ad-format, you don't have to do anything.


3. Run the code
   Use the command line to run script. For example, to run the 10X_PBMC dataset, execute:


   ```bash
   python main.py --name 10X_PBMC
   ```


## Recommended Hyperparameters of D-MELT
Many settings for training and model configuration are managed in `config.py`. This includes parameters such as the dataset name, number of training epochs, number of genes to select (`--select_gene`), learning rate, dropout rate, and more. For example, the following parameters are defined:


- `--name`: Dataset name (default: 10X_PBMC)
- `--cuda`: Cuda (default: 0)
- `--epoch`: Number of training epochs (default: 200)
- `--dropout`: Dropout rate (default: 0.9)
- `--temperature`: Sharpness of similarity scores for instance-level contrastive module (default: 0.07)


These arguments allow for flexible configuration of the training process and are parsed using Python's `argparse` module.

