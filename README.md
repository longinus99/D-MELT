## Data format and output

* Applying to new data, Data must include at least the following two components: a raw gene–cell expression matrix and cell type annotations.
* Unless unavoidable circumstances arise, the **h5ad** format is recommended.
* During training, the model uses the **cell_type** column in **obs** from the h5ad file. The resulting embeddings are stored in **result["features"]**, which can then be saved in **.obsm** for further downstream task.


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

## Recommended Hyperparameters of D-MELT
Many settings for training and model configuration are managed in `config.py`. This includes parameters such as the dataset name, number of training epochs, number of genes to select (`--select_gene`), learning rate, dropout rate, and more. For example, the following parameters are defined:

- `--name`: Dataset name (default: 10X_PBMC)
- `--cuda`: Cuda (default: 0)
- `--epoch`: Number of training epochs (default: 200)
- `--dropout`: Dropout rate (default: 0.9)
- `--temperature`: Sharpness of similarity scores for instance-level contrastive module (default: 0.07)

These arguments allow for flexible configuration of the training process and are parsed using Python's `argparse` module. For contrastive learning, larger batch sizes tend to improve training stability. 
