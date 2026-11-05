import os
import h5py
import config
import numpy as np
import train
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
from utils import preprocess

from sklearn.preprocessing import LabelEncoder


def save_embedding_plot(embedding, labels, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    embedding = np.asarray(embedding)

    if embedding.shape[1] == 1:
        embedding_2d = np.column_stack([embedding[:, 0], np.zeros(embedding.shape[0])])
    elif embedding.shape[1] == 2:
        embedding_2d = embedding
    else:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=9999)
        embedding_2d = reducer.fit_transform(embedding)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels,
        cmap="tab20",
        s=8,
        alpha=0.85,
        linewidths=0,
    )
    plt.title(name)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    out_png = os.path.join(save_dir, f"{name}_umap.png")
    plt.savefig(out_png, dpi=300)
    plt.close()


if __name__ == "__main__":
    h5_datasets = []
    mat_datasets = []

    path = "YOUR DATASET FOLDER PATH"
    dataset = config.args.name
    gene_exp = []
    real_label = []
    if dataset in h5_datasets:
        file_path = os.path.join(path, f"{dataset}.h5")
        data_h5 = h5py.File(file_path, 'r')

        gene_exp = np.array(data_h5.get('X'))
        real_label = np.array(data_h5.get('Y')).reshape(-1)
        gene_exp = preprocess(gene_exp, config.args.select_gene)

    elif dataset in mat_datasets:
        file_path = os.path.join(path, f"{dataset}.mat")
        data_mat = sio.loadmat(file_path)

        gene_exp = np.array(data_mat['feature'])
        real_label = np.array(data_mat['label']).reshape(-1)
        gene_exp = preprocess(gene_exp, config.args.select_gene)

    else:
        file_path = os.path.join(path, f"{dataset}.h5ad")
        adata = sc.read_h5ad(file_path)

        gene_expr = adata.X
        label_key = config.args.label_key
        if label_key not in adata.obs.columns:
            raise KeyError(
                f"Label column '{label_key}' not found in {file_path}. "
                f"Available obs columns: {list(adata.obs.columns)}"
            )
        string_labels = np.array(adata.obs[label_key]).reshape(-1)
        encoder = LabelEncoder()
        real_label = encoder.fit_transform(string_labels)
        gene_exp = preprocess(gene_expr, config.args.select_gene)
        gene_exp = np.asarray(gene_exp)

        
    print(f"The gene expression matrix shape is {gene_exp.shape}...")
    true_cluster_number = np.unique(real_label).shape[0]
    initial_k = config.args.initial_k
    print(f"The real clustering num is {true_cluster_number}...")
    print(f"The configured initial K is {initial_k}...")

    results = train.run(gene_exp=gene_exp, 
                        initial_k=initial_k,
                        dataset=config.args.name,
                        real_label=real_label, 
                        epochs=config.args.epoch, 
                        lr=config.args.lr,
                        temperature=config.args.temperature, 
                        dropout=config.args.dropout,
                        layers=[config.args.enc_1, config.args.enc_2, config.args.enc_3, config.args.mlp_dim],
                        save_pred=True, 
                        cluster_methods=config.args.cluster_methods, 
                        batch_size=config.args.batch_size,
                        m=config.args.m, 
                        noise=config.args.noise)
    print("==================== Performance Metrics ===================")  
    print("NMI:" + str(results["nmi"]))
    print("ARI:" + str(results["ari"]))
    final_pred_k = np.unique(np.argmax(results["q_cluster"], axis=1)).shape[0]
    print("Predicted K:" + str(final_pred_k))
    print("True label K:" + str(true_cluster_number))

    # save_dir = os.path.join(os.getcwd(), "save", config.args.save_name, "embeddings")
    # save_embedding_plot(results["q_instance"], real_label, save_dir, "q_instance")
    # if dataset not in h5_datasets and dataset not in mat_datasets:
    #     adata.obsm["embedding"] = results["features"]
    #     adata.obsm["soft_assignment"] = results["q_cluster"]
    #     adata.obs["prediction"] = np.argmax(results["q_cluster"], axis=1).astype(str)
    #     out_h5ad = os.path.join(os.getcwd(), "save", config.args.save_name, f"{config.args.save_name}_embedding.h5ad")
    #     adata.write_h5ad(out_h5ad)
