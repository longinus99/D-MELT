import os
import D_MELT
import config
import time
import numpy as np
import torch
import loss_module
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from utils import get_device, adjust_learning_rate, save_model, cluster_embedding


def run(gene_exp, initial_k, dataset, real_label, epochs, lr, temperature, dropout, layers, batch_size, m,
        save_pred=True, noise=None, use_cpu=None, cluster_methods=None):
    results = {}

    start = time.time()
    embedding, q_cluster_embedding, best_model = train_model(gene_exp=gene_exp, initial_k=initial_k, real_label=real_label,
                                                             epochs=epochs, lr=lr, temperature=temperature,
                                                             dropout=dropout, layers=layers, batch_size=batch_size,
                                                             m=m, save_pred=save_pred, noise=noise, use_cpu=use_cpu)

    if save_pred:
        results[f"features"] = embedding
        results[f"q_instance"] = embedding
        results[f"q_cluster"] = q_cluster_embedding
        results[f"max_epoch"] = best_model
    elapsed = time.time() - start
    pred = np.argmax(q_cluster_embedding, axis=1)
    res_eval = {
        "ari": round(adjusted_rand_score(real_label, pred), 4),
        "nmi": round(normalized_mutual_info_score(real_label, pred), 4),
        "pred": pred,
    }
    results = {**results, **res_eval, "dataset": dataset, "time": elapsed}

    return results


def train_model(gene_exp, initial_k, real_label, epochs, lr,
                temperature, dropout, layers, batch_size, m,
                save_pred=False, noise=None, use_cpu=None, evaluate_training=True):
    device = get_device(use_cpu, gpu_id=config.args.cuda)

    print(device)

    dims = np.concatenate([[gene_exp.shape[1]], layers])
    data_aug_model = D_MELT.DataAug(dropout=dropout)
    encoder_q = D_MELT.BaseEncoder(dims)
    encoder_k = D_MELT.BaseEncoder(dims)
    instance_projector = D_MELT.MLP(layers[2], layers[2] + layers[3], layers[2] + layers[3])
    cluster_cfg = [layers[2], layers[3]]
    
    model = D_MELT.D_MELT(encoder_q, encoder_k, instance_projector, cluster_cfg, initial_k, m=m)
    data_aug_model.to(device)
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    criterion_instance = loss_module.FN_InstanceLoss(temperature=temperature , phi=config.args.phi, debug_opt=False)
    criterion_cluster = loss_module.Corr_ClusterLoss()
    criterion_dp = loss_module.DPMMLoss(
        n_components=config.args.dp_components,
        max_iter=config.args.dp_max_iter,
        tol=config.args.dp_tol,
        suppress_convergence_warning=bool(config.args.dp_suppress_convergence_warning),
    )
    criterion_dp_eval = loss_module.DPMMLoss(
        n_components=config.args.dp_components,
        max_iter=config.args.dp_max_iter,
        tol=config.args.dp_tol,
        suppress_convergence_warning=bool(config.args.dp_suppress_convergence_warning),
    )

    max_value, best_model = -1, -1
    prev_dp_k = None
    active_dp_components = None
    dp_k_log_path = os.path.join(os.getcwd(), "save", config.args.save_name, "dp_k_change_log.txt")
    dp_update_interval = max(1, config.args.dp_update_interval)

    idx = np.arange(len(gene_exp))
    for epoch in range(epochs):
        model.train()
        adjust_learning_rate(optimizer, epoch, lr)
        use_dp = epoch >= config.args.dp_start_epoch
        refit_dp = use_dp and (epoch == config.args.dp_start_epoch or
                               (epoch - config.args.dp_start_epoch) % dp_update_interval == 0)
        if refit_dp:
            model.eval()
            with torch.no_grad():
                full_q_instance, _, _, _ = model(torch.FloatTensor(gene_exp).to(device), None)
            criterion_dp.fit(full_q_instance)
            with torch.no_grad():
                full_dp_posterior = criterion_dp.posterior(full_q_instance)
                active_dp_components = torch.unique(
                    full_dp_posterior.argmax(dim=1), sorted=True
                )
                current_dp_k = int(active_dp_components.numel())
            if prev_dp_k is not None and current_dp_k != prev_dp_k:
                os.makedirs(os.path.dirname(dp_k_log_path), exist_ok=True)
                with open(dp_k_log_path, "a") as f:
                    f.write(f"epoch {epoch}: K {prev_dp_k} -> {current_dp_k} (delta {current_dp_k - prev_dp_k:+d})\n")
            prev_dp_k = current_dp_k
            model.train()
        np.random.shuffle(idx)
        loss_instance_ = 0
        loss_cluster_ = 0
        loss_isotropy_ = 0
        for pre_index in range(len(gene_exp) // batch_size + 1):
            c_idx = np.arange(pre_index * batch_size,
                              min(len(gene_exp), (pre_index + 1) * batch_size))
            if len(c_idx) == 0:
                continue
            c_idx = idx[c_idx]
            c_inp = gene_exp[c_idx]
            input1 = data_aug_model(torch.FloatTensor(c_inp).to(device))
            input2 = data_aug_model(torch.FloatTensor(c_inp).to(device))

            if noise is None or noise == 0:
                input1 = torch.FloatTensor(input1).to(device)
                input2 = torch.FloatTensor(input2).to(device)
            else:
                noise1 = torch.normal(mean=0.0, std=noise, size=input1.shape, device=input1.device, dtype=input1.dtype)
                input1 = input1 + noise1
                noise2 = torch.normal(mean=0.0, std=noise, size=input2.shape, device=input2.device, dtype=input2.dtype)
                input2 = input2 + noise2
            q_instance, q_cluster, k_instance, k_cluster = model(input1, input2)

            features_instance = torch.cat(
                [q_instance.unsqueeze(1),
                 k_instance.unsqueeze(1)],
                dim=1)
            
            loss_instance = criterion_instance(features_instance)

            if use_dp:
                q_dp_cluster = criterion_dp.posterior(q_instance)[:, active_dp_components]
                k_dp_cluster = criterion_dp.posterior(k_instance)[:, active_dp_components]
                q_dp_cluster = q_dp_cluster / q_dp_cluster.sum(dim=1, keepdim=True).clamp_min(1e-8)
                k_dp_cluster = k_dp_cluster / k_dp_cluster.sum(dim=1, keepdim=True).clamp_min(1e-8)
                loss_cluster = criterion_cluster(
                    q_dp_cluster, k_dp_cluster, use_entropy=False
                ) * config.args.dp_cluster_weight
                loss_isotropy = (criterion_dp.isotropy_loss(q_instance)
                                 * config.args.isotropy_loss_weight)
            else:
                loss_cluster = criterion_cluster(q_cluster, k_cluster)
                loss_isotropy = q_instance.new_tensor(0.0)

            # eps = 1e-8
            # p = q_cluster.clamp_min(eps)
            # p = p / (p.sum(dim=1, keepdim=True) + eps)
            # q = k_cluster.clamp_min(eps)
            # q = q / (q.sum(dim=1, keepdim=True) + eps)

            # kl_qk = (p * (p.add(eps).log() - q.add(eps).log())).sum(dim=1).mean()
            # kl_kq = (q * (q.add(eps).log() - p.add(eps).log())).sum(dim=1).mean()
            # loss_consistency = 0.5 * (kl_qk + kl_kq)


            loss = loss_instance + loss_cluster + loss_isotropy
            loss_instance_ += loss_instance.item()
            loss_cluster_ += loss_cluster.item()
            loss_isotropy_ += loss_isotropy.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if evaluate_training and real_label is not None:
            model.eval()
            with torch.no_grad():
                q_instance, _, _, _ = model(torch.FloatTensor(gene_exp).to(device), None)
                if use_dp:
                    criterion_dp_eval.fit(q_instance)
                    dp_posterior = criterion_dp_eval.posterior(q_instance)
                    dp_pred = dp_posterior.argmax(dim=1).cpu().numpy()
                    res = {
                        "ari": round(adjusted_rand_score(real_label, dp_pred), 4),
                        "nmi": round(normalized_mutual_info_score(real_label, dp_pred), 4),
                    }
                    metric_name = "DPMM"
                else:
                    features = q_instance.detach().cpu().numpy()
                    res = cluster_embedding(
                        features, initial_k, real_label, save_pred=save_pred
                    )
                    metric_name = "KMeans"
            print(
                f"Epoch {epoch}: Loss: {loss_instance_ + loss_cluster_ + loss_isotropy_}, "
                f"Instance:{loss_instance_}, Cluster:{loss_cluster_}, "
                f"Isotropy:{loss_isotropy_}, "
                f"{metric_name} NMI: {res['nmi']}, {metric_name} ARI: {res['ari']}"
            )

            if use_dp and res['ari'] + res['nmi'] >= max_value:
                max_value = res['ari'] + res['nmi']
                save_model(
                    config.args.save_name, model, optimizer, epoch, best_model,
                    dp_posterior=dp_posterior.detach().cpu()
                )
                best_model = epoch

    if best_model < 0:
        raise RuntimeError(
            "No DPMM checkpoint was selected. Set --epoch greater than --dp_start_epoch."
        )

    model.eval()
    model_fp = os.path.join(os.getcwd(), "save", config.args.save_name, f"checkpoint_{best_model}.tar")
    checkpoint = torch.load(model_fp, map_location=device)
    model.load_state_dict(checkpoint['net'])
    model.to(device)

    with torch.no_grad():
        q_instance, q_cluster, _, _ = model(torch.FloatTensor(gene_exp).to(device), None)
        q_np = q_instance.detach().cpu().numpy()
        if epochs > config.args.dp_start_epoch:
            q_cluster = checkpoint['dp_posterior'].to(device)
        q_cluster_np = q_cluster.detach().cpu().numpy()

    return q_np, q_cluster_np, best_model
