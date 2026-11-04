import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import BayesianGaussianMixture
from sklearn.exceptions import ConvergenceWarning
                
EPS = 1e-8

class FN_InstanceLoss(nn.Module):
    def __init__(self, temperature, phi=0.95, debug_opt=True):
        super(FN_InstanceLoss, self).__init__()
        self.temperature = temperature
        self.phi = phi
        self.debug_opt = debug_opt

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask.bool()


    def forward(self, features):
        
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('features needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        
        batch_size = features.shape[0]
        N = 2 * batch_size
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = F.normalize(contrast_feature, dim=-1)

        raw_sim = torch.matmul(contrast_feature, contrast_feature.T)
        raw_sim = torch.clamp(raw_sim, min=-1.0, max=1.0)

        sim = raw_sim / self.temperature

        #base negative_samples
        neg_mask = self.mask_correlated_samples(batch_size).to(device)
        negative_samples = sim[neg_mask].reshape(N, -1)  
        

        #phi_base negative_samples
        neg_raw_sim = raw_sim[neg_mask].reshape(N, -1)

        # print(neg_raw_sim.shape)
        # cosine similarity가 phi 이상인 false negative는
        # 가중치를 0으로 두어 분모에서 제거한다.
        neg_weights = torch.where(
            neg_raw_sim >= self.phi,
            torch.zeros_like(neg_raw_sim),
            torch.ones_like(neg_raw_sim),
        ).detach()

        # print(neg_raw_sim)

        #positive_samples 
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        pos = positive_samples.squeeze(1)
    
        #loss
        # log(weight * exp(logit)) = logit + log(weight)
        weighted_negative_logits = negative_samples + torch.log(neg_weights)
        den = torch.logsumexp(
            torch.cat([pos.unsqueeze(1), weighted_negative_logits], dim=1),
            dim=1,
        )

        # 최종 손실:
        # -log(exp(pos) / (exp(pos) + Σ weight_j * exp(neg_j)))
        loss = (-pos + den).mean()

        if self.debug_opt:
            # false_neg_mask: "neg 후보" 중에서 cosine >= phi 인 위치만 True
            false_neg_mask = neg_mask & (raw_sim >= self.phi)
            debug = {
                "contrast_feature": contrast_feature.detach(),
                "raw_sim": raw_sim.detach(),
                "neg_mask_full": neg_mask,         # [N,N] bool
                "false_neg_mask": false_neg_mask,       # [N,N] bool
                "batch_size": batch_size,
            }
            return loss, debug


        return loss

def off_diag(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class Corr_ClusterLoss(nn.Module):
    def __init__(self):
        super(Corr_ClusterLoss, self).__init__()

    def forward(self, q_cluster, k_cluster, use_entropy=True):

        q_cluster_t = q_cluster.t()
        k_cluster_t = k_cluster.t()

        if use_entropy:
            p_i = q_cluster_t.sum(1).view(-1)
            p_i /= p_i.sum() + EPS
            ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i + EPS)).sum()
            p_j = k_cluster_t.sum(1).view(-1)
            p_j /= p_j.sum() + EPS
            ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j + EPS)).sum()
            ne_loss = ne_i + ne_j
        else:
            ne_loss = q_cluster.new_tensor(0.0)

        S = torch.mm(
            F.normalize(q_cluster.t(), dim=1, p=2),
            F.normalize(k_cluster, dim=0, p=2)
        )
        
        diagonal_loss = (torch.diagonal(S) - 1).pow(2).mean()
        if S.size(0) > 1:
            non_diagonal_loss = off_diag(S).pow(2).mean()
        else:
            non_diagonal_loss = S.new_tensor(0.0)
        loss = diagonal_loss + non_diagonal_loss
        return ne_loss + loss


class DPMMLoss(nn.Module):
    def __init__(self, n_components=50, weight_concentration_prior=1e-10,
                 mean_precision_prior=80, covariance_type="diag", max_iter=1000,
                 tol=1e-3, suppress_convergence_warning=True):
        super(DPMMLoss, self).__init__()
        self.n_components = n_components
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.suppress_convergence_warning = suppress_convergence_warning
        self.mean_dp = None
        self.weight_concentration_dp = None
        self.precisions_cholesky_dp = None
        self.degrees_of_freedom_dp = None
        self.mean_precision_dp = None

    def fit(self, z):
        z_np = z.detach().cpu().numpy()
        n_components = min(self.n_components, z_np.shape[0])
        with warnings.catch_warnings():
            if self.suppress_convergence_warning:
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
            bgm = BayesianGaussianMixture(
                n_components=n_components,
                weight_concentration_prior=self.weight_concentration_prior,
                mean_precision_prior=self.mean_precision_prior,
                covariance_type=self.covariance_type,
                init_params="kmeans",
                max_iter=self.max_iter,
                tol=self.tol,
                warm_start=True,
            ).fit(z_np)

        device = z.device
        dtype = z.dtype
        self.mean_dp = torch.as_tensor(np.array(bgm.means_), device=device, dtype=dtype)
        self.weight_concentration_dp = torch.as_tensor(np.array(bgm.weight_concentration_), device=device, dtype=dtype)
        self.precisions_cholesky_dp = torch.as_tensor(np.array(bgm.precisions_cholesky_), device=device, dtype=dtype)
        self.degrees_of_freedom_dp = torch.as_tensor(np.array(bgm.degrees_of_freedom_), device=device, dtype=dtype)
        self.mean_precision_dp = torch.as_tensor(np.array(bgm.mean_precision_), device=device, dtype=dtype)

    def _check_fitted(self):
        if self.mean_dp is None:
            raise RuntimeError("DPMMLoss.fit(z) must be called before posterior/loss.")

    def log_prob_and_weights(self, z):
        self._check_fitted()
        mean_dp = self.mean_dp.to(device=z.device, dtype=z.dtype)
        weight_concentration_dp = self.weight_concentration_dp.to(device=z.device, dtype=z.dtype)
        precisions_cholesky_dp = self.precisions_cholesky_dp.to(device=z.device, dtype=z.dtype)
        degrees_of_freedom_dp = self.degrees_of_freedom_dp.to(device=z.device, dtype=z.dtype)
        mean_precision_dp = self.mean_precision_dp.to(device=z.device, dtype=z.dtype)

        n_features = z.size(1)
        pi = 3.141592653589793
        digamma_sum_dp = torch.special.digamma(weight_concentration_dp[0] + weight_concentration_dp[1])
        digamma_a_dp = torch.special.digamma(weight_concentration_dp[0])
        digamma_b_dp = torch.special.digamma(weight_concentration_dp[1])
        log_weights_dp_b = torch.cat(
            (
                torch.zeros(1, device=z.device, dtype=z.dtype),
                torch.cumsum((digamma_b_dp - digamma_sum_dp), dim=0)[:-1],
            ),
            dim=0,
        )
        log_weights_dp = digamma_a_dp - digamma_sum_dp + log_weights_dp_b
        log_det_dp = torch.sum(precisions_cholesky_dp.log(), dim=1)
        precisions_dp = precisions_cholesky_dp ** 2
        log_prob_dp = (
            torch.sum((mean_dp ** 2 * precisions_dp), dim=1)
            - 2.0 * torch.mm(z, (mean_dp * precisions_dp).T)
            + torch.mm(z ** 2, precisions_dp.T)
        )
        log_gauss_pre_dp = -0.5 * (n_features * math.log(pi * 2) + log_prob_dp) + log_det_dp
        log_gauss_dp = log_gauss_pre_dp - 0.5 * n_features * degrees_of_freedom_dp.log()
        feature_range = torch.arange(0, n_features, device=z.device, dtype=z.dtype).unsqueeze(1)
        log_lambda_dp = n_features * math.log(2.0) + torch.sum(
            torch.special.digamma(0.5 * (degrees_of_freedom_dp - feature_range)),
            0,
        )
        log_prob_z_dp = log_gauss_dp + 0.5 * (log_lambda_dp - n_features / mean_precision_dp)
        return log_prob_z_dp, log_weights_dp

    def posterior(self, z):
        log_prob_z_dp, log_weights_dp = self.log_prob_and_weights(z)
        return F.softmax(log_prob_z_dp + log_weights_dp, dim=1)

    def effective_k(self, z):
        posterior = self.posterior(z)
        assignments = posterior.argmax(dim=1)
        return int(torch.unique(assignments).numel())

    def isotropy_loss(self, z, min_cluster_size=3, eps=1e-8):
        """Penalize anisotropic covariance within each hard-assigned DPMM component."""
        self._check_fitted()
        with torch.no_grad():
            assignments = self.posterior(z).argmax(dim=1)

        feature_dim = z.size(1)
        identity_target = torch.eye(
            feature_dim, device=z.device, dtype=z.dtype
        ) / feature_dim
        cluster_losses = []

        for cluster_id in torch.unique(assignments):
            cluster_z = z[assignments == cluster_id]
            if cluster_z.size(0) < min_cluster_size:
                continue
            centered = cluster_z - cluster_z.mean(dim=0, keepdim=True)
            covariance = centered.T @ centered / cluster_z.size(0)
            normalized_covariance = covariance / (torch.trace(covariance) + eps)
            cluster_losses.append(
                (normalized_covariance - identity_target).pow(2).sum()
            )

        if not cluster_losses:
            return z.sum() * 0.0
        return torch.stack(cluster_losses).mean()
