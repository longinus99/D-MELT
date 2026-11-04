import argparse

parser = argparse.ArgumentParser(description='D-MELT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', type=str, default='10X_PBMC')
parser.add_argument('--label_key', type=str, default='cell_type',
                    help='obs column used as the true label for H5AD datasets')
parser.add_argument('--save_name', type=str, default=None,
                    help='Output folder name under save/; defaults to --name')
parser.add_argument('--cuda', type=int, default=0 , help='GPU id to use.')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--select_gene', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--dropout', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.2)
parser.add_argument('--m', type=float, default=0.5)
parser.add_argument('--noise', type=float, default=0.1)
parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--enc_1', type=int, default=200)
parser.add_argument('--enc_2', type=int, default=40)
parser.add_argument('--enc_3', type=int, default=60)
parser.add_argument('--mlp_dim', type=int, default=40)
parser.add_argument('--cluster_methods', type=str, default="KMeans")
parser.add_argument('--initial_k', type=int, default=10,
                    help='Initial K for clu_head and evaluation; default is 10')
parser.add_argument('--phi', type=float, default=0.95,
                    help='Threshold for false negative filtering in FN_InstanceLoss')
parser.add_argument('--dp_components', type=int, default=50,
                    help='Number of components for BayesianGaussianMixture DPMM assignment')
parser.add_argument('--dp_max_iter', type=int, default=1000,
                    help='Maximum EM iterations for BayesianGaussianMixture')
parser.add_argument('--dp_tol', type=float, default=1e-3,
                    help='Convergence tolerance for BayesianGaussianMixture')
parser.add_argument('--dp_suppress_convergence_warning', type=int, default=1,
                    help='Set 1 to hide sklearn BayesianGaussianMixture convergence warnings')
parser.add_argument('--dp_start_epoch', type=int, default=40,
                    help='Epoch to start fitting DPMM posterior assignments')
parser.add_argument('--dp_update_interval', type=int, default=1,
                    help='Fit DPMM every N epochs after dp_start_epoch; 1 updates every epoch')
parser.add_argument('--dp_cluster_weight', type=float, default=1.0,
                    help='Weight for cluster contrastive loss using DPMM posterior assignments')
parser.add_argument('--isotropy_loss_weight', type=float, default=1,
                    help='Weight for cluster-wise trace-normalized covariance isotropy loss')

args = parser.parse_args()
if args.save_name is None:
    args.save_name = args.name
