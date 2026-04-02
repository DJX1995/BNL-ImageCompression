import yaml
from easydict import EasyDict as edict


config = edict()

#----------------path paramerters-------------------------
config.log_path = '../experiments/logs/'
config.model_path = f'../experiments/models/'
config.dataset_path = '../data/CBASS/split2/'
config.save_path = f'../experiments/data_recon/'

#----------------model paramerters------------------------
config.model_name = 'bmshj2018-hyperprior'  # 'tic_hp'    # 'cheng2020-attn'  # 'bmshj2018-hyperprior'
config.model_quality = 4
config.save_name = 'adaptive_hyperprior_bgrestrict' # 'adaptive_hyperprior_bgrestrict'
config.random_seed = 42
config.dim_in = 1

#----------------hyper paramerters------------------------
config.lmbda = 10.
config.lmbda_fg = 150.
config.lmbda_bg_over = 0.01
config.lmbda_fg_under = 0.5
config.spot_radius = 9
config.spot_fg_radius = 9
config.Rcut = 12
config.spot_type = 'diamond'
config.spot_gaus_sigma = 0.3
config.spot_mask_type = 'gaus_plat'  # 'binary'  # 'gaus_plat'
config.tau_fg = 0.  # percentage of the tolerance

#----------------training paramerters---------------------
config.learning_rate = 1e-4
config.aux_learning_rate = 1e-4
config.batch_size = 24
config.num_epoch = 3
config.num_workers = 4
config.clip_max_norm = 1.0
config.save = True
config.resume = False
config.eval = False

#----------------visualization paramerters---------------------
config.nbins = 20