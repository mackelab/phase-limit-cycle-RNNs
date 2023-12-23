import numpy as np
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from model import *
from train import *
from tasks.seqDS_lfp import seqDS
import time

print("cuda available = " + str(torch.cuda.is_available()))

# Set up the output dir where the output model will be saved
out_dir = os.path.dirname(os.path.realpath(__file__)) + "/../models"
data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../data"

# Check if LFP data is available
for fname in os.listdir(data_dir):
    if fname.endswith('.eeg'):
        print("Found LFP file")
        break
else:
    print("First download the LFP data from http://crcns.org/data-sets/hc/hc-2"
          +" , http://dx.doi.org/10.6080/K0Z60KZ9, Mizuseki K, Sirota A, Pastalkova E, Buzsáki G. (2009)")
    exit()
if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)
n_out = 1
n_inp = 3
n_osc = 1
dt = 1
loadings = None

params = {
    "nonlinearity": "tanh",
    "out_nonlinearity": "identity",
    "rank": 2,
    "train_meanfield": False,
    "train_cov": False,
    "n_supports": 5,
    "n_inp": n_inp,
    "p_inp": 1,
    "n_rec": 512,
    "p_rec": 1,
    "n_out": n_out,
    "cov": None,
    "loadings": loadings,
    "scale_w_inp": 1,
    "scale_w_out": 4,
    "scale_n": 1,
    "scale_m": 1,
    "row_balance_dale": False,
    "apply_dale": False,
    "1overN_out_scaling": True,
    "train_w_inp": True,
    "train_w_inp_scale": False,
    "train_m": True,
    "train_n": True,
    "train_taus": False,
    "train_w_out": False,
    "train_w_out_scale": True,
    "train_b_out": False,
    "train_x0": False,
    "tau_lims": [20],
    "dt": dt,
    "noise_std": 0.05,
    "scale_x0": 1,
    "randomise_x0": True,
    "readout_kappa": False,
}

training_params = {
    "n_epochs": 50,
    "lr": 10e-3,
    "batch_size": 128,
    "clip_gradient": 1,
    "cuda": True,
    "loss_fn": "mse",
    "optimizer": "adam",
    "osc_reg_cost": 0,
    "offset_reg_cost_masked": 0,
    "offset_reg_cost": 0, 
    "l2_rates_cost": 0,
    "l2_cov_cost": 0,
    "orth_indices": 0,
    "osc_reg_freq": 0.2,
    "osc_reg_LFP": True,
    "validation_split": 0.1,
}


task_params = {
    "dt": dt,
    "filter_freqs": [7],
    "phase_freqs": np.arange(7, 9, 0.2),
    "offsets": -1 * np.linspace(0.1, 0.85, (n_inp - n_osc)),
    "stim_on": [0.125, 0.25],
    "stim_dur": [0.125, 0.175],
    "probe_dur": 0.375,
    "n_out": n_out,
    "out_scale": 1,
    "resample_rate": 1.25,
    "rat": 2,
    "max_dur": 4,
    "buffer": 1,
    "artifact_tol": 3,
}

timestr = time.strftime("%m%d-%H%M%S")
fname = "N" + str(params["n_rec"]) + "_T" + timestr

rnn = RNN(params)

ds = seqDS(task_params, path=data_dir)

rnn.train()
loss, _ = train_rnn(
    rnn,
    training_params,
    ds,
    sync_wandb=False,
    params=params,
    task_params=task_params,
    out_dir=out_dir,
    fname=fname,
)
