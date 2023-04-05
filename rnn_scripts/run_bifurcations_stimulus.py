import numpy as np
from torch.utils.data import DataLoader
import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
print(os.path.dirname(os.path.realpath(__file__)))
from model import *
from train import *
from tasks.seqDS import *
import torch
from utils import *
import matplotlib.pyplot as plt
from bifurcations import *

# --------- CONTROL-----------------
n_pix = 100
model_name = "N512_T0217-151523"
freqs = [8]
amps = np.arange(0,.251,.0025)
dt = 0.5
n_periods = 100
k_ph_range = torch.arange(-np.pi, np.pi, np.pi / 3)
ph0_range = torch.ones(1)*-np.pi
device = "cpu"
model_dir = os.getcwd()+"/../models/"
out_dir = os.getcwd()+"/../data/"

config = {
    "freqs": freqs,
    "amps": amps,
    "dt": dt / 1000,
    "n_periods": n_periods,
    "k_ph_range": k_ph_range,
    "ph0_range": ph0_range,
    "model_name": model_name,
}


# ---Assert thresholds-----
max_loss = 0.1
loss_tol = 2

# Load model
rnn, params, task_params, training_params = load_rnn(
    model_dir + model_name, device=device
)
set_dt(task_params, rnn, dt)
make_deterministic(task_params, rnn)

# Initialise test dataset
ds = seqDS(task_params)

dataloader = DataLoader(ds, batch_size=16, shuffle=True)
test_input, test_target, test_mask = next(iter(dataloader))

# Compute accuracy
_, _, loss1 = predict(
    rnn, test_input, mse_loss, test_target, test_mask, return_loss=True
)

assert loss1 < max_loss, "The loss: {:.2E} is higher then the threshold: {:.2E}".format(
    loss1, max_loss
)

# Orthogonalise singular vectors and scale input weights
rnn.rnn.svd_orth()
weight_scalers_to_1(rnn)

_, _, loss2 = predict(
    rnn, test_input, mse_loss, test_target, test_mask, return_loss=True
)

assert loss2 < (
    loss1 * loss_tol
), "The loss after orthogonalisation: {:.2E} is more \
                                then {:.2f} times the original loss {:.2E}".format(
    loss2, loss_tol, loss1
)

# Extract connectivity
I, n, m, W = extract_loadings(rnn, orth_I=False, split=True)
alpha, I_orth = orthogonolise_Im(I, m)
alpha, I_orth, m, n = to_torch(alpha, I_orth, m, n, device=device)

# initialise connectivity
bifur = bifurcation(alpha, I_orth, m, n, rnn.rnn.tau / 1000, config)
print("device = " + str(bifur.m.device))

# create initial_conditions
rad = calculate_mean_radius(task_params["freq"], rnn)
print("Radius is " + str(rad))
phase_0, k_0 = create_bifur_ICs(ph0_range, k_ph_range, rad, device)
stim1 = torch.zeros(2,dtype=torch.float32)
stim1[0]=1
stim2 = torch.zeros(2,dtype=torch.float32)
stim2[1]=1
stims=[stim1,stim2]

# Do calculations
Ks, phases = bifur.run_sims_stim(False,phase_0,k_0,stims)
print("Ran simulations")
evs,evs_c,k_diff,dK_norms=bifur.calc_floquet_stim(False,Ks,phases,stims)
print("Calculated Floquet Multipliers")


# store config and results

config["evs"] = evs
config["Ks"] = Ks
config["phases"] = phases
config["dK_norms"] = dK_norms
config["k_diff"] = k_diff
out_name = "stim_bifur_dat-"+model_name+".pkl"
file = os.path.join(out_dir, out_name)
with open(file, "wb") as f:
    pickle.dump(config, f)

