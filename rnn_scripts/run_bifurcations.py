import numpy as np
from torch.utils.data import DataLoader
import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
print(os.path.dirname(os.path.realpath(__file__)))
from model import *
from train import *
from tasks.seqDS import *
import torch
import wandb
from utils import *
import matplotlib.pyplot as plt
from bifurcations import *

# --------- CONTROL-----------------
n_pix = 100
model_name = "N512_T0217-151523"
freqs = np.logspace(np.log2(2), np.log2(16), num=n_pix, base=2)
amps = np.arange(0, 3, 3 / n_pix)
sync_wandb = False
dt = 0.5
n_periods = 50
k_ph_range = torch.arange(-np.pi, np.pi, np.pi / 3)
ph0_range = torch.zeros(1)
n_jobs = 2
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

# ---initialise WandB---
if sync_wandb:
    wandb.init(
        project="phase-coding",
        group="pytorch",
        config=config,
    )
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

# Do calculations
Ks, phases = bifur.run_sims(sync_wandb, phase_0, k_0, n_jobs=n_jobs)
print("Ran simulations")
evs, k_diff, dK_norms = bifur.calc_floquet(Ks, phases, n_jobs=n_jobs)
print("Calculated Floquet Multipliers")


# create plot

fig = create_plot(evs, freqs, amps)

# store config and results

config["evs"] = evs
config["Ks"] = Ks
config["phases"] = phases
config["dK_norms"] = dK_norms
config["k_diff"] = k_diff
out_name = "bifur_dat-"+model_name+".pkl"
file = os.path.join(out_dir, out_name)

with open(file, "wb") as f:
    pickle.dump(config, f)

if sync_wandb:
    plt.close()
    figureb = {
        "BIFURC": fig,
    }
    wandb.save(file)
    wandb.log(figureb)
    wandb.finish()
