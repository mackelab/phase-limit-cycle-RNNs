import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import mixture
import torch
from matplotlib import colors
import os, sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from model import *
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from tasks.seqDS import  seqDS as seqDS
from tasks.seqDS_lfp import seqDS as seqDS_LFP

from train import *



def poincare_map(Ks, period, t0, phases, ph0=0):
    """
    Calculate the Poincare map of a given RNN trajectory with constant sinusoidal input.

    Args:
        Ks: RNN trajectory projected on the left singular vectors
        period: period of the sinusoidal input
        t0: time index of the first point in the Poincare map
        phases: phases of the sinusoidal input at t0
    
    Returns:
        pm: Poincare map

    """
    n_tr, n_k, n_t = Ks.shape
    n_pm_t = int((n_t - t0) / period) - 1

    # align poincare maps by phases of the reference oscillation
    phase_ind = np.int_(period * (-phases + np.min(phases) + ph0) / (np.pi * 2))
    pm = np.zeros((n_tr, n_k, n_pm_t))
    # loop through different trials
    for i in range(len(Ks)):
        t0i = t0 + phase_ind[i]
        pm_inds = np.arange(t0i, period * n_pm_t + t0i, period)
        pm[i] = Ks[i, :, pm_inds].T
    return pm

def get_traj(rnn, task_params, freq, amp_scale=1,apply_tanh_rates=True,batch_size=50, stim_scale=1):
    """
    Return the RNN trajectory for a given frequency and amplitude of the reference oscillation

    Args:
        rnn: RNN model
        task_params: task parameters
        freq: frequency of the reference oscillation
        amp_scale: amplitude of the reference oscillation
        apply_tanh_rates: whether to apply tanh to the RNN rates
        batch_size: batch size for the dataloader

    Returns:
        ks: RNN trajectories projected on the left singular vectors
        rates_aligned: RNN rates, aligned by phase
        phases: phases of the sinusoidal reference oscillation

    """
    task_params_ = task_params.copy()
    task_params_["freq"] = freq
    ds = seqDS(task_params_)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    ts = task_params["dt"] / 1000
    ps = 1 / freq
    period = ps / ts
    qp = int(period / 4)
    period = int(period)

    # create task input
    test_input, test_target, test_mask = next(iter(dataloader))
    test_input[:, :, 0] *= amp_scale
    labels = extract_labels(test_input, rnn.params["n_inp"] - 1)

    # simulate RNN
    test_input[:, :, 1:] *= stim_scale
    rates, _ = predict(rnn, test_input, mse_loss, test_target, test_mask)
    # extract input phase and project RNN activity on left singular vectors
    trials = [np.where(labels == i)[0][0] for i in range(rnn.params["n_inp"] - 1)]
    n_periods = 1
    ks = np.zeros((len(trials), rnn.rnn.rank, period + 1))
    rates_aligned = np.zeros((len(trials), period,rnn.rnn.N))
    phases = np.linspace(0, np.pi * 2 * n_periods, period + 1)
    _, _, m, _ = extract_loadings(rnn, orth_I=False, split=True)

    # loop through different trials
    for i, ind in enumerate(trials):
        k = np.array(proj(m, rates[ind, -period:]))
        r = rates[ind, -period:]
        phase = (
            np.arctan2(
                test_input[ind, -(qp + period) : -qp, 0], test_input[ind, -period:, 0]
            )
            .cpu()
            .numpy()
        )
        phase = wrap(phase)
        # sort such that all trajectories start with the same phase
        time_ind = np.arange(len(phase))
        time_ind = np.roll(time_ind, -np.argmin(phase))
        k = k[:, time_ind]
        r = r[time_ind]
        phase = phase[time_ind]
        ks[i, :, :period] = k
        rates_aligned[i,:period]=r
    ks[:, :, -1] = ks[:, :, 0]
    if apply_tanh_rates:
        rates_aligned=np.tanh(rates_aligned)
    return ks, phases,rates_aligned




def plot_covs(covs, vm, labels=None, titles=None, figsize=None, dpi=100, fontsize=8, float_labels=False,atol=0.1, float_lims=5,label_fs=8,numbers_fs=8):
    """
    Plot coviarance matrices

    Args:
        covs: list of covariance matrices
        vm: maximum value of the colorbar
        labels: labels of the covariance matrix entries
        titles: titles of the covariance matrices
        figsize: figure size
        dpi: figure resolution
        fontsize: font size of the numbers in the covariance matrix plots
    
    Returns:
        fig: figure handle
        axs: axis handle
    """
    
    n_cov, n_l = np.shape(covs)[:2]
    if not figsize:
        figsize = (6, n_cov * 6)
    fig, axs = plt.subplots(1, n_cov, figsize=figsize, dpi=dpi)
    for i, cov in enumerate(covs):
        data = np.ma.masked_where(np.triu(np.ones((n_l, n_l))) < 0.01, cov[:n_l, :n_l])
        axs[i].imshow(data, cmap="coolwarm", vmin=-vm, vmax=vm)
        add_numbers(axs[i], data, numbers_fs,float_labels,float_lims,atol)

        axs[i].yaxis.tick_right()
        axs[i].xaxis.tick_top()
        axs[i].spines["left"].set_visible(False)
        axs[i].spines["bottom"].set_visible(False)
        axs[i].spines["top"].set_visible(True)
        axs[i].spines["right"].set_visible(True)
        if titles is not None:
            axs[i].set_title(titles[i])
        if labels is not None:

            axs[i].set_xticks(np.arange(n_l), labels,fontsize=label_fs)
            axs[i].set_yticks([], labels=[])
    if labels is not None:
        axs[i].set_yticks(np.arange(n_l), labels,fontsize=label_fs)
    return fig, axs

def add_numbers(ax, grid, fontsize,float_labels=False,float_lims=5,atol=0.1):
    """add numbers to covariance matrix plots"""
    for (j, i), label in np.ndenumerate(grid):
        if not np.isclose(label, 0, atol=atol):
            if j <= i:
                if float_labels and abs(label)<float_lims:
                    ax.text(i, j, "{:.1f}".format(label), ha="center", va="center", fontsize=fontsize)
                else:
                    ax.text(i, j, int(label), ha="center", va="center", fontsize=fontsize)


def create_ICs(r_range, phi_range, theta_range, tau, T, dt, w, m, I_orth, stim_ind=0):
    """
    Create ICs for RNN simulations

    Args:
        r_range: range of the radius in the k1 k2 plane
        phi_range: range of the angle / phase of the RNN in the k1 k2 plane
        theta_range: range of the initial phase of the reference oscillation
        tau: time constant of the RNN in ms
        T: simulation time in s
        dt: simulation time step in ms
        w: frequency of the reference oscillation
        m: left singular vectors of the RNN
        I_orth: orthogonalised input vectors of the RNN
        stim_ind: index of the input vector that is used for stimulus input

    Returns:
        input_ICs: input to the RNN
        x0s: RNN states at t=0
        phases: phases of the reference oscillation
   
    """
    tm = np.arange(0, T, dt / 1000)
    total = len(r_range) * len(phi_range) * len(theta_range)
    input_ICs = torch.zeros((total, len(tm), len(I_orth)))
    x0s = torch.zeros((total, len(m[0])))
    phases = np.zeros((total, len(tm)))

    i = 0
    if stim_ind:
        input_ICs[:, :, stim_ind] = 1
    for r1 in r_range:
        for k_ph in phi_range:
            for ph0 in theta_range:
                k1 = np.cos(k_ph) * r1
                k2 = np.sin(k_ph) * r1
                phases[i] = tm * w + ph0
                v1 = (
                    1
                    / np.sqrt(1 + (tau * w / 1000) ** 2)
                    * np.sin(ph0 - np.arctan2(tau * w / 1000, 1))
                )
                u = np.sin(tm * w + ph0)
                x0 = np.array(k1 * m[0] + k2 * m[1] + v1 * I_orth[0])
                x0s[i] = torch.from_numpy(x0)
                input_ICs[i, :, 0] = torch.from_numpy(u)
                i += 1

    return x0s, input_ICs, phases
def create_ICs_MF(r_range, phi_range, theta_range, tau, T, dt, w, n_inp=3):
    """
    Create ICs for mean-field RNN simulations

    Args:
        r_range: range of the radius in the k1 k2 plane
        phi_range: range of the angle / phase of the RNN in the k1 k2 plane
        theta_range: range of the initial phase of the reference oscillation
        tau: time constant of the RNN in ms
        T: simulation time in s
        dt: simulation time step in ms
        w: frequency of the reference oscillation

    Returns:
        input_ICs: input to the RNN
        x0s: RNN states at t=0
        phases: phases of the reference oscillation
   
    """
    tm = np.arange(0, T, dt / 1000)
    total = len(r_range) * len(phi_range) * len(theta_range)
    input_ICs = torch.zeros((total, len(tm), n_inp))
    x0s = torch.zeros((total, n_inp+2))
    phases = np.zeros((total, len(tm)))

    i = 0
    for r1 in r_range:
        for k_ph in phi_range:
            for ph0 in theta_range:
                k1 = np.cos(k_ph) * r1
                k2 = np.sin(k_ph) * r1
                phases[i] = tm * w + ph0
                v1 = (
                    1
                    / np.sqrt(1 + (tau * w / 1000) ** 2)
                    * np.sin(ph0 - np.arctan2(tau * w / 1000, 1))
                )
                v2 = (
                    1
                    / np.sqrt(1 + (tau * w / 1000) ** 2)
                    * np.cos(ph0 - np.arctan2(tau * w / 1000, 1))
                )
                u = np.array([np.sin(tm * w + ph0), np.cos(tm * w + ph0)]).T
                x0 = [k1, k2, v1]
                for _ in range(n_inp-1):
                    x0.append(0)
                x0 = np.array(x0)
                x0s[i] = torch.from_numpy(x0)
                input_ICs[i, :, 0] = torch.from_numpy(u[:, 0])
                i += 1
    return x0s, input_ICs, phases


def make_deterministic(task_params, rnn):
    """
    Make the RNN and task deterministic by setting all noise parameters to 0
    """
    task_params["freq_var"] = 0
    task_params["freq_amp_covar"] = 0
    task_params["amp_var"] = 0
    rnn.params["noise_std"] = 0
    task_params["noise_sin"] = 0
    task_params["signal"] = torch.sin


def calculate_mean_radius(freq, rnn, n_period=10, n_period_stim=2,n_osc=1):
    """
    Calculate the mean radius of the RNN in the k1, k2 plane
    for a given frequency of the reference oscillation
    
    Args:
        freq: frequency of the reference oscillation
        rnn: RNN object
        n_period: number of periods of the reference oscillation for which we run the simulation
        n_period_stim: number of periods of the reference oscillation in which stimulus input is presented
                
    Returns: 
        rad: mean radius of the RNN in the k1, k2 plane

    """
    n_inp = rnn.params["n_inp"]
    dt = rnn.rnn.dt / 1000
    period = 1 / (dt * freq)
    trial_len = int(n_period * period)
    stim_len = int(n_period_stim * period)
    period = int(period)
    w = dt * freq * np.pi * 2
    input = torch.zeros((n_inp - 1, trial_len, n_inp))
    input[:, :, 0] = torch.sin(torch.arange(0, trial_len) * w)
    if n_osc == 2:
            input[:, :, 0] = torch.cos(torch.arange(0, trial_len) * w)
    for i in range(n_inp - 1):
        input[i, :stim_len, i + 1] = 1
    rates, _ = predict(rnn, input)
    kappas = np.array(proj(rnn.rnn.m.detach().numpy().T, rates[:, -period:]))
    rad = np.mean(np.linalg.norm(kappas, axis=0))
    return rad


def set_dt(task_params, rnn, dt):
    """ Utility function to change the simulation time step"""
    task_params["dt"] = dt
    rnn.rnn.dt = dt
    rnn.params["dt"] = dt


def weight_scalers_to_1(rnn):
    """ Utility function to set the weight scalers to 1"""
    with torch.no_grad():
        scale = torch.clone(rnn.rnn.w_inp_scale.detach())
        rnn.rnn.w_inp_scale = rnn.rnn.w_inp_scale.copy_(torch.ones_like(scale))
        rnn.rnn.w_inp = rnn.rnn.w_inp.copy_(rnn.rnn.w_inp * scale)
        scale = torch.clone(rnn.rnn.w_out_scale.detach())
        rnn.rnn.w_out_scale = rnn.rnn.w_out_scale.copy_(torch.ones_like(scale))
        rnn.rnn.w_out = rnn.rnn.w_out.copy_(rnn.rnn.w_out * scale)


def proj(m, x):
    """
    Orthogonal projection of x on m, for matrices 

    Args:
        m in RankxN
        x in TxN

    Returns:
        k in RankxT
    """
    k = []
    for mr in m:
        k.append(1 / (np.linalg.norm(mr) ** 2) * x @ mr)
    return k


def orth_proj(a, b, ret_comp=False):
    """
    Orthogonal projection of b on a, for vectors

    Args:
        a in Nx1
        b in Nx1
        
    Returns:
        b_par in Nx1
        b_orth in Nx1
    """
    alpha = (a.T.dot(b)) / (a.T.dot(a))
    b_par = alpha * a

    if ret_comp:
        b_orth = b - b_par
        return b_par, b_orth, alpha
    else:
        return b_par, alpha


def orthogonolise_Im(I, m, n_I=None):
    """
    Orthogonalisze the first n_I vectors of I with respect to the connectivity vectors m

    Args:
        I in n_inpxN
        m in RankxN
        n_I: number of vectors of I to orthogonalise

    Returns:
        alphas, projections of I on m
        I_orth, orthogonalised I
    """

    if not n_I:
        n_I = len(I)
    overlaps_m = m @ m.T

    assert (
        np.mean(overlaps_m - np.diag(np.diagonal(overlaps_m))) < 1e-3
    ), "connectivity vectors not orthogonal"
    alphas = np.zeros((len(m), n_I))

    # TODO: matrixify this
    for i, m_i in enumerate(m):
        for j, I_j in enumerate(I[:n_I]):
            _, alphas[i, j] = orth_proj(m_i, I_j)

    I_orth = np.copy(I)
    I_orth[:n_I] = I[:n_I] - alphas[:, :n_I].T @ m
    return alphas, I_orth


def align_phase(rates, input, period):
    """
    Given the rates of an RNN and the input, align the rates to the input phase

    Args:
        rates: rates of the RNN
        input: input to the RNN
        period: period of the input oscillation
    
    Returns:
        rates_al: aligned rates

    """
    qp = period // 4
    rates_al = np.zeros_like(rates[:, period:])
    for ind in np.arange(rates.shape[0]):
        phase = np.arctan2(input[ind, qp, 0], input[ind, 0, 0]) + np.pi
        phase = int((phase * 25) / np.pi)
        rates_al[ind] = rates[ind, phase : -(period - phase)]
    return rates_al


def inter_from_256(x):
    """Converts a value from [0, 255] to [0, 1] """
    return np.interp(x=x, xp=[0, 255], fp=[0, 1])


def to256(x):
    """Converts a value from [0, 1] to [0, 256]"""
    return [xi * 256 for xi in x]


def extract_labels(task_input, n_stim=2):
    """Extracts the labels from the task input"""
    n_trials = task_input.shape[0]
    labels = np.zeros(n_trials, dtype=int)
    for ind in np.arange(n_trials):
        for st in range(n_stim):
            if torch.sum(task_input[ind, :, -(st + 1)]) > 2:
                labels[ind] = st
    return labels


def build_custom_continuous_cmap(*rgb_list):
    """
    Builds a continuous colormap from a list of RGB value

    Args:
        *rgb_list: list of RGB values

    Returns:
        cmap: the colormap
    """
    all_red = []
    all_green = []
    all_blue = []
    for rgb in rgb_list:
        all_red.append(rgb[0])
        all_green.append(rgb[1])
        all_blue.append(rgb[2])
    # build each section
    n_section = len(all_red) - 1
    red = tuple(
        [
            (1 / n_section * i, inter_from_256(v), inter_from_256(v))
            for i, v in enumerate(all_red)
        ]
    )
    green = tuple(
        [
            (1 / n_section * i, inter_from_256(v), inter_from_256(v))
            for i, v in enumerate(all_green)
        ]
    )
    blue = tuple(
        [
            (1 / n_section * i, inter_from_256(v), inter_from_256(v))
            for i, v in enumerate(all_blue)
        ]
    )
    cdict = {"red": red, "green": green, "blue": blue}
    new_cmap = colors.LinearSegmentedColormap("new_cmap", segmentdata=cdict)
    return new_cmap


def circ_dif(x, y):
    """ Circular difference between two angles"""
    return np.arctan2(np.sin(x - y), np.cos(x - y))


def circ_mean(a, w=1):
    """
    Calculating a circular mean

    Args:
        a: phases
        w: magnitudes

    Returns:
        phase of summed vector
        magnitude of summed vector
    """
    sum_sin = np.sum(np.sin(a) * w)
    sum_cos = np.sum(np.cos(a) * w)
    return np.arctan2(sum_sin, sum_cos), np.linalg.norm([sum_sin, sum_cos])


def lut_to_grey(lut):
    """Converts a LUT to a grey scale LUT"""
    grey = np.copy(lut)
    values = 0.3 * lut[:, 0] + 0.58 * lut[:, 1] + 0.11 * lut[:, 2]
    grey[:, :3] = np.expand_dims(values, 1)
    return grey


def cluster(
    loadings,
    n_components,
    bayes=True,
    n_init=100,
    random_state=None,
    mean_precision_prior=10e5,
    mean_prior=None,
    weight_concentration_prior_type="dirichlet_process",
    weight_concentration_prior=None,
    max_iter=1000,
    init_params="random",
):
    """
    Clusters the loadings (connectivity of LR RNN) using a Gaussian Mixture Model
    Ref: Dubreuil, Valente et al. 2022

    Args:
        loadings: the loadings
        n_components: number of mixture components
        n_init: number of initializations
        random_state: random state
        mean_precision_prior: prior on the precision of the mean
        mean_prior: prior on the mean
        weight_concentration_prior_type: type of prior on the weight concentration
        weight_concentration_prior: prior on the weight concentration

    Returns:
        z: cluster assignment
    """
    if not bayes:
        gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full',max_iter=max_iter,n_init=n_init,init_params=init_params)
    else:
        gmm = mixture.BayesianGaussianMixture(
            n_components=n_components,
            n_init=n_init,
            random_state=random_state,
            mean_prior=mean_prior,
            init_params=init_params,
            mean_precision_prior=mean_precision_prior,
            weight_concentration_prior_type=weight_concentration_prior_type,
            weight_concentration_prior=weight_concentration_prior,
        )

    gmm = gmm.fit(loadings.T)
    z = gmm.predict(loadings.T)
    return z, gmm


def plot_loadings(loadings, z, alpha=1, colors = ["purple", "green", "orange", "red"],hist_lims=None):
    """
    Pair plots of the loadings, colored by their cluster assignment

    Args:
        loadings: the loadings
        z: cluster assignment
        alpha: transparency of the points
        colors: colors of the clusters
    
    Returns:
        fig: the figure
    """
    num_loading = len(loadings)
    fig, axs = plt.subplots(num_loading, num_loading, figsize=(14, 14))
    num_cov = np.max(z) + 1
    for x in range(num_loading):
        for y in range(x):
            axs[x, y].set_visible(False)
        for y in range(x + 1, num_loading):
            for cov in range(num_cov):
                axs[x, y].scatter(
                    loadings[x, z == cov],
                    loadings[y, z == cov],
                    color=colors[cov],
                    alpha=alpha,
                )
            axs[x, y].set_xticks([])
            axs[x, y].set_yticks([])
            axs[x, y].spines["bottom"].set_position("zero")
            axs[x, y].spines["left"].set_position("zero")
            vm = np.max(np.abs([loadings[x,z==cov], loadings[y,z==cov]]))
            axs[x,y].set_ylim(-vm,vm)
            axs[x,y].set_xlim(-vm,vm)

            
        for cov in range(num_cov):
            axs[x, x].hist(
                loadings[x, z == cov], color=colors[cov], alpha=alpha
            )  # , loadings[y], alpha = 0.5s)
            axs[x,x].set_xticks([])
            axs[x,x].set_yticks([])
            if hist_lims:
                axs[x,x].set_xlim(-hist_lims,hist_lims)

    fig.tight_layout()
    return fig


def extract_loadings(rnn, orth_I=False, zero_center=False, orth_n=False, split=False):
    """
    Returns the loadings of a (trained) low rank RNN
    
    Args:
        rnn: the low rank RNN
        orth_I: whether to orthogonalize the input weights
        zero_center: whether to zero center the loadings
        orth_n: whether to orthogonalize the right recurrent connectivity vectors
        split: whether to return one matrix with all loadings, or seperate matrices 
        for the input, recurrent and output loadings

    Returns:
        m: left recurrent connectivity vectors
        n: right recurrent connectivity vectors
        I: input weights
        W: output weights

    """
    m = np.copy(rnn.rnn.m.cpu().detach().numpy()).T
    n = np.copy(rnn.rnn.n.cpu().detach().numpy())
    I = np.copy(
        rnn.rnn.w_inp.cpu().detach().numpy() * rnn.rnn.w_inp_scale.detach().numpy()
    )
    W = np.copy(
        rnn.rnn.w_out.cpu().detach().numpy() * rnn.rnn.w_out_scale.detach().numpy()
    ).T

    if zero_center:
        m -= np.mean(m, axis=1, keepdims=True)
        n -= np.mean(n, axis=1, keepdims=True)
        I -= np.mean(I, axis=1, keepdims=True)
        W -= np.mean(W, axis=1, keepdims=True)

    if orth_I:
        for mi in m:
            for Ii in I:
                _, alpha = orth_proj(mi, Ii)
                Ii -= alpha * mi
    # keep only n parallel to m and I?
    if orth_n:
        orthogonalised_n = np.copy(n)

        for ni in orthogonalised_n:
            for mi in m:
                _, alpha = orth_proj(mi, ni)
                # substract part paralel to m
                ni -= alpha * mi
            for Ii in I:
                for ni in n:
                    _, alpha = orth_proj(Ii, ni)
                    # substract part paralel to I
                    ni -= alpha * Ii
            for Wi in W:
                for ni in n:
                    _, alpha = orth_proj(Wi, ni)
                    # substract part paralel to W
                    ni -= alpha * Wi

        n -= orthogonalised_n

    loadings = np.concatenate([I, n, m, W])

    if split:
        return I, n, m, W
    return loadings


def red_yellow_colours():
    """Returns a list of 4 colors, from red to yellow"""
    pltcolors = [
        [c / 255 for c in [255, 201, 70, 255]],
        [c / 255 for c in [253, 141, 33, 255]],
        [c / 255 for c in [227, 26, 28, 255]],
        [c / 255 for c in [142, 23, 15, 255]],
    ]
    return pltcolors

def green_blue_colours():
    """Returns a list of 4 colors, from green to blue"""
    pltcolors = [
        [c / 255 for c in [161, 218, 180, 255]],
        [c / 255 for c in [65, 182, 196, 255]],
        [c / 255 for c in [34, 94, 168, 255]],
        [c / 255 for c in [10, 30, 69, 255]],
    ]
    return pltcolors


def PC_traj(state_traj, n_comp=3):
    """
    Projects (typical high dimensional) currents on (typical low dimensional) PC space

    args:
        state_traj, numpy array of size (batch_size, seq_len, n_rec)
    n_comp:
        number of PC components to project on

    returns:
        z: trajectories, numpy array of size (batch_size, seq_len, n_comp)
        varexp: variance explained by each component
    """

    [batch_size, seq_len, n_rec] = state_traj.shape

    # typically we simply 'append' all trials, treating all time steps as a sample
    # and units as features
    state_traj_btxd = np.reshape(state_traj, (batch_size * seq_len, n_rec))

    pca = PCA(n_components=n_comp)
    pca.fit(state_traj_btxd)
    varexp = pca.explained_variance_ratio_
    z = np.zeros((batch_size, seq_len, n_comp))

    # project on n_comp first PC components
    for batch_idx in range(batch_size):
        x_idx = state_traj[batch_idx]
        z[batch_idx] = pca.transform(x_idx)
    return z, varexp

def tor(x, y, z, r=1):
    """
    Converts cartesian coordinates to torus coordinates
    """
    xn = (r - x) * np.cos(z)
    yn = (r - x) * np.sin(z)
    zn = y
    return [xn, yn, zn]


def tor_from_angles(phi, theta, R, r):
    """
    Converts spherical coordinates to torus coordinates
    """
    xn = (R + (r * np.cos(phi))) * np.cos(theta)
    yn = (R + (r * np.cos(phi))) * np.sin(theta)
    zn = r * np.sin(phi)
    return [xn, yn, zn]


def def_torus(R, r):
    """
    Returns a torus with specified radii
    Args:
        R: major radius
        r: minor radius
    Returns:
        torus: torus 
    """
    theta = np.linspace(0, 2 * np.pi, 200)
    phi = np.linspace(0, 2 * np.pi, 200)
    torus = np.zeros((3, 200, 200))

    for i in range(0, 200):
        for j in range(0, 200):
            # phi+=0.2
            torus[0][i][j] = (R + r * np.cos(phi[j])) * np.cos(theta[i])
            torus[1][i][j] = (R + r * np.cos(phi[j])) * np.sin(theta[i])
            torus[2][i][j] = r * np.sin(phi[j])
    return torus

def vector_field(x, y, z, F, *args):
    """Returns the vector field of a given function F"""
    dx,dy,dz = F(x,y,z, *args)
    return dx, dy, dz

def convert_vector_field_to_tor(x,y,z,dx,dy,dz, r = 1):
    """Converts a vector field in cartesian coordinates to torus coordinates"""
    du = -np.sin(z)*(r-x)*dz-np.cos(z)*dx
    dv = np.cos(z)*(r-x)*dz-np.sin(z)*dx
    dw=dy
    return du, dv, dw




def F_phase_space(x,y, z, r0=1, w=1, dz_=1):
    """Utils function for phase space plot"""
    r = np.sqrt(x**2+y**2)
    dr=r0 - r
    dth = w*r
    r+=1e-10
    dx = dr * x/r - dth * y/(r**2)
    dy = dr * y/r + dth * x/(r**2)
    dz = np.ones_like(z)*dz_
    return dx, dy, dz



def rotate(x,y, angle):
    """Utility for phase space arrows -> Rotate origin by an angle"""
    w = np.sin(angle)*(y-1)
    v = np.cos(angle)*(y-1)+1
    u = x
    return u,v,w

def rotate_vectors(dx,dy, angle):
    """Utility for phase space arrow -> Rotate derivative by an angle"""
    w = np.sin(angle)*dy
    v = np.cos(angle)*dy
    u = dx
    return u,v,w

def arg_is_close(array, value):
    """Returns index of closest value in array"""
    idx = np.argmin(np.abs(array - value))
    return idx

def wrap(x):
    """returns angle with range [-pi, pi]"""
    return np.arctan2(np.sin(x), np.cos(x))

def get_dataset_stats(task_params,training_params, data_path, dt,file):
    """
    Stores stats regarding the power and main frequency for each training trial in the dataset
    Args:
        task_params: dictionary of task parameters
        training_params: dictionary of training parameters
        data_path: path to dataset
        dt: simulation timestep
        file: file name  
    """

    # initialise dataset, takes some time!
    ds = seqDS_LFP(task_params, data_path)
    n_steps=len(ds[0][0])
    indices = np.arange(ds.len)
    val_indices=training_params['val_indices']
    train_indices = np.delete(indices,val_indices)

    amps= []
    freqs= []
    pows= []

    # loop through all training trials
    for i in train_indices:
        sig=ds[i][0][:,0].numpy()
        amps.append(torch.std(ds[i][0][:,0]).squeeze()*np.sqrt(2))
        f=task_params['main_freq'][i]
        freqs.append(f)
        s=np.sin(f*np.pi*2*np.arange(0,n_steps*dt,dt))
        c=np.cos(f*np.pi*2*np.arange(0,n_steps*dt,dt))
        pows.append(2*abs(sig@s+sig@c*1j)/n_steps)
    train_stats=np.array([amps,pows,freqs])
    np.save(file,train_stats)


def create_MF_covs(cov_params, plot=False,vm=3):
    
    

    eps = 1e-7

    w = cov_params["osc_w"]
    r = cov_params["osc_r"]


    sdm=cov_params["osc_sdm"]
    sdn=cov_params["osc_sdn"]
    sd = np.sqrt(1+ np.pi/2 *(np.sqrt(sdm)*r))
    sdw=sd*w
    sdmW = cov_params["osc_sdmW"]
    sdW = cov_params["osc_sdW"]

    cov1 = np.array([[ eps, 0,   0,   0,   0,   0,   0,   0],
                    [ 0,   eps, 0,   0,   0,   0,   0,   0],
                    [ 0,   0,   eps, 0,   0,   0,   0,   0],
                    [ 0,   0,   0,   sdn,0,   sd,  -sdw, 0],
                    [ 0,   0,   0,   0,   sdn,sdw,sd,  0],
                    [ 0,   0,   0,   sd,  sdw,sdm, 0,   0],
                    [ 0,   0,   0,   -sdw, sd,  0, sdm,   sdmW],
                    [ 0,   0,   0,   0,   0,   0, sdmW,     sdW]],
                dtype = np.float32)

    



    IIn1 = cov_params["coupl_sdIn"]
    sdIm = cov_params["coupl_sdIm"]
    Iss = cov_params["coupl_sdIstim"]
    diagn=cov_params["coupl_sdn"]
    diagm=cov_params["coupl_sdm"]
    II = cov_params["coupl_sdIosc"]
    sdmW=cov_params["coupl_sdmW"]
    sdW = cov_params["coupl_sdW"]
    ofd2 = cov_params["coupl_sdnimj"]
    ofd = cov_params["coupl_sdnimi"]


    cov21 = np.array([[ II,  0,  0,   IIn1, -IIn1, sdIm, sdIm, 0],#   0,   0],
                    [ 0,   Iss, 0,   0,    0,    0,    0,   0],
                    [ 0,   0,   eps, 0,    0,    0,    0,   0],
                    [ IIn1,0,   0,   diagn,0,    ofd,  ofd2,0],
                    [ -IIn1,0,   0,  0,    diagn,ofd2,-ofd, 0],
                    [ sdIm,   0,   0,   ofd,  ofd2, diagm,0,   0],
                    [ sdIm,   0,   0,   ofd2, -ofd, 0,    diagm,sdmW],
                    [ 0,   0,   0,   0,    0,    0,    sdmW,   sdW]],
                dtype = np.float32)
    cov22 = np.array([[ II,  0,  0,   -IIn1, IIn1, -sdIm, -sdIm, 0],#   0,   0],
                    [ 0,   eps, 0,   0,    0,    0,    0,   0],
                    [ 0,   0,   Iss, 0,    0,    0,    0,   0],
                    [ -IIn1,0,   0,   diagn,0,    ofd, ofd2,0],
                    [ IIn1,0,   0,   0,    diagn,ofd2, -ofd,0],
                    [ -sdIm,   0,   0,   ofd,  ofd2, diagm,0,   0],
                    [ -sdIm,   0,   0,   ofd2, -ofd, 0,    diagm,sdmW],
                    [ 0,   0,   0,   0,    0,    0,    sdmW,   sdW]],
                dtype = np.float32)


    if plot:
        """
        Plot cov
        """
        titles=["oscillator (w=.5)","coupling a (w=.25)","coupling b (w=.25)"]

        labels = ["$I_{osc}$", "$I_{s_a}$", "$I_{s_b}$", "$n_1$", "$n_2$", "$m_1$", "$m_2$", "w"]


        _,_=plot_covs([cov1,cov21,cov22],vm,labels,titles)
    chol_cov1 = np.float32(np.linalg.cholesky(cov1[:,:]))
    chol_cov21 = np.float32(np.linalg.cholesky(cov21[:,:]))
    chol_cov22 = np.float32(np.linalg.cholesky(cov22[:,:]))


    chol_covs = np.concatenate([[chol_cov1],[chol_cov1],[chol_cov21],[chol_cov22]])
    return chol_covs


def create_ICs_phase_prec(r_range,phi_range,theta_range, stim_range,tau, T, dt,w,amp,n_inp=2):
    """
    Create initial conditions for a phase precession model
    Args:
        r_range: range of the radius in the k1 k2 plane
        phi_range: range of the angle / phase of the RNN in the k1 k2 plane
        theta_range: range of the initial phase of the reference oscillation
        stim_range: range of the position inputs to the RNN
        tau: time constant of the RNN in ms
        T: simulation time in s
        dt: simulation time step in ms
        w: frequency of the reference oscillation
        amp: amplitude of the reference oscillation

    Returns:
        x0s: RNN states at t=0
        input_ICs: input to the RNN
        phases: phases of the reference oscillation
   
    """
    tm = np.arange(0,T,dt/1000)
    total = len(r_range)*len(phi_range)*len(theta_range)*len(stim_range)
    input_ICs = torch.zeros((total,len(tm),4))
    x0s = torch.zeros((total, 6))
    us = np.zeros((total,len(tm),2))
    i = 0
    phases = np.zeros((total,len(tm)))

    for r1 in r_range:
        for k_ph in phi_range:
            for ph0 in theta_range:
                for st in stim_range:
                    k1 = np.cos(k_ph)*r1
                    k2 = np.sin(k_ph)*r1
                    phases[i]=tm*w+ph0
                    v1 = amp/np.sqrt(1+(tau*w/1000)**2) * np.cos(ph0-np.arctan2(tau*w/1000,1))
                    v2 = amp/np.sqrt(1+(tau*w/1000)**2) * np.sin(ph0-np.arctan2(tau*w/1000,1))
                    u = np.array([amp*np.cos(tm*w+ph0), 
                                 amp*np.sin(tm*w+ph0)]).T
                    x0 = np.array([k1,k2,v1,v2,0,0])
                    us[i]=u
                    x0s[i]=torch.from_numpy(x0)
                    input_ICs[i,:,:2]=torch.from_numpy(u)
                    input_ICs[i,:,2]=np.cos(st)
                    input_ICs[i,:,3]=np.sin(st)
                    i+=1
    return x0s, input_ICs, phases


def create_MF_covs_phase_prec(cov_params, plot=False):

    """
    Create covariance matrices for a phase precession model

    Args:
        cov_params:dictionary of covariance parameters
        plot: plot the covariance matrices. Defaults to False.
    
    Returns:
        chol_covs: Cholesky decomposition of the covariance matrices
    """
    
    eps = 1e-7
    sdm=cov_params["osc_sdm"]
    sdn=cov_params["osc_sdn"]
    sd = cov_params["osc_sd"]
    sdw=cov_params["osc_sdw"]
    sdmW = cov_params["osc_sdmW"]
    sdW = cov_params["osc_sdW"]

    cov1 = np.array([[ eps, 0,   0,   0,   0,   0,   0,   0,   0],
                     [ 0,   eps, 0,   0,   0,   0,   0,   0,   0],
                     [ 0,   0,   eps, 0,   0,   0,   0,   0,   0],
                     [ 0,   0,   0,   eps, 0,   0,   0,   0,   0],
                     [ 0,   0,   0,   0,   sdn,0,   sd,  -sdw, 0],
                     [ 0,   0,   0,   0,   0,   sdn,sdw,sd,  0],
                     [ 0,   0,   0,   0,   sd,  sdw,sdm, 0,   0],
                     [ 0,   0,   0,   0,   -sdw, sd,  0, sdm,   sdmW],
                     [ 0,   0,   0,   0,   0,   0,   0, sdmW,     sdW]],
                   dtype = np.float32)  

    IIn1 = cov_params["coupl_sdIn"]
    sdIm = cov_params["coupl_sdIm"]
    Iss = cov_params["coupl_sdIstim"]
    diagn=cov_params["coupl_sdn"]
    diagm=cov_params["coupl_sdm"]
    II = cov_params["coupl_sdIosc"]
    sdmW=cov_params["coupl_sdmW"]
    sdW = cov_params["coupl_sdW"]
    ofd2 = cov_params["coupl_sdnimj"]
    ofd = cov_params["coupl_sdnimi"]


    cov21 = np.array([[ II,  0,   0,   0,   0,    IIn1,sdIm, sdIm, 0],#   0,   0],
                     [ 0,    II,  0,   0,    -IIn1, 0,    0,    0,   0],
                     [ 0,    0,   eps, 0,   0,    0,    0,    0,   0],
                     [ 0,    0,   0,   Iss, 0,   0,   0,   0,   0],
                     [ 0,    -IIn1,0,   0,   diagn,0,    ofd,  -ofd2,0],
                     [ IIn1,0,   0,   0,   0,    diagn,ofd2,ofd, 0],
                     [ sdIm, 0,   0,   0,   ofd,  ofd2, diagm,0,   0],
                     [ sdIm, 0,   0,   0,  - ofd2, ofd, 0,    diagm,sdmW],
                     [ 0,    0,   0,   0,   0,    0,    0,    sdmW, sdW]],
                   dtype = np.float32)
    
    cov22 = np.array([[ II,  0,   0,   0,   IIn1,0, -sdIm, -sdIm,0],#   0,   0],
                     [ 0,    II,  0,   0,   0,    IIn1, 0,     0,    0],
                     [ 0,    0,   Iss, 0,   0,    0,     0,     0,    0],
                     [ 0,    0,   0,   eps, 0,    0,     0,     0,   0],
                     [ IIn1,0,   0,   0,   diagn,0,     ofd,   -ofd2, 0],
                     [ 0,   IIn1,0,   0,   0,    diagn, ofd2,  ofd, 0],
                     [ -sdIm,0,   0,   0,   ofd,  ofd2,  diagm, 0,    0],
                     [ -sdIm,0,   0,   0,   -ofd2, ofd,  0,     diagm, sdmW],
                     [ 0,    0,   0,   0,   0,    0,     0,     sdmW,  sdW]],
                   dtype = np.float32)

    if plot:
        """
        Plot cov
        """
        vm = 4
        labels=["$I_{osc_1}$", "$I_{osc_2}$", "$I_{s_1}$", "$I_{s_1}$","$n_1$", "$n_2$", "$m_1$", "$m_2$"]
        titles=["oscillator (w=.5)","coupling a (w=.25)","coupling b (w=.25)"]



        _,_=plot_covs([cov1[:-1],cov21[:-1],cov22[:-1]],vm,labels,titles)

    chol_cov1 = np.float32(np.linalg.cholesky(cov1[:,:]))
    chol_cov21 = np.float32(np.linalg.cholesky(cov21[:,:]))
    chol_cov22 = np.float32(np.linalg.cholesky(cov22[:,:]))

    chol_covs = np.concatenate([[chol_cov1],[chol_cov1],[chol_cov21],[chol_cov22]])
    return chol_covs


def create_MF_covs_R1(cov_params, plot=False,vm=3):
   
    """
    Create covariance matrices for a rank 1 network

    Args:
        cov_params:dictionary of covariance parameters
        plot: plot the covariance matrices. Defaults to False.
    
    Returns:
        chol_covs: Cholesky decomposition of the covariance matrices
    """ 
    

    eps = 1e-4
    sdII1 = cov_params["sdII1"]
    sdIs1 = cov_params["sdIs1"]
    sdIw1 = cov_params["sdIw1"]
    sdn1 = cov_params["sdn1"]
    sdmn1 = cov_params["sdmn1"]
    sdm1 = cov_params["sdm1"]
    sdW1 = cov_params["sdW1"]

      

    cov1 = np.array([[sdII1, 0,    0,   0,    0,     sdIw1],
                    [ 0,     sdIs1,0,   0,    0,     0   ],
                    [ 0,     0,    eps, 0,    0,     0   ],
                    [ 0,     0,    0,   sdn1, sdmn1, 0   ],
                    [ 0,     0,    0,   sdmn1,sdm1,  0   ],
                    [  sdIw1,0,    0,   0,    0,     sdW1]],
                dtype = np.float32)

    sdII2 = cov_params["sdII2"]
    sdIs2 = cov_params["sdIs2"]
    sdIw2 = cov_params["sdIw2"]
    sdn2 = cov_params["sdn2"]
    sdmn2 = cov_params["sdmn2"]
    sdm2 = cov_params["sdm2"]
    sdW2 = cov_params["sdW2"]

    cov2 = np.array([[sdII2, 0,    0,   0,    0,     sdIw2],
                    [ 0,     eps,  0,   0,    0,     0   ],
                    [ 0,     0,    sdIs2, 0,    0,     0   ],
                    [ 0,     0,    0,   sdn2, sdmn2, 0   ],
                    [ 0,     0,    0,   sdmn2,sdm2,  0   ],
                    [  sdIw2,0,    0,   0,    0,     sdW2]],
                dtype = np.float32)

    if plot:
        """
        Plot cov
        """
        titles=["P1 (w=.67)","P2 (w=.33)"]

        labels = ["$I_{osc}$", "$I_{s_a}$", "$I_{s_b}$", "$n$",  "$m$", "$w$"]


        _,_=plot_covs([cov1,cov2],vm,labels,titles, float_labels=True)
    chol_cov1 = np.float32(np.linalg.cholesky(cov1[:,:]))
    chol_cov2 = np.float32(np.linalg.cholesky(cov2[:,:]))


    chol_covs = np.concatenate([[chol_cov1],[chol_cov2]])
    return chol_covs

def create_ICs_MF_R1(K_range, theta_range, tau, T, dt, w):
    """
    Create ICs for mean-field RNN simulations of rank 1 network

    Args:
        K_range: range of the radius in the k line
        theta_range: range of the initial phase of the reference oscillation
        tau: time constant of the RNN in ms
        T: simulation time in s
        dt: simulation time step in ms
        w: frequency of the reference oscillation

    Returns:
        input_ICs: input to the RNN
        x0s: RNN states at t=0
        phases: phases of the reference oscillation
   
    """
    tm = np.arange(0, T, dt / 1000)
    total = len(K_range) * len(theta_range)
    input_ICs = torch.zeros((total, len(tm), 3))
    x0s = torch.zeros((total, 4))
    phases = np.zeros((total, len(tm)))

    i = 0

    for k in K_range:
        for ph0 in theta_range:
            phases[i] = tm * w + ph0
            v1 = (
                1
                / np.sqrt(1 + (tau * w / 1000) ** 2)
                * np.sin(ph0 - np.arctan2(tau * w / 1000, 1))
            )
            v2 = (
                1
                / np.sqrt(1 + (tau * w / 1000) ** 2)
                * np.cos(ph0 - np.arctan2(tau * w / 1000, 1))
            )
            u = np.array([np.sin(tm * w + ph0), np.cos(tm * w + ph0)]).T
            x0 = np.array([k, v1, 0, 0])
            x0s[i] = torch.from_numpy(x0)
            input_ICs[i, :, 0] = torch.from_numpy(u[:, 0])
            i += 1
                
    return x0s, input_ICs, phases

def create_MF_covs_4Stims(cov_params, plot=False,vm=3):
   
    """
    Create covariance matrices for a rank 2 network coding for 4 stimuli

    Args:
        cov_params:dictionary of covariance parameters
        plot: plot the covariance matrices. Defaults to False.
    
    Returns:
        chol_covs: Cholesky decomposition of the covariance matrices
    """ 

    eps = 1e-7

    sdw = cov_params["osc_w"]
    sd = cov_params["osc_r"]
    sdm=cov_params["osc_sdm"]
    sdn=cov_params["osc_sdn"]
    sdmW = cov_params["osc_sdmW"]
    sdW = cov_params["osc_sdW"]

    cov1 = np.array([[ eps, 0,   0,   0,   0,   0,   0,   0,   0,   0],
                    [ 0,   eps, 0,   0,   0,   0,   0,   0,   0,   0],
                    [ 0,   0,   eps, 0,   0,   0,   0,   0,   0,   0],
                    [ 0,   0,   0,  eps,   0,   0,   0,   0,   0,   0],
                    [ 0,   0,   0,   0,  eps,   0,   0,   0,   0,   0],
                    [ 0,   0,   0,   0,   0,   sdn,0,   sd,  -sdw, 0],
                    [ 0,   0,   0,   0,   0,   0,   sdn,sdw,sd,  0],
                    [ 0,   0,   0,   0,   0,   sd,  sdw,sdm, 0,   0],
                    [ 0,   0,   0,   0,   0,   -sdw, sd,  0, sdm,   sdmW],
                    [ 0,   0,   0,   0,   0,   0,   0,   0, sdmW,     sdW]],
                dtype = np.float32)

    IIn1 = cov_params["coupl_sdIn"]
    sdIm = cov_params["coupl_sdIm"]
    Iss = cov_params["coupl_sdIstim"]
    diagn=cov_params["coupl_sdn"]
    diagm1=cov_params["coupl1_sdm1"]
    diagm2=cov_params["coupl1_sdm2"]
    II = cov_params["coupl1_sdIosc"]
    sdmW=cov_params["coupl_sdmW"]
    sdW = cov_params["coupl_sdW"]
    ofd12 = cov_params["coupl1_sdn1m2"]
    ofd21 = cov_params["coupl1_sdn2m1"]
    ofd11 = cov_params["coupl1_sdn1m1"]
    ofd22 = cov_params["coupl1_sdn2m2"]

    cov21 = np.array([[ II,   0,  0,    0,    0, IIn1,0,sdIm,  sdIm,   0],#   0,   0],
                    [ 0,    eps,  0,    0,    0,    0,    0,    0,    0,   0],
                    [ 0,    0,    Iss,  0,    0,    0,    0,    0,    0,   0],
                    [ 0,    0,    0,  Iss,    0,    0,    0,    0,    0,   0],
                    [ 0,    0,    0,    0,  Iss,    0,    0,    0,    0,   0],
                    [ IIn1, 0,    0,    0,    0,diagn,    0,  ofd11, ofd12,   0],
                    [ 0,0,    0,    0,    0,    0,diagn, ofd21, ofd22,   0],
                    [ sdIm, 0,    0,    0,    0,  ofd11, ofd21,diagm2,   0,  0],
                    [ sdIm, 0,    0,    0,    0, ofd12, ofd22,    0,diagm1,-sdmW],
                    [ 0,   0,     0,    0,    0,    0,    0,    0,  -sdmW, sdW]],
                dtype = np.float32)

    cov22 = np.array([[ II,   0,  0,    0,    0,-IIn1,0,sdIm,  sdIm,   0],#   0,   0],
                    [ 0,    Iss,  0,    0,    0,    0,    0,    0,    0,   0],
                    [ 0,    0,    eps,  0,    0,    0,    0,    0,    0,   0],
                    [ 0,    0,    0,  Iss,    0,    0,    0,    0,    0,   0],
                    [ 0,    0,    0,    0,  Iss,    0,    0,    0,    0,   0],
                    [ -IIn1,0,    0,    0,    0,diagn,    0,  ofd11, ofd12,   0],
                    [ 0, 0,    0,    0,    0,    0,diagn, ofd21, ofd22,   0],
                    [ sdIm, 0,    0,    0,    0,  ofd11, ofd21,diagm2,   0,   0],
                    [ sdIm, 0,    0,    0,    0, ofd12, ofd22,    0,diagm1,-sdmW],
                    [ 0,   0,     0,    0,    0,    0,    0,    0,  -sdmW, sdW]],
                dtype = np.float32)
    
    II = cov_params["coupl1_sdIosc"]
    diagm1=cov_params["coupl2_sdm1"]
    diagm2=cov_params["coupl2_sdm2"]
    ofd12 = cov_params["coupl2_sdn1m2"]
    ofd21 = cov_params["coupl2_sdn2m1"]
    ofd11 = cov_params["coupl2_sdn1m1"]
    ofd22 = cov_params["coupl2_sdn2m2"]

    cov23 = np.array([[ II,   0,  0,    0,    0, 0,IIn1,sdIm,  sdIm,   0],#   0,   0],
                    [ 0,    Iss,  0,    0,    0,    0,    0,    0,    0,   0],
                    [ 0,    0,    Iss,  0,    0,    0,    0,    0,    0,   0],
                    [ 0,    0,    0,  eps,    0,    0,    0,    0,    0,   0],
                    [ 0,    0,    0,    0,  Iss,    0,    0,    0,    0,   0],
                    [ 0,0,    0,    0,    0,diagn,    0, ofd11, ofd12,   0],
                    [ IIn1, 0,    0,    0,    0,    0,diagn,ofd21, ofd22,   0],
                    [ sdIm, 0,    0,    0,    0, ofd11,ofd21,diagm1,   0,   sdmW],
                    [ sdIm, 0,    0,    0,    0, ofd12, ofd22,    0,diagm2,0],
                    [ 0,   0,     0,    0,    0,    0,    0,    sdmW, 0, sdW]],
                dtype = np.float32)
    
    cov24 = np.array([[ II,   0,  0,    0,    0, 0,-IIn1,sdIm,  sdIm,   0],#   0,   0],
                    [ 0,    Iss,  0,    0,    0,    0,    0,    0,    0,   0],
                    [ 0,    0,    Iss,  0,    0,    0,    0,    0,    0,   0],
                    [ 0,    0,    0,  Iss,    0,    0,    0,    0,    0,   0],
                    [ 0,    0,    0,    0,  eps,    0,    0,    0,    0,   0],
                    [ 0, 0,    0,    0,    0,diagn,    0, ofd11, ofd12,   0],
                    [ -IIn1,0,    0,    0,    0,    0,diagn,ofd21, ofd22,   0],
                    [ sdIm, 0,    0,    0,    0, ofd11,ofd21,diagm1,   0,   sdmW],
                    [ sdIm, 0,    0,    0,    0, ofd12, ofd22,    0,diagm2,0],
                    [ 0,   0,     0,    0,    0,    0,    0,    sdmW, 0, sdW]],
                dtype = np.float32)

    if plot:
        """
        Plot cov
        """
        titles=["oscillator","coupling a", "coupling b","coupling c","coupling d"]
        labels = ["$I_{osc}$", "$I_{s_a}$", "$I_{s_b}$","$I_{s_c}$","$I_{s_d}$", "$n_1$", "$n_2$", "$m_1$", "$m_2$", "w"]
        _,_=plot_covs([cov1,cov21,cov22,cov23,cov24],vm,labels,titles)

    # Cholesky decomposition
    chol_cov1 = np.float32(np.linalg.cholesky(cov1[:,:]))
    chol_cov21 = np.float32(np.linalg.cholesky(cov21[:,:]))
    chol_cov22 = np.float32(np.linalg.cholesky(cov22[:,:]))
    chol_cov23 = np.float32(np.linalg.cholesky(cov23[:,:]))
    chol_cov24 = np.float32(np.linalg.cholesky(cov24[:,:]))
    chol_covs = np.concatenate([[chol_cov1],[chol_cov21],[chol_cov22],[chol_cov23],[chol_cov24]])
    return chol_covs

def su_stats(task_params,rnn,normalize = True, out_nonlinearity=False):
    ds = seqDS(task_params)

    dataloader = DataLoader(
        ds, batch_size=128, shuffle=True
    )
    test_input, test_target, test_mask = next(iter(dataloader))
    labels = extract_labels(test_input)
    rates, _ = predict(rnn, test_input,mse_loss, test_target, test_mask)
    #rates/=np.mean(rates**2,axis=(0,1),keepdims=True)

    ind0=np.arange(128)[labels==0]
    ind1=np.arange(128)[labels==1]

    period = 1000/(task_params['freq']*task_params['dt'])
    qp = int(period/4)
    period=int(period)
    sin = np.sin(np.linspace(0,np.pi*2,period))
    cos = np.cos(np.linspace(0,np.pi*2,period))

    phase= np.arctan2(test_input[:,0,0],test_input[:,qp,0]).cpu().numpy()
    phase_int=np.int_(((phase+np.pi)/(np.pi*2))*period)+1

    angs_dist=[]
    means_dist = []

    if out_nonlinearity:
        norm_rates = rnn.rnn.out_nonlinearity(torch.from_numpy(rates).to(device =rnn.rnn.w_inp.device)).cpu().numpy()
    else:
        norm_rates = np.copy(rates)
    if normalize:
        norm_rates =  (norm_rates - np.mean(norm_rates[:,-period:],axis=(0,1)))/np.std(norm_rates[:,-period:],axis=(0,1))
    


    for ni in range(rates.shape[-1]):
        
        #RNN 1
        angs0 = []
        angs1 = []
        means0= []
        means1= []
        for ind in ind0:
            mean =  np.mean(norm_rates[ind,-period-phase_int[ind]:-phase_int[ind],ni])
            angs0.append(np.angle(1j*np.inner(sin,rates[ind,-period-phase_int[ind]:-phase_int[ind],ni]-mean)+
                        np.inner(cos,rates[ind,-period-phase_int[ind]:-phase_int[ind],ni]-mean)))
            means0.append(mean)
        for ind in ind1:
            mean =  np.mean(norm_rates[ind,-period-phase_int[ind]:-phase_int[ind],ni])
            angs1.append(np.angle(1j*np.inner(sin,rates[ind,-period-phase_int[ind]:-phase_int[ind],ni]-mean)+
                        np.inner(cos,rates[ind,-period-phase_int[ind]:-phase_int[ind],ni]-mean)))
            means1.append(mean)
        angs_dist.append(circ_dif(circ_mean(angs1)[0],circ_mean(angs0)[0]))
        means_dist.append(np.mean(means0)-np.mean(means1))
    return angs_dist, means_dist

def resample(gmm,params):
    """Resample loadings of an RNN, using a fitted mixture model"""
    loadings = gmm.sample(params['n_rec'])
    params['loadings']=loadings[0].T
    params['scale_w_out']=1#w_out 
    params['scale_w_inp']=1
    rnn_rs =RNN(params)
    return rnn_rs

def resample_emp(z,loadings,params, keep_means=True):
    """Resample loadings of an RNN, using a fitted mixture model"""
    loadings_new = np.zeros_like(loadings)
    n_z = np.max(z)+1
    covs = [np.cov(loadings[:,z==i]) for i in np.arange(n_z )]
    if keep_means:
        for i in np.arange(n_z):
            loadings_new[:,z==i] = np.random.multivariate_normal(loadings[:,z==i].mean(axis=1),covs[i],size=np.sum(z==i)).T
    else:
        for i in np.arange(n_z):
            loadings_new[:,z==i] = np.random.multivariate_normal(np.zeros(len(loadings)),covs[i],size=np.sum(z==i)).T
    params['loadings']=loadings_new
    params['scale_w_out']=1#w_out 
    params['scale_w_inp']=1
    rnn_rs =RNN(params)
    return rnn_rs