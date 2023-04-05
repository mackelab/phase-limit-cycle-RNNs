import numpy as np
from utils import *
import torch.nn as nn
import torch
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.vq import kmeans2
from joblib import Parallel, delayed


def get_evs_stim_bifurc(Ks, evs):
    """
    Sort the eigenvalues for the stimulus induced bifurcation,
    based on their corresponding fixed points and stimulus shown
    
    Args:   
        Ks: The Ks of the fixed point
        evs: The eigenvalues of the Jacobian at those fixed point
    Returns:
        e0l0: The eigenvalues of the first stimulus, first fixed point
        e0l1: The eigenvalues of the first stimulus, second fixed point
        e1l0: The eigenvalues of the second stimulus, first fixed point
        e1l1: The eigenvalues of the second stimulus, second fixed point
    
    """

    # Reshape the Ks to be 2D, split into stimulus 1 and 2
    Ks0_r = Ks[0].reshape(-1, 2)
    Ks1_r = Ks[1].reshape(-1, 2)

    # For each of the two stimulus, cluster the Ks into two groups (fixed points)
    _, labels0 = kmeans2(Ks0_r, 2, seed=1)
    labels0 = labels0.reshape(Ks.shape[1], Ks.shape[2])

    _, labels1 = kmeans2(Ks1_r, 2, seed=1)
    labels1 = labels1.reshape(Ks.shape[1], Ks.shape[2])

    # initialise arrays to store the eigenvalues, for stimulus 1
    e0l0 = np.zeros(evs.shape[1])
    e0l1 = np.zeros(evs.shape[1])

    # we only want the largest eigenvalue for each fixed point
    for i in range(evs.shape[1]):
        if sum(labels0[i] == 0):
            e0l0[i] = np.max(evs[0, i, labels0[i] == 0])
        else:
            e0l0[i] = -1

        if sum(labels0[i] == 1):
            e0l1[i] = np.max(evs[0, i, labels0[i] == 1])
        else:
            e0l1[i] = -1

    # initialise arrays to store the eigenvalues, for stimulus 2
    e1l0 = np.zeros(evs.shape[1])
    e1l1 = np.zeros(evs.shape[1])

    # again, we only want the largest eigenvalue for each fixed point
    for i in range(evs.shape[1]):
        if sum(labels1[i] == 0):
            e1l0[i] = np.max(evs[1, i, labels1[i] == 0])
        else:
            e1l0[i] = -1

        if sum(labels1[i] == 1):
            e1l1[i] = np.max(evs[1, i, labels1[i] == 1])
        else:
            e1l1[i] = -1

    return e0l0, e0l1, e1l0, e1l1

def create_plot(evs, freqs, amps):

    """
    Plot the maximum eigenvalue (floquet multiplier) norm for
    each frequency and amplitude combination
    
    Args:
        evs: The eigenvalues of the Jacobian at the fixed points
        freqs: The frequencies of the reference oscillation
        amps: The amplitudes of the reference oscillation
    
    Returns:    
        fig: The bifurcation plot
    """

    evs_max = np.max(evs, axis=-1)
    vmax = 1
    tick_ind = 20
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    im = axs.imshow(
        np.max(evs_max, axis=-1), cmap="GnBu", vmax=vmax, interpolation="None"
    )

    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    fig.colorbar(im, cax=cax)  # , orientation='vertical')
    axs.set_yticks(np.arange(len(freqs))[::tick_ind])
    axs.set_yticklabels(["{:.1f}".format(i) for i in freqs[::tick_ind]])
    axs.set_xticks(np.arange(len(amps))[::tick_ind])
    axs.set_xticklabels(["{:.1f}".format(i) for i in amps[::tick_ind]])
    return fig

def create_bifur_ICs(ph0_range, k_ph_range, rad, device):
    """
    Create initial conditions for the bifurcation analysis' first step:
    finding all fixed points by running simulations till convergence from
    various initial conditions.
    
    Args:
        ph0_range: The range of initial reference oscillation phases to use
        k_ph_range: The range of initial RNN phases to use
        rad: The radius of the RNN in the k1,k2 plane
        device: The device to put the tensors on

    Returns:
        phase_0: The initial phases of the reference oscillation
        k_0: The state of the RNN
    """

    k_0 = (
        torch.stack(
            [
                torch.sin(k_ph_range).repeat_interleave(len(ph0_range)),
                torch.cos(k_ph_range).repeat_interleave(len(ph0_range)),
            ]
        ).t()
        * rad
    )

    phase_0 = torch.tile(ph0_range, (len(k_ph_range), 1)).flatten()
    return phase_0.to(device=device), k_0.to(device=device)


def to_torch(alpha, I_orth, m, n, device="cpu"):
    """
    Move the numpy arrays to torch tensors
    
    Args:
        alpha: The projection of the input matrix on m
        I_orth: The orthogonalised input matrix
        m: The left connectivity vectors of the recurrent matrix
        n: The right connectivity vectors of the recurrent matrix
        device: The device to put the tensors on
    
    Returns:
        alpha: The projection of the input matrix on m
        I_orth: The orthogonalised input matrix
        m: The left connectivity vectors of the recurrent matrix
        n: The right connectivity vectors of the recurrent matrix

    """
    alpha = torch.from_numpy(alpha).to(dtype=torch.float32, device=device)
    I_orth = torch.from_numpy(I_orth).to(device=device)
    m = torch.from_numpy(m).to(device=device)
    n = torch.from_numpy(n).to(device=device)
    return alpha, I_orth, m, n


class bifurcation(nn.Module):
    """
    The bifurcation analysis class. This module is used to find the fixed points of the Poincare map
    (limit cycles) and the eigenvalues of the Jacobian at these fixed points.
    """
    def __init__(self, alpha, I, m, n, tau, config):
        """
        Initialise the bifurcation analysis class

        Args: 
            alpha: The projection of the input matrix on m
            I: The orthogonalised input matrix
            m: The left connectivity vectors of the recurrent matrix
            n: The right connectivity vectors of the recurrent matrix
            tau: The time constant of the RNN
            config: The configuration dictionary      
        """
        self.alpha = alpha
        self.I = I
        self.m = m
        self.n = n
        self.tau = tau
        self.N = n.size()[1]
        self.nonlinearity = torch.tanh
        self.config = config
        self.identity = torch.eye(2, device=self.m.device).unsqueeze(0)

    def calc_x_u(self, amp, w, phase, k_t, stim):
        """
        Calculate the input to and state of the RNN units at a given time step

        Args:
            amp: The amplitude of the reference oscillation
            w: The frequency of the reference oscillation
            phase: The phase of the reference oscillation
            k_t: The state of the RNN projected on the m vectors
            stim: Additional stimulus input to the RNN
        
        Returns:
            x: The state of the RNN units
            u: The input to the RNN units
        """
        tau_sw = w * self.tau

        # Reference oscillation u projected on I, closed for msolution
        v_osc = (
            amp / np.sqrt(1 + tau_sw**2) * torch.sin(phase - np.arctan2(tau_sw, 1))
        )
        u_osc = amp * torch.sin(phase)

        u_stim = torch.tile(stim, dims=(phase.size()[0], 1))
        u = torch.cat((u_osc.unsqueeze(-1), u_stim), dim=-1)
        v = torch.cat((v_osc.unsqueeze(-1), u_stim), dim=-1)
        x = k_t @ self.m + v @ self.I

        return x, u

    def dKdt(self, k_t, x, u):
        """
        Calculate the change in K, the state of the RNN projected on the m vectors

        Args:
            k_t: The state of the RNN projected on the m vectors
            x: The state of the RNN units
            u: The input to the RNN units
        
        Returns:
            dK: The change in K
        """
        dK = (1 / self.tau) * (
            -k_t + ((self.nonlinearity(x) @ self.n.t()) / self.N) + u @ self.alpha.t()
        )
        return dK

    def dphi(self, x):
        """
        Derivative of the nonlinaerity tanh
        """
        return 1 - torch.tanh(x) ** 2

    def Jacobian(self, x):
        """
        Evaluate the jacobian of the RNN at a given state
        """
        return (1 / self.tau) * (
            -self.identity
            + (1 / self.N)
            * torch.einsum(
                "BNK,ND->BKD",
                (self.dphi(x).unsqueeze(-1) * self.m.t().unsqueeze(0)),
                self.n.t(),
            )
        )

    def forward(self, k_t, phase, amp, w, stim):
        """
        Do a forward step using euler integration

        Args:
            k_t: The state of the RNN projected on the m vectors
            phase: The phase of the reference oscillation
            amp: The amplitude of the reference oscillation
            w: The frequency of the reference oscillation
            stim: Additional stimulus input to the RNN
        
        Returns:
            k_t: The state of the RNN projected on the m vectors, one step later
            phase: The phase of the reference oscillation, one step later
        """

        x, u = self.calc_x_u(amp, w, phase, k_t, stim=stim)
        dK = self.dKdt(k_t, x, u)
        k_t += self.config["dt"] * dK
        phase += self.config["dt"] * w
        return k_t, phase

    def forward_Jac(self, k_t, phase, monodromy, dK_norms, amp, w, stim):
        """
        Do a forward step using euler integration, and linearise around the state

        Args:
            k_t: The state of the RNN projected on the m vectors
            phase: The phase of the reference oscillation
            monodromy: The monodromy matrix, for linearising the Poincare map
            dK_norms: The norm of the change in K
            amp: The amplitude of the reference oscillation
            w: The frequency of the reference oscillation
            stim: Additional stimulus input to the RNN
        
        Returns:
            k_t: The state of the RNN projected on the m vectors, one step later
            phase: The phase of the reference oscillation, one step later
            monodromy: The monodromy matrix, for linearising the Poincare map, one step later
            dK_norms: The norm of the change in K, one step later
        """

        x, u = self.calc_x_u(amp, w, phase, k_t, stim=stim)
        dK = self.dKdt(k_t, x, u)
        Jac = self.Jacobian(x)
        k_t += self.config["dt"] * dK
        phase += self.config["dt"] * w
        monodromy += self.config["dt"] * torch.bmm(Jac, monodromy)
        dK_norms += torch.norm(dK, dim=-1)

        return k_t, phase, monodromy, dK_norms

    def run_sim(self, IC):
        """
        Run a simulation for a given set of initial conditions
        
        Args:
            IC: initial conditions
        Returns:
            phase: The phase of the reference oscillation at the end of the simulation
            k_t: The state of the RNN projected on the m vectors at the end of the simulation

        """
        with torch.no_grad():
            freq = IC[0]
            amp = IC[1]
            w = np.pi * 2 * freq
            period = int(1 / (8 * self.config["dt"]))

            phase = torch.clone(self.phase_0)
            k_t = torch.clone(self.k_0)

            for _ in range(period * self.config["n_periods"]):
                k_t, phase[:] = self.forward(
                    k_t, phase[:], amp, w, stim=torch.zeros(self.I.size()[0] - 1)
                )
        print({"sim_progress": IC[2]})

        return [phase, k_t]

    def run_sims(self, sync_wandb, phase_0, k_0, n_jobs=2):
        """
        Run multiple simulations for given initial conditions

        Args:
            sync_wandb: Boolean indicating hether to sync wandb
            phase_0: The initial phases of the reference oscillation
            k_0: The initial state of the RNN projected on the m vectors
            n_jobs: The number of jobs to run in parallel
        Returns:
            phase: The phases of the reference oscillation at the end of the simulation
            k_t: The states of the RNN projected on the m vectors at the end of the simulation

        """

        n_freqs = len(self.config["freqs"])
        n_amps = len(self.config["amps"])
        n_trials = len(phase_0)
        self.phase_0 = phase_0
        self.k_0 = k_0
        self.sync_wandb = sync_wandb
        with torch.no_grad():
            Ks = torch.zeros(
                (n_freqs, n_amps, n_trials, 2),
                dtype=torch.float32,
                device=self.m.device,
            )
            phases = torch.zeros(
                (n_freqs, n_amps, n_trials), dtype=torch.float32, device=self.m.device
            )
            ICs = []
            ind = 0
            for _, freq in enumerate(self.config["freqs"]):
                for _, amp in enumerate(self.config["amps"]):
                    perc = ind / (n_freqs * n_amps)
                    ICs.append([freq, amp, perc])
                    ind += 1

            Ks_phases = Parallel(n_jobs=n_jobs)(delayed(self.run_sim)(IC) for IC in ICs)

            ind = 0
            for i in range(len(self.config["freqs"])):
                for j in range(len(self.config["amps"])):
                    phases[i, j] = Ks_phases[ind][0]
                    Ks[i, j] = Ks_phases[ind][1]
                    ind += 1
        return Ks, phases

    def calc_one_floquet(self, IC, stim=None):
        """
        Calculate floquet multipliers at a given state
        
        Args:
            IC: contains frequency, phase and amplitude of the reference, and the state of the RNN
            stim: Additional stimulus input to the RNN
        Returns:
            evs: The floquet multipliers norm
            k_diff_i: The norm of the difference between the initial and state 
                      after one cycle / applying the poincare map once (should be close to 0)
            dK_norm: The average norm of the change in Ks per time step during one cycle
        """

        if stim is None:
            stim = torch.zeros(self.I.size()[0] - 1)

        with torch.no_grad():
            freq = IC[0]
            amp = IC[1]
            K = IC[3]
            phase = IC[2]
            n_trials = len(phase)
            w = np.pi * 2 * freq
            period = int(1 / (freq * self.config["dt"]))
            dK_norm = torch.zeros(n_trials)
            
            # initialize monodromy as the identity matrix
            monodromy = (
                torch.eye(2, device=self.m.device).unsqueeze(0).tile(n_trials, 1, 1)
            )
            k_t = K
            k_0 = torch.clone(k_t)
            for _ in np.arange(0, period):
                k_t, phase, monodromy, dK_norm = self.forward_Jac(
                    k_t, phase, monodromy, dK_norm, amp, w, stim=stim
                )

            k_diffi = torch.norm(k_t - k_0, dim=-1)
            dK_norm /= period

            evs = torch.zeros(n_trials, 2)
            for ind in range(n_trials):
                ev = torch.linalg.eigvals(monodromy[ind])
                evs[ind] = abs(ev)

            print({"FM_progress": IC[4]})

        return [evs, k_diffi, dK_norm]

    def calc_floquet(self, Ks, phases, n_jobs=2):
        """
        Calculate floquet multipliers at given states
        
        Args:
            Ks: The states of the RNN projected on the m vectors
            phases: The phases of the reference oscillation
        Returns:
            evs: The floquet multipliers norm
            k_diff: The norm of the difference between the initial and state 
                      after one cycle / applying the poincare map once (should be close to 0)
            dK_norm: The average norm of the change in Ks per time step during one cycle
        """
        n_freqs = len(self.config["freqs"])
        n_amps = len(self.config["amps"])
        n_trials = phases.size()[-1]

        evs = torch.zeros((n_freqs, n_amps, n_trials, 2), device=self.m.device)
        k_diff = torch.zeros((n_freqs, n_amps, n_trials), device=self.m.device)
        dK_norms = torch.zeros((n_freqs, n_amps, n_trials), device=self.m.device)
        with torch.no_grad():
            evs = torch.zeros((n_freqs, n_amps, n_trials, 2), device=self.m.device)
            k_diff = torch.zeros((n_freqs, n_amps, n_trials), device=self.m.device)
            dK_norms = torch.zeros((n_freqs, n_amps, n_trials), device=self.m.device)
            ICs = []
            ind = 0
            for i, freq in enumerate(self.config["freqs"]):
                for j, amp in enumerate(self.config["amps"]):
                    perc = ind / (n_freqs * n_amps)
                    ICs.append([freq, amp, phases[i, j], Ks[i, j], perc])
                    ind += 0

            k_t_evs_k_diffi_dK_norm = Parallel(n_jobs=n_jobs)(
                delayed(self.calc_one_floquet)(IC) for IC in ICs
            )

            ind = 0
            for i, freq in enumerate(self.config["freqs"]):
                for j, amp in enumerate(self.config["amps"]):
                    evs[i, j] = k_t_evs_k_diffi_dK_norm[ind][0]
                    k_diff[i, j] = k_t_evs_k_diffi_dK_norm[ind][1]
                    dK_norms[i, j] = k_t_evs_k_diffi_dK_norm[ind][2]
                    ind += 1
        return (
            evs.detach().cpu().numpy(),
            k_diff.detach().cpu().numpy(),
            dK_norms.detach().cpu().numpy(),
        )

    def run_sims_stim(self, sync_wandb, phase_0, k_0, stims):
        """
        Run simulations with a given stimulus input

        Args:
            sync_wandb: Whether to sync with wandb
            phase_0: The initial phases of the reference oscillation
            k_0: The initial states of the RNN projected on the m vectors
            stims: The stimulus input to the RNN

        Returns:
            Ks: The states of the RNN projected on the m vectors
            phases: The phases of the reference oscillation
        """
        n_amps = len(self.config["amps"])
        n_trials = len(phase_0)
        n_stim = len(stims)
        Ks = torch.zeros(
            (n_stim, n_amps, n_trials, 2), dtype=torch.float32, device=self.m.device
        )
        phases = torch.zeros(
            (n_stim, n_amps, n_trials), dtype=torch.float32, device=self.m.device
        )
        freq = self.config["freqs"][0]

        with torch.no_grad():
            for i, stim in enumerate(stims):
                print("PROGRESS: " + str(i / n_stim))
                if sync_wandb:
                    wandb.log({"ls_progress": i / n_stim})

                w = np.pi * 2 * freq
                period = int(1 / (freq * self.config["dt"]))

                for j, amp in enumerate(self.config["amps"]):
                    phase = torch.clone(phase_0)
                    k_t = torch.clone(k_0)

                    for _ in range(period * self.config["n_periods"]):
                        k_t, phase[:] = self.forward(k_t, phase[:], 1, w, stim * amp)
                    phases[i, j] = phase
                    Ks[i, j] = k_t

        return Ks, phases

    def calc_floquet_stim(self, sync_wandb, Ks, phases, stims):
        """
        Calculate floquet multipliers at given states, with a given stimulus input

        Args:  
            sync_wandb: Whether to sync with wandb
            Ks: The states of the RNN projected on the m vectors
            phases: The phases of the reference oscillation
            stims: The stimulus input to the RNN
        
        Returns:
            evs: The floquet multipliers norm
            evs_c: The floquet multipliers in complex form
            k_diff: The norm of the difference between the initial and state
                        after one cycle / applying the poincare map once (should be close to 0)
            dK_norm: The average norm of the change in Ks per time step during one cycle
        
        """
        freq = self.config["freqs"][0]
        n_amps = len(self.config["amps"])
        n_stim = len(stims)
        n_trials = phases.size()[-1]

        evs = torch.zeros((n_stim, n_amps, n_trials, 2), device=self.m.device)
        evs_c = torch.zeros(
            (n_stim, n_amps, n_trials, 2), device=self.m.device, dtype=torch.complex64
        )

        k_diff = torch.zeros((n_stim, n_amps, n_trials), device=self.m.device)
        dK_norms = torch.zeros((n_stim, n_amps, n_trials), device=self.m.device)

        with torch.no_grad():
            for i, stim in enumerate(stims):
                print("PROGRESS: " + str(i / n_stim))
                if sync_wandb:
                    wandb.log({"FM_progress": i / n_stim})
                w = np.pi * 2 * freq
                period = int(1 / (freq * self.config["dt"]))

                for j, amp in enumerate(self.config["amps"]):
                    # initialize
                    monodromy = (
                        torch.eye(2, device=self.m.device)
                        .unsqueeze(0)
                        .tile(n_trials, 1, 1)
                    )
                    k_t = Ks[i, j]
                    k_0 = torch.clone(k_t)
                    phase = phases[i, j]
                    # for period_no in np.arange(selfn_periods):
                    for t in np.arange(0, period):
                        k_t, phase, monodromy, dK_norms[i, j] = self.forward_Jac(
                            k_t, phase, monodromy, dK_norms[i, j], 1, w, stim=stim * amp
                        )

                    k_diffi = torch.norm(k_t - k_0, dim=-1)
                    k_diff[i, j] = k_diffi
                    dK_norms[i, j] /= period

                    for ind in range(n_trials):
                        # print("periods: " + str(period_no))
                        ev = torch.linalg.eigvals(monodromy[ind])
                        evs[i, j, ind] = abs(ev)
                        evs_c[i, j, ind] = ev

        return (
            evs.detach().cpu().numpy(),
            evs_c.detach().cpu().numpy(),
            k_diff.detach().cpu().numpy(),
            dK_norms.detach().cpu().numpy(),
        )
