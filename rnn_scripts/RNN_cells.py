import torch
import torch.nn as nn
import numpy as np
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from initializers import *


class RNNCell(nn.Module):
    def __init__(self, params):
        """
        Full rank RNN cell

        Args:
            params: dictionary with model params
        """
        super(RNNCell, self).__init__()

        # activation function
        self.nonlinearity = set_nonlinearity(params["nonlinearity"])
        self.out_nonlinearity = set_nonlinearity(params["out_nonlinearity"])

        # declare network parameters
        self.w_inp = nn.Parameter(torch.Tensor(params["n_inp"], params["n_rec"]))
        self.w_rec = nn.Parameter(torch.Tensor(params["n_rec"], params["n_rec"]))
        self.w_out = nn.Parameter(torch.Tensor(params["n_rec"], params["n_out"]))
        self.w_inp_scale = nn.Parameter(torch.Tensor(1))
        self.w_out_scale = nn.Parameter(torch.Tensor(1, params["n_out"]))
        if not params["train_w_out_scale"]:
            self.w_out_scale.requires_grad = False
        if not params["train_w_inp_scale"]:
            self.w_inp_scale.requires_grad = False
        if not params["train_w_inp"]:
            self.w_inp.requires_grad = False
        if not params["train_w_out"]:
            self.w_out.requires_grad = False
        if not params["train_w_rec"]:
            self.w_rec.requires_grad = False

        # time constants
        self.dt = params["dt"]
        self.tau = params["tau_lims"]
        if len(params["tau_lims"]) > 1:
            self.taus_gaus = nn.Parameter(torch.Tensor(params["n_rec"]))
            if not params["train_taus"]:
                self.taus_gaus.requires_grad = False

        # initialize parameters
        with torch.no_grad():
            w_inp = initialize_w_inp(params)
            self.w_inp = self.w_inp.copy_(torch.from_numpy(w_inp))
            self.w_inp_scale = self.w_inp_scale.fill_(params["scale_w_inp"])
            self.w_out_scale = self.w_out_scale.fill_(params["scale_w_out"])
            w_rec, dale_mask = initialize_w_rec(params)
            self.dale_mask = torch.from_numpy(dale_mask)

            self.w_rec = self.w_rec.copy_(torch.from_numpy(w_rec))

            # deep versus shallow learning?
            if params["1overN_out_scaling"]:
                self.w_out = self.w_out.normal_(std=1 / params["n_rec"])
            else:
                self.w_out = self.w_out.normal_(std=1 / np.sqrt(params["n_rec"]))

            # connection mask
            if params["apply_dale"]:
                self.mask = mask_dale
            else:
                self.mask = mask_none

            # possibly initialize tau with distribution
            # (this is then later projected to be within preset limits)
            if len(params["tau_lims"]) > 1:
                self.taus_gaus.normal_(std=1)

    def forward(self, input, x, noise=0):
        """
        Do a forward pass through one timestep

        Args:
            input: tensor of size [batch_size, seq_len, n_inp]
            x: hidden state at current time step, tensor of size [batch_size, n_rec]
            noise: noise at current time step, tensor of size [batch_size, n_rec]

        Returns:
            x: hidden state at next time step, tensor of size [batch_size, n_rec]
            output: linear readout at next time step, tensor of size [batch_size, n_out]

        """

        # apply mask to weight matrix
        w_eff = self.mask(self.w_rec, self.dale_mask)

        # compute alpha (dt/tau), and scale noise accordingly
        if len(self.tau) == 1:
            alpha = self.dt / self.tau[0]
            noise_t = np.sqrt(2 * alpha) * noise
        else:
            taus_sig = project_taus(self.taus_gaus, self.tau[0], self.tau[1])
            alpha = self.dt / taus_sig
            noise_t = torch.sqrt(2 * alpha) * noise

        # calculate input to units
        rec_input = torch.matmul(
            self.nonlinearity(x), w_eff.t()
        ) + self.w_inp_scale * input.matmul(self.w_inp)
        # update hidden state
        x = (1 - alpha) * x + alpha * rec_input + noise_t

        # linear readout of the rates
        output = self.w_out_scale * self.out_nonlinearity(x).matmul(self.w_out)

        return x, output


class LR_RNNCell(nn.Module):
    def __init__(self, params):
        """
        RNN cell with rank of the recurrent weight matrix constrained
        (contains parameters and computes one step forward)

        Args:
            params: dictionary with model params
        """
        super(LR_RNNCell, self).__init__()

        # activation function
        self.nonlinearity = set_nonlinearity(params["nonlinearity"])
        self.out_nonlinearity = set_nonlinearity(params["out_nonlinearity"])

        # declare network parameters
        self.w_inp = nn.Parameter(torch.Tensor(params["n_inp"], params["n_rec"]))
        if not params["train_w_inp"]:
            self.w_inp.requires_grad = False
        self.w_inp_scale = nn.Parameter(torch.Tensor(1))
        if not params["train_w_inp_scale"]:
            self.w_inp_scale.requires_grad = False

        self.m = nn.Parameter(torch.Tensor(params["n_rec"], params["rank"]))
        if not params["train_m"]:
            self.m.requires_grad = False
        self.n = nn.Parameter(torch.Tensor(params["rank"], params["n_rec"]))
        if not params["train_n"]:
            self.n.requires_grad = False

        self.w_out = nn.Parameter(torch.Tensor(params["n_rec"], params["n_out"]))
        if not params["train_w_out"]:
            self.w_out.requires_grad = False
        self.w_out_scale = nn.Parameter(torch.Tensor(1, params["n_out"]))
        if not params["train_w_out_scale"]:
            self.w_out_scale.requires_grad = False

        self.dt = params["dt"]
        self.tau = params["tau_lims"][0]
        if len(params["tau_lims"]) > 1:
            print("WARNING: distribution of Tau currently not supported for LR RNN")
        self.N = params["n_rec"]
        self.rank = params["rank"]
        self.readout_kappa = params["readout_kappa"]

        # initialize network parameters
        with torch.no_grad():
            if params["loadings"] is None:
                loadings = initialize_loadings(params)
            else:
                loadings = params["loadings"]

            self.w_inp = self.w_inp.copy_(torch.from_numpy(loadings[: params["n_inp"]]))
            self.w_out = self.w_out.copy_(
                torch.from_numpy(loadings[-params["n_out"] :]).T
            )

            # n and m are parallel to the left and right singular vectors
            # of the recurrent weight matrix

            self.n = self.n.copy_(
                torch.from_numpy(
                    loadings[params["n_inp"] : params["n_inp"] + params["rank"]]
                    * np.sqrt(params["scale_n"])
                )
            )
            self.m = self.m.copy_(
                torch.from_numpy(
                    loadings[
                        params["n_inp"]
                        + params["rank"] : params["n_inp"]
                        + params["rank"] * 2
                    ].T
                    * np.sqrt(params["scale_m"])
                )
            )
            self.w_inp_scale = self.w_inp_scale.fill_(params["scale_w_inp"])
            self.w_out_scale = self.w_out_scale.fill_(params["scale_w_out"])

    def forward(self, input, x, noise=0):
        """
        Do a forward pass through one timestep

        Args:
            input: tensor of size [batch_size, n_inp]
            x: hidden state at current time step, tensor of size [batch_size, n_rec]
            noise: noise at current time step, tensor of size [batch_size, n_rec]

        Returns:
            x: hidden state at next time step, tensor of size [batch_size, n_rec]
            output: linear readout at next time step, tensor of size [batch_size, n_out]

        """

        alpha = self.dt / self.tau

        # input to units
        rec_input = torch.matmul(
            torch.matmul(self.nonlinearity(x), self.n.t()), self.m.t()
        ) / self.N + self.w_inp_scale * input.matmul(self.w_inp)

        # update hidden state
        x = (1 - alpha) * x + alpha * rec_input + np.sqrt(2 * alpha) * noise

        # linear readout
        output = self.readout(x)

        return x, output

    def readout(self, x):
        if self.readout_kappa:
            output = x.matmul(self.m)
            output /= torch.norm(self.m, dim=0, keepdim=True)
        else:
            output = (
                self.w_out_scale * self.out_nonlinearity(x).matmul(self.w_out) / self.N
            )
        return output

    def svd_orth(self):
        """
        Orthogonalize m and n via SVD
        """
        with torch.no_grad():
            J = (self.m @ self.n).cpu().numpy()
            m, s, n = np.linalg.svd(J, full_matrices=False)
            m, s, n = m[:, : self.rank], s[: self.rank], n[: self.rank, :]
            m *= np.sqrt(s)
            device = self.m.device
            m = torch.from_numpy(m).to(device)
            n = torch.from_numpy((n.T * np.sqrt(s)).T).to(device)
            self.m.set_(m)
            self.n.set_(n)


class Cov_RNNCell(nn.Module):
    """
    Train covariance matrix, using reparameterisation trick
    """

    def __init__(self, params):
        super(Cov_RNNCell, self).__init__()

        # activation function
        self.nonlinearity = set_nonlinearity(params["nonlinearity"])
        self.out_nonlinearity = set_nonlinearity(params["out_nonlinearity"])

        # assign network parameters
        self.dt = params["dt"]
        self.tau = params["tau_lims"][0]
        self.n_supports = params["n_supports"]
        if len(params["tau_lims"]) > 1:
            print("WARNING: distribution of Tau currently not supported for LR RNN")
        self.loading_dim = 2 * params["rank"] + params["n_inp"] + params["n_out"]

        # covariance matrices need to be SPD, so we train the cholesky decomposition
        self.cov_chols = nn.Parameter(
            torch.Tensor(params["n_supports"], self.loading_dim, self.loading_dim)
        )
        self.N = params["n_rec"]
        self.n_inp = params["n_inp"]
        self.rank = params["rank"]
        self.w_inp_scale = nn.Parameter(torch.Tensor(1))
        self.readout_kappa = params["readout_kappa"]
        if not params["train_w_inp_scale"]:
            self.w_inp_scale.requires_grad = False

        self.w_out_scale = nn.Parameter(torch.Tensor(1, params["n_out"]))
        if not params["train_w_out_scale"]:
            self.w_out_scale.requires_grad = False

        # Initialize parameters
        covs = np.zeros(
            (params["n_supports"], self.loading_dim, self.loading_dim), dtype=np.float32
        )
        self.weights = nn.Parameter(torch.Tensor(params["n_supports"]))
        self.weights.requires_grad = False
        with torch.no_grad():
            self.weights.copy_(
                torch.from_numpy(np.ones(params["n_supports"]) / params["n_supports"])
            )
            for i in range(params["n_supports"]):
                cov_chol = (
                    initialize_loadings(params, return_loadings=False)
                    + np.random.randn(self.loading_dim, self.loading_dim)
                    * params["cov_init_noise"]
                )
                covs[i] = cov_chol
            self.cov_chols.copy_(torch.from_numpy(covs))
            self.w_inp_scale = self.w_inp_scale.fill_(params["scale_w_inp"])
            self.w_out_scale = self.w_out_scale.fill_(params["scale_w_out"])

        self.Gauss = torch.distributions.Normal(0, 1)
        self.resample()

    def resample(self):
        """
        Resample entries in the network connectivity from the trainable covariance matrix
        """
        loadings = []
        for cov_chol, w in zip(self.cov_chols,self.weights):
            gaussian_basis = self.Gauss.sample(
                (self.loading_dim, int(self.N * w))
            )
            loadings.append(cov_chol@gaussian_basis)
        self.loadings = torch.cat(loadings, dim=1)
        #print(self.cov_chols.device)
        self.n = self.loadings[self.n_inp : self.n_inp + self.rank]#.to(device = self.cov_chols.device)
        #pritn(s)
        self.m = self.loadings[
            self.n_inp + self.rank : self.n_inp + self.rank + self.rank
        ].t()#.to(device = self.cov_chols.device)
        self.w_inp = self.loadings[: self.n_inp]#.to(device = self.cov_chols.device)
        self.w_out = self.loadings[self.n_inp + self.rank + self.rank :].t()#.to(device = self.cov_chols.device)

    def forward(self, input, x, noise=0):
        """
        Do a forward pass through one timestep

        Args:
            input: tensor of size [batch_size, n_inp]
            x: hidden state at current time step, tensor of size [batch_size, n_rec]
            noise: noise at current time step, tensor of size [batch_size, n_rec]

        Returns:
            x: hidden state at next time step, tensor of size [batch_size, n_rec]
            output: linear readout at next time step, tensor of size [batch_size, n_out]

        """

        alpha = self.dt / self.tau

        # input to units
        rec_input = torch.matmul(
            torch.matmul(self.nonlinearity(x), self.n.t()), self.m.t()
        ) / self.N + self.w_inp_scale * input.matmul(self.w_inp)

        # update hidden state
        x = (1 - alpha) * x + alpha * rec_input + np.sqrt(2 * alpha) * noise

        # linear readout
        output = self.readout(x)

        return x, output

    def readout(self, x):
        if self.readout_kappa:
            output = x.matmul(self.m)
            output /= torch.norm(self.m, dim=0, keepdim=True)
        else:
            output = (
                self.w_out_scale * self.out_nonlinearity(x).matmul(self.w_out) / self.N
            )
        return output


class Meanfield_RNNCell(nn.Module):
    """
    Simulate reduced equations:
    directly train parametrization of covariance matrix
    """

    def __init__(self, params):
        super(Meanfield_RNNCell, self).__init__()

        # activation function
        self.nonlinearity = set_nonlinearity(params["nonlinearity"])
        self.out_nonlinearity = set_nonlinearity(params["out_nonlinearity"])

        # assign network parameters
        self.dt = params["dt"]
        self.tau = params["tau_lims"][0]
        self.n_supports = params["n_supports"]
        if len(params["tau_lims"]) > 1:
            print("WARNING: distribution of Tau currently not supported for LR RNN")
        loading_dim = 2 * params["rank"] + params["n_inp"] + params["n_out"]
        self.loading_dim = loading_dim
        self.rank = params["rank"]
        self.n_inp = params["n_inp"]
        self.n_out = params["n_out"]
        self.w_out = nn.Parameter(torch.Tensor(params["rank"], params["n_out"]))
        self.weights = nn.Parameter(torch.Tensor(params["n_supports"]))
        self.weights.requires_grad = False

        covs = np.zeros(
            (params["n_supports"], self.loading_dim, self.loading_dim), dtype=np.float32
        )
        self.cov_chols = nn.Parameter(
            torch.Tensor(params["n_supports"], self.loading_dim, self.loading_dim)
        )

        # Initialize parameters
        with torch.no_grad():
            self.weights.copy_(
                torch.from_numpy(np.ones(params["n_supports"]) / params["n_supports"])
            )
            for i in range(params["n_supports"]):
                cov_chol = (
                    initialize_loadings(params, return_loadings=False)
                    + np.random.randn(self.loading_dim, self.loading_dim)
                    * params["cov_init_noise"]
                )
                covs[i] = cov_chol
            self.cov_chols.copy_(torch.from_numpy(covs))
            self.w_out.copy_(
                torch.ones(params["rank"], params["n_out"])
                / (params["rank"] * params["n_out"])
            )

        self.gaussian_norm = 1 / np.sqrt(np.pi)
        self.gauss_points, self.gauss_weights = np.polynomial.hermite.hermgauss(50)
        self.gauss_points = self.gauss_points * np.sqrt(2)
        self.gauss_points = torch.from_numpy(self.gauss_points)
        self.gauss_weights = torch.from_numpy(self.gauss_weights)

        # just because we refer device with this:
        self.w_inp = nn.Parameter(torch.Tensor(params["n_inp"], params["n_rec"]))
        self.readout_kappa = params["readout_kappa"]
        self.orth_indices = params["orth_indices"]
        self.out_nonlinearity = params["out_nonlinearity"]
    def forward(self, input, x, noise=0):
        """
        Do a forward pass through one timestep

        Args:
            input: tensor of size [batch_size, n_inp]
            x: hidden state at current time step, tensor of size [batch_size, rank+n_inp]
        Returns:
            x: hidden state at next time step, tensor of size [batch_size, rank+n_inp]
            output: linear readout at next time step, tensor of size [batch_size, n_out]

        """

        K = x[:, : self.rank]
        V = x[:, self.rank :]
        alpha = self.dt / self.tau
        C = self.cov_chols
        covs = torch.bmm(C, C.permute((0, 2, 1)))
        n_ind = self.n_inp
        m_ind = self.rank + self.n_inp
        #print(covs)
        #print(n_ind)
        #print(m_ind)
        # Calculate gains
        delta0 = torch.sum(
            torch.stack(
                [
                    (K[:, i] ** 2).outer(covs[:, m_ind + i, m_ind + i])
                    for i in range(self.rank)
                ]
            ),
            axis=0,
        )
        delta0 += torch.sum(
            torch.stack(
                [(V[:, i] ** 2).outer(covs[:, i, i]) for i in range(self.n_inp)]
            ),
            axis=0,
        )
        gains = self.gain(delta0)
        # Recurrent decay
        dK = -K

        # Recurrent input
        for i in range(self.rank):
            ni_inp = torch.sum(
                torch.stack(
                    [
                        K[:, j].outer(covs[:, n_ind + i, m_ind + j])
                        for j in range(self.rank)
                    ]
                ),
                axis=0,
            )
            ni_inp += torch.sum(
                torch.stack(
                    [V[:, j].outer(covs[:, n_ind + i, j]) for j in range(self.n_inp)]
                ),
                axis=0,
            )

            # inp has shape [BATCH_SIZE, N_SUPPORTS]
            dK[:, i] += torch.sum((ni_inp) * gains * self.weights, axis=1)

        # External input
        dV = -V + input

        # update hidden state
        K_n = K + alpha * dK
        V_n = V + alpha * dV

        # output
        if self.readout_kappa:
            out = K_n[:,0].unsqueeze(1)
        else:
            out = torch.zeros((K.size()[0], self.n_out))
            for i in range(self.n_out):
                w_out = torch.sum(
                    torch.stack(
                        [
                            K_n[:, j].outer(covs[:, -(i + 1), m_ind + j])
                            for j in range(self.rank)
                        ]
                    ),
                    axis=0,
                )
                w_out += torch.sum(
                    torch.stack(
                        [
                            V_n[:, j].outer(covs[:, -(i + 1), j])
                            for j in range(self.n_inp)
                        ]
                    ),
                    axis=0,
                )
                if self.out_nonlinearity=="tanh":
                    out[:, i] += torch.sum(w_out * gains * self.weights, axis=1)
                else:
                    out[:, i] += torch.sum(w_out * self.weights, axis=1)

        x = torch.cat((K_n, V_n), 1)
        return x, out

    def gain(self, delta0):
        """
        return expectation of tanh'(x)
        Use close form solution obtained by approximating tanh with the error function
        """
        return 1 / torch.sqrt(1 + np.pi / 2 * delta0)


def project_taus(x, lim_low, lim_high):
    """
    Apply a non linear projection map to keep x within bounds

    Args:
        x: Tensor with unconstrained range
        lim_low: lower bound on range
        lim_high: upper bound on range

    Returns:
        x_lim: Tensor constrained to have range (lim_low, lim_high)
    """

    x_lim = torch.sigmoid(x) * (lim_high - lim_low) + lim_low
    return x_lim


# Note make these callable classes so we don't have to pass the mask each time?
# implement __cal__ method
def mask_dale(w_rec, mask):
    """Apply Dale mask"""
    return torch.matmul(torch.relu(w_rec), mask)


def mask_none(w_rec, mask):
    """Apply no mask"""
    return w_rec


def set_nonlinearity(param):
    """utility returning activation function"""
    if param == "tanh":
        return torch.tanh
    elif param == "identity":
        return lambda x: x
    elif param == "logistic":
        return torch.sigmoid
    elif param == "relu":
        return nn.ReLU()
    elif param == "softplus":
        softplus_scale = 1  # Note that scale 1 is quite far from relu
        nonlinearity = (
            lambda x: torch.log(1.0 + torch.exp(softplus_scale * x)) / softplus_scale
        )
        return nonlinearity
    elif type(param) == str:
        print("Nonlinearity not yet implemented.")
        print("Continuing with identity")
        return lambda x: x
    else:
        return param


def orth_rows(rnn, row_indices):
    "Orthogonalize rows, without gradients"
    "projection for projected gradient descent"
    C = rnn.cov_chols.detach()
    for cov in range(rnn.n_supports):
        for pairs in row_indices:
            i = pairs[0]
            j = pairs[1]
            Cj_on_i, _ = orth_proj(C[cov, i], C[cov, j])
            C[cov, j] -= Cj_on_i
        with torch.no_grad():
            rnn.cov_chols.copy_(C)


def orth_rows_forward(cov_chols, row_indices):
    "Orthogonalize rows, with gradients"
    C = cov_chols.clone()
    for cov in range(cov_chols.size()[0]):
        for pairs in row_indices:
            i = pairs[0]
            j = pairs[1]
            Cj_on_i, _ = orth_proj(cov_chols[cov, i], cov_chols[cov, j])
            C[cov, j] = cov_chols[cov, j].clone() - Cj_on_i
    return C


def orth_proj(a, b, ret_comp=False):
    """
    Orthogonal projection of b on a
    Args:
        a in Nx1
        b in Nx1
    Returns:
        b_par in Nx1
        b_orth in Nx1
    """
    alpha = torch.matmul(a.t(), b) / torch.matmul(a.t(), a)
    b_par = alpha * a

    if ret_comp:
        b_orth = b - b_par
        return b_par, b_orth, alpha
    else:
        return b_par, alpha
