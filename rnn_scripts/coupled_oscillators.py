import numpy as np


class coupled_oscillators:
    """
    Simulate coupled oscillators, with parameters extracted from trained LRRNNs
    """

    def __init__(self, tau, freq, m, n, I_orth, alphas, rad, amp,n_osc=1):
        """
        Initialise coupled oscillators

        Args:
            tau: time constant of the internal oscillator / RNN
            freq: frequency of the external, reference oscillation
            m: left connectivity vectors of the RNN
            n: right connectivity vectors of the RNN
            I_orth: orthogonalised input weights of the RNN
            alphas: input weights of the RNN projected on m
            rad: mean radius of the RNN in the k1, k2 plane
            amp: amplitude of the external, reference oscillation

        """
        self.tau = tau
        self.w = np.pi * 2 * freq
        self.m = m
        self.n = n
        self.I_orth = I_orth
        self.alphas = alphas
        self.rad = rad
        self.N = len(m[0])
        self.amp = amp
        self.n_osc = n_osc

    def coupling_function(self, theta, phi, inp):
        """
        Given thetas, phi and input, return dtheta and dphi
        
        Args:
            theta: phase of the external, reference oscillation
            phi: phase of the internal oscillation / RNN
            inp: stimulus input to the internal oscillator / RNN

        Returns:
            dtheta: change in theta
            dphi: change in phi
        """

        # transfer from polar to cartesian coordinates
        k1 = np.cos(phi) * self.rad
        k2 = np.sin(phi) * self.rad

        tau_sw = self.tau * self.w
        
        # input to the internal oscillator 
        v_osc = (
            self.amp / np.sqrt(1 + tau_sw**2) * np.sin(theta - np.arctan2(tau_sw, 1))
        )
        v_osc_cos = (
            self.amp / np.sqrt(1 + tau_sw**2) * np.cos(theta - np.arctan2(tau_sw, 1))
        )

        # calculate RNN unit activity
        x = (
            k1 * self.m[0]
            + k2 * self.m[1]
            + inp @ self.I_orth[self.n_osc:]
            +  v_osc * self.I_orth[0]
        )
        if self.n_osc>1:
            x += v_osc_cos * self.I_orth[1]

        # calculate change in cartesian coordinates
        dk = (1 / self.tau) * (
            -np.array([k1, k2])
            + (1 / self.N) * self.n.dot(np.tanh(x))
            + self.alphas[:, 0] * np.sin(theta) * self.amp
        )
        if self.n_osc>1:
            dk += self.alphas[:, 1]*np.cos(theta) * self.amp

        for i in range(len(inp)):
            dk += (1 / self.tau) * self.alphas[:, self.n_osc + i] * inp[i]

        # convert back to polar coordinates
        dr = (k1 * dk[0] + k2 * dk[1]) / (self.rad)
        dt = (k1 * dk[1] - k2 * dk[0]) / (self.rad**2)
        return dr, dt

    def plot_coupling(self, thetas, phis, inp):
        """
        Plot the coupling function for a given input and range of thetas and phis

        Args:
            thetas: range of thetas to plot
            phis: range of phis to plot
            inp: input to the internal oscillator / RNN
        
        Returns:
            grid: grid of coupling function values
        
        """
        grid = np.zeros((len(thetas), len(phis)))
        for i, theta in enumerate(thetas):
            for j, phi in enumerate(phis):
                grid[j, i] = self.coupling_function(theta, phi, inp=inp)[1]
        return grid

    def F(self, state, inp):
        """
        Calculate the derivative of the state vector, containing theta and phi
        
        Args:
            state: state vector containing theta and phi
            inp: input to the internal oscillator / RNN

        Returns:
            dx: derivative of the state vector containging theta and phi    
        """
        dx = np.zeros_like(state)
        dx[0] = self.w
        dx[1] = self.coupling_function(state[0], state[1], inp)[1]
        return dx

    def euler(self, init_state, dur, dt, inp):
        """
        Do a forward euler integration of the coupled oscillators

        Args:
            init_state: initial state of the system
            dur: duration of the simulation
            dt: timestep of the simulation
            inp: input to the internal oscillator / RNN
        
        Returns:
            states: array of states of the system
        """
        states = np.zeros((int(dur / dt), 2))
        states[0] = init_state
        for i in range(len(states) - 1):
            states[i + 1] = states[i] + dt * self.F(states[i], inp)
        return states

    def euler_backwards(self, init_state, dur, dt, inp):
        """
        Do a backwards euler integration of the coupled oscillators,
        which is equivalent to a forward euler integration of the
        negative of the derivative. This is used to find unstable
        fixed points / limit cycles.
        """
        states = np.zeros((int(dur / dt), 2))
        states[0] = init_state
        for i in range(len(states) - 1):
            states[i + 1] = states[i] - dt * self.F(states[i], inp)
        return states

    def run_sims(self, init_phis, init_thetas, dur, dt, inp, forward=True):
        """
        Run a number of simulations of the coupled oscillators

        Args:
            init_phis: initial values of phi to simulate
            init_thetas: initial values of theta to simulate
            dur: duration of the simulation in seconds
            dt: timestep of the simulation in seconds
            inp: input to the internal oscillator / RNN
            forward (optional): whether to do a forward euler integration or a backwards euler integration
        
        Returns:
            all_states: array of states of the system
        """

        if forward:
            sim = self.euler
        else:
            sim = self.euler_backwards

        init_state = np.zeros(2)
        all_states = np.zeros((len(init_phis) * len(init_thetas), int(dur / dt), 2))
        si = 0
        for i in init_phis:
            for j in init_thetas:
                init_state[1] = i
                init_state[0] = j
                all_states[si] = sim(init_state, dur, dt, inp)
                si += 1
        return all_states


def rescale(x, size):
    """
    Rescale trajectories so that they can overlay an image of size size
    """
    return size * (x + np.pi) / (np.pi * 2)


def detect_breaks(x, y, tol=3):
    """
    Utility function to plot trajectories with circular boundary conditions
    """
    abs_d_x = np.abs(np.diff(x))
    abs_d_y = np.abs(np.diff(y))

    mask1 = np.hstack([abs_d_x > abs_d_x.mean() + tol * abs_d_x.std(), [False]])
    mask2 = np.hstack([abs_d_y > abs_d_y.mean() + tol * abs_d_y.std(), [False]])
    mask = mask1 + mask2
    masked_x = np.ma.MaskedArray(x, mask)
    masked_y = np.ma.MaskedArray(y, mask)

    return masked_x, masked_y
