import torch
import torch.nn as nn
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from RNN_cells import *


class RNN(nn.Module):
    def __init__(self, params):
        """
        Initializes a continuous time recurrent neural network model

        Args:
            params: python dictionary containing network parameters
        """

        super(RNN, self).__init__()
        self.params = params

        # choose between full and low rank RNN cell
        if params["rank"]:
            # there are multiple options for low-rank network

            # train reduced description of connectivity in terms of Gaussians
            if self.params["train_meanfield"]:
                self.rnn = Meanfield_RNNCell(params)

            # train Gaussian decomposition of connectivity where entries are sampled
            elif self.params["train_cov"]:
                self.rnn = Cov_RNNCell(params)

            # or train standard low rank RNN
            else:
                self.rnn = LR_RNNCell(params)
        else:
            # standard full rank RNN
            self.rnn = RNNCell(params)

        # make hidden state at t = 0 (x0) a parameter, as it could be learned
        self.x0 = nn.Parameter(torch.Tensor(1, params["n_rec"]))
        if not params["train_x0"]:
            self.x0.requires_grad = False

        # initialise x0 normally distributed with specified variance
        with torch.no_grad():
            if params["scale_x0"]:
                self.x0 = self.x0.normal_(std=params["scale_x0"])
            else:
                self.x0 = self.x0.copy_(
                    torch.zeros(1, params["n_rec"], dtype=torch.float32)
                )

    def forward(self, input, x0=None):
        """
        Do a forward pass through all time steps

        Args:
            input: tensor of size [batch_size, seq_len, n_inp]
            x0 (optional): hidden state at t=0
        """

        batch_size = input.size(0)
        seq_len = input.size(1)

        # precompute noise
        noise = (
            torch.randn(
                batch_size,
                seq_len,
                self.params["n_rec"],
                device=self.rnn.w_inp.device,
                dtype=torch.float32,
            )
            * self.params["noise_std"]
        )

        # allocate tensors for hidden state and output
        outputs = torch.zeros(
            batch_size,
            seq_len,
            self.params["n_out"],
            device=self.rnn.w_inp.device,
            dtype=torch.float32,
        )
        hidden = torch.zeros(
            batch_size,
            seq_len + 1,
            self.params["n_rec"],
            device=self.rnn.w_inp.device,
            dtype=torch.float32,
        )

        # initialize current x0 at t=0

        # option 1: use x0 that was used as input to this function
        if x0 is not None:
            if x0.shape[0] == 1:
                h_t = torch.tile(x0, dims=(batch_size, 1))
            else:
                h_t = x0

        # option 2: use random x0
        elif self.params["randomise_x0"]:
            if self.params["rank"] > 0 and self.params["train_meanfield"] == False:
                # random k (in space spanned by singular vector)
                h_t = (
                    torch.outer(
                        torch.randn(
                            batch_size,
                            device=self.rnn.w_inp.device,
                            dtype=torch.float32,
                        ),
                        self.rnn.m[:, 0],
                    )
                    + torch.outer(
                        torch.randn(
                            batch_size,
                            device=self.rnn.w_inp.device,
                            dtype=torch.float32,
                        ),
                        self.rnn.m[:, 1],
                    )
                ) * self.params["scale_x0"]
            else:
                h_t = (
                    torch.randn(
                        batch_size,
                        self.params["n_rec"],
                        device=self.rnn.w_inp.device,
                        dtype=torch.float32,
                    )
                    * self.params["scale_x0"]
                )

        # option 3: use constant (but potentially trainable) x0
        else:
            h_t = torch.tile(self.x0, dims=(batch_size, 1))

        hidden[:, 0] = h_t

        # run through all timesteps
        for i, input_t in enumerate(input.split(1, dim=1)):
            h_t, output = self.rnn(input_t.squeeze(dim=1), h_t, noise[:, i])
            hidden[:, i + 1] = h_t
            outputs[:, i] = output

        return hidden, outputs


def predict(
    rnn, _input, loss_fn=None, _target=None, _mask=None, x0=None, return_loss=False
):
    """
    Do a forward pass with an RNN
    Utility function to call outside of training

    Args:
        rnn: Initialized RNN
        _input: input tensor of size [batch_size, seq_len, n_inp]
        loss_fn (optional): loss function
        _target (optional), tensor of size [batch_size, seq_len, n_out]
        _mask(optional), tensor of size [batch_size, seq_len, n_out]
        _x0(optional), tensor of size [batch_size, n_rec]

    Returns:
        rates: tensor of size [batch_size, seq_len, n_rec]
        predict: tensor of size [batch_size, seq_len, n_out]

    """
    # disable gradients
    rnn.eval()

    # if single trial, add batch dimension
    if _input.dim() < 3:
        _input = _input.unsqueeze(0)

    device = rnn.rnn.w_inp.device
    input = _input.to(device=device)
    if loss_fn is not None:
        if _target.dim() < 3:
            _target = _target.unsqueeze(0)
        if _mask.dim() < 3:
            _mask = _mask.unsqueeze(0)
        target = _target.to(device=device)
        mask = _mask.to(device=device)

    with torch.no_grad():
        rates, predict = rnn(input, x0=x0)

        if loss_fn is not None:
            loss = loss_fn(predict, target, mask)
            print("test loss:", loss.item())
            print("==========================")
            if return_loss:
                return (
                    rates.cpu().detach().numpy(),
                    predict.cpu().detach().numpy(),
                    loss.item(),
                )
    return rates.cpu().detach().numpy(), predict.cpu().detach().numpy()


