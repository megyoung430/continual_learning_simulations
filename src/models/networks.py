"""This file defines the class for the different networks trained on the auditory discrimination task, designed for the continual learning experiments."""

import torch
from torch import nn

class AuditoryDiscriminationNetwork(nn.Module):
    """This class specifies a network to be trained on our auditory discrimination task."""
    def __init__(self, rpe=True, rpe_type="full", tonotopy=False, num_notes=7, num_actions=2):
        """This function initializes one of the networks.

        This class is set up to instantiate three different types of networks for comparison:
        1) An 
        2) 
        3) 

        Args:
            rpe (bool, optional): If true, then the network learns via reinforcement learning; if false, the network learns via supervised learning. Defaults to True.
            rpe_type (str, optional): Specifies the type of the RPE signal, either "full" or "partial". Defaults to "full".
            tonotopy (bool, optional): If true, then the first layer weights are diagonal, motivated by the existence of tonotopy in auditory cortex. Defaults to False.
            num_notes (int, optional): Size of the input layer (excluding a constant bias term), here reflecting the spectrogram of the input. Defaults to 7.
            num_actions (int, optional): Size of the output layer, here reflecting the Q-values for left and right choices. Defaults to 2.
        """
        super().__init__()
        self.rpe = rpe
        self.tonotopy = tonotopy
        if tonotopy:
            self.l1_weights = nn.Parameter(torch.ones(num_notes + 1))
        else:
            self.l1_weights = nn.Linear(num_notes + 1, num_notes + 1, bias=False)
        if rpe:
            self.rpe_type = rpe_type
            if rpe_type == "full":
                self.l2_weights = nn.Linear(num_notes + 1, num_actions, bias=False)
            elif rpe_type == "partial":
                self.l2_weights_const = nn.Parameter(torch.ones(num_actions, 1))
                self.l2_weights_stim = nn.Parameter(torch.ones(num_actions, num_notes))
        else:
            self.l2_weights = nn.Linear(num_notes + 1, num_actions, bias=False)

    def forward(self, input):
        """This function implements a forward pass through the network

        Args:
            input (num_notes + 1 tensor): Input to the network, representing the spectrogram of the auditory stimulus + a constant bias term.

        Returns:
            l2_output (num_actions tensor): Output of the network, reflecting the probabilities of choosing left and right, respectively.
        """
        if self.tonotopy:
            l1 = torch.diag_embed(self.l1_weights)
            if self.rpe and self.rpe_type == "partial":
                l1_output = torch.matmul(input, l1)
                l2_weights = torch.cat((self.l2_weights_const, self.l2_weights_stim), dim=1)
                l2_output = torch.matmul(l1_output, l2_weights.T)
            else:
                l1_output = torch.matmul(input, l1)
                l2_output = self.l2_weights(l1_output)
        else:
            l1_output = self.l1_weights(input)
            if self.rpe and self.rpe_type == "partial":
                l2_weights = torch.cat((self.l2_weights_const, self.l2_weights_stim), dim=1)
                l2_output = torch.matmul(l1_output, l2_weights.T)
            else:
                l2_output = self.l2_weights(l1_output)
        return l2_output
