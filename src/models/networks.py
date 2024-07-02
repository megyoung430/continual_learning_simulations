"""This file defines the class for the different networks trained on the auditory discrimination task, designed for the continual learning experiments."""

import torch
from torch import nn

class DeepRLAuditoryDiscriminationNetwork(nn.Module):
    """This class specifies a deep linear network to be trained on our auditory discrimination task using reinforcement learning."""
    def __init__(self, rpe_type="full", tonotopy=False, num_notes=7, num_actions=2):
        """This function initializes one of the networks.

        This class is set up to instantiate several different types of networks for comparison:
        1) A tonotopic network where the weights from the input to the hidden layer are diagonal, motivated by the existence of tonotopy in
        auditory cortex.
        2) A fully-connected network, where both layers are all-to-all connected.
        Both networks can be trained with the full, global RPE signal or with partial RPE signals that determine independently the weight updates
        for the stimulus terms and the constant term, inspired by Liebana-Garcia et al, 2023 and motivated in our case by Znamenskiy and Zador, 2012.

        Args:
            rpe_type (str, optional): Specifies the type of the RPE signal, either "full" or "partial". Defaults to "full".
            tonotopy (bool, optional): If true, then the first layer weights are diagonal, motivated by the existence of tonotopy in auditory cortex. Defaults to False.
            num_notes (int, optional): Size of the input layer (excluding a constant bias term), here reflecting the spectrogram of the input. Defaults to 7.
            num_actions (int, optional): Size of the output layer, here reflecting the Q-values for left and right choices. Defaults to 2.
        """
        super().__init__()

        # If the network is "tonotopic", then the first-layer weights are diagonal. Otherwise,
        # the first layer is fully connected. 
        self.tonotopy = tonotopy
        if tonotopy:
            self.l1_weights = nn.Parameter(torch.ones(num_notes + 1))
            nn.init.uniform_(self.l1_weights, a=0.0, b=0.01)
        else:
            self.l1_weights = nn.Linear(num_notes + 1, num_notes + 1, bias=False)
            nn.init.uniform_(self.l1_weights.weight, a=0.0, b=0.01)
        
        # If the network uses full RPEs, then all of the weights are updated simultaneously from the global RPE.
        self.rpe_type = rpe_type
        if rpe_type == "full":
            self.l2_weights = nn.Linear(num_notes + 1, num_actions, bias=False)
            nn.init.uniform_(self.l2_weights.weight, a=0.0, b=0.01)
        # However, if the network uses partial RPEs, then the weights for the constant term are updated indepedently
        # from those of the stimulus using different partial RPEs.
        elif rpe_type == "partial":
            self.l2_weights_const = nn.Parameter(torch.ones(num_actions, 1))
            self.l2_weights_stim = nn.Parameter(torch.ones(num_actions, num_notes))
            nn.init.uniform_(self.l2_weights_const, a=0.0, b=0.01)
            nn.init.uniform_(self.l2_weights_stim, a=0.0, b=0.01)

    def forward(self, input):
        """This function implements a forward pass through the network.

        Args:
            input (num_notes + 1 tensor): Input to the network, representing the spectrogram of the auditory stimulus + a constant bias term.

        Returns:
            l2_output (num_actions tensor): Output of the network, reflecting the probabilities of choosing left and right, respectively.
        """
        if self.tonotopy:
            l1 = torch.diag_embed(self.l1_weights)
            l1_output = torch.matmul(input, l1)
        else:
            l1_output = self.l1_weights(input)
        
        if self.rpe_type == "partial":
                l2_weights = torch.cat((self.l2_weights_const, self.l2_weights_stim), dim=1)
                l2_output = torch.matmul(l1_output, l2_weights.T)
        else:
            l2_output = self.l2_weights(l1_output)

        return l2_output

class DeepSupervisedAuditoryDiscriminationNetwork(nn.Module):
    """This class specifies a deep linear network trained on our auditory discrimination task using supervised learning."""
    def __init__(self, tonotopy=False, num_notes=7, num_actions=2):
        """This function initializes one of the networks.

        This class is set up to instantiate several different types of networks for comparison:
        1) A tonotopic network where the weights from the input to the hidden layer are diagonal, motivated by the existence of tonotopy in
        auditory cortex.
        2) A fully-connected network, where both layers are all-to-all connected.
        Both networks are trained via supervised learning.

        Args:
            tonotopy (bool, optional): If true, then the first layer weights are diagonal, motivated by the existence of tonotopy in auditory cortex. Defaults to False.
            num_notes (int, optional): Size of the input layer (excluding a constant bias term), here reflecting the spectrogram of the input. Defaults to 7.
            num_actions (int, optional): Size of the output layer, here reflecting the Q-values for left and right choices. Defaults to 2.
        """
        super().__init__()

        # If the network is "tonotopic", then the first-layer weights are diagonal. Otherwise,
        # the first layer is fully connected.
        self.tonotopy = tonotopy
        if tonotopy:
            self.l1_weights = nn.Parameter(torch.ones(num_notes + 1))
            nn.init.uniform_(self.l1_weights, a=0.0, b=0.01)
        else:
            self.l1_weights = nn.Linear(num_notes + 1, num_notes + 1, bias=False)
            nn.init.uniform_(self.l1_weights.weight, a=0.0, b=0.01)
        
        self.l2_weights = nn.Linear(num_notes + 1, num_actions, bias=False)
        nn.init.uniform_(self.l2_weights.weight, a=0.0, b=0.01)

    def forward(self, input):
        """This function implements a forward pass through the network.

        Args:
            input (num_notes + 1 tensor): Input to the network, representing the spectrogram of the auditory stimulus + a constant bias term.

        Returns:
            l2_output (num_actions tensor): Output of the network, reflecting the probabilities of choosing left and right, respectively.
        """
        if self.tonotopy:
            l1 = torch.diag_embed(self.l1_weights)
            l1_output = torch.matmul(input, l1)
        else:
            l1_output = self.l1_weights(input)
        l2_output = self.l2_weights(l1_output)
        return l2_output

class ShallowRLAuditoryDiscriminationNetwork(nn.Module):
    """This class specifies a shallow linear network to be trained on our auditory discrimination task via reinforcement learning."""
    def __init__(self, rpe_type="full", num_notes=7, num_actions=2):
        """This function initializes one of the networks.

        This class is set up to instantiate a network with all-to-all connectivity between the input layer and the output layer.
        The network can be trained with the full, global RPE signal or with partial RPE signals that determine independently the weight updates
        for the stimulus terms and the constant term, inspired by Liebana-Garcia et al, 2023 and motivated in our case by Znamenskiy and Zador, 2012.

        Args:
            rpe_type (str, optional): Specifies the type of the RPE signal, either "full" or "partial". Defaults to "full".
            num_notes (int, optional): Size of the input layer (excluding a constant bias term), here reflecting the spectrogram of the input. Defaults to 7.
            num_actions (int, optional): Size of the output layer, here reflecting the Q-values for left and right choices. Defaults to 2.
        """
        super().__init__()
        self.rpe_type = rpe_type
        if rpe_type=="partial":
            self.l1_weights_const = nn.Parameter(torch.ones(num_actions, 1))
            self.l1_weights_stim = nn.Parameter(torch.ones(num_actions, num_notes))
            nn.init.uniform_(self.l1_weights_const, a=0.0, b=0.01)
            nn.init.uniform_(self.l1_weights_stim, a=0.0, b=0.01)
        else:
            self.l1_weights = nn.Linear(num_notes + 1, num_actions, bias=False)
            nn.init.uniform_(self.l1_weights.weight, a=0.0, b=0.01)

    def forward(self, input):
        """This function implements a forward pass through the network.

        Args:
            input (num_notes + 1 tensor): Input to the network, representing the spectrogram of the auditory stimulus + a constant bias term.

        Returns:
            l1_output (num_actions tensor): Output of the network, reflecting the probabilities of choosing left and right, respectively.
        """
        if self.rpe_type == "partial":
            l1_weights = torch.cat((self.l1_weights_const, self.l1_weights_stim), dim=1)
            l1_output = torch.matmul(input, l1_weights.T)
        else:
            l1_output = self.l1_weights(input)
        return l1_output

class ShallowSupervisedAuditoryDiscriminationNetwork(nn.Module):
    """This class specifies a shallow linear network to be trained on our auditory discrimination task via supervised learning."""
    def __init__(self, num_notes=7, num_actions=2):
        """This function initializes one of the networks.

        This class is set up to instantiate a network with all-to-all connectivity between the input layer and the output layer and trained
        via supervised learning.

        Args:
            num_notes (int, optional): Size of the input layer (excluding a constant bias term), here reflecting the spectrogram of the input. Defaults to 7.
            num_actions (int, optional): Size of the output layer, here reflecting the Q-values for left and right choices. Defaults to 2.
        """
        super().__init__()
        self.l1_weights = nn.Linear(num_notes + 1, num_actions, bias=False)
        nn.init.uniform_(self.l1_weights.weight, a=0.0, b=0.01)

    def forward(self, input):
        """This function implements a forward pass through the network.

        Args:
            input (num_notes + 1 tensor): Input to the network, representing the spectrogram of the auditory stimulus + a constant bias term.

        Returns:
            l1_output (num_actions tensor): Output of the network, reflecting the probabilities of choosing left and right, respectively.
        """
        l1_output = self.l1_weights(input)
        return l1_output