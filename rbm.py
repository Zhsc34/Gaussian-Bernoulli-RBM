import torch

cuda0 = torch.device("cuda:0")

class rmb():
    def __init__(self, 
        visible_size=100, hidden_size=120, weight_init=None, hidden_bias_init=None, visible_bias_init=None, 
        learning_rate=1e-4, n_epoch=None, batch_size=100, n_gibbs_sampling=1, 
        hidden_type="bernoulli", gaussian_noise_variance=10):
        """
        Gaussian-Bernoulli Restricted Boltzmann Machine with SSUs

        Parameters
        ----------
        visible_size: int, default=100
            size of visible layer
        hidden_size: int, default=120
            size of hidden layer
        weight_init: (visible_size, hidden_size) sized tensor
            defaults to sampling from normal distribution with std=0.01
            initializes weight matrix
        hidden_bias_init: (hidden_size) sized tensor, defaults to all 0
            initializes bias for hidden layer
        visible_bias_init: (visible_size) sized tensor, defaults to all 0
            initializes bias for visible layer
        learning_rate: float, default=0.0001
            learning rate for gradient descent
        n_epoch: int
            number of epochs to train for
        batch_size: int, default=100
            size of mini-batch
        n_gibbs_sampling: int, default = 1
            iterations of alternating gibbs sampling to be done
            before entering contrastive divergence stage
        hidden_type: str, default="bernoulli"
            types of hidden layer nodes
            possible types: "bernoulli", "SSU", "NReLu"
        gaussian_noise_variance: float, default=10
            variance for gaussian noise to be used in NReLu
            only works with hidden_type="NReLu"
        """
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        