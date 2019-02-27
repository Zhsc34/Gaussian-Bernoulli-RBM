import torch
import numpy as np
from preprocessing import preprocessing
from nist_to_wav import NIST_to_wav

class rbm():
    def __init__(self, 
        visible_size=100, hidden_size=120, weights_init=None, hidden_bias_init=None, visible_bias_init=None, 
        learning_rate=1e-4, n_epoch=None, batch_size=100, n_gibbs_sampling=1, 
        hidden_type="bernoulli", gaussian_noise_variance=10, use_cuda=False):
        """
        Gaussian-Bernoulli Restricted Boltzmann Machine with SSUs

        Parameters
        ----------
        visible_size: int, default=100
            size of visible layer
        hidden_size: int, default=120
            size of hidden layer
        weights_init: (visible_size, hidden_size) sized numpy array
            defaults to sampling from normal distribution with std=0.01
            initializes weights matrix
        hidden_bias_init: (hidden_size) sized numpy array, defaults to all 0
            initializes bias for hidden layer
        visible_bias_init: (visible_size) sized numpy array, defaults to all 0
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
            possible types: "bernoulli", "NReLu"
        gaussian_noise_variance: float, default=10
            variance for gaussian noise to be used in NReLu
            only works with hidden_type="NReLu"
        use_cuda: bool, default=False
            if true, tensors will use cuda; otherwise cpu will be used
        """
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.weights_init = weights_init
        self.hidden_bias_init = hidden_bias_init
        self.visible_bias_init = visible_bias_init
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.n_gibbs_sampling = n_gibbs_sampling
        self.hidden_type = hidden_type
        self.gaussian_noise_variance = gaussian_noise_variance
        self.visible_mean = 0
        self.visible_std = 1

        if use_cuda:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)

        self.weights = torch.empty(visible_size, hidden_size)
        if weights_init is None:
            self.weights.normal_(mean=0, std=0.01)
        else:
            self.weights = torch.from_numpy(weights_init)

        self.hidden_bias = torch.empty(hidden_size)
        if hidden_bias_init is None:
            self.hidden_bias.zero_()
        else:
            self.hidden_bias = torch.from_numpy(hidden_bias_init)
        
        self.visible_bias = torch.empty(visible_size)
        if visible_bias_init is None:
            self.visible_bias.zero_()
        else:
            self.visible_bias = torch.from_numpy(visible_bias_init)

        self.hidden_layer = torch.empty(self.hidden_size)
        self.visible_layer = torch.empty(self.visible_size)

    def train(self, input_data):
        """
        Train RBM with input data
        """
        pass

    def energy(self):
        """
        Calculate energy of the current configuration
        """
        return -1 * torch.dot(torch.mv(self.weights, self.hidden_layer), self.visible_layer) + 1/(self.visible_std * self.visible_std) * torch.sum(torch.pow(2, self.visible_layer - self.visible_bias)) - torch.dot(self.hidden_layer, hidden_bias)

    def gradient_descent(self):
        """
        Perform one iteration of gradient descent on bias and weights using gibbs sampling
        """
        positive_gradient_weight = torch.ger(self.visible_layer, self.hidden_layer)
        original_visible_layer = self.visible_layer.clone()
        original_hidden_layer = self.hidden_layer.clone()
        self.gibbs_sampling()

        negative_gradient_weight = torch.ger(self.visible_layer, self.hidden_layer)
        gradient_visible_bias = original_visible_layer - self.visible_layer 
        gradient_hidden_bias = original_hidden_layer - self.hidden_layer

        self.weights += self.learning_rate * (positive_gradient_weight - negative_gradient_weight)
        self.hidden_bias += self.learning_rate * gradient_hidden_bias
        self.visible_bias += self.learning_rate * gradient_visible_bias

    def gibbs_sampling(self):
        """
        Perform one iteration of alternating Gibbs sampling on hidden and visible layers
        """
        # sample hidden layer
        hidden_probabilities = torch.addmv(self.hidden_bias, self.weights.transpose(0, 1), self.visible_layer).sigmoid_()
        torch.bernoulli(hidden_probabilities, out=self.hidden_layer)
        
        # TODO sample visible layer from normal distribution according to visible_probabilities

if __name__ == "__main__":
    # NIST_to_wav('./TIMIT/TEST/*/*/*.wav', './TEST/')
    # input_data = preprocessing('./TEST/*.wav')
    rbm = rbm()