import torch
import numpy as np
from preprocessing import preprocessing
from nist_to_wav import NIST_to_wav



torch.set_default_dtype(torch.double)

class rbm():
    def __init__(self, 
        visible_size=100, hidden_size=120, weights_init=None, hidden_bias_init=None, visible_bias_init=None, 
        learning_rate=1e-4, n_epoch=30, batch_size=100, n_gibbs_sampling=1, 
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
        n_epoch: int, default=30
            number of epochs to train for
        batch_size: int, default=100
            size of mini-batch
        n_gibbs_sampling: int, default = 1
            iterations of alternating gibbs sampling to be done
            before updating weight and bias
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
        self.visible_std = 1

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

        
        if use_cuda:
            self.hidden_layer = self.hidden_layer.to(torch.device("cuda"))
            self.hidden_bias = self.hidden_bias.to(torch.device("cuda"))
            self.visible_layer = self.visible_layer.to(torch.device("cuda"))
            self.visible_bias = self.visible_bias.to(torch.device("cuda"))
            self.weights = self.weights.to(torch.device("cuda"))

    def train(self, input_data):
        """
        Train RBM with input data

        input_data: numpy array
            numpy array containing the training data
        """
        input_size = input_data.shape[0]
        
        for i in range(0, int(input_size * self.n_epoch / self.batch_size)):
            batch = input_data[np.random.choice(input_size, size=self.batch_size)]
            print("Batch #" + str(i + 1))
            self.gradient_descent(torch.from_numpy(batch))

    def energy(self):
        """
        Calculate energy of the current configuration
        """
        return -1 * torch.dot(torch.mv(self.weights, self.hidden_layer), self.visible_layer) + 1/(self.visible_std * self.visible_std) * torch.sum(torch.pow(2, self.visible_layer - self.visible_bias)).item() - torch.dot(self.hidden_layer, hidden_bias)

    def gradient_descent(self, batch):
        """
        Perform one iteration of gradient descent on bias and weights using gibbs sampling with mini-batches

        batch: (batch_size, visible_size) sized tensor
            the batch used in gradient descent
        """
        # TODO clarify std learning?
        self.visible_layer = batch[0]
        self.visible_std = self.visible_layer.std().item()
        self.gibbs_sampling()
        positive_gradient_weight = torch.ger(self.visible_layer, self.hidden_layer.double())
        original_visible_layer = self.visible_layer.clone()
        original_hidden_layer = self.hidden_layer.clone()
        original_visible_bias = self.visible_bias.clone()

        for i in range(1, self.batch_size):
            self.visible_layer = batch[i]
            self.visible_std = self.visible_layer.std().item()
            for k in range(0, self.n_gibbs_sampling):
                self.gibbs_sampling()

        negative_gradient_weight = torch.ger(self.visible_layer, self.hidden_layer.double())
        gradient_visible_bias = original_visible_layer - self.visible_layer 
        gradient_hidden_bias = original_hidden_layer - self.hidden_layer

        self.weights += self.learning_rate * (positive_gradient_weight - negative_gradient_weight)
        self.hidden_bias += self.learning_rate * gradient_hidden_bias
        self.visible_bias += self.learning_rate * gradient_visible_bias
        # self.visible_std = self.learning_rate * 1e-6 * -2 * (torch.sum(torch.pow(2, self.visible_layer - self.visible_bias)).item() - torch.sum(torch.pow(2, original_visible_layer - original_visible_bias)).item()) * self.visible_std ** -3
        
    def gibbs_sampling(self):
        """
        Perform one iteration of alternating Gibbs sampling on hidden and visible layers
        """
        # sample hidden layer
        # TODO NReLu
        hidden_probabilities = torch.sigmoid(torch.addmv(self.hidden_bias, self.weights.transpose(0, 1), self.visible_layer))
        torch.bernoulli(hidden_probabilities, out=self.hidden_layer)
        
        # sample visible layer
        visible_normal_mean = torch.addmv(self.visible_bias, self.weights, self.hidden_layer, alpha=self.visible_std * self.visible_std)
        torch.normal(mean=visible_normal_mean.double(), std=self.visible_std * self.visible_std, out=self.visible_layer)

if __name__ == "__main__":
    # NIST_to_wav('./TIMIT/TEST/*/*/*.wav', './TEST/')
    input_data = preprocessing('./TEST/*.wav')
    rbm = rbm()
    rbm.train(input_data)
    