import torch
import numpy as np
from preprocessing import preprocessing
from nist_to_wav import NIST_to_wav

# torch.set_default_dtype(torch.double)

torch.set_default_tensor_type(torch.cuda.DoubleTensor)

class rbm():
    def __init__(self, 
        visible_size=100, hidden_size=120, weights_init=None, hidden_bias_init=None, visible_bias_init=None, 
        learning_rate=1e-4, momentum=0.5, n_epoch=30, batch_size=100, visible_std_init=10,
        n_gibbs_sampling=1, hidden_type="ssu", use_cuda=False):
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
        momentum: float, default=0.5
            momentum for gradient descent
        n_epoch: int, default=30
            number of epochs to train for
        batch_size: int, default=100
            size of mini-batch
        visible_std_init: float, default=10
            standard deviation for visible layer
        n_gibbs_sampling: int, default = 1
            iterations of alternating gibbs sampling to be done
            before updating weight and bias
        hidden_type: str, default="ssu"
            types of hidden layer nodes
            possible types: "bernoulli", "ssu"
        use_cuda: bool, default=False
            if true, tensors will use cuda; otherwise cpu will be used
        """
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.weights_init = weights_init
        self.hidden_bias_init = hidden_bias_init
        self.visible_bias_init = visible_bias_init
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.n_gibbs_sampling = n_gibbs_sampling
        self.hidden_type = hidden_type
        self.visible_std = torch.full((1, visible_size), visible_std_init)

        self.visible_bias_momentum = torch.zeros(visible_size)
        self.hidden_bias_momentum = torch.zeros(hidden_size)
        self.weights_momentum = torch.zeros(visible_size, hidden_size)
        self.visible_std_momentum = torch.zeros(visible_size)

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
    
        # if use_cuda:
        #     self.hidden_bias = self.hidden_bias.to(torch.device("cuda"))
        #     self.visible_bias = self.visible_bias.to(torch.device("cuda"))
        #     self.weights = self.weights.to(torch.device("cuda"))
        #     self.visible_std = self.visible_std.to(torch.device("cuda"))

        #     self.weights_momentum = self.weights_momentum.to(torch.device("cuda"))
        #     self.visible_bias_momentum = self.visible_bias_momentum.to(torch.device("cuda"))
        #     self.hidden_bias_momentum = self.hidden_bias_momentum.to(torch.device("cuda"))
        #     self.visible_std_momentum = self.visible_std_momentum.to(torch.device("cuda"))


    def train(self, input_data):
        """
        Train RBM with input data

        input_data: numpy array
            numpy array containing the training data
        """
        input_size = input_data.shape[0]
        num_batch_in_input = int(input_size/self.batch_size)
        
        batch_num = 1
        for i in range(self.n_epoch):
            for j in range(num_batch_in_input):
                batch = input_data[np.random.choice(input_size, size=self.batch_size)]
                print("Batch #" + str(batch_num), end='')
                self.contrastive_divergence(torch.from_numpy(batch).cuda())
                batch_num += 1

        torch.save(self.visible_bias, "visible_bias.pt")
        torch.save(self.hidden_bias, "hidden_bias.pt")
        torch.save(self.weights, "weights.pt")
        torch.save(self.visible_std, "std.pt")

    def contrastive_divergence(self, batch):
        """
        Perform one iteration of contrastive divergence on bias and weights using gibbs sampling with mini-batches

        batch: (batch_size, visible_size) sized tensor
            the batch used in gradient descent
        """
        visible_layer = batch
        hidden_layer = self.sample_hidden(visible_layer)
        if(self.hidden_type == "bernoulli"):
            hidden_layer = torch.bernoulli(hidden_layer)
        positive_association = torch.bmm(visible_layer.unsqueeze(2), hidden_layer.unsqueeze(1))
        positive_hidden_association = hidden_layer
        positive_std_association = torch.pow(visible_layer - self.visible_bias, 2) 

        for i in range(self.n_gibbs_sampling):
            visible_layer = self.sample_visible(hidden_layer)
            hidden_layer = self.sample_hidden(visible_layer)
            if(self.hidden_type == "bernoulli"):
                hidden_layer = torch.bernoulli(hidden_layer)
        
        negative_association = torch.bmm(visible_layer.unsqueeze(2), hidden_layer.unsqueeze(1))

        self.weights_momentum = self.weights_momentum * self.momentum + torch.sum(positive_association - negative_association, dim=0)
        self.visible_bias_momentum = self.visible_bias_momentum * self.momentum + torch.sum(batch - visible_layer, dim=0)
        self.hidden_bias_momentum = self.hidden_bias_momentum * self.momentum + torch.sum(positive_hidden_association - hidden_layer, dim=0)

        self.weights += self.weights_momentum * self.learning_rate / self.batch_size
        self.visible_bias += self.visible_bias_momentum * self.learning_rate / self.batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / self.batch_size

        
        negative_std_association = torch.pow(visible_layer - self.visible_bias, 2)  
        self.visible_std_momentum = self.visible_std_momentum * self.momentum + torch.div(torch.sum(negative_std_association - positive_std_association, dim=0), torch.pow(self.visible_std, 3))
        self.visible_std += self.visible_std_momentum * self.learning_rate / self.batch_size
        # self.visible_std += torch.div(torch.sum(negative_std_association - positive_std_association, dim=0), torch.pow(self.visible_std, 3)) * self.learning_rate / self.batch_size
        error = torch.sum((batch - visible_layer) ** 2)
        print("             Error: " + str(error.item()))
    

    def sample_hidden(self, visible_layer):
        """
        Sample hidden layer based on visible layer and hidden unit type

        visible_layer: (visible_size) sized tensor
            the visible layer used to sample the hidden layer
        
        Returns
            tensor containing sampled hidden layer if hidden unit type is ssu
            tensor containing hidden probabilities if hidden unit type is bernoulli
        """
        if self.hidden_type == "bernoulli":
            hidden_probabilities = torch.sigmoid(torch.mm(visible_layer, self.weights) + self.hidden_bias)
            return hidden_probabilities
        else:
            x = torch.mm(visible_layer, self.weights) + self.hidden_bias
            hidden_layer = torch.clamp(torch.normal(mean=x, std=torch.sigmoid(x)), min=0.0)
            return hidden_layer

    def sample_visible(self, hidden_layer):  
        """
        Sample visible layer based on hidden layer

        hidden_layer: (hidden_size) sized tensor
            the hidden layer used to sample the visible layer
        
        Returns
            tensor containing sampled visible layer
        """ 
        visible_normal_mean = torch.mul(torch.mm(hidden_layer, self.weights.transpose(0, 1)), torch.pow(self.visible_std, 2)) + self.visible_bias
        visible_layer = torch.normal(mean=visible_normal_mean, std=self.visible_std.expand(self.batch_size, -1))
        return visible_layer

if __name__ == "__main__":
    # NIST_to_wav('./TIMIT/TEST/*/*/*.wav', './TEST/')
    input_data = preprocessing('./TEST/*.wav')
    rbm = rbm()
    rbm.train(input_data)
    