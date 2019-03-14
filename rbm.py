import torch
import numpy as np
import glob
from preprocessing import preprocessing
from torch.utils import data
from nist_to_wav import NIST_to_wav
from Dataset import Dataset
from Dataset import RBMSampler

# torch.set_default_dtype(torch.double)

torch.set_default_tensor_type(torch.cuda.DoubleTensor)

class rbm():
    def __init__(self, 
        visible_size=100, hidden_size=120, weights_init=None, hidden_bias_init=None, visible_bias_init=None, 
        learning_rate=1e-4, momentum=0.5, n_epoch=30, batch_size=100, visible_std_init=10,
        n_gibbs_sampling=1, hidden_type="ssu", use_cuda=True):
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
        self.use_cuda = use_cuda

        self.weights = torch.empty(visible_size, hidden_size)
        if weights_init is None:
            self.weights.normal_(mean=0, std=0.01)
        else:
            self.weights = torch.from_numpy(weights_init)

        self.hidden_bias = torch.empty(hidden_size)
        if hidden_bias_init is None:
            # self.hidden_bias.zero_()
            self.hidden_bias.exponential_(lambd=2)
        else:
            self.hidden_bias = torch.from_numpy(hidden_bias_init)
        
        self.visible_bias = torch.empty(visible_size)
        if visible_bias_init is None:
            self.visible_bias.zero_()
        else:
            self.visible_bias = torch.from_numpy(visible_bias_init)

        self.visible_bias_momentum = self.visible_bias.clone()
        self.hidden_bias_momentum = self.hidden_bias.clone()
        self.weights_momentum = self.weights.clone()
        self.visible_std_momentum = self.visible_std.clone()

    def train(self, train_directory, test_directory, validation_size):
        """
        Train RBM with input data

        input_data: numpy array
            numpy array containing the training data
        """
        training_set = Dataset(train_directory)
        testing_set = Dataset(test_directory)
        num_batch_in_input = len(training_set)//self.batch_size

        free_energy_gaps = torch.empty(self.n_epoch * num_batch_in_input) 
        training_generator = data.DataLoader(training_set, sampler=data.RandomSampler(training_set), batch_size=self.batch_size, drop_last=True, pin_memory=True)
        training_representative_generator = data.DataLoader(training_set, sampler=data.RandomSampler(training_set), batch_size=validation_size, drop_last=True, pin_memory=True)
        validation_generator = data.DataLoader(testing_set, sampler=data.RandomSampler(testing_set), batch_size=validation_size, drop_last=True, pin_memory=True)

        training_set_representative = next(iter(training_representative_generator)).to(torch.device('cuda'))

        batch_num = 1
        for i in range(self.n_epoch):
            for batch in training_generator:

                batch = batch.to(torch.device('cuda'))
                print("Batch #" + str(batch_num), end='')

                self.contrastive_divergence(batch) 
                feg = self.free_energy(training_set_representative) - self.free_energy(next(iter(validation_generator)).to(torch.device('cuda')))
                # feg = self.free_energy(training_set_representative) - self.free_energy(validation_set_representative.to(torch.device('cuda')))
                free_energy_gaps[batch_num - 1] = feg
                print("             FEG: " + str(feg.item()))
                batch_num += 1

        torch.save(self.visible_bias, "visible_bias.pt")
        torch.save(self.hidden_bias, "hidden_bias.pt")
        torch.save(self.weights, "weights.pt")
        torch.save(self.visible_std, "std.pt")
        torch.save(free_energy_gaps, "feg.pt")

    def free_energy(self, visible_layers):
        """
        Calculate average free energy of given visible layers

        visible_layer: (validation_set_size, visible_size) sized tensor
            visible layer used to calculate free energy

        Returns
            average free energy of the given input
        """
        return (-1 * torch.sum(torch.mv(visible_layers, self.visible_bias)) - torch.sum(torch.log1p(torch.exp(torch.matmul(visible_layers, self.weights) + self.hidden_bias))))/self.batch_size


    def contrastive_divergence(self, batch):
        """
        Perform one iteration of contrastive divergence on bias and weights using gibbs sampling with mini-batches

        batch: (batch_size, visible_size) sized tensor
            the batch used in gradient descent
        """
        visible_layer = batch.clone()
        hidden_layer = self.sample_hidden(visible_layer)
        if(self.hidden_type == "bernoulli"):
            hidden_layer = torch.bernoulli(hidden_layer)
        positive_association = torch.bmm(visible_layer.unsqueeze(2), hidden_layer.unsqueeze(1))
        # positive_association = torch.matmul(visible_layer.t(), hidden_layer)
        positive_hidden_association = hidden_layer
        positive_std_association = torch.pow(visible_layer - self.visible_bias, 2) 

        for i in range(self.n_gibbs_sampling):
            visible_layer = self.sample_visible(hidden_layer)
            hidden_layer = self.sample_hidden(visible_layer)
            if(self.hidden_type == "bernoulli"):
                hidden_layer = torch.bernoulli(hidden_layer)
        
        negative_association = torch.bmm(visible_layer.unsqueeze(2), hidden_layer.unsqueeze(1))
        # negative_association = torch.matmul(visible_layer.t(), hidden_layer)

        self.weights_momentum = self.weights_momentum * self.momentum + torch.sum(positive_association - negative_association, dim=0)
        # self.weights_momentum = self.weights_momentum * self.momentum + positive_association - negative_association
        self.visible_bias_momentum = self.visible_bias_momentum * self.momentum + torch.sum(batch - visible_layer, dim=0)
        self.hidden_bias_momentum = self.hidden_bias_momentum * self.momentum + torch.sum(positive_hidden_association - hidden_layer, dim=0)

        self.weights += self.weights_momentum * self.learning_rate / self.batch_size
        self.visible_bias += self.visible_bias_momentum * self.learning_rate / self.batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / self.batch_size

        
        negative_std_association = torch.pow(visible_layer - self.visible_bias, 2)  
        self.visible_std_momentum = self.visible_std_momentum * self.momentum + torch.div(torch.sum(negative_std_association - positive_std_association, dim=0), torch.pow(self.visible_std, 3))
        self.visible_std += self.visible_std_momentum * self.learning_rate / self.batch_size
        # self.visible_std += torch.div(torch.sum(negative_std_association - positive_std_association, dim=0), torch.pow(self.visible_std, 3)) * self.learning_rate / self.batch_size
        error = torch.sum((batch - visible_layer) ** 2) / (self.batch_size * self.visible_size)
        print("             Error: " + str(error.item()), end='')
    

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
            x = torch.mm(visible_layer, self.weights)
            hidden_layer = torch.clamp(torch.normal(mean=x, std=torch.sigmoid(x)), min=0.0) + self.hidden_bias
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
    # NIST_to_wav('./TIMIT/TRAIN/*/*/*.WAV', './TRAIN/')
    # input_data = preprocessing('./TRAIN/*.wav')
    # preprocessing('./TEST/*.wav')
    # validation_data = preprocessing('./TEST/*.wav')
    rbm = rbm(n_epoch=30)
    rbm.train("./TRAIN_PT/*.pt", "./TEST_PT/*.pt", 10)
    
