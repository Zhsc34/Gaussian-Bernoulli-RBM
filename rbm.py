import torch

dfloat = torch.float
device = torch.device("cuda:0")



class rmb():

    def __init__(self, 
        visible_size=100, hidden_size=120, weight_init, hidden_bias_init, visible_bias_init, 
        learning_rate=0.0001, n_epoch, batch_size=100, n_gibbs_sampling=1, 
        hidden_type="bernoulli", n_ssu, gaussian_noise_variance=10):
        """
        Gaussian-Bernoulli Restricted Boltzmann Machine with SSUs

        Parameters
        ----------
        visible_size: int, default=100
            size of visible layer
        hidden_size: int, default=120
            size of hidden layer
        weight_init: (visible_size, hidden_size) sized tensor
            initializes weight matrix
        hidden_bias_init: (hidden_size) sized tensor
            initializes bias for hidden layer
        visible_bias_init: (visible_size) sized tensor
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
        n_ssu: int
            number of SSUs for each bernoulli node
            only works when hidden_type="SSU"
        gaussian_noise_variance: float, default=10
            variance for gaussian noise to be used in NReLu
            only works with hidden_type="NReLu"
        """
        pass

if __name__ == "__main__":
    main()  