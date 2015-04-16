import GPy
import numpy as np
# try:
#     from mpi4py import MPI
# except:
#     print "mpi not found"

# from ABM import ABM
# d={'first':1,'second':2}
# a=ABM.pSAM(d)


"""
SAM based on Latent Feature Models
"""
class LFM(object):
    def __init__(self):
        self.__type = []
        self.model = []

    def store(self,observed, inputs=None, Q=None, kernel=None, num_inducing=None):
        """
        Store events.
        ARG: obserbved: A N x D matrix, where N is the number of points and D the number
        of features needed to describe each point.
        ARG: inputs: A N x Q matrix, where Q is the number of features per input. 
        Leave None for unsupervised learning.
        ARG: Q: Leave None for supervised learning (Q will then be the dimensionality of
        inputs). Otherwise, specify with Q the dimensionality (number of features) for the
        compressed space that acts as "latent" inputs.
        ARG: kernel: for the GP. can be left as None for default.
        ARG: num_inducing: says how many inducing points to use. Inducing points are
        a fixed number of variables through which all memory is filtered, to achieve
        full compression. E.g. it can correspond to the number of neurons.
        Of course, this is not absolutely fixed, but it also doesn't grow necessarily
        proportionally to the data, since synapses can make more complicated combinations
        of the existing neurons. The GP is here playing the role of "synapses", by learning
        non-linear and rich combinations of the inducing points.
        """
        assert(isinstance(observed,dict))
        self.observed = observed
        self.__num_views = len(self.observed.keys())
        self.Q = Q
        #self.D = observed.shape[1]
        self.N = observed[observed.keys()[0]].shape[0]
        self.num_inducing = num_inducing
        if num_inducing is None:
            self.num_inducing = self.N
        if inputs is None:
            if self.Q is None:
                self.Q = 2#self.D
            if self.__num_views == 1:
                assert(self.__type == [] or self.__type == 'bgplvm')
                self.__type = 'bgplvm'
            else:
                assert(self.__type == [] or self.__type == 'mrd')
                self.__type = 'mrd'
        else:
            assert(self.__type == [] or self.__type == 'gp')
            assert(self.__num_views == 1)
            self.Q = inputs.shape[1]
            self.__type = 'gp'
            self.inputs = inputs

        if kernel is None:
            kernel = GPy.kern.RBF(self.Q, ARD=True) + GPy.kern.Bias(self.Q) + GPy.kern.White(self.Q)

        if self.__type == 'bgplvm':
            self.model = GPy.models.BayesianGPLVM(self.observed[self.observed.keys()[0]], self.Q, kernel=kernel, num_inducing=self.num_inducing)
        elif self.__type == 'mrd':
            pass
        elif self.__type == 'gp':
            self.model = GPy.models.SparseGPRegression(self.inputs, self.observed[self.observed.keys()[0]], kernel=kernel, num_inducing=self.num_inducing)
        
        self.model.data_labels = None
    
    def add_labels(self, labels):
        """
        If observables are associated with labels, they can be added here.
        labels has to be a matrix of size N x K, where K is the total number
        of different labels. If e.g. the i-th row of L is [1 0 0] (or [1 -1 -1])
        then this means that there are K=3 different classes and the i-th row
        of the observables belongs to the first class.
        """
        self.model.data_labels = labels

    def learn(self, optimizer='bfgs',max_iters=1000, verbose=True):
        """
        Learn the model (analogous to "forming synapses" after perveiving data).
        """
        self.model.optimize(optimizer, messages=verbose, max_iters=max_iters)

    def visualise(self):
        """
        Show the internal representation of the memory
        """
        if self.__type == 'bgplvm' or self.__type == 'mrd':
            if self.model.data_labels is not None:
                ret = self.model.plot_latent(labels=self.model.data_labels)
            else:
                ret = self.model.plot_latent()
        elif self.__type == 'gp':
            ret = self.model.plot()
        return ret

    def recall(self, locations):
        """
        Recall stored events. This is closely related to performing pattern pattern_completion
        given "training" data.
        """
        # TODO
        pass

    def pattern_completion(self, test_data):
        """
        In the case of supervised learning, pattern completion means that we 
        give new inputs and infer their correspondin outputs. In the case of
        unsupervised leaerning, pattern completion means that we give new
        outputs and we infer their corresponding "latent" inputs, ie the internal
        compressed representation of the new outputs in terms of the already
        formed "synapses".
        """
        if self.__type == 'bgplvm':
            #tmp = self.model.infer_newX(test_data)[0]
            #pred_mean = tmp.mean
            #pred_variance = tmp.variance #np.zeros(pred_mean.shape)
            tmp = self.model.infer_newX(test_data, optimize=False)[1]
            tmp.optimize(max_iters=2000, messages=True)
            pred_mean = tmp.X.mean
            pred_variance = tmp.X.variance
        elif self.__type == 'mrd':
            pred_mean =[] # TODO
            pred_variance = [] # TODO
        elif self.__type == 'gp':
            pred_mean, pred_variance = self.model.predict(test_data)
        return pred_mean, pred_variance

    def _get_inducing(self):
        # TODO
        pass

    def _get_latent(self):
        # TODO
        pass
