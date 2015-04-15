import GPy
import numpy as np
# try:
#     from mpi4py import MPI
# except:
#     print "mpi not found"

# from ABM import ABM
# d={'first':1,'second':2}
# a=ABM.pSAM(d)

class LFM(object):
    def __init__(self):
        self.__type = []
        self.model = []

    def store(self,observed, inputs=None, Q=None, kernel=None, num_inducing=None):
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
        self.model.data_labels = labels

    def learn(self, optimizer='bfgs',max_iters=1000, verbose=True):
        self.model.optimize(optimizer, messages=verbose, max_iters=max_iters)

    def visualise(self):
        if self.__type == 'bgplvm' or self.__type == 'mrd':
            if self.model.data_labels is not None:
                ret = self.model.plot_latent(labels=self.model.data_labels)
            else:
                ret = self.model.plot_latent()
        elif self.__type == 'gp':
            ret = self.model.plot()
        return ret

    def recall(self, locations):
        # TODO
        pass

    def pattern_completion(self, test_data):
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
