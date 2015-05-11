import GPy
import numpy as np
import matplotlib.cm as cm
import itertools
import pylab as pb
from GPy.plotting.matplot_dep import dim_reduction_plots as dredplots

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
        self.type = []
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
                assert(self.type == [] or self.type == 'bgplvm')
                self.type = 'bgplvm'
            else:
                assert(self.type == [] or self.type == 'mrd')
                self.type = 'mrd'
        else:
            assert(self.type == [] or self.type == 'gp')
            assert(self.__num_views == 1)
            self.Q = inputs.shape[1]
            self.type = 'gp'
            self.inputs = inputs

        if kernel is None:
            kernel = GPy.kern.RBF(self.Q, ARD=True) + GPy.kern.Bias(self.Q) + GPy.kern.White(self.Q)

        if self.type == 'bgplvm':
            self.model = GPy.models.BayesianGPLVM(self.observed[self.observed.keys()[0]], self.Q, kernel=kernel, num_inducing=self.num_inducing)
        elif self.type == 'mrd':
            # Create a list of observation spaces (aka views)
            self.Ylist = []
            self.namesList = []
            for k in self.observed.keys():
                self.Ylist = [self.Ylist, self.observed[k]]
                self.namesList = [self.namesList, k]
            self.Ylist[0]=self.Ylist[0][1]
            self.namesList[0]=self.namesList[0][1]
            self.model = GPy.models.MRD(self.Ylist, input_dim=self.Q, num_inducing=self.num_inducing, kernel=kernel, initx="PCA_concat", initz='permute')
            self.model['.*noise']=[yy.var() / 100. for yy in self.model.Ylist]
        elif self.type == 'gp':
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

    def learn(self, optimizer='bfgs',max_iters=1000, init_iters=300, verbose=True):
        """
        Learn the model (analogous to "forming synapses" after perveiving data).
        """
        if self.type == 'bgplvm' or self.type == 'mrd':
            self.model['.*noise'].fix()
            self.model.optimize(optimizer, messages=verbose, max_iters=init_iters)
            self.model['.*noise'].unfix()
            self.model['.*noise'].constrain_positive()\
        
        self.model.optimize(optimizer, messages=verbose, max_iters=max_iters)

    def visualise(self):
        """
        Show the internal representation of the memory
        """
        if self.type == 'bgplvm' or self.type == 'mrd':
            if self.model.data_labels is not None:
                ret = self.model.plot_latent(labels=self.model.data_labels)
            else:
                ret = self.model.plot_latent()
        elif self.type == 'gp':
            ret = self.model.plot()
        if self.type == 'mrd':
            ret2 = self.model.plot_scales()

        #if self.type == 'mrd':
        #    ret1 = self.model.X.plot("Latent Space 1D")
        #    ret2 = self.model.plot_scales("MRD Scales")
        
        return ret

    def visualise_interactive(self, dimensions=(20,28), transpose=True, order='F', invert=False, scale=False, colorgray=True, view=0, which_indices=(0, 1)):
        """
        Show the internal representation of the memory and allow the user to
        interact with it to map samples/points from the compressed space to the
        original output space
        """
        if self.type == 'bgplvm':
            ax = self.model.plot_latent(which_indices)
            y = self.model.Y[0, :]
            # dirty code here
            if colorgray:
                data_show = GPy.plotting.matplot_dep.visualize.image_show(y[None, :], dimensions=dimensions, transpose=transpose, order=order, invert=invert, scale=scale, cmap = cm.Greys_r)
            else:
                data_show = GPy.plotting.matplot_dep.visualize.image_show(y[None, :], dimensions=dimensions, transpose=transpose, order=order, invert=invert, scale=scale)
            lvm = GPy.plotting.matplot_dep.visualize.lvm(self.model.X.mean[0, :].copy(), self.model, data_show, ax)
            raw_input('Press enter to finish')
        elif self.type == 'mrd':
            """
            NOT TESTED!!!
            """
            ax = self.model.bgplvms[view].plot_latent(which_indices)
            y = self.model.bgplvms[view].Y[0, :]
            # dirty code here
            if colorgray:
                data_show = GPy.plotting.matplot_dep.visualize.image_show(y[None, :], dimensions=dimensions, transpose=transpose, order=order, invert=invert, scale=scale, cmap = cm.Greys_r)
            else:
                data_show = GPy.plotting.matplot_dep.visualize.image_show(y[None, :], dimensions=dimensions, transpose=transpose, order=order, invert=invert, scale=scale)
            lvm = GPy.plotting.matplot_dep.visualize.lvm(self.model.bgplvms[view].X.mean[0, :].copy(), self.model.bgplvms[view], data_show, ax)
            raw_input('Press enter to finish')

    def recall(self, locations):
        """
        Recall stored events. This is closely related to performing pattern pattern_completion
        given "training" data.
        """
        # TODO
        pass

    def pattern_completion(self, test_data, view=0, verbose=False, visualiseInfo=None):
        """
        In the case of supervised learning, pattern completion means that we 
        give new inputs and infer their correspondin outputs. In the case of
        unsupervised leaerning, pattern completion means that we give new
        outputs and we infer their corresponding "latent" inputs, ie the internal
        compressed representation of the new outputs in terms of the already
        formed "synapses".
        """
        if self.type == 'bgplvm':
            #tmp = self.model.infer_newX(test_data)[0]
            #pred_mean = tmp.mean
            #pred_variance = tmp.variance #np.zeros(pred_mean.shape)
            tmp=self.model.infer_newX(test_data,optimize=False)[1]
            tmp.optimize(max_iters=2000, messages=verbose)
            pred_mean = tmp.X.mean
            pred_variance = tmp.X.variance
        elif self.type == 'mrd':
            tmp = self.model.bgplvms[view].infer_newX(test_data, optimize=False)[1]
            tmp.optimize(max_iters=2000, messages=verbose)
            pred_mean = tmp.X.mean
            pred_variance = tmp.X.variance
        elif self.type == 'gp':
            pred_mean, pred_variance = self.model.predict(test_data)
        if (self.type == 'mrd' or self.type == 'bgplvm') and visualiseInfo is not None:
            ax = visualiseInfo['ax']
            inds0, inds1=dredplots.most_significant_input_dimensions(self.model, None)
            pp=ax.plot(pred_mean[:,inds0], pred_mean[:,inds1], 'om', markersize=12, mew=12)
            pb.draw()
        else:
            pp=None
	    
        return pred_mean, pred_variance, pp

    def _get_inducing(self):
        # TODO
        pass

    def _get_latent(self):
        # TODO
        pass
