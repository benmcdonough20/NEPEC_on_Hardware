from scipy.optimize import nnls
from noisemodel import NoiseModel
from termdata import TermData
import numpy as np
from matplotlib import pyplot as plt
from primitives.circuit import Circuit
from primitives.pauli import Pauli
from benchmarkinstance import BenchmarkInstance
import logging

logger = logging.getLogger("experiment")

from benchmarkinstance import SINGLE, PAIR

class LayerNoiseData:
    """This class is responsible for aggregating the data associated with a single layer,
    processing it, and converting it into a noise model to use for PER"""

    def __init__(self, layer : Circuit):
        self._term_data = {} #keys are terms and the values are TermDatas
        self.cliff_layer = layer #LayerNoiseData is assocaited with a clifford layer

    def add_expectation(self, term : Pauli, instance : BenchmarkInstance):
        """Add the result of a benchmark instance to the correct TermData object"""

        pair = self.cliff_layer.conjugate(term) #get the pair of the pauli term
        if not term in self._term_data: #add key to dictionary if it does not exist
            self._term_data[term] = TermData(term, pair)

        #add the expectation value to the data for this term
        logger.info("adding value to "+str(instance.type)) 
        if instance.type == SINGLE:
            self._term_data[pair].add_single(instance.get_expectation(pair))
        elif instance.type == PAIR:
            self._term_data[term].add_pair(instance.depth, instance.get_expectation(term))

        
    def fit_noise_model(self):
        """Fit all of the terms, and then use obtained SPAM coefficients to make degerneracy
        lifting estimates"""

        for term in self._term_data.values(): #perform all pairwise fits
            term.fit()
        
        #get noise model from fits
        self.fit_noisemodel()
   
    def fit_noisemodel(self):
        """Generate a noise model corresponding to the Clifford layer being benchmarked
        for use in PER"""

        def sprod(a,b): #simplecting inner product between two Pauli operators
            return int(not a.commutes(b))

        F1 = [] #First list of terms
        F2 = [] #List of term pairs
        fidelities = [] # list of fidelities from fits

        for datum in self._term_data.values():
            F1.append(datum.pauli)
            fidelities.append(datum.fidelity)
            #If the Pauli is conjugate to another term in the model, a degeneracy is present
            if datum.pauli == datum.pair and datum.pair in self._term_data:
                pair = datum.pair
                F2.append(pair)
            else:
                F2.append(datum.pauli)

        #create commutativity matrices
        M1 = [[sprod(a,b) for a in F1] for b in F1]
        M2 = [[sprod(a,b) for a in F1] for b in F2]

        #check to make sure that there is 
        if np.linalg.matrix_rank(np.add(M1,M2)) != len(F1):
            raise Exception("Matrix is not full rank, something went wrong!")
        
        coeffs,_ = nnls(np.add(M1,M2), -np.log(fidelities)) 
        self.noisemodel = NoiseModel(self.cliff_layer, F1, coeffs)

    def plot_coeffs(self, *terms):
        fig, ax = plt.subplots()
        coeffs = [self.noisemodel.coeffs_dict[term] for term in terms]
        ax.bar([term.to_label() for term in terms], coeffs)

    def graph(self, *terms):
        fig, ax = plt.subplots()
        for term in terms:
            termdata = self._term_data[term]
            termdata.graph(ax)

        return ax

    def plot_infidelitites(self, *terms):
        fig, ax = plt.subplots()
        bars = []
        names = []
        for term in terms:
            termdata = self._term_data[term]
            names.append(term.to_label())
            bars.append(1-termdata.fidelity)

        ax.bar(names, bars)
        return ax