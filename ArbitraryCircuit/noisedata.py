from scipy.optimize import nnls
from noisemodel import NoiseModel
from termdata import SingleData, DoubleData
import numpy as np

class NoiseData:
    def __init__(self):
        self._term_data = {}

    def add_expectation(self, term, instance):
        if not term in self._term_data:
            if instance.type == "single":
                self._term_data[term] = SingleData(term, term.conjugate(self.cliff_layer))
            elif instance.type == "double":
                self._term_data[term] = DoubleData(term, term.conjugate(self.cliff_layer))

        self._term_data[term].add_expectation(instance.depth, instance.expectation)

    def noise_model():
        pass

    def fit_noise_model(self, termdata):

        #call fit on each termdata
        for term in self._term_data:
            if not self.is_single(term.term):
                term.fit()
                self._term_data[term.pair].spam = term.spam
        
        for term in self._term_data:
            if self.is_single(term.term):
                self._term_data[term.pair].fit()

        F1 = []
        F2 = []
        fidelities = []
        for datum in termdata:
            F1.append(datum.term)
            fidelities.append(datum.fidelity)
            if datum.ispair():
                pair = datum.pair
                F2.append(pair)
            else:
                F2.append(datum.term)

        sprod = lambda a,b: int(a.anticommutes(b))
        M1 = [[sprod(a,b) for a in self.model_terms] for b in F1]
        M2 = [[sprod(a,b) for a in self.model_terms] for b in F2]

        if np.linalg.matrix_rank(np.add(M1,M2)) != len(self.model_terms):
            raise Exception("Matrix is not full rank, something went wrong!")

        coeffs,_ = nnls(np.add(M1,M2), -np.log(fidelities))
        #noise_model = NoiseModel([Term(p, prob=coeff) for p,coeff in zip(F1,coeffs)])