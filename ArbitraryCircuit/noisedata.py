from scipy.optimize import nnls
from noisemodel import NoiseModel
from termdata import TermData
import numpy as np

class NoiseData:
    def __init__(self, layer):
        self._term_data = {}
        self.cliff_layer = layer

    def add_expectation(self, term, instance):
        pair = term.conjugate(self.cliff_layer)
        if not term in self._term_data:
            self._term_data[term] = TermData(term, pair)
        
        self._term_data[term].add_expectation(instance.depth, instance.get_expectation(pair), instance.type)
        
    def noise_model():
        pass

    def fit_noise_model(self):

        def sprod(a,b):
            return int(not a.commutes(b))

        for term in self._term_data.values():
            term.fit()

        for term in self._term_data.values():
            if term.pair in self._term_data:
                meas_err = self._term_data[term.pair].spam
                term.fit_single(meas_err)

        model_terms = list(self._term_data.keys())

        F1 = []
        F2 = []
        fidelities = []
        for datum in self._term_data.values():
            F1.append(datum.term)
            fidelities.append(datum.fidelity)
            if datum.ispair():
                pair = datum.pair
                F2.append(pair)
            else:
                F2.append(datum.term)

        M1 = [[sprod(a,b) for a in model_terms] for b in F1]
        M2 = [[sprod(a,b) for a in model_terms] for b in F2]

        if np.linalg.matrix_rank(np.add(M1,M2)) != len(model_terms):
            raise Exception("Matrix is not full rank, something went wrong!")

        coeffs,_ = nnls(np.add(M1,M2), -np.log(fidelities))
        self.coeffs = coeffs
        #noise_model = NoiseModel([Term(p, prob=coeff) for p,coeff in zip(F1,coeffs)])