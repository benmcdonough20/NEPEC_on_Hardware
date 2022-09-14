import numpy as np
from random import random
from scipy.optimize import nnls
from primitives.term import Term

class NoiseModel:
    def __init__(self, cliff_layer, model_terms = None):
        self.cliff_layer = cliff_layer
        self.model_terms = model_terms
        #self.overhead = np.exp(2*sum(noise_coefficients))
    
    def scaled_noise(self, noise_parameter):
        pass 

    def sample(self):
        operator = Term("I"*self.n)
        sgn = 0
        noise_model = self.noise_model
        for term, prob in self.noise_model:
            if random() < prob:
                operator = operator.compose(term)
                sgn ^= 1
        return operator, sgn
    
    def issingle(self, p):
        return p == p.conjugate(self.cliff_layer)