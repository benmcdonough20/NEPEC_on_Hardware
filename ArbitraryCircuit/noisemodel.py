import numpy as np
from random import random
from scipy.optimize import nnls

class NoiseModel:
    def __init__(self, cliff_layer, model_terms, coefficients):
        self.cliff_layer = cliff_layer
        self.coeffs_dict = dict(zip(model_terms, coefficients))
    
    def scaled_noise(self, noise_parameters):
        pass 

    def sample(self):
        operator = self.cliff_layer.pauli_type("I"*self.n)
        sgn = 0
        noise_model = self.noise_model
        for term, prob in self.noise_model:
            if random() < prob:
                operator = operator.compose(term)
                sgn ^= 1
        return operator, sgn