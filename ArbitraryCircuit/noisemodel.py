import numpy as np
from random import random

NO_NOISE = 0

class NoiseModel:
    def __init__(self, cliff_layer, model_terms, coefficients):
        self.cliff_layer = cliff_layer
        self.coeffs = list(zip(model_terms, coefficients))
        self.pauli_type = cliff_layer.pauli_type
        self.init_scaling(NO_NOISE)

    def init_scaling(self, strength):
        new_coeffs = [(term, strength*coeff) for term,coeff in self.coeffs]
        self.init_tuning(new_coeffs)

    def init_tuning(self, noise_params):
        new_coeffs =  dict(noise_params)
        new_probs = []
        for pauli, lambdak in self.coeffs:
            phik = new_coeffs.get(pauli, 0)
            new_prob = .5*(1-np.exp(-2*abs(phik-lambdak)))
            sgn = 0
            if phik < lambdak:
                sgn = 1
            new_probs.append((pauli, new_prob, sgn))
        
        overhead = 1
        for pauli, lambdak in self.coeffs:
            phik = new_coeffs[pauli]
            if phik < lambdak:
                overhead *= np.exp(2*(lambdak-phik))

        self.probs = new_probs
        self.overhead = overhead

    def sample(self):
        operator = self.pauli_type("I"*self.cliff_layer.num_qubits())
        sgn_tot = 0
        for term, prob, sgn in self.probs:
            if random() < prob:
                operator *= term
                sgn_tot ^= sgn

        return operator, sgn_tot

    def terms(self):
        return list(zip(*self.coeffs))[0]

    def coeffs(self):
        return list(zip(*self.coeffs))[1]
    
    def overhead(self, strength):
        if strength >= 1:
            return np.exp(2*(1-strength)*sum(self.coeffs))
            