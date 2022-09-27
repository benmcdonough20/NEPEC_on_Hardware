import numpy as np
from random import random

SCALING = "scaling"
LOCAL_DEPOL = "ldepol"
CROSSTALK_SCALING = "ctscale"


class NoiseModel:
    """ Stores noise parameters, computes probabilities and overhead to perform noise scaling and
    tuning, provides a sampling method to automatically sample from the partial noise inverse
    """

    def __init__(self, cliff_layer, model_terms, coefficients):
        """Initalizes a noise model with the associate clifford layer, model terms, and the
        coefficients learned from tomography.

        Args:
            cliff_layer (_type_): _description_
            model_terms (_type_): _description_
            coefficients (_type_): _description_
        """

        self.cliff_layer = cliff_layer
        self.coeffs = list(zip(model_terms, coefficients))
        self.pauli_type = cliff_layer.pauli_type

    def init_scaling(self, strength):
        new_coeffs = [(term, strength*coeff) for term,coeff in self.coeffs]
        self._init_tuning(new_coeffs)

    def _init_tuning(self, noise_params):
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