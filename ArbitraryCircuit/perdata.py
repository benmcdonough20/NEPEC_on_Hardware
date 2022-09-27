import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

class PERData:
    def __init__(self, pauli, spam):
        self.pauli = pauli
        self.spam = spam
        self._data = {}
        self._counts = {}

    def add_data(self, inst):
        strength = inst.noise_strength
        expectation = inst.get_adjusted_expectation(self.pauli)
        expectation /= self.spam 
        self._data[strength] = self._data.get(strength, 0) + expectation
        self._counts[strength] = self._counts.get(strength, 0)+1

    def get_expectations(self):
        return [self._data[s]/self._counts[s] for s in self.get_strengths()]

    def get_strengths(self):
        return list(sorted(self._data.keys()))

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.get_strengths(), self.get_expectations())
        return ax

    def fit(self):
        expfit = lambda x,a,b: a*np.exp(-b*x)
        popt, pcov = curve_fit(expfit, self.get_strengths(), self.get_expectations())