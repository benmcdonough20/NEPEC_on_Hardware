from scipy.optimize import curve_fit, OptimizeWarning
import warnings
import numpy as np

warnings.simplefilter('error', OptimizeWarning)

class TermData:
    def __init__(self, term, pair):
        self.term = term
        self.pair = pair
        self._expectations = {}
        self._count = {}
        self._spam = 1
        self._fidelity = 1
    
    def add_expectation(self, depth, expectation):
        self._expectations[depth] = expectation
        self._count[depth] += self._counts.get(depth, 0)+1

    def _depths(self):
        return sorted(self._expectations.keys())

    def _list_expectations(self):
        return [self._expectations[d]/self._count[d] for d in self._depths()]

    def ispair(self):
        return self.term == self.pair

class SingleData(TermData):

    def set_SPAM(self, spam):
        self.spam = spam

    def fit(self):
        expectation = abs(sum(self._expectations[1])/self._count[1])
        self.fidelity = min([1,expectation/self.spam])

        return self.fidelity

class DoubleData(TermData):

    def fit(self):
        expfit = lambda x,a,b: a*np.exp(-b*x)
        try:
            (a,b),_ = curve_fit(expfit, self._depths(), self._list_expectations())
        except:
            (a,b) = 1,0

        self.spam = a
        self.fidelity = np.exp(-b)