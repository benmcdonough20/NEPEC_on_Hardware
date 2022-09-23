from scipy.optimize import curve_fit, OptimizeWarning
import warnings
import numpy as np

warnings.simplefilter('error', OptimizeWarning)

class TermData:
    def __init__(self, term, pair):
        self.term = term
        self.pair = pair
        self._expectations = {}
        self._single_vals = []
        self._single_count = 0
        self._count = {}
        self._spam = 1
        self._fidelity = 1
    
    def add_expectation(self, depth, expectation, type):
        if type == "single":
            self._single_vals.append(expectation)
            self._single_count += 1
        elif type == "double":
            self._expectations[depth] = self._expectations.get(depth, 0) + expectation
            self._count[depth] = self._count.get(depth, 0) + 1

    def _depths(self):
        return sorted(self._expectations.keys())

    def _list_expectations(self):
        return [self._expectations[d]/self._count[d] for d in self._depths()]

    def ispair(self):
        return self.term == self.pair

    def fit(self):
        a,b = self._fit()
        self.spam = a
        self.fidelity = np.exp(-b)

    def _fit(self):
        expfit = lambda x,a,b: a*np.exp(-b*x)
        try:
            (a,b),_ = curve_fit(expfit, self._depths(), self._list_expectations(), p0 = [.8, .01])
        except:
            (a,b) = 1,0
        return a,b

    def fit_single(self, meas_err):
        if any(self._single_vals):
            expectation = abs(sum(self._single_vals)/self._single_count)
            self.fidelity = min([1,expectation/meas_err])

    def plot(self, ax):
        xline = np.linspace(0,max(self._depths()), 100)
        a,b = self._fit()
        ax.plot(self._depths(), self._list_expectations(), 'x')
        ax.plot(xline, [a*np.exp(-b*x) for x in xline])

    