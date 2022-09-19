from matplotlib import pyplot as plt
import numpy as np
from qiskit.quantum_info import Pauli
from scipy.optimize import curve_fit
from itertools import cycle

class SparsePauliTomographyResult:
    def __init__(self, model):
        for term in model:
            self.add_term(term, **model[term])

    def add_term(self):
        pass

    #graph a subset of the measured expectation values and plot fits
    def graph(self,*paulis):
        expfit = lambda x,a,b : a*np.exp(x*-b)
        colcy = cycle(["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:cyan", "tab:brown", "tab:pink", "tab:gray", "tab:olive"])
        for p in paulis:
            c = next(colcy)
            data = self.ordered_data[p]['expectation']
            popt, pcov = curve_fit(expfit, self.depths, data, p0=[.9,.01])
            xrange = np.linspace(0,np.max(self.depths))
            plt.plot(xrange, [expfit(x, *popt) for x in xrange], color=c)
            plt.plot(self.depths, data, color = c, marker="o", linestyle = 'None')
        plt.title("Expectation vs Depth")
        plt.xlabel("Depth")
        plt.ylabel("Fidelity")
        plt.show()

    #display the measured fidelities plotted against the ideal fidelitites
    def display(self,*paulis, factor = 10):
        def nophase(pauli):
            return Pauli((pauli.z, pauli.x))

        def conjugate(pauli):
            return nophase(pauli.evolve(self.layer))
        basis_dict = self.ordered_data 
        ax = np.arange(len(paulis))
        fidelities = []
        for p in paulis:
            fid = basis_dict[p]['fidelity']
            if(basis_dict[p]['type'] == 'pair'):
                pair = conjugate(p)
                fid = fid**2/basis_dict[pair]['fidelity']
            fidelities.append(fid)
        plt.bar(ax, [10*(1-f) for f in fidelities],color='tab:blue')
        plt.xticks(ax, paulis)
        plt.title("Measured Fidelitites")
        plt.xlabel("Term")
        plt.ylabel(str(factor)+" x (1-f)")
        plt.legend(["Measured", "Ideal"])
        plt.show()

    def display_model(self, *paulis, labels = None):
        if not labels:
            labels = paulis
        ax = np.arange(len(paulis))
        coeffs,_ = zip(*self.noise_model)
        plt.bar(ax, coeffs, color='tab:blue')
        plt.xticks(ax, labels);
        plt.title("Measured Model Terms")
        plt.xlabel("Term")
        plt.ylabel("Coefficient")
        plt.show()