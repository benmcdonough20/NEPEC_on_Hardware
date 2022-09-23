from noisedata import NoiseData
from matplotlib import pyplot as plt
import pickle

class Analysis:
    def __init__(self, benchmarkinstances, procspec):
        self.benchmarkinstances = benchmarkinstances
        self.procspec = procspec
        self._data = {}

    def sim_meas(self, inst, pauli):
        pair = pauli.conjugate(inst.cliff_layer)
        if inst.type == "single":
            return [term for term in self.procspec.model_terms if pauli.disjoint(term) and pair.disjoint(term.conjugate(inst.cliff_layer))]
        elif inst.type == "double":
            return [term for term in self.procspec.model_terms if pauli.disjoint(term)]

    def analyze(self):
        sim_measurements = {}
        for inst in self.benchmarkinstances:
            if inst.cliff_layer not in  self._data:
                self._data[inst.cliff_layer] = NoiseData(inst.cliff_layer)

            basis = inst.prep_basis
            
            if not basis in sim_measurements:
                sim_measurements[basis] = self.sim_meas(inst, basis)

            for pauli in sim_measurements[basis]:
                self._data[inst.cliff_layer].add_expectation(pauli, inst)

        for noisedata in self._data.values():
            noisedata.fit_noise_model()
            
    def profiles(self):
        return list(self._data)

    def plot(self, layer):
        fig, ax = plt.subplots()
        self._data[layer].plot(ax)
        return ax

    def save(self):
        with open("results.dat", "wb") as f:
            pickle.dump(self,f)

    @staticmethod
    def load():
        with open("results.dat", "rb") as f:
            return pickle.load(f)
