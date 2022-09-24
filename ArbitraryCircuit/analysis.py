from noisedata import LayerNoiseData
from matplotlib import pyplot as plt
import pickle
import logging

from benchmarkinstance import SINGLE, PAIR

logger = logging.getLogger("experiment")

class Analysis:
    def __init__(self, benchmarkinstances, procspec):
        self.benchmarkinstances = benchmarkinstances
        self.procspec = procspec
        self._data = {}

    def sim_meas(self, inst, pauli):
        pair = inst.cliff_layer.conjugate(pauli)
        if inst.type == SINGLE:
            return [term for term in self.procspec.model_terms if pauli.simultaneous(term) and pair.simultaneous(inst.cliff_layer.conjugate(term))]
        elif inst.type == PAIR:
            return [term for term in self.procspec.model_terms if pauli.simultaneous(term)]

    def analyze(self):
        logging.info("Running analysis...")
        sim_measurements = {}
        for inst in self.benchmarkinstances:
            if inst.cliff_layer not in  self._data:
                self._data[inst.cliff_layer] = LayerNoiseData(inst.cliff_layer)

            basis = inst.prep_basis
            
            if not basis in sim_measurements:
                sim_measurements[basis] = self.sim_meas(inst, basis)

            for pauli in sim_measurements[basis]:
                self._data[inst.cliff_layer].add_expectation(pauli, inst)

        for noisedata in self._data.values():
            noisedata.fit_noise_model()
            
    def noise_profiles(self):
        return list(self._data.values())

    def save(self):
        with open("results.dat", "wb") as f:
            pickle.dump(self,f)
    
    def model_terms(self, *qubits):
        paulis = []
        for pauli in self.procspec.model_terms:
            overlap = [pauli[q].to_label() != "I" for q in qubits]
            support = [p.to_label() == "I" or q in qubits for q,p in enumerate(pauli)]
            if all(overlap) and all(support):
                paulis.append(pauli)

        return paulis

    @staticmethod
    def load():
        with open("results.dat", "rb") as f:
            return pickle.load(f)
