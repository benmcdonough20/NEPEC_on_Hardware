from layernoisedata import LayerNoiseData
from matplotlib import pyplot as plt
import pickle
import logging

from noisedataframe import NoiseDataFrame

logger = logging.getLogger("experiment")

class Analysis:
    """This class is responsible for aggregating the data obtained from running the list of
    of benchmarkinstances generated by the experiment class, extracting the Pauli data that
    can be obtained simultaneously from the measurement basis, and assigning this data to the correct layer"""

    def __init__(self, procspec):
        """Takes as input the benchmark instances generated by the experiment class containing
        the results of running the underlying circuit on the qpu"""

        self.procspec = procspec
        self._data = {}
        self.spam = {}

    def analyze(self, benchmarkinstances):
        """Sorts the data into different layers represented by LayerNoiseData objects,
        runs the analysis on each once data is collected"""

        logging.info("Running analysis...")
        #sort instance data into layers
        for inst in benchmarkinstances:
            if inst.cliff_layer not in  self._data:
                self._data[inst.cliff_layer] = LayerNoiseData(inst.cliff_layer)

            self._data[inst.cliff_layer].add_expectation(inst, self.procspec)

        #run analysis on layers
        for noisedata in self._data.values():
            noisedata.fit_noise_model()

        #collect SPAM coefficients for readout error mitigation use in PER
        self.spam = {term:0 for term in self.procspec.model_terms if term.weight() == 1}
        for noisedata in self._data.values():
            spam_coeffs = noisedata.get_spam_coeffs()
            for term in self.spam:
                self.spam[term] += spam_coeffs[term]/len(self._data)

        return self.noise_profiles()
            
    def noise_profiles(self):
        """Get the noise profiles used to run PER. This is the link from the analysis
        portion to the PER portion"""
        noise_models = [noisedata.noisemodel for noisedata in self._data.values()]
        self.noisedataframe = NoiseDataFrame(noise_models, self.spam)
        return self.noisedataframe
    
    def layer_data(self):
        """Returns a list of LayerNoiseData objects that can be used for plotting and
        inspecting the full result of a run"""

        return list(self._data.values())

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()
