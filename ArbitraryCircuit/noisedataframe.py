class NoiseDataFrame:
    """Aggregates the noise models and spam errors from the learning procedure for use in PER."""

    def __init__(self, noisemodels, spam_fidelities):
        self.noisemodels = {}
        for nm in noisemodels:
            self.noisemodels[nm.cliff_layer]  = nm
        self.spam = spam_fidelities

    def init_scaling(self, noise_strength):
        for nm in self.noisemodels.values():
            nm.init_scaling(noise_strength)