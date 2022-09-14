from noisemodel import NoiseModel
from benchmarkinstance import BenchmarkInstance
from itertools import product

class LayerLearning:
    def __init__(self, layer, inst_map, depths, samples, single_samples):
        self._cliff_layer = layer.cliff_layer(layer)
        self._inst_map = inst_map
        self.noise_model = NoiseModel(self._cliff_layer)
        self.samples = samples
        self.depths = depths
        self.single_samples = single_samples

    def procedure(self, procspec):

        if isinstance(samples, int):
            samples = [samples]*len(self.depths)
        if not single_samples:
            single_samples = samples[0]

        #find disjoint operators that can be measured simultaneously to find six bases
        pairs = set([frozenset([p,p.conjugate(self._cliff_layer)]) for p in procspec.model_terms if not self.noise_model.issingle(p)])
        single_bases = []
        for p1,p2 in pairs:
            for i,pauli in enumerate(single_bases):
                if pauli.disjoint(p1) and pauli.disjoint(p2):
                    single_bases[i] = pauli * p2
                    break
            else:
                single_bases.append(p2)

        print("bases for singles: ",single_bases)

        SINGLE = 1
        instances = []

        for basis, (d,s) in product(procspec.meas_bases, zip(self.depths,samples)):
            for i in range(s):
                inst = BenchmarkInstance(basis, basis, d, procspec)
                inst.type = "double"
                instances.append(inst)

        for basis, s in product(single_bases, range(single_samples)):
            inst = BenchmarkInstance(basis.conjugate(self._cliff_layer), basis, SINGLE, procspec)
            inst.type = "single"
            instances.append(inst)

        self.instances = instances 
        return instances