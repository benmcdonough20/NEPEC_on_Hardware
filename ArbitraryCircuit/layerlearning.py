from noisemodel import NoiseModel
from benchmarkinstance import BenchmarkInstance
from itertools import product

class LayerLearning:
    def __init__(self, layer, samples, single_samples, depths):
        self._cliff_layer = layer.cliff_layer
        self.samples = samples
        self.depths = depths
        self.single_samples = single_samples

    def procedure(self, procspec):
        
        def issingle(term):
            pair = term.conjugate(self._cliff_layer)
            return pair == term or pair not in procspec.model_terms

        #find disjoint operators that can be measured simultaneously to find six bases
        pairs = set([frozenset([p,p.conjugate(self._cliff_layer)]) for p in procspec.model_terms if not issingle(p)])
        single_bases = []
        for p1,p2 in pairs:
            for i,pauli in enumerate(single_bases):
                if pauli.disjoint(p1) and pauli.disjoint(p2):
                    single_bases[i] = pauli * p2
                    break
            else:
                single_bases.append(p2)

        SINGLE = 1
        instances = []

        for basis,d,s in product(procspec.meas_bases, self.depths, range(self.samples)):
            inst = BenchmarkInstance(basis, basis, d, procspec, self._cliff_layer)
            inst.type = "double"
            instances.append(inst)

        for basis,s in product(single_bases, range(self.single_samples)):
            inst = BenchmarkInstance(basis.conjugate(self._cliff_layer), basis, SINGLE, procspec, self._cliff_layer)
            inst.type = "single"
            instances.append(inst)

        self.instances = instances 
        return instances