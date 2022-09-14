from noisedata import NoiseData

class LayerAnalysis:
    def __init__(self, layer, benchmarkinstances, procspec):
        self.benchmarkinstances = benchmarkinstances
        self.procspec = procspec
        self.noisedata = NoiseData(layer)

    def is_single(self, pauli):
        pair = self.cliff_layer.conjugate(pauli)
        return (pauli in self.model_terms and pair in self.model_terms) and pauli != pair

    def sim_meas(self, pauli):
        if pauli.is_single():
            return [term for term in self.model_terms if pauli.simultaneous(term)]
        else:
            return [term for term in self.model_terms if pauli.simultaneous(term)]

    def analyze(self):

        depths = self.depths
        single_samples = self.single_samples

        singles = []
        doubles = []
        for circ in self.benchmarkinstances:
            if circ.type == "single":
                singles.append(circ)
            elif circ.type == "double":
                doubles.append(circ)
        
        #improve execution time by storing runs of all_sim_meas for each basis
        sim_measurements = {}
        for inst in self.benchmarkinstances:
            #get run data
            basis = inst.prep_basis
            depth = inst.depth
            #find simultaneous measurements
            if not basis in sim_measurements:
                sim_measurements[basis] = self.noisedata.sim_meas(pauli)

            #aggregate expectation value data for each simultaneous measurement
            for pauli in sim_measurements[basis]:
                self.noisedata.add_expectation(pauli, inst)      

        expfit = lambda x,a,b : a*np.exp(x*-b)
        #for each of the simultaneous measurements
        for key in basis_dict.keys():
            for i,d in enumerate(depths):
                #divide by total
                basis_dict[key]["expectation"][i] /= basis_dict[key]["total"][i]
            #try finding exponential fit, default to ideal if no fit found
            try:
                popt, pcov = curve_fit(expfit, depths, basis_dict[key]["expectation"], p0=[.9,.01])
            except:
                popt = 1,0

            #store fidelity and SPAM coefficients
            basis_dict[key]["fidelity"] = expfit(1,1,popt[1])
            basis_dict[key]["SPAM"] = popt[0]

            #record whether measurement appears as a pair or as a single fidelity
            if key != conjugate(key):
                basis_dict[key]["type"] = "pair"
            else:
                basis_dict[key]["type"] = "single"
        
        singles_dict = {} #store results of single measurements
        sim_measurements = {}
        for datum in singles:
            meas_basis = datum['meas_basis']
            prep_basis = datum['prep_basis']
            #find terms that can be measured simultaneously
            if not meas_basis in sim_measurements:
                sim_measurements[meas_basis] = []
                for term in self.model_terms:
                    if simultaneous(meas_basis, term) and simultaneous(prep_basis, conjugate(term)) and is_single(term):
                        sim_measurements[meas_basis].append(term)
            #aggregate data together
            for meas in sim_measurements[meas_basis]:
                if meas not in singles_dict:
                    singles_dict[meas] = 0
                expectation = self.get_expectation(meas, **datum)
                #the measurement basis SPAM coefficients are closer because the readout noise, combined
                #with the noise from the last layer, is greater than the state preparation noise
                fidelity = np.min([1.0,np.abs(expectation)/basis_dict[meas]["SPAM"]])
                singles_dict[meas] += fidelity/single_samples

        #add singles data to basis_dict
        for key in singles_dict.keys():
            basis_dict[key]['fidelity'] = singles_dict[key]
            basis_dict[key]['type'] = "single"
        
        #fit model 
        self.noise_model = self.get_noise_model(basis_dict)
        for coeff, pauli in self.noise_model:
            basis_dict[pauli]["coefficient"] = coeff

        self.ordered_data = basis_dict
        return basis_dict
    
