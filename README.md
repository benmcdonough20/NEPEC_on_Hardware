# NEPEC on Hardware
A project conducted as part of my Fermilab Summer 2022 SQMS internship

Research Advisor Dr. Peter Orth

## QiskitProcessTomography
I wrote a short script to carry out process tomography and return the channel in terms of Kraus operators. Qiskit is currently deprecating the `qiskit.ignis` package in favor of `qiskit_experiments`. I wrote the example using `qiskit_experiments`. This worked well on the simulated noisy backend, but returned error code `7001`, unsupported instruction, when run on a quantum backend. I think the circuit may have been compiled to have resets during the run, which was not supported. I wrote another example using the `qiskit.ignis` package, and this time it worked when I tested it on the hardware.
