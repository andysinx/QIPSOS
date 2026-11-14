from qiskit import QuantumCircuit
from qiskit_ibm_runtime import SamplerV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2


import numpy as np

def define_backend(use_noise=False):

    if use_noise:

        backend_fake = FakeWashingtonV2()

        noise_model = NoiseModel.from_backend(
            backend_fake
        )

        gpu_instance = AerSimulator(
            noise_model=noise_model,
            method="statevector",
            device="GPU"
        )

        gpu_instance.set_options(
            precision='single'
        )

        return gpu_instance

    else:

        gpu_instance = AerSimulator(
            method="statevector",
            device="GPU"
        )

        gpu_instance.set_options(
            precision='single'
        )

        return gpu_instance

backend = define_backend(use_noise=False) 
sampler = SamplerV2(backend) 


'''def build_circuit(theta): #Hardware Efficient Ansatz -> RY + Rz + alternating entanglement lineare (HEA5)
    n = len(theta)
    qc = QuantumCircuit(n, n)

    qc.h(range(n))

    for i in range(n):
        qc.ry(theta[i], i)
    
    for i in range(n):
        qc.rz(theta[i], i)

    for i in range(n - 1):
        qc.cx(i, i + 1)

    for i in range(n):
        qc.ry(theta[i], i)
    
    for i in range(n - 1):
        qc.cx(i, i + 1)

    qc.measure(range(n), range(n))
    return qc'''

def build_circuit(theta):
    n = len(theta)
    qc = QuantumCircuit(n, n)

    for layer in range(2):

        for i in range(n):
            qc.ry(theta[i], i)
            qc.rz(theta[i], i)

        for i in range(n - 1):
            qc.cx(i, i + 1)

    qc.measure(range(n), range(n))
    return qc

def quantum_sample(theta):
    qc = build_circuit(theta)
    result = sampler.run([qc], shots=512).result()

    counts = result[0].data.c.get_counts()

    bitstrings = list(counts.keys())
    probs = np.array(list(counts.values()))
    probs = probs / probs.sum()

    b = bitstrings[np.random.choice(len(bitstrings), p=probs)]

    return np.array([1 if bit == '1' else -1 for bit in b[::-1]])


def decode_quantum(z, bounds):
    z = np.array(z)

    x = np.zeros(len(bounds))

    for i in range(len(bounds)):
        low, high = bounds[i]
        x[i] = low + (z[i] + 1) * 0.5 * (high - low)

    return x

