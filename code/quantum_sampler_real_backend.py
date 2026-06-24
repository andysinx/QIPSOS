import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2


#NOTA: RICORDATI DI USARE L" AMBIENTE NUOVO DI CONDA NON USARE QUANTUM_ML MA QISKIT_NEW!!!
'''QiskitRuntimeService.save_account(
    channel="ibm_cloud",
    token="fOyMgA8eRRAVR1kNlmrwiu5NOmA4Exrd5zK0Fh9cz_S_",
    instance="time_free_instance_plan",
    overwrite=True
)'''

#IBM BACKEND AVAILABLE IN MY INSTANCE: [<IBMBackend('ibm_fez')>, <IBMBackend('ibm_marrakesh')>, <IBMBackend('ibm_kingston')>] usare marrakesh e' il piu scarico di job.
class QuantumSamplerIBM:
    global service
    def __init__(self,
                 backend_name="ibm_marrakesh",
                 shots=2048,
                 batch_size=10,
                 use_least_busy=True):

        self.shots = shots
        self.batch_size = batch_size

        # -------------------------
        # CONNECT TO IBM
        # -------------------------
        self.service = QiskitRuntimeService()
        
        if use_least_busy:
            self.backend = self.service.least_busy(operational=True)
        elif backend_name is not None:
            self.backend = self.service.backend(backend_name)
        else:
            raise ValueError("Provide backend_name or set use_least_busy=True")

        print(f"[IBM backend selected]: {self.backend.name}")

        # IMPORTANT: create sampler ONCE (FREE TIER COMPATIBLE)
        self.sampler = SamplerV2(mode=self.backend)


    # -------------------------
    # CIRCUIT DEFINITION
    # -------------------------
    def build_circuit(self, theta):
        n = len(theta)
        qc = QuantumCircuit(n, n)

        qc.h(range(n))

        # encoding layer
        for i in range(n):
            qc.ry(theta[i], i)
            qc.rz(0.5 * theta[i], i)

        # entanglement
        for i in range(n - 1):
            qc.cx(i, i + 1)

        # refinement layer
        for i in range(n):
            qc.ry(0.3 * theta[i], i)

        qc.measure(range(n), range(n))

        return qc

    # -------------------------
    # SINGLE SAMPLE (DEBUG / SMALL SCALE)
    # -------------------------
    def sample_single(self, theta):
        qc = self.build_circuit(theta)

        qc_t = transpile(qc, self.backend, optimization_level=1)
        result = self.sampler.run([qc_t], shots=self.shots).result()

        counts = result[0].data.c.get_counts()

        return self._decode(counts)

    # -------------------------
    # BATCHED SAMPLING (IBM OPTIMIZED)
    # -------------------------
    def sample_batch(self, theta_list):

        chunks = [
            theta_list[i:i + self.batch_size]
            for i in range(0, len(theta_list), self.batch_size)
        ]

        outputs = []

        for chunk in chunks:

            circuits = []
            n_qubits = None

            for t in chunk:
                qc = self.build_circuit(t)
                circuits.append(qc)

                if n_qubits is None:
                    n_qubits = qc.num_qubits

            circuits_t = transpile(
                circuits,
                self.backend,
                optimization_level=1
            )

            job = self.sampler.run(circuits_t, shots=self.shots)
            result = job.result()

            for pub in result:
                counts = pub.data.c.get_counts()
                outputs.append(self._decode(counts, n_qubits))

        return np.vstack(outputs)

    # -------------------------
    # DECODE BITSTRING → {-1, +1}
    # -------------------------
    def _decode(self, counts, n_qubits):
        bitstrings = list(counts.keys())
        probs = np.array(list(counts.values()), dtype=float)
        probs /= probs.sum()

        chosen = bitstrings[np.random.choice(len(bitstrings), p=probs)]

        vec = np.array([1 if b == "1" else -1 for b in chosen[::-1]])

        # 🔥 PAD/TRIM per sicurezza
        if len(vec) < n_qubits:
            vec = np.pad(vec, (0, n_qubits - len(vec)))
        else:
            vec = vec[:n_qubits]

        return vec
    
    def close(self):
        pass