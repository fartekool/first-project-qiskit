import numpy as np
from numpy import random
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import qiskit.quantum_info
from qiskit.visualization import plot_histogram, plot_state_city
import matplotlib.pyplot as plt
from qiskit import ClassicalRegister, QuantumRegister
from collections import Counter


def makeError(phi: float, disp: float) -> float:
  random_number = np.random.normal(0, (disp*disp), None)
  #print(random_number)
  return (phi + random_number)


# В симуляторе это QFT_Range
def QFT(circuit: QuantumCircuit, start: int, n: int, disp: float = 0) -> QuantumCircuit:
  for i in range(n - 1, start - 1, -1):
    circuit.h(i)
    for j in range(i - 1, start - 1, -1):
      phase = 2 * np.pi / pow(2, i - j + 1)
      phase = makeError(phase, disp)
      circuit.cp(phase, j, i)
  return circuit

# В симуляторе это IQFT_Range
def IQFT(circuit: QuantumCircuit, start: int, n: int, disp: float = 0) -> QuantumCircuit:
  for i in range(start, n, 1):
    for j in range(start, i, 1):
      phase = (-2) * np.pi / pow(2, i - j + 1)
      phase = makeError(phase, disp)
      circuit.cp(phase, j, i)
    circuit.h(i)
  return circuit


def reverse_qubit_order(circuit: QuantumCircuit, start: int, n: int):
  reversed_circuit: QuantumCircuit = circuit.reverse_bits()
  new_circuit: QuantumCircuit = circuit
  for i in range(start, n, 1):
    new_circuit[i] = reversed_circuit[i]

def QFT_Sub(circuit: QuantumCircuit, a_start: int, b_start: int, n: int, disp: float = 0) -> QuantumCircuit:
  reversed_circuit: QuantumCircuit = circuit
  reversed_circuit = reverse_qubit_order(circuit, a_start, n)
  reversed_circuit = reverse_qubit_order(circuit, b_start, n)
  QFT(reversed_circuit, b_start, b_start + n, disp)

  for i in range(0, n, 1):
    for j in range(i, n, 1):
      phase = -np.pi / (1 << (j - i))
      phase = makeError(phase, disp)
      reversed_circuit.cp(phase, a_start + (n - 1 - j), b_start + (n - 1 - i))

  IQFT(reversed_circuit, b_start, b_start + n, disp)
  return reversed_circuit


def copy_qubit(circuit: QuantumCircuit, qubit: int, an1: int, an2: int) -> QuantumCircuit:
  circuit.cx(qubit, an1)
  circuit.cx(qubit, an2)
  return circuit


def qubit_flip_correction(circuit: QuantumCircuit, qubit: int, an1: int, an2: int) -> QuantumCircuit:
  circuit.cx(qubit, an1)
  circuit.cx(qubit, an2)

  circuit.ccx(an2, an1, qubit)
  return circuit

def QFT_adder(circuit: QuantumCircuit, a_start: int, b_start: int, n:int, disp: float = 0) -> QuantumCircuit:
  circuit = QFT(circuit, b_start, b_start + n, disp)
  for i in range(0, n, 1):
    for j in range(i, n, 1):
      phase = np.pi / (1 << (j - i))
      phase = makeError(phase, disp)
      circuit.cp(phase, a_start + (n - 1 -j), b_start + (n - 1 - i))

  circuit = IQFT(circuit, b_start, b_start + n, disp)
  return circuit


def QFT_adder_with_correction(circuit: QuantumCircuit, a_start: int, b_start: int, n: int, disp: float = 0) -> QuantumCircuit:
  new_qubits_count = circuit.num_qubits
  new_qr = QuantumRegister(new_qubits_count, 'ancilla')
  circuit.add_register(new_qr)
  for i in range(0, n):
    circuit = copy_qubit(circuit, b_start + i, b_start + i + n, b_start + i + 2*n)

  circuit.barrier()

  circuit = QFT_adder(circuit, 0, b_start, n, disp)
  circuit = QFT_adder(circuit, 0, 2 * b_start, n, disp)
  circuit = QFT_adder(circuit, 0, 3 * b_start, n, disp)

  circuit.barrier()

  for i in range(0, n):
    circuit = qubit_flip_correction(circuit, b_start + i, b_start + i + n, b_start + i + 2 * n)
  return circuit

def copy_qubit_and_sign(circuit: QuantumCircuit, qubit: int, an1: int, an2: int, an3: int, an4: int, an5: int,
                        an6: int, an7: int, an8: int) -> QuantumCircuit:
  circuit.cx(qubit, an3)
  circuit.cx(qubit, an6)

  circuit.h(qubit)
  circuit.h(an3)
  circuit.h(an6)

  circuit.cx(qubit, an1)
  circuit.cx(qubit, an2)

  circuit.cx(an3, an4)
  circuit.cx(an3, an5)

  circuit.cx(an6, an7)
  circuit.cx(an6, an8)

  return circuit

def shor_correction(circuit: QuantumCircuit, qubit: int, an1: int, an2: int, an3: int, an4: int, an5: int,
                        an6: int, an7: int, an8: int) -> QuantumCircuit:
  circuit.cx(qubit, an1)
  circuit.cx(qubit, an2)

  circuit.cx(an3, an4)
  circuit.cx(an3, an5)

  circuit.cx(an6, an7)
  circuit.cx(an6, an8)

  circuit.ccx(an2, an1, qubit)
  circuit.ccx(an5, an4, an3)
  circuit.ccx(an8, an7, an6)

  circuit.h(qubit)
  circuit.h(an3)
  circuit.h(an6)

  circuit.cx(qubit, an3)
  circuit.cx(qubit, an6)

  circuit.ccx(an6, an3, qubit)

  return circuit

def QFT_adder_with_shor_correction(circuit: QuantumCircuit, a_start: int, b_start: int, n: int, disp: float = 0) -> QuantumCircuit:
  new_qr = QuantumRegister(8 * n, 'ancilla')
  circuit.add_register(new_qr)

  circuit = copy_qubit_and_sign(circuit, 2, 4, 6, 8, 10, 12, 14, 16, 18)
  circuit = copy_qubit_and_sign(circuit, 3, 5, 7, 9, 11, 13, 15, 17, 19)


  circuit = QFT_adder(circuit, 0, 2, 2, disp)
  circuit = QFT_adder(circuit, 0, 4, 2, disp)
  circuit = QFT_adder(circuit, 0, 6, 2, disp)
  circuit = QFT_adder(circuit, 0, 8, 2, disp)
  circuit = QFT_adder(circuit, 0, 10, 2, disp)
  circuit = QFT_adder(circuit, 0, 12, 2, disp)
  circuit = QFT_adder(circuit, 0, 14, 2, disp)
  circuit = QFT_adder(circuit, 0, 16, 2, disp)
  circuit = QFT_adder(circuit, 0, 18, 2, disp)

  circuit = shor_correction(circuit, 2, 4, 6, 8, 10, 12, 14, 16, 18)
  circuit = shor_correction(circuit, 3, 5, 7, 9, 11, 13, 15, 17, 19)

  return circuit

def Experiment(circuit: QuantumCircuit, count_of_mes: int, correction: bool, disp: float = 0) -> dict:

  final_counts = Counter()

  aer_sim = AerSimulator(method='statevector')

  for n in range(0, count_of_mes):
    q = circuit.copy()
    if (correction):
      q = QFT_adder_with_correction(q, 0, circuit.num_qubits // 2, circuit.num_qubits // 2, disp)
      #q = QFT_adder_with_shor_correction(q, 0, circuit.num_qubits // 2, circuit.num_qubits // 2, disp)
    else:
      q = QFT_adder(q, 0, circuit.num_qubits // 2, circuit.num_qubits // 2, disp)
    q.add_register(ClassicalRegister(circuit.num_qubits))
    q.measure(range(circuit.num_qubits), range(circuit.num_qubits))

    result_aer = aer_sim.run(q, shots = 1).result()
    current_counts = result_aer.get_counts()
    final_counts += Counter(current_counts)


  return dict(final_counts)



#-------------------------------------------------------------------------------



q = QuantumCircuit(4) 

65536


q.x(0)
q.x(2)

q.barrier()
q.draw('mpl')



result_dict = Experiment(q, count_of_mes = 100, correction=True, disp=0)
print(result_dict)
plot_histogram(result_dict)
plt.show()



