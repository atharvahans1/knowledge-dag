from theory import *
import numpy as np

class test(theory):

	def __init__(self, V, E, S_nodes):
		super().__init__(V, E, S_nodes)

	def create_question(self, V_m):
		"""
			X: List of variables relevant to problem m
		"""
		K_true = self.K_true
		
		V = self.V # Every node in the theory

		ind = np.array([True if v in V_m else False for v in V])

		Q_m = np.zeros(K_true.shape)
		Q_m[ind, :] = 1
		Q_m[:, ~ind] = 0

		return Q_m

	def questions(self, V_ms):
		"""
			V_ms: List of lists for variables
		"""
		Q_ms = list()
		for i in range(len(V_ms)):
			Q_m = self.create_question(V_ms[i])
			Q_ms.append(Q_m)
		return Q_ms

