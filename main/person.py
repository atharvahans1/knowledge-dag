from theory import *
import numpy as np

class person(theory):

	def __init__(self, V, E, S_nodes, A_s):
		super().__init__(V, E, S_nodes)
		
		#A_s = Probability matrix of links between the subgraphs
		self.A_s = A_s

	def sample_K_s(self):
		A_s = self.A_s
		K_s = np.random.binomial(n=1, p=A_s, size=A_s.shape)
		K_s[K_s==0]=-1
		return K_s

	def sample_K(self):

		K_s = self.sample_K_s()
		
		#Map from K_s to K
		K_true = self.K_true
		K_s_true = self.K_s_true
		V_idx = self.V_idx
		S_idx = self.S_idx

		K = -1*np.ones(K_true.shape)
		K[V_idx[:,0], V_idx[:, 1]] = K_s[S_idx[:,0], S_idx[:, 1]]

		return K