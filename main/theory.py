import numpy as np
import theano.tensor as tt

class theory(object):

	def __init__(self, V, E, S_nodes):
		self.V = V #Nodes in the knowledge graph
		self.E = E #Edges between the nodes in V
		self.S_nodes = S_nodes #Node in different subgraphs of the knowledge graph

		#Row index and column index of an edge in NxN knowledge matrix (K_true)
		E_to_Ktrue = list()
		for e in E:
			E_to_Ktrue.append([V.index(e[0]), V.index(e[1])])
		E_to_Ktrue = np.array(E_to_Ktrue)

		#Defining K_true
		K_true = -1*np.ones((len(V), len(V)))
		K_true[E_to_Ktrue[:, 0], E_to_Ktrue[:, 1]] = 1

		#Row index and column index of an edge in N_sxN_s subgraph knowledge matrix (K_s_true)
		E_to_Kstrue = list() 
		for e in E:
			id_0= [i for i, s in enumerate(S_nodes) if e[0] in s][0]
			id_1= [i for i, s in enumerate(S_nodes) if e[1] in s][0]
			E_to_Kstrue.append([id_0, id_1])
		E_to_Kstrue = np.array(E_to_Kstrue)

		#Defining connections between subgraphs using matrix K_s
		N_s = len(S_nodes) #Number of subgraphs
		K_s_true = -1*np.ones((N_s, N_s))
		K_s_true[E_to_Kstrue[:, 0], E_to_Kstrue[:, 1]] = 1

		self.K_s_true = K_s_true
		self.K_true = K_true

		#Store indices
		self.E_to_Ktrue = E_to_Ktrue
		self.E_to_Kstrue = E_to_Kstrue

		#Mappings
		Sflat_to_S, Sflat_to_Aflat = np.unique(E_to_Kstrue, axis=0, return_inverse=True)
		self.Sflat_to_S = Sflat_to_S
		self.Sflat_to_Aflat = Sflat_to_Aflat



