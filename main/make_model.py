import pymc3 as pm3
from pymc3.math import logsumexp, erfinv, log1pexp
import theano
import theano.tensor as tt
import numpy as np
import scipy
#from pymc3.step_methods import smc
import dill as pickle

def phi_positives_match(Q_ms_flat, Kflats, K_true_flat):
	#Loop over all questions and return feature values
	phis, _ = theano.scan(fn=lambda Q_m_flat, Kflats, K_true_flat: tt.sum((Q_m_flat*(Kflats+1)/2)*(Q_m_flat*(K_true_flat+1)/2), axis=1),
									sequences=[Q_ms_flat],
									non_sequences=[Kflats, K_true_flat])
	return phis.T

def make_model1(fatigue_th, e_obs, Q_ms, num_chains, iteration, posterior_samples, max_tree_depth, early_max_tree_depth, tune_samples):
	"""
		This is GrL Model
		This is a model for explaining the probability of correct response using the fraction of correctly matched links.
	"""
	n_pars = np.sum(fatigue_th.K_s_true>0) #Number of links in the true knowledge matrix assuming subgraphs
	Sflat_to_Aflat = fatigue_th.Sflat_to_Aflat
	K_true = fatigue_th.K_true
	E_to_Ktrue = fatigue_th.E_to_Ktrue
	Q_ms_flat = Q_ms[:, E_to_Ktrue[:, 0], E_to_Ktrue[:, 1]]
	K_true_flat = K_true[E_to_Ktrue[:, 0], E_to_Ktrue[:, 1]]
	n_questions=len(Q_ms)
	n_students = e_obs.shape[0]

	data_shared = theano.shared(e_obs) 
	Q_ms_flat_shared=theano.shared(Q_ms_flat) 


	with pm3.Model() as model:
		alpha_m = pm3.Exponential('alpha_m', lam=0.1) 							#sigmoid slope parameter
		beta_m = pm3.Beta('beta_m', alpha=1, beta=1)							#sigmoid threshold parameter
		mu_mu = pm3.Normal('mu_mu', mu = 0, sigma = 1, shape=n_pars)			#population level knowledge
		tau_i = pm3.Normal('tau_i', mu=0, sigma=15, shape=(n_students,n_pars)) 	#individual offset from population

		Sflat = pm3.math.sigmoid(mu_mu+tau_i)									#link probability   						
		Sflat = Sflat.clip(0.01,.99) 											#ensures stability

		l_mu = tt.sqrt(2)*erfinv(2*Sflat-1) 									#reparametrizing binary link variable into uncontrained real variable using sigmoid function
		l = pm3.Normal('l', mu=l_mu, sd=1, shape=(n_students, n_pars))			#reparametrizing binary link variable into uncontrained real variable using sigmoid function
		K_s_flats = -1 + 2*(1/(1+tt.exp(-50*l)))								#reparametrizing binary link variable into uncontrained real variable using sigmoid function
		Kflats = K_s_flats[:, Sflat_to_Aflat]

		phi1 = phi_positives_match(Q_ms_flat_shared, Kflats, K_true_flat)
		phi = phi1/Q_ms_flat_shared.sum(axis=1)

		slip_factor = pm3.Beta('slip_factor', alpha=.5, beta=1)					#slip factor
		b_intercept = (1 - slip_factor) / (pm3.math.sigmoid(alpha_m*(1-beta_m)) - pm3.math.sigmoid(-alpha_m*beta_m))
		a_intercept = -b_intercept*pm3.math.sigmoid(-alpha_m*beta_m)
		p_correct = a_intercept + b_intercept * pm3.math.sigmoid(alpha_m*(phi-beta_m))	#probability of answering correctly
		p_correct = p_correct.clip(.01,.99)										#ensures stability
	
		
		e = pm3.Bernoulli('e', logit_p=tt.log(p_correct/(1-p_correct)), observed=data_shared)

		step = pm3.NUTS(max_treedepth=max_tree_depth, early_max_treedepth=early_max_tree_depth)
		trace = pm3.sample(draws=iteration, chains=num_chains, step=step, tune=tune_samples)
		
# 	trace_burn_thin = trace[20000::30]
	trace_burn_thin = trace

	### Posterior predictive training dataset
	ppc_train = pm3.sample_posterior_predictive(trace_burn_thin, samples=posterior_samples, model=model)

	return locals()
