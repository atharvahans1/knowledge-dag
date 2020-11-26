import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import scipy
from scipy.stats import ttest_ind, norm
from scipy.special import logsumexp
from matplotlib.cm import ScalarMappable
import graphviz
import dill as pickle
import pymc3 as pm3
import theano
import theano.tensor as tt
import plotly as pl
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from theano.tensor.shared_randomstreams import RandomStreams
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact, interact_manual, Layout, Button, fixed, Textarea, IntSlider, AppLayout,GridspecLayout, Output
from graphviz import Digraph
import io
from IPython.display import display_html, clear_output, display
import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)
from IPython.core.display import HTML
HTML("""
<style>
.output_png {
	display: table-cell;
	text-align: center;
	vertical-align: middle;
}
</style>
""")
sns.set_style('white')
sns.set_context('talk')

import requests
import os
def download(url, local_filename=None):
	"""
	Downloads the file in the ``url`` and saves it in the current working directory.
	"""
	data = requests.get(url)
	if local_filename is None:
		local_filename = os.path.basename(url)
	with open(local_filename, 'wb') as fd:
		fd.write(data.content)

url_theory = ''
download(url_theory)

url_question = '' 
download(url_question)

url_person = ''
download(url_person)

url_make_model = '' 
download(url_make_model)

from theory import *
from question import *
from person import *
from make_model import *




style = {'description_width': 'initial'}
node_input = widgets.FileUpload(
	accept='.txt',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
	multiple=False , # True to accept multiple files upload else False
	description='Nodes',
	button_style='', #'success', 'info', 'warning', 'danger'
	layout=Layout(height='auto', width='auto')
)
edges_input = widgets.FileUpload(
	accept='.txt',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
	multiple=False , # True to accept multiple files upload else False
	description='Edges',
	button_style='', #'success', 'info', 'warning', 'danger'
	layout=Layout(height='auto', width='auto')
)
same_knowledge_input = widgets.FileUpload(
	accept='.txt',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
	multiple=False , # True to accept multiple files upload else False
	description='Sub-Graphs Nodes',
	button_style='', #'success', 'info', 'warning', 'danger'
	layout=Layout(height='auto', width='auto')
)
# Questions
training_question_nodes = widgets.FileUpload(
	accept='.txt',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
	multiple=False , # True to accept multiple files upload else False
	description='Question-Specific Nodes',
	button_style='', #'success', 'info', 'warning', 'danger'
	layout=Layout(height='auto', width='auto')
)
# testing_question_nodes = widgets.FileUpload(
#     accept='.txt',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
#     multiple=False , # True to accept multiple files upload else False
#     description='Testing Question-Specific Nodes',
#     button_style='', #'success', 'info', 'warning', 'danger'
#     layout=Layout(height='auto', width='auto')
# )
training_data = widgets.FileUpload(
	accept='.csv',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
	multiple=False , # True to accept multiple files upload else False
	description='Question Responses',
	button_style='', #'success', 'info', 'warning', 'danger'
	layout=Layout(height='auto', width='auto')
)
# style = {'description_width': 'initial'}
# num_individuals = widgets.BoundedIntText(
#     value=1,
#     min=1,
#     max=200,
#     step=1,
#     description='Number of Individuals',
#     disabled=False,
#     layout=Layout(height='auto', width='auto')
#     style=style
# )
rbut1=Button(description="Upload Data", 
			layout=Layout(height='auto', width='auto'),
			button_style='danger')

def run_button_clicked1(b1):
	results_output_vent.clear_output()
	node_input.disabled=True
	edges_input.disabled=True
	same_knowledge_input.disabled=True
	training_question_nodes.disabled=True
	training_data.disabled=True
	
	numchains.disabled=False
	tunesamples.disabled=False
	mcmciteration.disabled=False
	posterioriteration.disabled=False
	maxtreedepth.disabled=False
	earlymaxtreedepth.disabled=False
	
	b1.button_style='danger'
	b1.description='Uploading'
	with results_output_vent:
		problem_specific_inputs()
	b1.disabled=True
	rbut2.disabled=False
	b1.button_style='success'
	b1.description='Data Uploaded'
rbut1.on_click(run_button_clicked1)



def problem_specific_inputs():
	global V, E, S_nodes, V_ms1, fatigue_th, fatigue_test, Q_ms, e_obs
	
	V = eval(list(node_input.value.values())[0]['content'].decode("utf-8"))
	E = eval(list(edges_input.value.values())[0]['content'].decode("utf-8"))
	S_nodes = eval(list(same_knowledge_input.value.values())[0]['content'].decode("utf-8"))
	V_ms1 = eval(list(training_question_nodes.value.values())[0]['content'].decode("utf-8"))
#     V_ms2 = eval(list(testing_question_nodes.value.values())[0]['content'].decode("utf-8"))

	filename_e_obs = list(training_data.value.keys())[0]
	content_e_obs = training_data.value[filename_e_obs]['content']
	df = pd.read_csv(io.BytesIO(content_e_obs), encoding='utf8', sep=",", header=None, index_col=False, dtype={"switch": np.int8})
	e_obs = df.values
	
	fatigue_th = theory(V, E, S_nodes)
	fatigue_test = test(V, E, S_nodes)
	Q_ms = np.array(fatigue_test.questions(V_ms1)) # training data-set
#     Q_ms_test = np.array(fatigue_test.questions(V_ms2)) # testing data-set   



# MCMC Sampler (NUTS) Inputs
numchains=widgets.IntSlider(
	value=2,
	max=4,
	min=1,
	step=1,
	description='Chains',
	disabled=True,
	layout=Layout(height='auto', width='auto'),
	style=style
)
tunesamples=widgets.IntSlider(
	value=500,
	max=2000,
	min=0,
	step=100,
	description='Tune Samples',
	disabled=True,
	layout=Layout(height='auto', width='auto'),
	style=style
)
mcmciteration=widgets.IntSlider(
	value=50000,
	max=100000,
	min=500, ########################################
	step=500,
	description='MCMC Samples',
	disabled=True,
	layout=Layout(height='auto', width='auto'),
	style=style
)
posterioriteration=widgets.IntSlider(
	value=2000,
	max=5000,
	min=1000, ########################################
	step=100,
	description='Posterior Samples',
	disabled=True,
	layout=Layout(height='auto', width='auto'),
	style=style
)
maxtreedepth=widgets.IntSlider(
	value=8,
	min=6,
	max=10,
	step=1,
	description='Maximum Tree Depth',
	disabled=True,
	layout=Layout(height='auto', width='auto'),
	style=style
)
earlymaxtreedepth=widgets.IntSlider(
	value=8,
	min=6,
	max=10,
	step=1,
	description='Early Maximum Tree Depth',
	disabled=True,
	layout=Layout(height='auto', width='auto'),
	style=style
)



def sigmoid(x):
	return 1/(1+np.exp(-x))

def simulate():
	global env1, trace, model, ppc_train, e_pred_training, Sflat_to_Aflat, mu_mu, tau_i, l, K_s_flats, K_flat_sampled_205, Aflats, Aflats_clustering
	
	env1 = make_model1(fatigue_th, e_obs, Q_ms, num_chains=numchains.value,
					   iteration=mcmciteration.value, 
					   posterior_samples=posterioriteration.value,
					   max_tree_depth=maxtreedepth.value, early_max_tree_depth=earlymaxtreedepth.value,
					   tune_samples=tunesamples.value)
	
	trace = env1['trace_burn_thin']
	model = env1['model']
	ppc_train = env1['ppc_train']
	e_pred_training = ppc_train['e']
	Sflat_to_Aflat = fatigue_th.Sflat_to_Aflat
#     K_true=fatigue_th.K_true
#     E_to_Ktrue = fatigue_th.E_to_Ktrue
	mu_mu = trace['mu_mu']
	tau_i = trace['tau_i']
	l = trace['l']
	## K graphs
	K_s_flats = -1 + 2*(1/(1+np.exp(-500000*l)))
	K_flat_sampled_205 = K_s_flats[:,:, Sflat_to_Aflat]
	K_flat_sampled_205[K_flat_sampled_205<0] = 0
	# Population Graph
	Sflats = sigmoid(mu_mu) 
	Aflats = Sflats[:,Sflat_to_Aflat]
	Aflats = np.array(Aflats) 
	# For clustering 
	Sflats_clustering = sigmoid(mu_mu[:,None,:] + tau_i)
	Aflats_clustering = Sflats_clustering[:,:,Sflat_to_Aflat]



# Population Knowledge Graph
def getpopknowledgegraph():
	global figpop
	figpop, ax=plt.subplots(sharex=True, sharey=True, figsize=(12,20))
	figpop.text(0.5,1, 'Population-level Link Probability sigm($\mu_{ij}$)',
			 fontsize=25, ha="center", va="center",fontweight='bold')
	ax.axes.xaxis.set_ticks([])
	ax.axes.yaxis.set_ticks([])
	figpop.text(0.0,0.5, "#Posterior Samples",
			 fontsize=20, ha="center", va="center", rotation=90)
	for i in range(len(E)):
		ax = figpop.add_subplot(8,3,i+1)
		cm = plt.cm.get_cmap('Reds')
		n, bins, patches = plt.hist(Aflats[:,i], bins=np.linspace(0,1,25)) 
		bin_centers = 0.5 * (bins[:-1] + bins[1:])
		# scale values to interval [0,1]
		col = bin_centers - min(bin_centers)
		col /= max(col)
		for c, p in zip(col, patches):
			plt.setp(p, 'facecolor', cm(c))
	
		ax.set_title('Link '+ str(E[i]), fontsize=20)
		ax.set_xlim([0,1])
		ax.tick_params(axis='y', colors='black')
		
	position=figpop.add_axes([.98,.3,0.02,.4])
	scales = np.linspace(0, 1, 7)
	cmap = plt.get_cmap("Reds")
	norm = plt.Normalize(scales.min(), scales.max())
	sm =  ScalarMappable(norm=norm, cmap=cmap)
	sm.set_array([])
	cbar = figpop.colorbar(sm,cax=position)
	
	plt.tight_layout()
	
def noneedslider():
	getpopknowledgegraph()

rbut4=Button(description="Get Population Knowledge Graph", 
			layout=Layout(height='auto', width='auto'),
			tooltip="Population Knowledge Graph",
			disabled=True)

def run_button_clicked4(b4):  
	results_output_vent.clear_output()
	rbut5.button_style=''
	rbut5_2.button_style=''
	rbut6.button_style=''
	rbut7.button_style=''
	b4.button_style='danger'
	b4.description='Processing'
	with results_output_vent:
		display(download_rbut4)
		_=interact(noneedslider,Individual=widgets.IntSlider(min=1, max=len(e_obs), step=1, value=1))
	b4.disabled=False
	b4.button_style='success'
	b4.description="Get Population Knowledge Graph"
rbut4.on_click(run_button_clicked4)

# Downloading population plot
def down_pop():
	figpop.savefig('Figures/Population/Pop_Link_Prob.pdf', dpi=300, bbox_inches='tight')

def download_pop(b_pop_download):
	b_pop_download.button_style='danger'
	b_pop_download.description='Processing'
	down_pop()
	b_pop_download.disabled=False
	b_pop_download.button_style=''
	b_pop_download.description="Download Population Knowledge Graph"
	

download_rbut4=Button(description="Download Population Knowledge Graph", 
			layout=Layout(height='auto', width='auto'),
					  style=style,
			tooltip="Download Population Knowledge Graph",
			disabled=False,
			button_style='')
download_rbut4.on_click(download_pop)



# Posterior Predictive Checking on training dataset----Visual Checks
def posterior_predictive_checks():
	global fig_post
	
	fig_post, ax = plt.subplots(2,2, sharey=True, sharex=True, figsize=(8,8))
	
	fig_post.text(0.5,1, 'Posterior predictive checking on the training dataset. Black color represent\n incorrect responses and ivory color represent correct responses.',
			 fontsize=20, ha="center", va="center",fontweight='bold')

	idx_train = np.argsort(e_obs.sum(axis=1))

	ax[0,0].imshow(e_obs[idx_train,], aspect='auto')
	ax[0,0].set_title('Observed')
	ax[0,0].set_xticks(range(e_obs.shape[1]))
	ax[0,0].set_xticklabels(range(1,1+e_obs.shape[1]))
	ax[0,1].patch.set_visible(False)
	ax[0,1].axis('off')
	
	import matplotlib.ticker as ticker
	ax[0,0].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

	ax[1,0].imshow(e_pred_training[-1,idx_train,], aspect='auto')
	ax[1,0].set_title('GrL Posterior Sample 1')

	ax[1,1].imshow(e_pred_training[-3,idx_train,], aspect='auto')
	ax[1,1].set_title('GrL Posterior Sample 2')
	
	ax[1,0].set_xlabel('Question number')
	ax[1,1].set_xlabel('Question number')

	ax[1,0].set_ylabel('Individuals ordered by\n #of correct responses')
	ax[0,0].set_ylabel('Individuals ordered by\n #of correct responses')
	
	plt.tight_layout()
	
def noneedslider2():
	posterior_predictive_checks()
	
rbut5=Button(description="Get Posterior Predictive Checks for Training Dataset (Visual Checks)", 
			layout=Layout(height='auto', width='auto'),
			tooltip="Posterior Predictive Checks for Training Dataset (Visual Checks)",
			disabled=True)

def run_button_clicked5(b5):  
	results_output_vent.clear_output()
	rbut4.button_style=''
	rbut5_2.button_style=''
	rbut6.button_style=''
	rbut7.button_style=''
	b5.button_style='danger'
	b5.description='Processing'
	with results_output_vent:
		display(download_rbut5)
		_=interact(noneedslider2,Individual=widgets.IntSlider(min=1, max=len(e_obs), step=1, value=1))
	b5.disabled=False
	b5.button_style='success'
	b5.description="Get Posterior Predictive Checks for Training Dataset (Visual Checks)"
rbut5.on_click(run_button_clicked5)


# Downloading ppc
def down_ppc():
	fig_post.savefig('Figures/Post_Pred_Check_Training/ppc_training_visual_checks.pdf', dpi=300, bbox_inches='tight')

def download_ppc(b_ppc_download):
	b_ppc_download.button_style='danger'
	b_ppc_download.description='Processing'
	down_ppc()
	b_ppc_download.disabled=False
	b_ppc_download.button_style=''
	b_ppc_download.description="Download Posterior Predictive Plot (Visual Checks)"

download_rbut5=Button(description="Download Posterior Predictive Plot (Visual Checks)",
					  layout=Layout(height='auto', width='auto'),
					  tooltip="Download Posterior Predictive Plot (Visual Checks)",
					  button_style='',
					  disabled=False
					 )
download_rbut5.on_click(download_ppc)


# Posterior Predictive Checking on training dataset----# of correct responses
def population_test(e_pred_training):
	return np.sum(e_pred_training, axis=(1,2))

def posterior_predictive_checks_correct_res():
	global fig_post_correct_res
	
	fig_post_correct_res, ax = plt.subplots(figsize=(8,8))
	fig_post_correct_res.text(0.5,1, 'Posterior predictive checking on the training dataset. Blue line represents\n the observed number of correct responses for the population.',
			 fontsize=20, ha="center", va="center",fontweight='bold')
	
	test_grlm = population_test(e_pred_training)
	test_obs = np.sum(e_obs)
	bayes_p_value = np.sum(test_grlm>test_obs)/test_grlm.shape[0]

	_=sns.distplot(test_grlm, ax=ax, color='red', label='GrL\n(Bayes p=%.2f)'%bayes_p_value)
	ax.axvline(test_obs, 0,4000, lw=4, label='Observed')
	ax.set_xlabel('Total correct responses\n by the population')
	ax.set_ylabel('Frequency')
	ax.legend(fontsize=14, loc='best')
	plt.tight_layout()
	
def noneedslider3():
	posterior_predictive_checks_correct_res()
	
rbut5_2=Button(description="Get Posterior Predictive Checks for Training Dataset (#Correct Responses)", 
			layout=Layout(height='auto', width='auto'),
			tooltip="Posterior Predictive Checks for Training Dataset (#Correct Responses)",
			disabled=True)

def run_button_clicked5_2(b5_2):  
	results_output_vent.clear_output()
	rbut4.button_style=''
	rbut5.button_style=''
	rbut6.button_style=''
	rbut7.button_style=''
	b5_2.button_style='danger'
	b5_2.description='Processing'
	with results_output_vent:
		display(download_rbut5_2)
		_=interact(noneedslider3,Individual=widgets.IntSlider(min=1, max=len(e_obs), step=1, value=1))
	b5_2.disabled=False
	b5_2.button_style='success'
	b5_2.description="Get Posterior Predictive Checks for Training Dataset (#Correct Responses)"
rbut5_2.on_click(run_button_clicked5_2)


# Downloading ppc
def down_ppc_correct_res():
	fig_post_correct_res.savefig('Figures/Post_Pred_Check_Training/ppc_training_correct_res.pdf', dpi=300, bbox_inches='tight')

def download_ppc_correct_res(b_ppc_correct_res_download):
	b_ppc_correct_res_download.button_style='danger'
	b_ppc_correct_res_download.description='Processing'
	down_ppc_correct_res()
	b_ppc_correct_res_download.disabled=False
	b_ppc_correct_res_download.button_style=''
	b_ppc_correct_res_download.description="Download Posterior Predictive Plot (#Correct Responses)"

download_rbut5_2=Button(description="Download Posterior Predictive Plot (#Correct Responses)",
					  layout=Layout(height='auto', width='auto'),
					  tooltip="Download Posterior Predictive Plot (#Correct Responses)",
					  button_style='',
					  disabled=False
					 )
download_rbut5_2.on_click(download_ppc_correct_res)


# Individual Causal Graph
def individualcausalgraph(xx):
	global gcp, download_number_dag
	download_number_dag = xx
	
	penwidth_diagraph = 3*Aflats_clustering
	
	gcp = Digraph('Causal_Graph')
	for i in range(len(V)):
		gcp.node('%s'%V[i],label=V[i])
	for i in range(len(E)):
		gcp.edge(*E[i], label='%.2f'%Aflats_clustering.mean(axis=0)[(xx)-1][i], 
				 penwidth='%.2f'%penwidth_diagraph.mean(axis=0)[(xx)-1][i])
	gcp.attr(label=r'\n\nKnowledge Graph for Individual#%.0f'%xx, labelloc ='t')
	gcp.attr(fontsize='20')
	display(gcp)
	
def studentslider6(Individual):
	individualcausalgraph(Individual)
	
rbut6=Button(description="Get Individuals' Knowledge Graph (DAG)", 
			layout=Layout(height='auto', width='auto'),
			tooltip="Individuals' Knowledge Graph (DAG)",
			disabled=True)

def run_button_clicked6(b6):
	results_output_vent.clear_output()
	rbut4.button_style=''
	rbut5.button_style=''
	rbut5_2.button_style=''
	rbut7.button_style=''
	b6.button_style='danger'
	b6.description='Processing'
	with results_output_vent:
		display(download_rbut6)
		interact(studentslider6,Individual=widgets.IntSlider(min=1, max=len(e_obs), step=1, value=1,
															description='Individual#', layout=Layout(height='auto', width='auto'),
															style=style));
	b6.disabled=False
	b6.button_style='success'
	b6.description="Get Individuals' Knowledge Graph (DAG)"
rbut6.on_click(run_button_clicked6)

# Downloading DAG
def down_dag():
	gcp.render('Figures/Individual_DAGs/DAG-Individual-%.0f'%download_number_dag)

def download_dag(b_download):
	b_download.button_style='danger'
	b_download.description='Processing'
	down_dag()
	b_download.disabled=False
	b_download.button_style=''
	b_download.description="Download Knowledge Graph (DAG)"

download_rbut6=Button(description="Download Knowledge Graph (DAG)",
					  layout=Layout(height='auto', width='auto'),
					  tooltip="Download Knowledge Graph (DAG)",
					  button_style='',
					  disabled=False
					 )
download_rbut6.on_click(download_dag)


#Parallel Cordination Plot
def plotting_link_specific_cluster_plot(num_clusters, width, height):
	### Parallel Coordination Based on a_ij's
	## Clustering
	parallel_coor_aij = Aflats_clustering.mean(axis=0) #shape=num_students X num_links
	kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(parallel_coor_aij)
	# kmeans.labels_
 

	## Automating Dictionary
	par_cord_list = []
	for num_links in range(len(E)):
		values = [[0,1], str(E[num_links]), parallel_coor_aij[:,num_links]]
		par_cord_list.append(values)

	parallel_cord_list = []
	keys= ["range","label","values"]   
	for i in range(len(E)):
		intermediary_dict = {}
		for j in range(len(keys)):
			intermediary_dict[keys[j]]=par_cord_list[i][j]
		parallel_cord_list.append(intermediary_dict)

	# Adding individual numbering 
	dicts_naming = {}
	keys= ["range","label","values","tickvals"] 
	values_naming = [[1,e_obs.shape[0]], "Select Individual", list(range(1, 1+e_obs.shape[0])),list(range(1, 1+e_obs.shape[0])) ]
	for i in range(len(keys)):
		dicts_naming[keys[i]] = values_naming[i]

	parallel_cord_list.insert(0,dicts_naming)


	## plotting parallel coordination a_ij's
	fig_aij = go.Figure(data=
	go.Parcoords(
		line = dict(color = kmeans.labels_),# colorscale = 'rdylgn', showscale = True, cmin = 0, cmax = 1),
		dimensions = parallel_cord_list))
	
	fig_aij.update_layout(
	plot_bgcolor = 'white',
	paper_bgcolor = 'white')

	fig_aij.update_layout(
#         title='Parallel Coordination Plot + Cluster wrt Posterior Link Probabilities',
	autosize=False,
	width=width, #1800
	height=height,) #500


	fig_aij.show();

#Buttons
	
rbut7=Button(description="Get Parallel Coordination Plot + Clusters wrt Posterior Link Probabilities", 
			layout=Layout(height='auto', width='auto'),
			tooltip="Parallel Coordination Plot + Clusters wrt Posterior Link Probabilities",
			disabled=True)


def studentslider7(Clusters,Width,Height):
	plotting_link_specific_cluster_plot(Clusters, Width, Height)


def run_button_clicked7(b7):
	results_output_vent.clear_output()
	rbut4.button_style=''
	rbut5.button_style=''
	rbut5_2.button_style=''
	rbut6.button_style=''
	b7.button_style='danger'
	b7.description='Processing'
	with results_output_vent:
		interact(studentslider7, Clusters=widgets.IntSlider(min=1, max=len(e_obs)/2, step=1, value=2,
															description='#Clusters Across Population:',
															layout=Layout(height='auto', width='auto'),
															style=style), 
				 Width=widgets.IntSlider(min=1500, max=2500, step=100, value=1800, 
										 description='Figure Width',
										 layout=Layout(height='auto', width='auto'),
										 style=style),
				Height=widgets.IntSlider(min=500, max=2500, step=100, value=500, description='Figure Height',
										 layout=Layout(height='auto', width='auto'),
										 style=style))
	b7.disabled=False
	b7.button_style='success'
	b7.description="Compare Individual's Across Causal Links"
rbut7.on_click(run_button_clicked7)


# Running MCMC 
rbut2=Button(description="Run MCMC Sampler", 
			layout=Layout(height='auto', width='auto'),
			tooltip="MCMC Sample",
			button_style='danger',
			disabled=True)

def run_button_clicked2(b2):
	results_output_vent.clear_output()
	rbut2.disabled=True
	numchains.disabled=True
	tunesamples.disabled=True
	mcmciteration.disabled=True
	posterioriteration.disabled=True
	maxtreedepth.disabled=True
	earlymaxtreedepth.disabled=True
	b2.button_style='danger'
	b2.description='Sampling'
	with results_output_vent:
		print(simulate())
		results_output_vent.clear_output()
	b2.disabled=True
	rbut4.disabled=False
	rbut5.disabled=False
	rbut5_2.disabled=False
	rbut6.disabled=False
	rbut7.disabled=False
	b2.button_style='success'
	b2.description='Sampling Complete'
rbut2.on_click(run_button_clicked2)


tool_name = widgets.HTML(
	value="<b><font size='5'><u>Hierarchical Bayesian Inference Tool for Knowledge DAGs</u><font></b>: <font size='5'>Infer individuals' causal scientific knowledge from the individuals' responses to a set of questions.<font>",
	placeholder='Some HTML',
	description='',
)

header_1 = widgets.HTML(
	value="<b><font size='3'>1) Theory Specific Inputs<font></b>",
	placeholder='Some HTML',
	description='',
)

header_2 = widgets.HTML(
	value="<b><font size='3'>2) MCMC Sampler (NUTS)<font></b>",
	placeholder='Some HTML',
	description=''
)

header_3 = widgets.HTML(
	value="<b><font size='3'>3) Begin Sampling<font></b>",
	placeholder='Some HTML',
	description='',
	layout=Layout(height='auto', width='auto')
)

header_4 = widgets.HTML(
	value="<b><font size='3'>4) Analyze Results<font></b>",
	placeholder='Some HTML',
	description='',
	layout=Layout(height='auto', width='auto')
)

header_5 = widgets.HTML(
	value="<b><font size='3'>5) Outputs<font></b>",
	placeholder='Some HTML',
	description='',
	layout=Layout(height='auto', width='auto')
)


results_output_vent = Output() 


grid = GridspecLayout(48, 2)

grid[0,:] = tool_name

grid[1,0] = header_1
grid[2,0] = node_input
grid[3,0] = edges_input
grid[4,0] = same_knowledge_input
grid[5,0] = training_question_nodes
grid[6,0] = training_data
grid[7,0] = rbut1

grid[1,1] = header_2
grid[2,1] = numchains
grid[3,1] = tunesamples
grid[4,1] = mcmciteration
grid[5,1] = posterioriteration
grid[6,1] = maxtreedepth
grid[7,1] = earlymaxtreedepth

grid[8,:] = header_3
grid[9,:] = rbut2

grid[10,:] = header_4
grid[11,:] = rbut5
grid[12,:] = rbut5_2
grid[13,:] = rbut4
grid[14,:] = rbut6
grid[15,:] = rbut7


grid[16,:] = header_5
grid[17:45,:] = results_output_vent

display(grid)