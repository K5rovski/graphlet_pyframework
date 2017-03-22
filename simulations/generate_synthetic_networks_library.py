import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import networkx as nx
import numpy as np
import os
import sys

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean as distance
from scipy.spatial.distance import cosine
from mds import cmdscale


sys.path.append("..")
from multiplex_pygraph.Multiplex_Graph import GraphFunc


plex_corrsize={2:36,3:280,4:2160,5:5544}

def generate_barabasi_albert_graph(nodes, m,ind=0):
	graph=[]
	graph.extend([(i,j,ind) for (i,j) in nx.barabasi_albert_graph(nodes,m,seed=np.random.randint(0,1000000) ).edges()])

	graph_edges=np.array(graph)
	return graph_edges

def generate_powerlaw_graph(nodes,d,p,ind=1):
	graph=[]
	graph.extend([(i,j,ind) for (i,j) in nx.powerlaw_cluster_graph(nodes,d,p,seed=np.random.randint(0,10000000)).edges()])

	graph_edges=np.array(graph)
	return graph_edges

def generate_watts_strogatz_graph(nodes,k,p,ind=2):
	graph=[]
	graph.extend([(i,j,ind) for (i,j) in nx.watts_strogatz_graph(nodes,k,p,seed=np.random.randint(0,1000000) ).edges()])

	graph_edges=np.array(graph)
	return graph_edges

def generate_erdos_renyi_graph(nodes,p,ind=3):
	graph=[]
	graph.extend([(i,j,ind) for (i,j) in nx.erdos_renyi_graph(nodes,p,seed=np.random.randint(0,100) ).edges()])

	graph_edges=np.array(graph)
	return graph_edges

def calculate_corrmatrix(graph_edges,ind_g=0,subOne=False):
	graphF=GraphFunc(doDirected=False)
	graphF.make_graph_table(graph_edges)
	graphF.make_direct_neighbours(subOne=subOne)# : self.graphNeigh

	graphF.make_zero_orbit()
	graphF.count_tri_graphs()# : self.triGNMulti

	orb_mat=graphF.return_orbits_fast()
	orb_mat_corr=np.corrcoef(orb_mat.T)
	which_nan=np.where(np.all(np.isnan( orb_mat_corr) ,axis=0)  )[0]

	for i in which_nan:
		orb_mat_corr[i,:]=0
		orb_mat_corr[:,i]=0

	for i,j in itertools.combinations_with_replacement(which_nan,2):
		orb_mat_corr[i,j]=1
		orb_mat_corr[j,i]=1

	
	return True,orb_mat_corr

def calculate_graph_orbit_summary_distribution(graph_edges,ind_g=0,subOne=False):
	graphF=GraphFunc(doDirected=False)
	graphF.make_graph_table(graph_edges)
	graphF.make_direct_neighbours(subOne=subOne)# : self.graphNeigh
	print('made neighbor Dikt: ',ind_g,flush=True)

	graphF.make_zero_orbit()
	graphF.count_tri_graphs()# : self.triGNMulti

	print('made orbits: ',ind_g,flush=True)

	orb_mat=graphF.return_orbits_fast()

	g_2node=graphF.graphNeigh
	neig_counter={}

	for ind,row in enumerate(g_2node):
		for node_key in row:
			orb_key=row[node_key]
			neig_counter[orb_key]=neig_counter.get(orb_key,0)+1

	sum_neig=sum(neig_counter.values())
	neig_distribution=np.array([(orb_key,neig_counter[orb_key]/sum_neig) for orb_key in neig_counter])

	g_3node=graphF.triGNMulti
	node3_counter={}

	edgesC,wedge1C,wedge2C,triC=graphF.return_full_orbit_counts()

	for i,offs in zip(range(3),(0,wedge1C,wedge1C+wedge2C)):
		for ind,row in enumerate(g_3node[i]):
			for orb_key in row:
				node3_counter[(i,orb_key)]=node3_counter.get((i,orb_key),0)+1

	sum_node3=sum(node3_counter.values())
	node3_distribution=np.array([(orb_key[0],orb_key[1],node3_counter[orb_key]/sum_node3) for orb_key in node3_counter])

	return neig_distribution,node3_distribution


def save_orbits(graph_edges,filename,ind_g=0,subOne=False):
	graphF=GraphFunc(doDirected=False)
	graphF.make_graph_table(graph_edges)
	graphF.make_direct_neighbours(subOne=subOne)# : self.graphNeigh
	print('made neighbor Dikt: ',ind_g,flush=True)

	graphF.make_zero_orbit()
	graphF.count_tri_graphs()# : self.triGNMulti

	print('made orbits: ',ind_g,flush=True)

	graphF.save_orbits(filename)

def learn_embeddings_node2vec(nx_G,walks,vec_size=36, window_size=6, sgd_iter=1 ):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [map(str, walk) for walk in walks]
	model = Word2Vec(walks, size=vec_size, window_size=6, min_count=0,
	 sg=1, workers=2, sgd_iter=1)
	
	vec_mat=np.array([model[str(node)] for node in nx_G.nodes()])

	return True,vec_mat

def get_node2vec_distance_mat(vec_mat):
	dist_mat=np.zeros((vec_mat.shape[0],vec_mat.shape[0]))
	for ind,(n1,n2) in enumerate(itertools.combinations( range(vec_mat.shape[0]),2) ):
		distance_cos=cosine(vec_mat[n1],vec_mat[n2])
		dist_mat[n1,n2]=dist_mat[n2,n1]=distance_cos


def calculate_cmdscale_node2vec(graph_edges,p=1,q=1,num_walks=30,walk_length=20, **kwargs):
	nx_G=nx.Graph()
	nx_G.add_edges_from(graph_edges[:,:2])
	G = node2vec.Graph(nx_G, False, p,q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(num_walks, walk_length)
	vec_mat=learn_embeddings_node2vec(nx_G,walks,**kwargs)
	dist_mat=get_node2vec_distance_mat(vec_mat)

	Y_original,evals=cmdscale(dist_mat)
	Y=Y_original[:,:3]

	print('Y shape: ',Y.shape[0])
	return Y


def run_multiplex_simulations_pl_ba(nodes,degrees,probs,how_many_g,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,36,36),dtype=int)

	for ind,(n,d_s,p) in zip(range(how_many_g),itertools.product(nodes,degrees,probs)):
		if real_ind==break_point_g:
			break
		graph_edges_ba=generate_barabasi_albert_graph(n,d_s[0],ind=0)
		graph_edges_pl=generate_powerlaw_graph(n,d_s[1],p,ind=1)
		graph_edges=np.vstack((graph_edges_pl,graph_edges_ba))
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list
	
	
def run_multiplex_simulations_ba_ws(nodes,probs,how_many_g,ws_prob=0.01,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,36,36),dtype=int)

	for ind,(p,n) in zip(range(how_many_g),itertools.product(probs,nodes)):
		if real_ind==break_point_g:
			break
		
		d_s=int(n*p)
		graph_edges_ba_1=generate_barabasi_albert_graph(n,d_s,ind=0)
		graph_edges_ws_2=generate_watts_strogatz_graph(n,d_s,ws_prob,ind=1)
		
		graph_edges=np.vstack((graph_edges_ba_1,graph_edges_ws_2))
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		

		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list	

def run_multiplex_simulations_2ba(nodes,degrees,probs,how_many_g,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,36,36),dtype=int)

	for ind,(n,d_s,p) in zip(range(how_many_g),itertools.product(nodes,degrees,probs)):
		if real_ind==break_point_g:
			break
		graph_edges_ba_1=generate_barabasi_albert_graph(n,d_s[0],ind=0)
		graph_edges_ba_2=generate_barabasi_albert_graph(n,d_s[0],ind=1)
		graph_edges=np.vstack((graph_edges_ba_1,graph_edges_ba_2))
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		

		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list

def run_singleplex_simulations_ba(nodes,probs,how_many_g,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,4,4),dtype=int)

	for ind,(p,n) in zip(range(how_many_g),itertools.product(probs,nodes)):
		if real_ind==break_point_g:
			break
		
		d_s=int(n*p)
		graph_edges_ba_1=generate_barabasi_albert_graph(n,d_s,ind=0)
		
		status,corr=calculate_corrmatrix(graph_edges_ba_1, real_ind)
		

		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list	
	
def run_multiplex_simulations_2ba_new(nodes,probs,how_many_g,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,36,36),dtype=int)

	for ind,(p,n) in zip(range(how_many_g),itertools.product(probs,nodes)):
		if real_ind==break_point_g:
			break
		
		d_s=int(n*p)
		graph_edges_ba_1=generate_barabasi_albert_graph(n,d_s,ind=0)
		graph_edges_ba_2=generate_barabasi_albert_graph(n,d_s,ind=1)
		graph_edges=np.vstack((graph_edges_ba_1,graph_edges_ba_2))
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		

		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list	

def run_multiplex_simulations_more_ba(nodes,probs,how_many_g,numPlex=3,break_point_g=100):
	real_ind=0
	if numPlex not in plex_corrsize:
		raise Exception('not supported plex number')
		
	corr_list=np.zeros((0,plex_corrsize[numPlex],plex_corrsize[numPlex]),dtype=int)

	for ind,(p,n) in zip(range(how_many_g),itertools.product(probs,nodes)):
		if real_ind==break_point_g:
			break
		
		d_s=int(p*n)
		graph_edges_ba=[ generate_barabasi_albert_graph(n,d_s,ind=pl_ind) for pl_ind in range(numPlex)]
		
		graph_edges=np.vstack(graph_edges_ba)
		status,corr=calculate_corrmatrix(graph_edges, real_ind)


		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list	

def run_multiplex_simulations_2er(nodes,degrees,probs,how_many_g,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,36,36),dtype=int)

	for ind,(n,p) in zip(range(how_many_g),itertools.product(nodes,probs)):
		if real_ind==break_point_g:
			break
		graph_edges_er_1=generate_erdos_renyi_graph(n,p,ind=0)
		graph_edges_er_2=generate_erdos_renyi_graph(n,p,ind=1)
		graph_edges=np.vstack((graph_edges_er_1,graph_edges_er_2))
		status,corr=calculate_corrmatrix(graph_edges, real_ind)

		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {}, p: {}'\
				.format(n,p))
			real_ind+=1

	return real_ind,corr_list

def run_singleplex_simulations_er(nodes,probs,how_many_g,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,4,4),dtype=int)

	for ind,(p,n) in zip(range(how_many_g),itertools.product(probs,nodes)):
		if real_ind==break_point_g:
			break
		graph_edges_er_1=generate_erdos_renyi_graph(n,p,ind=0)

		status,corr=calculate_corrmatrix(graph_edges_er_1, real_ind)

		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {}, p: {}'\
				.format(n,p))
			real_ind+=1

	return real_ind,corr_list
	
def run_multiplex_simulations_2er_new(nodes,probs,how_many_g,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,36,36),dtype=int)

	for ind,(p,n) in zip(range(how_many_g),itertools.product(probs,nodes)):
		if real_ind==break_point_g:
			break
		graph_edges_er_1=generate_erdos_renyi_graph(n,p,ind=0)
		graph_edges_er_2=generate_erdos_renyi_graph(n,p,ind=1)
		graph_edges=np.vstack((graph_edges_er_1,graph_edges_er_2))
		status,corr=calculate_corrmatrix(graph_edges, real_ind)

		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {}, p: {}'\
				.format(n,p))
			real_ind+=1

	return real_ind,corr_list
	
def run_multiplex_simulations_more_er(nodes,probs,how_many_g,numPlex=3,break_point_g=100):
	real_ind=0
	if numPlex not in plex_corrsize:
		raise Exception('not supported plex number')
		
	corr_list=np.zeros((0,plex_corrsize[numPlex],plex_corrsize[numPlex]),dtype=int)

	for ind,(p,n) in zip(range(how_many_g),itertools.product(probs,nodes)):
		if real_ind==break_point_g:
			break
		graph_edges_er=[generate_erdos_renyi_graph(n,p,ind=pl_ind) for pl_ind in  range(numPlex)]
		
		graph_edges=np.vstack(graph_edges_er)
		status,corr=calculate_corrmatrix(graph_edges, real_ind)

		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {}, p: {}'\
				.format(n,p))
			real_ind+=1

	return real_ind,corr_list

def run_multiplex_simulations_2pl(nodes,degrees,probs,how_many_g,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,36,36),dtype=int)

	for ind,(n,d_s,p) in zip(range(how_many_g),itertools.product(nodes,degrees,probs)):
		if real_ind==break_point_g:
			break
		graph_edges_pl_1=generate_powerlaw_graph(n,d_s[0],p,ind=0)
		graph_edges_pl_2=generate_powerlaw_graph(n,d_s[0],p,ind=1)
		graph_edges=np.vstack((graph_edges_pl_1,graph_edges_pl_2))
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list

def run_singleplex_simulations_pl(nodes,probs,how_many_g,tri_prob=0.6,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,4,4),dtype=int)

	for ind,(p,n) in zip(range(how_many_g),itertools.product(probs,nodes)):
		if real_ind==break_point_g:
			break
		d_s=int(p*n)
		graph_edges_pl_1=generate_powerlaw_graph(n,d_s,tri_prob,ind=0)
		
		status,corr=calculate_corrmatrix(graph_edges_pl_1, real_ind)
		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list
	
def run_multiplex_simulations_2pl_new(nodes,probs,how_many_g,tri_prob=0.6,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,36,36),dtype=int)

	for ind,(p,n) in zip(range(how_many_g),itertools.product(probs,nodes)):
		if real_ind==break_point_g:
			break
		d_s=int(p*n)
		graph_edges_pl_1=generate_powerlaw_graph(n,d_s,tri_prob,ind=0)
		graph_edges_pl_2=generate_powerlaw_graph(n,d_s,tri_prob,ind=1)
		graph_edges=np.vstack((graph_edges_pl_1,graph_edges_pl_2))
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list

	
def run_multiplex_simulations_more_pl(nodes,probs,how_many_g,tri_prob=0.6,numPlex=3,break_point_g=100):
	real_ind=0
	if numPlex not in plex_corrsize:
		raise Exception('not supported plex number')
		
	corr_list=np.zeros((0,plex_corrsize[numPlex],plex_corrsize[numPlex]),dtype=int)

	for ind,(p,n) in zip(range(how_many_g),itertools.product(probs,nodes)):
		if real_ind==break_point_g:
			break
			
		d_s=int(p*n)
		graph_edges_pl=[generate_powerlaw_graph(n,d_s,tri_prob,ind=pl_ind) for pl_ind in range(numPlex)]
		
		graph_edges=np.vstack(graph_edges_pl)
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list

def run_multiplex_simulations_2ws(nodes,degrees,probs,how_many_g,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,36,36),dtype=int)

	for ind,(n,d_s,p) in zip(range(how_many_g),itertools.product(nodes,degrees,probs)):
		if real_ind==break_point_g:
			break
		graph_edges_ws_1=generate_watts_strogatz_graph(n,d_s[0],p,ind=0)
		graph_edges_ws_2=generate_watts_strogatz_graph(n,d_s[0],p,ind=1)
		graph_edges=np.vstack((graph_edges_ws_1,graph_edges_ws_2))
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list

def run_singleplex_simulations_ws(nodes,probs,how_many_g,ws_prob=0.01,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,4,4),dtype=int)

	for ind,(p,n) in zip(range(how_many_g),itertools.product(probs,nodes)):
		if real_ind==break_point_g:
			break
		d_s=int(p*n)
		graph_edges_ws_1=generate_watts_strogatz_graph(n,d_s,ws_prob,ind=0)
		

		status,corr=calculate_corrmatrix(graph_edges_ws_1, real_ind)
		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list
	
def run_multiplex_simulations_2ws_new(nodes,probs,how_many_g,ws_prob=0.01,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,36,36),dtype=int)

	for ind,(p,n) in zip(range(how_many_g),itertools.product(probs,nodes)):
		if real_ind==break_point_g:
			break
		d_s=int(p*n)
		graph_edges_ws_1=generate_watts_strogatz_graph(n,d_s,ws_prob,ind=0)
		graph_edges_ws_2=generate_watts_strogatz_graph(n,d_s,ws_prob,ind=1)
		graph_edges=np.vstack((graph_edges_ws_1,graph_edges_ws_2))
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list
	
def run_multiplex_simulations_more_ws(nodes,probs,how_many_g,ws_prob=0.01,numPlex=3,break_point_g=100):
	real_ind=0
	if numPlex not in plex_corrsize:
		raise Exception('not supported plex number')
		
	corr_list=np.zeros((0,plex_corrsize[numPlex],plex_corrsize[numPlex]),dtype=int)

	for ind,(p,n) in zip(range(how_many_g),itertools.product(probs,nodes)):
		if real_ind==break_point_g:
			break
		
		d_s=int(p*n)
		graph_edges_ws=[generate_watts_strogatz_graph(n,d_s,ws_prob,ind=pl_ind) for pl_ind in range(numPlex)]
		
		graph_edges=np.vstack(graph_edges_ws)
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list

def run_multiplex_simulations_singleplex_pl(nodes,probs,how_many_g,tri_prob=0.6,howManyPlexes=2,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,4,4),dtype=int)

	for ind,(p,n) in zip(range(how_many_g),itertools.product(probs,nodes)):
		if real_ind==break_point_g:
			break
		
		d_s=int(n*p)
		graph_edges_pl=[generate_powerlaw_graph(n,d_s,tri_prob,ind=0) for ind_plex in range(howManyPlexes)]
	
		
		graph_edges=np.vstack(graph_edges_pl)
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list
	

def run_multiplex_simulations_singleplex_er(nodes,probs,how_many_g,howManyPlexes=2,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,4,4),dtype=int)

	for ind,(p,n) in zip(range(how_many_g),itertools.product(probs,nodes)):
		if real_ind==break_point_g:
			break
		
		
		graph_edges_er=[generate_erdos_renyi_graph(n,p,ind=0) for ind_plex in range(howManyPlexes)]
		
		graph_edges=np.vstack(graph_edges_er)
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list

def run_multiplex_simulations_singleplex_ws(nodes,probs,how_many_g,ws_prob=0.01,break_point_g=100,howManyPlexes=2):
	real_ind=0
	corr_list=np.zeros((0,4,4),dtype=int)

	for ind,(p,n) in zip(range(how_many_g),itertools.product(probs,nodes)):
		if real_ind==break_point_g:
			break
		d_s=int(p*n)
		
		graph_edges_ws=[generate_watts_strogatz_graph(n,d_s,ws_prob,ind=0) for ind_plex in range(howManyPlexes)]
		
		graph_edges=np.vstack(graph_edges_ws)
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		

		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list
	
def run_multiplex_simulations_singleplex_ba(nodes,probs,how_many_g,break_point_g=100,howManyPlexes=2):
	real_ind=0
	corr_list=np.zeros((0,4,4),dtype=int)

	for ind,(p,n) in zip(range(how_many_g),itertools.product(probs,nodes)):
		if real_ind==break_point_g:
			break
		d_s=int(p*n)
		graph_edges_ba=[generate_barabasi_albert_graph(n,d_s,ind=0) for ind_plex in range(howManyPlexes)]
		
		
		graph_edges=np.vstack(graph_edges_ba)
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		

		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list

def run_multiplex_simulations_singleplex_pl_ba(nodes,degrees,probs,how_many_g,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,4,4),dtype=int)

	for ind,(n,d_s,p) in zip(range(how_many_g),itertools.product(nodes,degrees,probs)):
		if real_ind==break_point_g:
			break
		graph_edges_ba=generate_barabasi_albert_graph(n,d_s[0],ind=0)
		graph_edges_pl=generate_powerlaw_graph(n,d_s[1],p,ind=0)
		graph_edges=np.vstack((graph_edges_pl,graph_edges_ba))
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list

def run_multiplex_simulations_cloud_size_2ba(nodes,degrees,probs,how_many_g,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,36,36),dtype=int)

	for ind,n,d_s,p in zip(range(how_many_g),nodes,degrees,probs):
		if real_ind==break_point_g:
			break
		graph_edges_ba_1=generate_barabasi_albert_graph(n,d_s[0],ind=0)
		graph_edges_ba_2=generate_barabasi_albert_graph(n,d_s[1],ind=1)
		graph_edges=np.vstack((graph_edges_ba_1,graph_edges_ba_2))
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		

		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list

def run_multiplex_simulations_cloud_size_2er(nodes,probs,how_many_g,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,36,36),dtype=int)

	for ind,n,p in zip(range(how_many_g),nodes,probs):
		if real_ind==break_point_g:
			break
		graph_edges_er_1=generate_erdos_renyi_graph(n,p,ind=0)
		graph_edges_er_2=generate_erdos_renyi_graph(n,p,ind=1)
		graph_edges=np.vstack((graph_edges_er_1,graph_edges_er_2))
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		# print('Not made')
		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {}, p: {}'\
				.format(n,p))
			real_ind+=1

	return real_ind,corr_list

def run_multiplex_simulations_cloud_size_2pl(nodes,degrees,probs,how_many_g,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,36,36),dtype=int)

	for ind,n,d_s,p in zip(range(how_many_g),nodes,degrees,probs):
		if real_ind==break_point_g:
			break
		graph_edges_pl_1=generate_powerlaw_graph(n,d_s[0],p,ind=0)
		graph_edges_pl_2=generate_powerlaw_graph(n,d_s[1],p,ind=1)
		graph_edges=np.vstack((graph_edges_pl_1,graph_edges_pl_2))
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list

def run_multiplex_simulations_cloud_size_2ws(nodes,degrees,probs,how_many_g,break_point_g=100):
	real_ind=0
	corr_list=np.zeros((0,36,36),dtype=int)

	for ind,n,d_s,p in zip(range(how_many_g),nodes,degrees,probs):
		if real_ind==break_point_g:
			break
		graph_edges_ws_1=generate_watts_strogatz_graph(n,d_s[0],p,ind=0)
		graph_edges_ws_2=generate_watts_strogatz_graph(n,d_s[1],p,ind=1)
		graph_edges=np.vstack((graph_edges_ws_1,graph_edges_ws_2))
		status,corr=calculate_corrmatrix(graph_edges, real_ind)
		if status:
			corr_list=np.vstack((corr_list,corr.reshape(1, corr.shape[0],-1) ))
			print('made Graph with nodes: {},d_s: {}, p: {}'\
				.format(n,d_s,p))
			real_ind+=1

	return real_ind,corr_list


def run_multiplex_simulations_node2vec_2ba(nodes,degrees,probs,how_many_g,
		break_point_g=100,vec_size=36,**kwargs):
	real_ind=0
	Y_list=np.zeros((0,vec_size))

	for ind,(n,p) in zip(range(how_many_g),itertools.product(nodes,probs)):
		if real_ind==break_point_g:
			break

		graph_edges_ba_1=generate_barabasi_albert_graph(n,d_s[0],ind=0)
		graph_edges_ba_2=generate_barabasi_albert_graph(n,d_s[1],ind=0)
		graph_edges=np.vstack((graph_edges_ba_1,graph_edges_ba_2))

		Y_single=calculate_cmdscale_node2vec(graph_edges,**kwargs)

		Y_list=np.vstack((Y_list,Y_single  ))

		real_ind+=1

	return real_ind,Y_list

def run_multiplex_simulations_node2vec_2er(nodes,degrees,probs,how_many_g,
		break_point_g=100,vec_size=36,**kwargs):
	real_ind=0
	Y_list=np.zeros((0,vec_size))

	for ind,(n,p) in zip(range(how_many_g),itertools.product(nodes,probs)):
		if real_ind==break_point_g:
			break
		graph_edges_er_1=generate_erdos_renyi_graph(n,p,ind=0)
		graph_edges_er_2=generate_erdos_renyi_graph(n,p,ind=0)
		graph_edges=np.vstack((graph_edges_er_1,graph_edges_er_2))

		Y_single=calculate_cmdscale_node2vec(graph_edges,**kwargs)

		Y_list=np.vstack((Y_list,Y_single  ))

		real_ind+=1

	return real_ind,Y_list

def run_multiplex_simulations_node2vec_2pl(nodes,degrees,probs,how_many_g,
		break_point_g=100,vec_size=36,**kwargs):
	real_ind=0
	Y_list=np.zeros((0,vec_size))

	for ind,(n,p) in zip(range(how_many_g),itertools.product(nodes,probs)):
		if real_ind==break_point_g:
			break
		
		graph_edges_pl_1=generate_powerlaw_graph(n,d_s[0],p,ind=0)
		graph_edges_pl_2=generate_powerlaw_graph(n,d_s[1],p,ind=0)
		graph_edges=np.vstack((graph_edges_pl_1,graph_edges_pl_2))

		Y_single=calculate_cmdscale_node2vec(graph_edges,**kwargs)

		Y_list=np.vstack((Y_list,Y_single  ))

		real_ind+=1

	return real_ind,Y_list


def run_multiplex_simulations_node2vec_2pl(nodes,degrees,probs,how_many_g,
		break_point_g=100,vec_size=36,**kwargs):
	real_ind=0
	Y_list=np.zeros((0,vec_size))

	for ind,(n,p) in zip(range(how_many_g),itertools.product(nodes,probs)):
		if real_ind==break_point_g:
			break
		
		graph_edges_ws_1=generate_watts_strogatz_graph(n,d_s[0],p,ind=0)
		graph_edges_ws_2=generate_watts_strogatz_graph(n,d_s[1],p,ind=0)
		graph_edges=np.vstack((graph_edges_ws_1,graph_edges_ws_2))


		Y_single=calculate_cmdscale_node2vec(graph_edges,**kwargs)

		Y_list=np.vstack((Y_list,Y_single  ))

		real_ind+=1

	return real_ind,Y_list	

def get_3D_projection_from_corrs(big_corr_list):
	distance_c=np.zeros((big_corr_list.shape[0],big_corr_list.shape[0]))

	for ind,(c1,c2) in enumerate(itertools.combinations(range(big_corr_list.shape[0]),2)):
		c1t=np.triu(big_corr_list[c1]).flatten()
		c2t=np.triu(big_corr_list[c2]).flatten()
		if c1>c2:
			c1,c2=c2,c1

		distance_c[c1,c2]=distance(c1t,c2t)
		distance_c[c2,c1]=distance_c[c1,c2]


	Y_original,evals=cmdscale(distance_c)
	Y=Y_original[:,:3]

	print('Y shape: ',Y.shape[0])
	return Y

def draw_3D_graphs(Y,cmap_colors,save_name='3D_graphs.plot',doshow=True,doSave=False,subplot_ind=(1,1,1),fig=None,graphShape='*',starSize=100):
	if doSave and os.path.exists(save_name): os.remove(save_name)

	print('Cmap len: ',cmap_colors.shape[0])

	if fig is None: fig = plt.figure()

	ax = fig.add_subplot(*subplot_ind, projection='3d')
	ax.scatter(Y[:,0],Y[:,1],Y[:,2],marker=graphShape,s=starSize,c=cmap_colors)
	
	if doSave and not os.path.exists(save_name):
		with open(save_name,'wb') as f:
			pickle.dump((Y,cmap_colors),f)
		print('saved ploting infos')
	
	if doshow: plt.show()

	print('Done, saved: ',save_name,flush=True)
	return fig

def get_colors(color_c=3,color_step=100):
	cmap_colors=np.vstack((
		cm.Oranges(np.linspace(0.4,1,color_step)),
		cm.Reds(np.linspace(0.4,1,color_step)),
		cm.Greys(np.linspace(0.4,1,color_step)),
		
		cm.Purples(np.linspace(0.4,1,color_step)),
		cm.Blues(np.linspace(0.4,1,color_step)),
		cm.Greens(np.linspace(0.4,1,color_step)),
		
		cm.pink(np.linspace(0.4,1,color_step)),
		cm.copper(np.linspace(0.4,1,color_step)),
	))
	return cmap_colors[np.arange(color_c*color_step)%(color_step*  8)]



def get_single_color(color_ind=3,color_step=100,starting=0.4):
	cmap_colors=[
		cm.Oranges,
		cm.Blues,
		
		cm.Purples,
		cm.Greens,

		cm.Reds,
		cm.Greys,
		
		cm.copper,
		cm.pink,
	]
	# print(help(cmap_colors[color_ind%8 ]))
	return cmap_colors[color_ind%8 ](np.linspace(starting,1,color_step))


def summarize_graph_orbitDikt(graph_edges,ind_g=0,subOne=False):
    graphF=GraphFunc(doDirected=False)
    graphF.make_graph_table(graph_edges)
    graphF.make_direct_neighbours(subOne=subOne)# : self.graphNeigh

    graphF.make_zero_orbit()
    graphF.count_tri_graphs()# : self.triGNMulti

    orbitDikt=graphF.return_orbits_dikt()
    summary_orbitDikt={}
    for node in orbitDikt:
        for orbit_key in orbitDikt[node]:
            orbit_val=orbitDikt[node][orbit_key]

            summary_orbitDikt[orbit_key]=summary_orbitDikt.get(orbit_key,0)+orbit_val

    return summary_orbitDikt

def add_graph_orbitDikts(orb1,orb2):
    sumary_orb={}
    sumary_orb.update(orb1)
    for key2 in orb2:
        if key2 in sumary_orb:
            sumary_orb[key2]+=orb2[key2]
        else:
            sumary_orb[key2]=orb2[key2]
    return sumary_orb

def draw_correlation_matrix(graph_edges,subOne=False,ind_g=0):
    graphF=GraphFunc(doDirected=False)
    graphF.make_graph_table(graph_edges)
    graphF.make_direct_neighbours(subOne=subOne)# : self.graphNeigh
    print('made neighbor Dikt: ',ind_g,flush=True)

    graphF.make_zero_orbit()
    graphF.count_tri_graphs()# : self.triGNMulti

    print('made orbits: ',ind_g,flush=True)

    orb_mat=graphF.return_orbits_fast()
    print('!!! orb mat size',orb_mat.shape)
    orb_mat_corr=np.corrcoef(orb_mat.T)
    which_nan=np.where(np.all(np.isnan( orb_mat_corr) ,axis=0)  )[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('PRGn', 30)
    labels=np.arange(orb_mat_corr.shape[0])
    ax1.set_xticks(labels)
    ax1.set_yticks(labels)
    cax = ax1.imshow(orb_mat_corr, interpolation="nearest", cmap=cmap)
    plt.title('Graph')
    
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=np.linspace(-1,0,1))
    # cbar.ax.set_yticklabels(['1','0','-1'])
    plt.show()
	
	
def iterate_multiple_configs(calc_synt_corrs,synt_type,nodes,probs,n_instances,corr_list):
	
	print('Calculating for Syntetic graph type: ',synt_type)
	for i in range(n_instances):
		print('instance num: {} of {}'.format(i+1,n_instances))
		
		real_ind,inner_corr_list = calc_synt_corrs[synt_type](*(nodes,probs))	

		corr_list=np.vstack((corr_list,inner_corr_list))
			

	return corr_list