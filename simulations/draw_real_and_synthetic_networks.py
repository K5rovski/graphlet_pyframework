import numpy as np
import pandas as pd
import os
import re
import time
import sys
import pandas as pd 
import pickle
import random
import matplotlib.pyplot as plt

from functools import partial
from scipy.cluster.hierarchy import dendrogram, linkage,cut_tree
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from generate_synthetic_networks_library import *
from scipy.spatial.distance import euclidean as distance




def remove_percent_from_mean(big_corrs,perc_remove=5):
	mean_cor=np.mean(big_corrs,axis=0)
	
	reduced_bymean=np.sum(np.abs(big_corrs-mean_cor).reshape((-1,big_corrs.shape[1]*big_corrs.shape[2])) ,axis=1) 
	
	
	low_bound,high_bound=np.percentile(reduced_bymean,(perc_remove,100-perc_remove))
	ret_mask=(reduced_bymean>low_bound) & (reduced_bymean<high_bound)
	
	
	# print('There are now: ',np.sum(ret_mask),low_bound,high_bound,perc_remove,mean_cor.shape)
	return big_corrs[ret_mask]
	

random.seed(8)
np.random.seed(8)

# -----------------------------------------------------
# Preparing for ploting
# ----------------------------------------------------

with open('results/economic_100bigcombs.cors','rb') as f:
	econ_cors=pickle.load(f)

with open('results/social_full.cors','rb') as f:
	soc_cors=pickle.load(f)

	
with open('results/3D_graphs_biplex.corr_list','rb') as f:
	synth_cors=pickle.load(f)


np.random.shuffle(econ_cors)
np.random.shuffle(soc_cors)

econ_cors=econ_cors[:1000]
soc_cors=soc_cors[:1000]

econ_cors=remove_percent_from_mean(econ_cors,5)
soc_cors=remove_percent_from_mean(soc_cors,5)



# Saving the plotfile---------------------------------------------
save_graph_name='results/full_plot.plot'
doSave=True

# ------------------------------------------------------------
# COLORS
# -----------------------------------------------------------

n_types=25
n_iters=1	
cmap_colors=np.vstack((

cm.Greens(np.linspace(0.4,1,econ_cors.shape[0])),
cm.cool(np.linspace(0.0,0.5,soc_cors.shape[0])),

cm.Blues( np.tile( np.linspace(0.4,1,n_types) ,n_iters) ),
cm.Purples(np.tile( np.linspace(0.4,1,n_types) ,n_iters) ),
cm.Oranges(np.tile( np.linspace(0.4,1,n_types) ,n_iters) ),
cm.Greys(np.tile( np.linspace(0.4,1,n_types) ,n_iters) ),

))

# -------------------------------------------------------

corr_list=np.zeros(( econ_cors.shape[0]+soc_cors.shape[0]+synth_cors.shape[0] 
					,econ_cors.shape[1],econ_cors.shape[2] ))

					
corr_list[:econ_cors.shape[0]]=econ_cors
corr_list[econ_cors.shape[0]:econ_cors.shape[0]+soc_cors.shape[0] ]=soc_cors
corr_list[econ_cors.shape[0]+soc_cors.shape[0]:]=synth_cors


print('Economic networks: {}, Social networks: {}, Synthetic networks: {}'\
		.format(econ_cors.shape[0],soc_cors.shape[0],synth_cors.shape[0] ))

distance_c=np.zeros((corr_list.shape[0],corr_list.shape[0]))
for ind,(c1,c2) in enumerate(itertools.combinations(range(corr_list.shape[0]),2)):
	c1t=np.triu(corr_list[c1]).flatten()
	c2t=np.triu(corr_list[c2]).flatten()
	if c1>c2:
		c1,c2=c2,c1

	distance_c[c1,c2]=distance(c1t,c2t)
	distance_c[c2,c1]=distance_c[c1,c2]

# from the upper triangle of the distance matrix with multidimensional scaling 
Y,evals=cmdscale(distance_c)
Y=Y[:,:3]





draw_3D_graphs(Y,cmap_colors,save_name=save_graph_name,
	doSave=doSave,doshow=True,subplot_ind=(1,1,1),fig=None,graphShape='o',starSize=20)



