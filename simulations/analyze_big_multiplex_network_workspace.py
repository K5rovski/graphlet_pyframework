import numpy as np
import pandas as pd
import os
import re
import time
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
from analyze_big_multiplex_network_library import *

random.seed(8)
np.random.seed(8)



# ==============================================================================
# ECONOMIC
# ==============================================================================


with open('../multiplex_data/economic.2productcode','rb') as f:
	prod2_list=pickle.load(f)
	
	
plexes_preps={}

plexes_preps[2]=[';'.join((a,b)) for a,b in itertools.combinations(prod2_list,2) ]
plexes_preps[3]=[';'.join((a,b,c)) for a,b,c in  itertools.combinations(prod2_list,3) ]


random.shuffle(plexes_preps[2])
random.shuffle(plexes_preps[3])


print('economic big networks before removing: ',len(plexes_preps[2]))


econ_rel_limit=100
doDraw3D=False
doPositiveCorrs=True
doSaveCorrMap=True
SampleLimit=2000




saved_countmaps_file='results/economic_100bigcombs.countmaps'
saved_corrmaps_file='results/economic_100bigcombs.corrmaps'
indv_combs_id='indivudual_class_combinations_{}'


# this saves the correlation matrices to diplay latter,
# you can choose the number N saved (N,corr_height,corr_width)
save_2corrs=np.zeros((16100,36,36))-100


# ----------------------------------------------------------------------
# This are variables specific to the network type, this signifies whether
 # we obtain the correlation and countmaps by summarizing the sampled individual products from product class networks (economic case)
 # we obtain the correlation and countmaps by summarizing specific village networks (social case)
 
net_filename='../multiplex_data/4digit_ITN.csv'
net_type='economic'
# ------------------------------------------------------------------

# corrmap dimensions for different plexes
corr_dim={2:36,3:280,4:2160,5:5544}


# Removing some of the big economic networks
plexes_preps[2]='\n'.join(plexes_preps[2][:econ_rel_limit])
plexes_preps[3]='\n'.join(plexes_preps[3][:econ_rel_limit])

# just 2-plex networks
del plexes_preps[3]

saved_corrmaps={
'summary_by_class_comb_combination':[],
'final_summary':[],
'unsummarized_maps':[],

}
saved_countmaps={
'final_summary':[],

}

# -----------------------------------------------------


start_time=time.time()

run_many_multiplex_combs(net_filename,net_type,plexes_preps, 
	saved_corrmaps,saved_corrmaps_file,saved_countmaps,saved_countmaps_file,corr_dim,
	indv_combs_id,itertools.product,doDraw3D,doSaveCorrMap,doPositiveCorrs,subSample=SampleLimit,save_2corrs=save_2corrs)

if save_2corrs is not None:
	with open('{}.cors'.format(saved_corrmaps_file[:saved_corrmaps_file.rindex('.')]) ,'wb') as f:
		pickle.dump(save_2corrs,f)

	
print('Analysis took: ',time.time()-start_time,' secs.')





# ==============================================================================
# SOCIAL
# ==============================================================================




all_soc_plexes=['borrowmoney',  'giveadvice',  'helpdecision',  'keroricecome',  
'keroricego',  'lendmoney',  'medic',  'nonrel',  'rel',  
'templecompany',  'visitcome',  'visitgo']


plexes_preps={}



plexes_preps[2]=[';'.join((a,b)) for a,b in itertools.combinations(all_soc_plexes,2) ]
plexes_preps[3]=[';'.join((a,b,c)) for a,b,c in  itertools.combinations(all_soc_plexes,3) ]

random.shuffle(plexes_preps[2])
random.shuffle(plexes_preps[3])





soc_rel_limit=100
doDraw3D=False
doPositiveCorrs=True
doSaveCorrMap=True


saved_countmaps_file='results/social_full.countmaps'
saved_corrmaps_file='results/social_full.corrmaps'
indv_combs_id='indivudual_social_class_combinations_{}'


# this saves the correlation matrices to diplay latter,
# you can choose the number N saved (N,corr_height,corr_width)
save_2corrs=np.zeros((12100,36,36))-100



# ----------------------------------------------------------------------
# This are variables specific to the network type, this signifies whether
 # we obtain the correlation and countmaps by summarizing the sampled individual products from product class networks (economic case)
 # we obtain the correlation and countmaps by summarizing specific village networks (social case)
 
net_filename='../multiplex_data/indian_social_summary_edges.csv'
net_type='social'
# ------------------------------------------------------------------

# corrmap dimensions for different plexes
corr_dim={2:36,3:280,4:2160,5:5544}


# Removing some of the big social networks
plexes_preps[2]='\n'.join(plexes_preps[2][:soc_rel_limit])
plexes_preps[3]='\n'.join(plexes_preps[3][:soc_rel_limit])

# just 2-plex networks
del plexes_preps[3]

saved_corrmaps={
'summary_by_class_comb_combination':[],
'final_summary':[],
'unsummarized_maps':[],

}
saved_countmaps={
'final_summary':[],

}


# -------------------------------------------------


start_time=time.time()

run_many_multiplex_combs(net_filename,net_type,plexes_preps, 
	saved_corrmaps,saved_corrmaps_file,saved_countmaps,saved_countmaps_file,corr_dim,indv_combs_id,zip,doDraw3D,doSaveCorrMap,doPositiveCorrs,save_2corrs=save_2corrs)

if save_2corrs is not None:
	with open('{}.cors'.format(saved_corrmaps_file[:saved_corrmaps_file.rindex('.')]) ,'wb') as f:
		pickle.dump(save_2corrs,f)
	
print('Analysis took: ',time.time()-start_time,' secs.')


