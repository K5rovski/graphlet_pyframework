'''

'''
import numpy as np
from functools import partial
from generate_synthetic_networks_library import *




# ========================================================================
# ========================================================================
# ========================================================================
	
nodes=np.linspace(100,500,5,dtype=int)#[100,200,300,400,500]
probs=np.linspace(0.2,0.8,5)#[0.2,0.35,0.5,0.75,0.8]

tri_prob=0.8
ws_prob=0.01





n_instances=1

howManyFlatten=2
howManyPlex=2

n_configs=len(nodes)*len(probs)
howManyG=n_configs#len(nodes)*len(degrees)*len(degrees_tri)


calc_synt_corrs_more={
'erdos_renyi' : partial(run_multiplex_simulations_more_er,how_many_g=howManyG,numPlex=howManyPlex,break_point_g=n_configs),
'watts_strogatz' : partial(run_multiplex_simulations_more_ws,ws_prob=ws_prob,numPlex=howManyPlex,how_many_g=howManyG,break_point_g=n_configs),
'barabasi_albert' : partial(run_multiplex_simulations_more_ba,numPlex=howManyPlex,how_many_g=howManyG,break_point_g=n_configs),
'power_law': partial(run_multiplex_simulations_more_pl,numPlex=howManyPlex,tri_prob=tri_prob,how_many_g=howManyG,break_point_g=n_configs),
}

calc_synt_corrs_flat={
'erdos_renyi' : partial(run_multiplex_simulations_singleplex_pl,howManyPlexes=howManyFlatten, how_many_g=howManyG,break_point_g=n_configs),
'watts_strogatz' : partial(run_multiplex_simulations_singleplex_ws,howManyPlexes=howManyFlatten, ws_prob=ws_prob,how_many_g=howManyG,break_point_g=n_configs),
'barabasi_albert' : partial(run_multiplex_simulations_singleplex_ba,howManyPlexes=howManyFlatten,how_many_g=howManyG,break_point_g=n_configs),
'power_law': partial(run_multiplex_simulations_singleplex_pl,howManyPlexes=howManyFlatten,tri_prob=tri_prob,how_many_g=howManyG,break_point_g=n_configs),
}




save_graph_name='results/3D_graphs_biplex.plot'
doSave=True

calc_synt_corr_chosen=calc_synt_corrs_more

corr_dim={1:4,2:36,3:280,4:2160,5:5544}
corr_list=np.zeros((0,corr_dim[2],corr_dim[2] )) # 0,4,4 for flattened case, otherwise use corr_dim

# ===========================================================================================
# ===========================================================================================
# ===========================================================================================




sorted_synths=sorted(list(calc_synt_corr_chosen.keys()))
for synt_type in sorted_synths:
	corr_list=iterate_multiple_configs(calc_synt_corr_chosen,synt_type,nodes,probs,n_instances,corr_list)




if doSave:
	with open(save_graph_name[:save_graph_name.rindex('.')]+'.corr_list','wb') as f:
		pickle.dump(corr_list,f)
	
	
	
# get distance matrix for all graphs that were generated 
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



cmap_colors=np.vstack((
cm.Blues(np.linspace(0.4,1,n_instances*n_configs)),
cm.Purples(np.linspace(0.4,1,n_instances*n_configs)),
cm.Oranges(np.linspace(0.4,1,n_instances*n_configs)),
cm.Greys(np.linspace(0.4,1,n_instances*n_configs)),
))


	

print('Drawing these graph types: ',sorted_synths)


draw_3D_graphs(Y,cmap_colors,save_name=save_graph_name,doSave=doSave,doshow=True,
	subplot_ind=(1,1,1),fig=None,graphShape='o',starSize=30)