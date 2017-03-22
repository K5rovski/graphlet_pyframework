
import numpy as np
from functools import partial
from generate_synthetic_networks_library import *
import pickle

saved_name='results/3D_graphs_biplex.plot'

with open(saved_name,'rb') as f:
	Y,cmap_colors=pickle.load(f)
	

n_types=25
n_iters=1	

cmap_colors=np.vstack((


cm.Blues( np.tile( np.linspace(0.4,1,n_types) ,n_iters) ),
cm.Purples(np.tile( np.linspace(0.4,1,n_types) ,n_iters) ),
cm.Oranges(np.tile( np.linspace(0.4,1,n_types) ,n_iters) ),
cm.Greys(np.tile( np.linspace(0.4,1,n_types) ,n_iters) ),


))


draw_3D_graphs(Y,cmap_colors,doSave=False,doshow=True,
	subplot_ind=(1,1,1),fig=None,graphShape='o',starSize=20)