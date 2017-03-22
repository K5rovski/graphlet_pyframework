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


random.seed(8)
np.random.seed(8)

def select_plexes_multiplex_net(read_graph,selected_plex_func,
	selected_plex_ind_func,layer_col=2,do_aggregate=False):

	if type(read_graph)==str:
		raw_graph=pd.read_csv(read_graph,delimiter=',',skiprows=0)
	else:
		raw_graph=read_graph


	howMany=raw_graph.shape[0]


	conv_graph=np.zeros((howMany,3),dtype=int)
	convDikt={}
	ind_node=0
	ind=0

	convEDikt={}
	ind_layer=0
	for (indxx,i,j,l) in raw_graph.itertuples():
		if not selected_plex_func(l):
			continue


		if i not in convDikt:
			convDikt[i]=ind_node
			ind_node+=1
		if j not in convDikt:
			convDikt[j]=ind_node
			ind_node+=1
		if l not in convEDikt:
			convEDikt[l]=ind_layer
			ind_layer+=1


		if do_aggregate:
			plex_ind=selected_plex_ind_func(l)
			if plex_ind is None:
				continue
		else:
			plex_ind=convEDikt[l]


		conv_graph[ind]=(convDikt[i],convDikt[j],plex_ind )
		ind+=1

	conv_graph=conv_graph[:ind]


	rev_conv_dikt={convDikt[k]:k for k in convDikt}
	rev_conv_e_dikt={convEDikt[k]:k for k in convEDikt}

	plexes_dikt={}
 
	for l in set(conv_graph[:,2]):
		pl_key=selected_plex_ind_func(l if do_aggregate else rev_conv_e_dikt[l] )
		if pl_key is None:
			continue

		if pl_key not in plexes_dikt:
			plexes_dikt[pl_key]=[]

		plexes_dikt[pl_key].append(l)

	return conv_graph,rev_conv_dikt,plexes_dikt,rev_conv_e_dikt

# ==================================================

def social_sorting(combs_list,layer_dikt):
	
	lis_len=0

	for plex in combs_list:
		lis=combs_list[plex]
		lis_len+=len(lis)
		lis.sort(key=lambda comb:
			int(layer_dikt[comb][layer_dikt[comb].index('_')+1:] ))

	if lis_len%len(combs_list)!=0:
		raise Exception("different number of villages for these plexes: "+str(combs_list))


def run_many_multiplex_combs(net_filename,net_type,plexes_preps, 
	saved_corrmaps,saved_corrmaps_file,saved_countmaps,saved_countmaps_file,corr_dim,indv_combs_id,iter_func,doDraw3D,doSave,doPositiveCorrs,subSample=None,save_2corrs=None):
	
	print ('iterating these plex combs',plexes_preps.keys())
	save_corr_ind=0

	for which_combs in plexes_preps.keys():

		corr_list=np.zeros((0,corr_dim[which_combs],corr_dim[which_combs]))

		all_important_corrs=np.zeros((corr_dim[which_combs],corr_dim[which_combs]),dtype=int)
		all_important_corrs_above60=np.zeros((corr_dim[which_combs],corr_dim[which_combs]),dtype=int)
		total_count_prod_comb=0
		
		
		
		total_countmaps=0
		all_sum_countmaps={}
		

		print('Im working on: ',which_combs)
		colors=np.zeros((0,4))

		cols=[]
		

		for pls_ind,pl_s in enumerate(plexes_preps[which_combs].splitlines()):
			pls=pl_s.split(';')
			pls=[pl.split(',') for pl in pls]

			graph,cd,cld,ced=select_plexes_multiplex_net(net_filename
			,partial(good_plexes_func,selected_plexes_preps=pls,net_type=net_type),
			partial(good_plexes_ind_func,selected_plexes_preps=pls,net_type=net_type),
			layer_col=3,do_aggregate=False)

			# !!!!!!!!!!!!!!!!!!!!!!!!!!!!! SOCIAL NETS
			# print(cld,ced ,set(graph[:,2]),pl_s)
			# input('testing graph made')

			# in_corr_list=np.zeros(( np.product( [len(cv) for cv in cld.values() ])\
			# ,corr_dim[which_combs],corr_dim[which_combs] ))

			sum_corrs=np.zeros((corr_dim[which_combs],corr_dim[which_combs] ),dtype=int)
			count_corrs=0

			num_countmaps=0
			sum_countmap={}
			
			
			if net_type=='economic':
				lenCombs=np.product( [len(cv) for cv in cld.values() ]) 
				

			elif net_type=='social':
				lenCombs=None
				social_sorting(cld,ced)
			
			
			
			if subSample is not None:
				if net_type=='social':
					raise Exception('Not implemented')
					
				if lenCombs>subSample:
					subSampleFac=subSample/(lenCombs*1.0)
				else:
					subSampleFac=1
			if net_type=='economic':
				print('These many combs in economic combs: ', 
						int(lenCombs*subSampleFac))
			
			if not doSave: continue
			
			for ind,pl_combs in enumerate(iter_func(*cld.values() )):
				
				# !!!!!!!!!!!SOCIAL MET
				# print (pl_combs,[ced[i] for i in pl_combs])
				# input('view zipped plexes: ')
				
				if subSample is not None and np.random.rand()>subSampleFac:
					continue
				
				if ind%100==99:
					print('passed {} from all combs'.format(ind,),flush=True)
					
				

				graph_used=graph.copy()

				pl_comb_map=False 
				
				for pl_comb in pl_combs:
					pl_comb_map=np.logical_or(pl_comb_map,(graph[:,2]== pl_comb) )

				if not np.any(pl_comb_map):
					continue

				graph_used=graph[pl_comb_map]

				
				
				graph_used2=graph_used.copy()
				for pl_comb_ind,pl_comb in enumerate(pl_combs):
					graph_used2[graph_used[:,2]==pl_comb,2 ]=pl_comb_ind


					
				in_countmap=summarize_graph_orbitDikt(graph_used2.copy())

				sum_countmap=add_graph_orbitDikts(sum_countmap,in_countmap)
				num_countmaps+=1
					
				# print(cld,set(graph[:,2]),set(graph_used[:,2]))
				in_corr=calculate_corrmatrix(graph_used2)[1]
				
				if save_2corrs is not None and in_corr.shape[0]==save_2corrs.shape[1]:
					save_2corrs[save_corr_ind%save_2corrs.shape[0] ]=in_corr
					save_corr_ind+=1
				
				elif save_2corrs is not None and save_2corrs.shape[1]==4:
					graph_used2[:,2]=0
					in_corr4=calculate_corrmatrix(graph_used2)[1]
					save_2corrs[save_corr_ind%save_2corrs.shape[0] ]=in_corr4
					save_corr_ind+=1
				
				if doPositiveCorrs:
					sum_corrs+=(in_corr> 0.7).astype(int)
				else:
					sum_corrs+=(in_corr < -0.7).astype(int)
				count_corrs+=1

			
			important_above60=((sum_corrs/count_corrs)>0.6).astype(int) 

			# important_above60,important_corrs,count_prod_comb=find_strong_corrs(in_corr_list,net_type+'_triplex_more_',pls_ind)
			
			print ('I passed: ',count_corrs,' combinations')
			# print(('verification_count:{}, '+
			# 	'verification important:{}, verification impabove60:{}, {}')\
			# .format(tcount_corrs==count_prod_comb
			# 	,np.all(important_corrs==tsum_corrs),
			# np.all(important_above60==timportant_above60),important_above60[:5,:5]  ))

			if doDraw3D: raise Exception("Not implemented") #corr_list=np.vstack((corr_list,in_corr_list))
			
			if indv_combs_id.format(which_combs) not in saved_countmaps:
				saved_countmaps[indv_combs_id.format(which_combs)]=[]
				saved_countmaps[indv_combs_id.format(which_combs)+'_config']=[]
				
			saved_countmaps[indv_combs_id.format(which_combs)].append((sum_countmap,num_countmaps))

			saved_countmaps[indv_combs_id.format(which_combs)+'_config'].append((cld,ced))

			# find the graphlet correlations that emerge in all economic networks
			all_sum_countmaps=add_graph_orbitDikts(sum_countmap,all_sum_countmaps)
			
			
			total_countmaps+=num_countmaps
			
			
			if indv_combs_id.format(which_combs) not in saved_corrmaps:
				saved_corrmaps[indv_combs_id.format(which_combs)]=[]
				saved_corrmaps[indv_combs_id.format(which_combs)+'_config']=[]
				
			saved_corrmaps[indv_combs_id.format(which_combs)].append(sum_corrs)

			saved_corrmaps[indv_combs_id.format(which_combs)+'_config'].append((cld,ced))

			# find the graphlet correlations that emerge in all economic networks
			all_important_corrs+=sum_corrs
			all_important_corrs_above60+=important_above60
			
			total_count_prod_comb+=count_corrs
			
			if doDraw3D:
				cols.append(in_corr_list.shape[0] )
				colors=np.vstack((colors,get_single_color(pls_ind%8,in_corr_list.shape[0] ,starting=0.8)  ))

			# if pls_ind==5:
				# break


		summary_important_corrs=((all_important_corrs/total_count_prod_comb)>0.6).astype(int)

		saved_corrmaps['summary_by_class_comb_combination'].append(all_important_corrs_above60)
		saved_corrmaps['final_summary'].append(summary_important_corrs)
		
		saved_corrmaps['unsummarized_maps'].append((all_important_corrs,total_count_prod_comb))
			
		
		
		saved_countmaps['final_summary'].append((all_sum_countmaps,total_countmaps))


		if doDraw3D:
			Y=get_3D_projection_from_corrs(corr_list)
			print(cols,Y.shape)
			draw_3D_graphs(Y,colors)


	if save_2corrs is not None and doSave:
		print('Actually Saved CorrMatrices',save_corr_ind)
		save_2corrs=save_2corrs[:save_corr_ind]
			
	if doSave:
		with open(saved_corrmaps_file,'wb') as f:
			pickle.dump(saved_corrmaps,f)
			
		with open(saved_countmaps_file,'wb') as f:
			pickle.dump(saved_countmaps,f)
			

def make_edgelist_from_adj(filename,plex_ind,threshold=0):

	raw_mat=np.loadtxt(filename,delimiter=',',skiprows=0,dtype=float)
	howMany=np.sum( np.triu(raw_mat,1).flatten()>threshold)
	graph_edges=pd.DataFrame({'i':np.zeros((howMany,),dtype=int),
		'j':np.zeros((howMany,),dtype=int),
		'edgetype':np.empty((howMany,),dtype=str),

	})

	ind=0
	for i,j in itertools.combinations(np.arange(raw_mat.shape[0],dtype=int),2):
		if raw_mat[i,j]>threshold:
			graph_edges.ix[ind]=(i,j,plex_ind)
			ind+=1

	return graph_edges

def iterate_indian_villages_adj_matrices(file_dir):
	summary_edges=pd.DataFrame()

	for find,filename in enumerate(os.listdir(file_dir)):
		if not filename.endswith('.csv'):
			continue
		if find%50==0: print(find,'im here')

		read_filename=os.path.join(file_dir,filename)
		re_match=re.match('adj_([a-zA-Z0-9]+)_vilno_([0-9]+)',filename)
		if re_match is None:
			continue
		relationship=re_match.group(1)
		vil_id=re_match.group(2)

		plex_ind=relationship+'_'+vil_id

		in_graph_edges=make_edgelist_from_adj(read_filename,plex_ind)

		summary_edges=pd.concat((summary_edges,in_graph_edges))

		# if find>40:
		# 	break
	
	summary_edges.to_csv('../multiplex_data/indian_social_summary_edges.csv'
		,sep=',',header=False,index=False)

def good_plexes_func(x,selected_plexes_preps,net_type='economic'):
	plex_map=good_plex_map(x,selected_plexes_preps,net_type)
	# print(plex_map)
	if np.any(plex_map):
		# if plex_map[0]:
		# 	print('im first')
		return True

	return False

def good_plex_map(x,plexes,net_type):
	plex_map=[]
	if net_type=='economic': x=str(x).zfill(4)
	for pls in plexes:
		
		plex_map.append(np.any([True if x.startswith(pl_in) else False \
			for pl_in in pls if pl_in ])   )

	return plex_map
		



def good_plexes_ind_func(x,selected_plexes_preps,default_val=None,net_type='economic'):
	plex_map=good_plex_map(x,selected_plexes_preps,net_type)
	
	# print(plex_map,x,selected_plexes_preps)
	# input('wait a minute')
	get_ind=np.where(plex_map)[0]
	if len(get_ind)>0:
		return get_ind[0]

	return None

def find_strong_corrs(corr_list,str_prep,index,doNegative=False):
	corr_size=corr_list.shape[1]
	sum_pos_corrs=np.zeros(corr_list.shape[1:],dtype=int)
	sum_neg_corrs=np.zeros(corr_list.shape[1:],dtype=int)
	for corr in corr_list:
		pos_corr=(corr > 0.7).astype(int)
		neg_corr=(corr < -0.7).astype(int)
		
		sum_pos_corrs+=pos_corr
		sum_neg_corrs+=neg_corr
	
	# plt.close('all')
	# fig = plt.figure(figsize=(20,20))
	
	# cmap = cm.get_cmap('viridis', 30)
	# # for i,j in itertools.product(range(corr_size),range(corr_size)):
		 # # plt.text(i, j, str(sum_pos_corrs[i,j]), va='center', ha='center')

	# # plt.subplot(111)
	# cax = plt.imshow(sum_pos_corrs, interpolation="nearest", cmap=cmap)
	# # plt.subplot(212)
	# # cax = plt.imshow(sum_neg_corrs, interpolation="nearest", cmap=cmap)
	# plt.title('Graph')

	# # Add colorbar, make sure to specify tick locations to match desired ticklabels
	# cbar = fig.colorbar(cax, ticks=np.linspace(-1,0,1))
	# # cbar.ax.set_yticklabels(['1','0','-1'])
	# plt.savefig(str_prep+str(index)+'.png',dpi=200,bbox_inches='tight')
	
	important_neg=sum_neg_corrs/corr_list.shape[0]
	important_neg_above60=(important_neg>0.6).astype(int)
	
	important_pos=sum_pos_corrs/corr_list.shape[0]
	important_pos_above60=(important_pos>0.6).astype(int)
	
	if doNegative:
		return important_neg_above60,sum_neg_corrs,corr_list.shape[0]
	
	return important_pos_above60,sum_pos_corrs,corr_list.shape[0]


