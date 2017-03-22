import sys

sys.path.append('..')
from multiplex_pygraph.Multiplex_Graph import GraphFunc





graphF=GraphFunc(doDirected=False)

print('Loading example multiplex network... ')
graphF.load_file('../multiplex_data/london_transport_multiplex.csv',delimiter=',',skiprows=0)


print('Calculating graphlet signature matrix... ')
graphF.make_direct_neighbours(subOne=True) # creates graphNeigh attribute, 
										   # dictionary of neighbors
										   # subOne removes 1 from edge attributes for 0 indexe
										   
graphF.make_zero_orbit() # calculates zero orbit part of SI
graphF.count_tri_graphs() # calculates 1,2,3 orbits part of SI


print('Saving example graphlet signature matrix... ')
graphF.save_orbits('results/london_multi_SIMatrix.csv') # Saves graphlet signature vectors 

