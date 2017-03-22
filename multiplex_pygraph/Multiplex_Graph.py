
# coding: utf-8

# In[105]:

import numpy as np
import itertools
import math
import pandas as pd


class GraphFunc:
    """
    GraphFunc To calculate multiplex graphlet orbitals:
    general flow is:
    graphF=GraphFunc(doDirected=False)
    graphF.load_file('london_transport_multiplex.csv',delimiter=',',skiprows=0)
    graphF.make_direct_neighbours(subOne=True) : self.graphNeigh
    graphF.make_zero_orbit()
    graphF.count_tri_graphs() : self.triGNMulti
    graphF.return_orbits_Mat()
    graphF.save_orbits('london_multi2.csv')

    ----------
    From edge list
    ----------
    graphF=GraphFunc(doDirected=False)
    graphF.make_graph_table(graph_edges)
    graphF.make_direct_neighbours(subOne=False)# : self.graphNeigh

    graphF.make_zero_orbit()
    graphF.count_tri_graphs()# : self.triGNMulti

    orb_mat=graphF.return_orbits_Mat().values


    """
    def __init__(self,doDirected=False):
        self.doDirected=doDirected
        self.factorialDikt={}
        self.combR2Dikt={}
        pass
    def load_file(self,inputfile,delimiter=',',dtype=np.int32,**kwargs):
        self.inputfile=inputfile
        
        graph_table=np.loadtxt(inputfile,delimiter=delimiter,dtype=dtype,
                    **kwargs)
        return self.make_graph_table(graph_table)

    def make_graph_table(self,graph_table):
        fix_graph_table=np.zeros((len(graph_table),3)
                ,dtype=np.int32)
        curVInd=0
        vertDikt={}
        for ind,(i,j,k) in enumerate(graph_table):
            if i not in vertDikt:
                vertDikt[i]=curVInd
                curVInd+=1
            if j not in vertDikt:
                vertDikt[j]=curVInd
                curVInd+=1
            
            if vertDikt[i]!=vertDikt[j]:
                fix_graph_table[ind]=(vertDikt[i],vertDikt[j],k)


        self.vertN=curVInd
        self.graph_t=fix_graph_table
        self.edgeN=(2**len(set(fix_graph_table[:,2])))-1
        self.vertDikt=vertDikt
        self.revvertDikt={vertDikt[i]:i for i in vertDikt}

        return fix_graph_table,curVInd
    def load_edge_list(self,edgeLists):
        graph_table=[]
        for k,eL in enumerate(edgeLists):
            graph_table.extend([(i,j,k) for (i,j) in eL])
            
        fix_graph_table=np.zeros((len(graph_table),3)
                ,dtype=np.int32)
        curVInd=0
        vertDikt={}
        for ind,(i,j,k) in enumerate(graph_table):
            if i not in vertDikt:
                vertDikt[i]=curVInd
                curVInd+=1
            if j not in vertDikt:
                vertDikt[j]=curVInd
                curVInd+=1
            
            if vertDikt[i]!=vertDikt[j]:
                fix_graph_table[ind]=(vertDikt[i],vertDikt[j],k)


        self.vertN=curVInd
        self.graph_t=fix_graph_table
        self.edgeN=(2**len(set(fix_graph_table[:,2])))-1
        self.vertDikt=vertDikt

        self.revvertDikt={vertDikt[i]:i for i in vertDikt}
        
        return fix_graph_table,curVInd
    def make_direct_neighbours(self,subOne=False,do_plex_wrap=True):
        graphHeigh=[{} for i in  range(self.vertN)]

        
        mMax=-1

        if do_plex_wrap:
            self.edgeN=(2**len(set(self.graph_t[:,2])))-1
            self.edgeN_nonF=len(set(self.graph_t[:,2]))
        else:
            self.edgeN=len(set(self.graph_t[:,2]))
            self.edgeN_nonF=int(math.log2(self.edgeN1) )


        #!!!!  edgeN ke e 3 bidejki e so directed
        if self.doDirected: self.edgeN=3
        

        for u,v,k in self.graph_t:
            if subOne: k-=1
            k1=k
            
            if k<2 and self.doDirected:
                k2=1-k1
            else:
                k2=k
            if v in graphHeigh[u]:
                lis=self.get_all_comb(self.edgeN_nonF,graphHeigh[u][v])
                lis=list(lis)
                lis.append(k1)
                lis.sort()
                # print(lis,self.edgeN_nonF,self.get_m_list(self.edgeN_nonF,lis))
                m=self.get_m_list(self.edgeN_nonF,lis)
                graphHeigh[u][v]=m
                # if m==3: print('im here',v,u,)

                if mMax<m: mMax=m
            else:
                # if k1==0: print('im here',v,u,)

                graphHeigh[u][v]=k1
            if u in graphHeigh[v]:
                lis=self.get_all_comb(self.edgeN_nonF,graphHeigh[v][u])
                lis=list(lis)
                lis.append(k2)
                lis.sort()
                # print(lis,self.edgeN_nonF,self.get_m_list(self.edgeN_nonF,lis))
                m=self.get_m_list(self.edgeN_nonF,lis)
                # if m==3: print('im here',v,u,)
                graphHeigh[v][u]=m
                if mMax<m: mMax=m
            else:
                # if k2==0: print('im here',v,u,)
                graphHeigh[v][u]=k2
            
            
            
        
#         if mMax!=-1: 
#             print()
#             self.edgeN
        self.graphNeigh=graphHeigh

    def count_tri_graphs(self):
        triGN=np.zeros((self.vertN,3), dtype=int)
        triGNMulti=[]
        for ind in range(3):
            triGNMulti.append([{} for i in  range(self.vertN)])
        
        
        for vind,vDikt in enumerate(self.graphNeigh):
            runnedThrough=set([])
            vind_set=set([vind])
            for k,j in itertools.combinations(vDikt.keys(),2):
                if k in self.graphNeigh[j]:
                    triGN[vind][2]+=1
                    
                    #Multiplex======================
                    indComb=self.calc_combR2(vDikt[k],vDikt[j])
                    indComb+=self.graphNeigh[j][k]*self.comb2_num()
                    triGNMulti[2][vind][indComb]=triGNMulti[2][vind].get(indComb,0)+1
                    
                else:
                    triGN[vind][1]+=1
                    
                    #Multiplex======================
                    indComb=self.calc_combR2(vDikt[k],vDikt[j])
                    triGNMulti[1][vind][indComb]=triGNMulti[1][vind].get(indComb,0)+1
                    
                    
                if k not in runnedThrough:
                    setK_diff=self.graphNeigh[k].keys() - vDikt.keys()-vind_set
                    triGN[vind][0]+=len(setK_diff)-1
                    runnedThrough.add(k)
                    
                    #Multiplex======================
                    indComb_base=vDikt[k]*self.edgeN
                    
                    for ik in setK_diff:
                        # print(vind,k,ik,vDikt)

                        indComb=indComb_base+ self.graphNeigh[k][ik]
#                         indComb=(vDikt[k],self.graphNeigh[k][ik])
                        triGNMulti[0][vind][indComb]=triGNMulti[0][vind].get(indComb,0)+1
                    
                if j not in runnedThrough:
                    setJ_diff=self.graphNeigh[j].keys() - vDikt.keys()-vind_set
                    
                    triGN[vind][0]+=len(setJ_diff)-1
                    runnedThrough.add(j)
                    
                    #Multiplex========================================
                    indComb_base=vDikt[j]*self.edgeN
                    
                    for ij in setJ_diff:
                        # print(vind,j,ij,vDikt)

                        indComb=indComb_base+ self.graphNeigh[j][ij]
                        
#                         indComb=(vDikt[j],self.graphNeigh[j][ij])
                        triGNMulti[0][vind][indComb]=triGNMulti[0][vind].get(indComb,0)+1
            
        self.triGN=triGN
        self.triGNMulti=triGNMulti
        return triGN
    def calc_combR2(self,a,b):
        if (a,b) in self.combR2Dikt:
            return self.combR2Dikt[(a,b)]
        if a>b:
            a,b=b,a
            
        row_n=self.sumrowsR(self.edgeN-a+1)
        combr2_ret=int(row_n+(b-a))

        if (a,b) not in self.combR2Dikt:
            self.combR2Dikt[(a,b)]=combr2_ret 

        return combr2_ret
    def show_iters(self):
        pass
    def sumrowsR(self,a):
        en=self.edgeN
        if en==a:
            return a
        return int((en+a)*(en-a+1)/2)
    def comb2_num(self):
        n=self.edgeN+1
        k=2
        return self.calc_binom(n,k)
    def calc_binom(self,n,k):
        if  (n,k) in self.factorialDikt:
            return self.factorialDikt[(n,k)]

        self.factorialDikt[(n,k)]=math.factorial(n)\
        //(math.factorial(n-k)*math.factorial(k))

        return self.factorialDikt[(n,k)]

    def return_orbits_Dikt(self):
        dikt_orbits={i:[1]*3 for i in self.vertDikt}
        
        for indG in range(3):
            for ind in self.vertDikt:
                dikt_orbits[ind][indG]= self.triGNMulti[indG][ self.vertDikt[ind] ]
        
        return dikt_orbits
    def make_zero_orbit(self):
        self.zeroGCMulti=np.zeros((self.vertN,self.edgeN),dtype=int)
        for v,vD in enumerate(self.graphNeigh):
            try:
                countsV=np.bincount(list(vD.values()))
            except:
                print(countsV,vD.values())
                sys.exit()
            # print('counts shape ',v,vD,countsV.shape[0])
            self.zeroGCMulti[v,:countsV.shape[0]]=countsV
            
    def return_full_orbit_counts(self):
        wedgepN=self.edgeN**2
        wedgesN=self.comb2_num()
        triN=self.comb2_num()*self.edgeN
        return self.edgeN,wedgepN,wedgesN,triN
    def return_orbits_Mat(self):
        # 9+6+18 za tri plexa
        wedgepN=self.edgeN**2
        wedgesN=self.comb2_num()
        triN=self.comb2_num()*self.edgeN
        sumOrb=wedgepN+wedgesN+triN+self.edgeN
        # print('sumorbs',wedgepN,wedgesN,triN,self.edgeN,self.zeroGCMulti.shape)
        orbs_indexes=np.repeat(np.arange(4),(self.edgeN,wedgepN,wedgesN,triN))
        orbs_start0=np.repeat((0,self.edgeN,wedgepN+self.edgeN,wedgesN+wedgepN+self.edgeN)\
            ,(self.edgeN,wedgepN,wedgesN,triN))
        # print(orbs_start0[:10])
        starting_mat_orbs=np.hstack((self.zeroGCMulti,            np.zeros((self.vertN,sumOrb-self.edgeN),dtype=int)  ))
        
        mat_orbits=pd.DataFrame(starting_mat_orbs,            index=self.vertDikt.keys(),
                columns=['orbital_[{},{}]'.format(i,j-k) for i,j,k in zip(orbs_indexes,range(sumOrb),orbs_start0)  ]   )
        
        count_start=[self.edgeN,self.edgeN+wedgepN,self.edgeN+wedgepN+wedgesN]
        for indG,cS in zip(range(3),count_start):
            
            for ind in self.vertDikt:
                ind_dikt=self.triGNMulti[indG][self.vertDikt[ind]]
                for ind_col in ind_dikt:
                    mat_orbits.ix[ind,ind_col+cS  ]=ind_dikt[ind_col]
        
        return mat_orbits

    def return_orbits_fast(self,internal_node_id=True):
        
        if not internal_node_id:
            raise Exception("Can't return numpy array with node ids outside of [0,n-1] range")

        wedgepN=self.edgeN**2
        wedgesN=self.comb2_num()
        triN=self.comb2_num()*self.edgeN
        sumOrb=wedgepN+wedgesN+triN+self.edgeN

        mat_orbits=np.zeros((self.vertN,sumOrb),dtype=int)
        # print('self vertdikt')

        mat_orbits[:,:self.edgeN]=self.zeroGCMulti

        count_start=[self.edgeN,self.edgeN+wedgepN,self.edgeN+wedgepN+wedgesN]

        for indG,cS in zip(range(3),count_start):     
            for ind in self.vertDikt:
                ind_dikt=self.triGNMulti[indG][self.vertDikt[ind]]
                for ind_col in ind_dikt:
                    if internal_node_id:
                        mat_orbits[ self.vertDikt[ind],ind_col+cS  ]=ind_dikt[ind_col]
        
        return mat_orbits


    def return_orbits_dikt(self):
        

        wedgepN=self.edgeN**2
        wedgesN=self.comb2_num()
        triN=self.comb2_num()*self.edgeN
        sumOrb=wedgepN+wedgesN+triN+self.edgeN

        dikt_orbits={ind:{} for ind in self.vertDikt}
        # print('self vertdikt')

        for ind in self.vertDikt:
            zeroG=self.zeroGCMulti[self.vertDikt[ind]]
            for conv_ind,conv_val in enumerate(zeroG):
                if conv_val!=0:
                    dikt_orbits[ind][conv_ind]=conv_val

            # print ('ind is ',ind,'dikt is ',self.graphNeigh[self.vertDikt[ind]] )


        count_start=[self.edgeN,self.edgeN+wedgepN,self.edgeN+wedgepN+wedgesN]
        for indG,cS in zip(range(3),count_start):
            
            for ind in self.vertDikt:
                ind_dikt=self.triGNMulti[indG][self.vertDikt[ind]]
                for ind_col in ind_dikt:
                    dikt_orbits[ind][ind_col+cS  ]=ind_dikt[ind_col]
        
        return dikt_orbits

    def save_orbits(self,filename):
        d_orbs=self.return_orbits_Mat()
        d_orbs.to_csv(filename)
        
    def get_combs(self,n,k):

        return self.calc_binom(n,k)

    def get_sum(self,c,i):
        return (c*i)-(i*(i+1))//2

    def get_comb2(self,c,m):
        if m>=self.get_combs(c,2):
            return
        root=np.roots([1,(1-(2*c)),2*m])
        row=min(root)
        row=int(round(row,4))
        col=m-self.get_sum(c,row)
        return row,col+row+1

    def get_combs_k(self,c,k,m):
        if k==2:
            return self.get_comb2(c,m)
        step_one=-1
        while m>=0:
            try:
                step_one+=1
                m-=self.get_combs(c-1-step_one,k-1)
            except:
                print('Error in get combs',c,k,m,step_one)
                sys.exit()
        m+=self.get_combs(c-1-step_one,k-1)
        retL=[step_one]
        lis=[i+1+step_one for i in self.get_combs_k(c-1-step_one,k-1,m)]
        retL.extend(lis)
        return retL

    def get_all_comb(self,c,m):
        if m+1>=2**c:
            return
        combs_k=1
        while combs_k<=c:
            if m<0:
                break
            m-=self.get_combs(c,combs_k)
            combs_k+=1
        combs_k-=1
        m+=self.get_combs(c,combs_k)
        if combs_k==1:
            return (m,)

        return self.get_combs_k(c,combs_k,m)

    def calc_comb2(self,en,a,b):
            row_n=0
            if a!=0:
                row_n=self.sumrows(en-1,en-a)
            return int(row_n+(b-a))-1

    def sumrows(self,en,a):
            if en==a:
                return a
            return int((en+a)*(en-a+1)/2)

    def sum_combs(self,N,k):
        return sum(self.get_combs(N,ik) for ik in range(1,k))


    def get_m_list(self,N,lis,indFromStart=True):
        if lis is None:
            return

        if type(lis)!=np.ndarray: lis=np.array(lis)
        combs_k=len(lis)

        m=0
        if  indFromStart:
            m=self.sum_combs(N,combs_k)


        if combs_k==1:
            return lis[0]
        if combs_k==2:
            return m+self.calc_comb2(N,*lis)
        first_ind=lis[0]

        m+=sum([self.get_combs(N-inrow,combs_k-1) for inrow in range(1,first_ind+1)])

        nN=N-first_ind-1
        m+=self.get_m_list(nN,lis[1:]-(first_ind+1),indFromStart=False)
        return m




