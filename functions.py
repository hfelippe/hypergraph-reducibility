import numpy as np
import itertools
import random
import math
import time
from collections import Counter
from scipy.special import loggamma

def logchoose(n,k):
    """
    log binomial coefficient
    """
    if len(str(n)) > 300: #stable approximation for n >> k
        if (k == 0) or (k == n): return 0
        else: return k*math.log(n) - k*math.log(k) + k
    else:
        n,k = float(n),float(k)
        return loggamma(n+1) - loggamma(k+1) - loggamma(n-k+1)

def logmultiset(n,k):
    """
    log multiset coefficient
    """
    return logchoose(n+k-1,k)

def projection(G,indices):
    """
    project hypergraph G (set of multiple tuple sizes) onto layers with tuple sizes in the list "indices"
    projects all tuples of size l' >= l onto layer l for the final projections in the "projections" dict
    """
    indices = sorted(indices)[::-1]
    projections = {}
    for l in indices: 
        projections[l] = set()
        for tup in G:
            for combo in itertools.combinations(tup,l):
                projections[l].add(combo)
            
    return projections

def get_num_nodes(G):
    """
    finds number of unique nodes in G
    """
    nodes = set([])
    for t in G:
        for i in t:
            nodes.add(i)
    return len(nodes)

def get_layers(G,indices='all'):
    """
    gets layers of hypergraph G (set of multiple tuple sizes) and puts them into a dict "layers"
    only grabs the layers corresponding to "indices"
    """
    if indices == 'all': indices = list(Counter([len(tup) for tup in G]).keys())
    layers = {}
    for l in indices:
        layers[l] = set()
    for tup in G:
        l = len(tup)
        if l in layers: 
            layers[l].add(tup)

    return layers

def powerset(iterable):
    """
    powerset of iterable, returned in the form of an itertools chain object
    removes empty set
    """
    s = list(iterable)
    pset = list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1)))
    return pset[1:]

def coarse_grain(G,b):
    """
    return coarse-grained hypergraph of G according to node partition b
    returns a dict of form {tuple:count} to represent a multiset
    """
    Gc = {}
    for e in G:
        
        ec = [b[i] for i in e]
        new_e = tuple(sorted(ec))
        
        if not(new_e in Gc): Gc[new_e] = 0
        Gc[new_e] += 1
        
    return Gc

def all_projections(G):
    """
    returns dict P such that P[k][l] is projection of layer k onto tuples of size l
    """
    layers = get_layers(G)
    P = {}
    indices = sorted(list(layers.keys()))[::-1]
    for index,k in enumerate(indices):
        
        P[k] = {}
        remaining_indices = indices[index+1:]
        for subindex,l in enumerate(remaining_indices):
            
            if subindex == 0: P[k][l] = projection(layers[k],[l])[l]        
            else:
                l_prev = remaining_indices[subindex-1]
                P[k][l] = projection(P[k][l_prev],[l])[l]  

    return P

def get_entropies_project(G,partition=None):
    """
    returns:
        dict M such that M[k][l] is conditional entropy of higher layer k to lower layer l
        dict Q such that Q[l] is entropy of layer l
    computes entropies under specified node partition if not None
    """  
    
    if partition is not None:
        B = len(set(partition))

    N = get_num_nodes(G)
        
    def entropy(l):
        """
        entropy of hypergraph layer l, in total bits
        """
        El = len(layers[l]) 
        if partition is not None: return logmultiset(math.comb(B+l-1,l),El)    
        else: return logchoose(math.comb(N,l),El)

    def conditional_entropy(k,l):
        """
        conditional entropy of hypergraph layer l when transmitted from layer k
        measured in total bits
        """
        El = len(layers[l])
        Ek2l = len(P[k][l])
        
        if partition is not None:
            proj_coarse = coarse_grain(P[k][l],partition)
            lower_coarse = coarse_grain(layers[l],partition)
            
            Eoverlap = 0
            for e in lower_coarse:
                if e in proj_coarse:
                    Eoverlap += min(proj_coarse[e],lower_coarse[e])
             
            return logchoose(Ek2l,Eoverlap) + logmultiset(math.comb(B+l-1,l),El-Eoverlap)
        
        else:
            Eoverlap = len(P[k][l].intersection(layers[l]))
            return logchoose(Ek2l,Eoverlap) + logchoose(math.comb(N,l)-Ek2l,El-Eoverlap)

    layers = get_layers(G) 
    L = len(layers)
    P = all_projections(G)

    M,Q = {},{}
    for k in layers:
        
        M[k] = {}
        for l in layers:
            if k > l: M[k][l] = conditional_entropy(k,l)
            
        Q[k] = entropy(k)

    return M,Q

def get_entropies_count(G,partition=None):
    """
    returns:
        dict M such that M[k][l] is conditional entropy of higher layer k to lower layer l
        dict Q such that Q[l] is entropy of layer l
    computes entropies under specified node partition if not None
    """  
    layers = get_layers(G)
    N = get_num_nodes(G)

    if partition is not None:
        B = len(set(partition))
        layers_coarse = {}
        for l in layers: layers_coarse[l] = coarse_grain(layers[l],partition)
            
    start = time.time()
    
    M,Q = {},{}
    indices = sorted(list(layers.keys()))[::-1]
    for ii,k in enumerate(indices):
        
        Ek = len(layers[k])
        if partition is not None: Q[k] = logmultiset(math.comb(B+k-1,k),Ek)    
        else: Q[k] = logchoose(math.comb(N,k),Ek)
        
        lower_layer_indices = indices[ii+1:]

        """
        compute Ek2l for all l < k
        """
        def get_sizes_proj(layer):
            """
            calculates size of projection of layer onto tuples of size l for all l < k
            recurses to get lower-level overlap sizes to subtract off 
            """
            sizes = Counter({l:0 for l in lower_layer_indices})
            
            if len(layer) == 0:
                return sizes
                
            checked = []
            layer = list(layer)
            for e in layer:
            
                overlaps = set()
                for epast in checked:
                    
                    inter = set(list(e)).intersection(set(list(epast))) 
                    overlaps.add(tuple(sorted(list(inter))))

                sizes_e_max = Counter({l:math.comb(len(e),l) for l in lower_layer_indices})
                sizes += sizes_e_max - get_sizes_proj(overlaps)
                checked.append(e)
            
            return sizes

        Ek2ls = get_sizes_proj(layers[k])
        
        M[k] = {}
        for l in lower_layer_indices:

            El = len(layers[l])
            Ek2l = Ek2ls[l]
                        
            """
            compute Eoverlap
            """
            Eoverlap = 0
            if partition is not None:
                """
                approximation of multiscale Eoverlap assuming tuples in layer k have small intersections
                equivalent to coarse-graining layer k then projecting onto l-tuples instead of vice-versa
                """
                Eoverlap = 0
                lower_coarse = coarse_grain(layers[l],partition)
                for lower_tup in lower_coarse:

                    lower_counts = Counter(list(lower_tup))
                    num_lower_in_higher = 0
                    for higher_tup in layers[k]:

                        higher_counts = Counter([partition[i] for i in higher_tup])
                        hl_counts = higher_counts & lower_counts

                        if not(hl_counts == lower_counts): 
                            num_lower_in_higher += 0
                        else:
                            num_lower_in_higher += np.prod([math.comb(higher_counts[i],lower_counts[i])\
                                                            for i in lower_counts])

                    Eoverlap += min(num_lower_in_higher,lower_coarse[lower_tup])
                
                M[k][l] = logchoose(Ek2l,Eoverlap) + logmultiset(math.comb(B+l-1,l),El-Eoverlap)
                
            else: 
                """
                exact Eoverlap for non-coarse-grained case
                """
                lower_tmp = layers[l].copy()
                for eh in layers[k]:
    
                    higher_set = set(list(eh))
                    overlapping_tups = set()
                    for el in lower_tmp:
                        
                        overlap = set(list(el)).intersection(higher_set)
                        if len(overlap) == l: overlapping_tups.add(el)
                        
                    for t in overlapping_tups: 
                        lower_tmp.remove(t)
                        Eoverlap += 1

                M[k][l] = logchoose(Ek2l,Eoverlap) + logchoose(math.comb(N,l)-Ek2l,El-Eoverlap)
    
    return M,Q
    
    
def reducibility(G,partition=None,optimization='exact',ent_method='count'):
    
    """
    hypergraph reducibility of G (set of all tuple sizes)
    set partition=b for returning multiscale reducibility under coarse-graining 
        with node partition b (default = None)
    returns reducibility and representative set of layers
    """

    """
    precompute matrices M and Q
    """
    if ent_method == 'project':
        M,Q = get_entropies_project(G,partition)
    else:
        M,Q = get_entropies_count(G,partition)
    
    """
    find best representative configuration Rstar and compute reducibility
    """
    if optimization == 'exact':
        """
        exact enumeration over powerset of layers
        """
        maxl = max(list(Q.keys()))
        DLs = {}
        for R in powerset(list(Q.keys())):
            
            R = set(R)
            if not(maxl in R):
                continue #max layer must be in representatives, since it cannot be transmitted from another layer
    
            H = sum(Q[r] for r in R)
            
            nonR = set(Q.keys()) - R
            for l in nonR:
                rs = [r for r in R if r > l]
                CEs = [M[r][l] for r in rs]
                rl = rs[np.argmin(CEs)]
                H += M[rl][l]
    
            DLs[tuple(sorted(R))] = H 
                
        Rstar = min(DLs, key=DLs.get)
        Hstar = DLs[Rstar]

    else:
        """
        greedy approximate method where best representative is added until all representatives considered
        faster, but not guaranteed to find exact minimum of H(R)
        """
        maxl = max(list(Q.keys()))
        Rs,Hs = [],[]
        R = set([maxl])
        LminusR = set(Q.keys()) - R
        Rs.append(list(R))
        Hs.append(Q[maxl] + sum(M[maxl][l] for l in LminusR))
        
        while len(LminusR) > 0:

            r_tmps,H_tmps = [],[]
            for r_tmp in LminusR:
                
                R_tmp = R.union(set([r_tmp]))
                nonR_tmp = LminusR - set([r_tmp])
                H_tmp = sum(Q[r] for r in R_tmp)
            
                for l in nonR_tmp:
                    rs = [r for r in R_tmp if r > l]
                    CEs = [M[r][l] for r in rs]
                    rl = rs[np.argmin(CEs)]
                    H_tmp += M[rl][l]

                r_tmps.append(r_tmp)
                H_tmps.append(H_tmp)

            best_index = np.argmin(H_tmps)
            R.add(r_tmps[best_index])
            LminusR.remove(r_tmps[best_index])
            Rs.append(list(R))
            Hs.append(H_tmps[best_index])

        best_num_reps = np.argmin(Hs)
        Rstar = tuple(sorted(Rs[best_num_reps]))
        Hstar = Hs[best_num_reps]
         
    H0 = sum(Q[l] for l in Q)
    if H0 == Q[maxl]:
        return 1.,Rstar
        
    return (H0 - Hstar)/(H0 - Q[maxl]), Rstar

def generate_reps_hypergraph(N,R,E,nonR,noise,constant_E=False):
    """
    generates random hypergraph over N nodes through specified representative layer indices R,
        which are each random hypergraphs of E hyperedges each for the specified hyperedge sizes 
    assigns each layer l in nonR to a layer r in R of higher order at random 
    E is #edges in each representative layer
    generates l from r by:
        -removing a fraction 'noise' of the hyperedges from r projected down to order l
        -adding a fraction 'noise' of hyperedges at random from all possible hyperedges of size l
        -(optional, if constant_E == True) keeping only E of these edges at random
    returns hypergraph G aggregating all the layers into a single set
    """
    def random_hyperedge(size):
        """
        generates random hyperedge of indicated size
        """
        e = set()
        while len(e) < size: e.add(np.random.randint(N))
        return tuple(sorted(list(e)))

    """
    add hyperedges for representative layers in R to G
    """
    G = set()
    for r in R: #not guaranteed to have E unique edges per layer, but should be close enough
        for edge in range(E): G.add(random_hyperedge(r)) 

    """
    add hyperedges for non-representative layers to G
    """
    layers = get_layers(G)
    for l in nonR:
        
        """
        select representative r for l
        """
        Rabove = [r for r in R if r > l]
        r = np.random.choice(Rabove)

        """
        sample (1-'noise') fraction of random edges from projected version of layer r to use as layer l
        add remaining 'noise' fraction of hyperedges at random to layer l so it has same # edges as projected layer
        update G with the new layer l
        """
        if constant_E:
            
            num_to_keep = int((1.-noise)*E)
            kept_from_r = set()
            rlayer = list(layers[r])
            while len(kept_from_r) < num_to_keep:
                tup_higher = rlayer[np.random.randint(len(rlayer))]
                tup_proj = np.random.choice(a=tup_higher,size=l,replace=False)
                kept_from_r.add(tuple(sorted(tup_proj)))
            to_add = set([random_hyperedge(l) for _ in range(E-num_to_keep)])
            
        else:
            
            rproj = list(projection(layers[r],[l])[l])
            Er = len(rproj)
            to_keep = np.random.choice(range(Er),size=int((1.-noise)*Er),replace=False)
            kept_from_r = [rproj[i] for i in to_keep]
            E_kept = len(kept_from_r)
            to_add = set([random_hyperedge(l) for _ in range(Er-E_kept)])
        
        layerl = set(kept_from_r).union(to_add)
        G = G.union(layerl)

    return G

def layer_reducibility(G,partition=None,ent_method='count'):

    if ent_method == 'project':
        M,Q = get_entropies_project(G,partition)
    else:
        M,Q = get_entropies_count(G,partition)

    lmax = max(l for l in Q)

    etas = {}
    for l in Q:
        
        Hl = Q[l]

        best_rep,best_CE = None,np.inf
        for k in Q:
            if k > l: 
                CE = M[k][l]
                if CE <= best_CE:
                    best_CE = CE
                    best_rep = k

        if l == lmax:
            etas[l] = 0.
        elif Hl == 0:
            if best_rep != l:
                etas[l] = 1.
            else:
                etas[l] = 0
        else:
            etas[l] = 1. - best_CE/Hl
            
    return etas

def noisy_simplex(lmax,noise_map,N=100):
    
    G = set(powerset(range(lmax))[lmax:])
    layers = get_layers(G)

    G_rand = set([])
    for l in layers:
        
        layerlist = list(layers[l])
        E = len(layerlist)
        E_replace = int(noise_map[l]*E)
        inds_to_replace = random.sample(range(E),E_replace)
        
        for i in inds_to_replace:
            new = tuple(sorted(random.sample(range(N),l)))
            layerlist[i] = new

        for e in layerlist:
            G_rand.add(e)
        
    return G_rand

def block_hypergraph(N,B,layers,layer_sizes,p):

    b = [i%B for i in range(N)]
    
    if B == N:
        
        G = set()
        for l in layers:
            
            E = layer_sizes[l]
            for _ in range(E):
                
                e = random.sample(range(N),l)
                e = tuple(sorted(e))
                G.add(e)
            
        return G,b
        
    groups = {}
    for i in range(N):
        bi = b[i]
        if not(bi in groups): groups[bi] = []
        groups[bi].append(i)

    G = set()
    for l in layers:
        
        E = layer_sizes[l]
        for _ in range(E):
            
            e = []
            i1 = np.random.randint(N)
            e.append(i1)
            be = b[i1]
            not_be = set(range(B)) - set([be])

            num_same = np.random.binomial(l-1,p)
            same = random.sample(list(set(groups[be])-set([i1])),num_same)
            for i in same:
                e.append(i)
            
            num_diff = l-1-num_same
            diff_counts = np.random.multinomial(n=num_diff,pvals=[1/(B-1) for c in not_be])
            diff_dict = dict(zip(list(not_be),diff_counts))
            for c in diff_dict:
                samp = random.sample(groups[c],diff_dict[c])
                for i in samp:
                    e.append(i)

            e = tuple(sorted(e))
            G.add(e)

    return G,b

def partition_shuffle(b,noise):

    N = len(b)
    num_shuffles = int(noise*N/2)
    pairs = list(zip(range(N),np.random.permutation(range(N))))
    for shuf in range(num_shuffles):
        i,j = pairs[shuf]
        bi,bj = b[i],b[j]
        b[i] = bj
        b[j] = bi

    return b
