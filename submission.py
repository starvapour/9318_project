import scipy
import numpy as np
from scipy.cluster.vq import vq
from scipy.spatial.distance import cdist

# --------------------- Part 1 ---------------------

def pq(data, P, init_centroids, max_iter):
    N, M = data.shape
    K = init_centroids.shape[1]
    
    # split data into P parts
    step = M // P
    data_parts=np.hsplit(data, [i * step for i in range(1,P)])
    codebooks = np.empty(init_centroids.shape, dtype = "float32")
    codes = np.empty([P, N], dtype = "uint8")
    # do cluster for each part
    for i in range(len(data_parts)):
        data = data_parts[i]
        centers = init_centroids[i]
        # start cluster
        iter_num = 0
        
        # for each iter
        while iter_num < max_iter:
            # store all the vector's cluster result
            clusters = [[] for _ in range(K)]
            
            # find the nearest center for each vector
            # then add vector into cluster
            for vector in data:
                #print(centers.shape)
                distances = cdist(centers, [vector], 'cityblock')
                cluster_num = np.unravel_index(np.argmin(distances),distances.shape)[0]
                # after get the final cluster_num
                clusters[cluster_num].append(vector)
            
            # caculate new centers
            for j in range(K):
                if len(clusters[j]) != 0:
                    # use k-median to update the centers
                    centers[j] = np.median(clusters[j], axis = 0)
            iter_num = iter_num + 1
        
        # end k-median, get centers
        codebooks[i] = centers
        # final cluster
        cluster = [np.argmin(cdist(centers, [data], 'cityblock')) for data in data_parts[i]]
        codes[i] = cluster
    
    # reshape codes
    codes = np.dstack(codes)[0]
    return codebooks, codes

# --------------------- Part 2 ---------------------

'''
# get location in ADs from location_index
def get_location(ADs, location_index):
    return [ADs[i][location_index[i]][0] for i in range(len(location_index))]

# get distance in ADs from location_index
def get_distance(ADs, location_index):
    return sum([ADs[i][location_index[i]][1] for i in range(len(location_index))])
'''

def query(queries, codebooks, codes, T):
    
    candidates = []
    
    _, M = queries.shape
    P, K, _ = codebooks.shape
    # print(M, P, K) get 128 2 256
    
    # create inverted index
    # inverted index is a dict with {(location), {v_index}}
    inverted_index = {}
    for i in range(codes.shape[0]):
        location = tuple(codes[i])
        if location not in inverted_index:
            inverted_index[location] = {i}
        else:
            inverted_index[location].add(i)
    
    #print(inverted_index)
        
    # for each query
    for query in queries:
        candidate_points = set()
        
        # break query into P parts
        query_parts=np.hsplit(query, [i * M // P for i in range(1,P)])
        
        # ADs is a list store P parts
        # each parts have K(256) data which is like [num, distance]
        ADs = []
        # caculate distance for P parts
        for i in range(P):
            index = np.arange(0, K, 1)
            distance = cdist(codebooks[i], [query_parts[i]], 'cityblock')
            AD_part = np.dstack([index, np.dstack(distance)[0][0]])[0]
            AD_part = AD_part[np.lexsort(AD_part.T)]
            ADs.append(AD_part)
        
        # use location_index to make the current location index in each AD
        location_index = np.zeros(P, dtype=int)
        
        # store information like [[location_index, distance]] in location_index_stack
        location_index_stack = [[location_index, sum([ADs[i][location_index[i]][1] for i in range(len(location_index))]) ]]
        # add candidate points of first location
        first_location = tuple([ADs[i][location_index[i]][0] for i in range(len(location_index))])
        
        # location_already_visit use 0 and 1
        # 0 means the location_index in data have not been used
        location_index_already_visit = np.zeros([K for num in range(P)], dtype = "uint8")
        
        if first_location in inverted_index:
            candidate_points = candidate_points | inverted_index[first_location]
        
        # set the first location index as 1
        location_index_already_visit[np.ix_(*np.vstack(location_index))] = 1
                
        # in each iteration
        # 1. pop location_index_stack[0] and add 1 to each number of the index in location_index_stack[0]
        # 2. then caculate their distance add these data into location_index_stack[0]]
        # 3. add these data into location_index_stack[0]] by order
        # 4. try to find point match location_index_stack[0] and add points into results
        while len(candidate_points) < T:
            # create all +1 on location_index_stack[0]
            location_index = location_index_stack.pop(0)[0]
            for i in range(P):
                # if still can +1 on this index
                if K > location_index[i] + 1:
                    temp_index = location_index.copy()
                    temp_index[i] = temp_index[i] + 1
                    temp_distance = sum([ADs[i][temp_index[i]][1] for i in range(len(temp_index))])
                    #print(type(location_index_stack))
                    #if the location index has not been added
                    if int(location_index_already_visit[np.ix_(*np.vstack(temp_index))]) == 0:
                        # add the data into location_index_stack by order
                        add_success = False
                        for j in range(len(location_index_stack)-1,-1,-1):
                            if temp_distance > location_index_stack[j][1]:
                                location_index_stack.insert(j+1, [temp_index, temp_distance])
                                add_success = True
                                break
                        # if it needs to be added to the first place
                        if add_success == False:
                            location_index_stack.insert(0, [temp_index, temp_distance])             
                        # set the location index as already added
                        location_index_already_visit[np.ix_(*np.vstack(temp_index))] = 1
            temp_location = tuple([ADs[i][location_index_stack[0][0][i]][0] for i in range(len(location_index_stack[0][0]))])
            # find points match the location
            if temp_location in inverted_index:
                candidate_points = candidate_points | inverted_index[temp_location]
        # end this query
        candidates.append(candidate_points)
    
    return candidates
