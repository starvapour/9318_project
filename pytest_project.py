#coding=utf-8

'''
L1正则化是指权值向量ww中各个元素的绝对值之和
L2正则化是指权值向量ww中各个元素的平方和然后再求平方根
'''
#########################################################################

# 作业部分
import scipy
import numpy as np
from scipy.cluster.vq import vq,whiten

# --------------------- Part 1 ---------------------

# caculate L1 distance
def L1_dis(v1, v2):
	return np.linalg.norm(v1-v2,ord=1)
	
# kmeans function, use L1 distance
def kmeans(data, K, max_iter, centers):
	iter_num = 0
	v_num = data.shape[0]
	change_exist = True
	
	# for each iter
	while iter_num <= max_iter and change_exist == True:
		#print("开始第",iter_num,"次迭代")
		change_exist = False
		# store all the vector's cluster result
		clusters = [[] for _ in range(K)]
		
		# find the nearest center for each vector
		# then add vector into cluster
		for vector in data:
			cluster_num = 0
			min_distance = L1_dis(vector, centers[0])
			# caculate distance between the vector and each center
			for i in range(1, len(centers)):
				distance = L1_dis(vector, centers[i])
				if distance < min_distance:
					min_distance = distance
					cluster_num = i
			# after get the final cluster_num
			clusters[cluster_num].append(vector)
		
		# caculate new centers
		for i in range(K):
			if len(clusters[i]) != 0:
				# 暂时使用平均数更新质心，不确定是否要改用中位数
				new_center = np.mean(clusters[i],axis = 0)
				if not (new_center == centers[i]).all():
					centers[i] = new_center
					change_exist = True
		'''
		#cluster,_ = vq(data,centers)
		#print(centers)
		#print()
		'''
		iter_num = iter_num + 1
	
	return centers

# data NxM, N个向量，M维
# 分成p块
# 【P,K,M/P】，K是256，将K，M/P作为初始质心
# 最大迭代次数上限
def pq(data, P, init_centroids, max_iter):
	#data = whiten(data)
	N, M = data.shape
	K = init_centroids.shape[1]
	
	# split data into P parts
	step = M // P
	data_parts=np.hsplit(data, [i * step for i in range(1,P)])
	codebooks = np.empty(init_centroids.shape, dtype = "float32")
	codes = np.empty([P, N], dtype = "uint8")
	# do k_means for each part
	for i in range(len(data_parts)):
		centers = kmeans(data_parts[i],K,max_iter,init_centroids[i])
		codebooks[i] = centers
		cluster,_ = vq(data_parts[i],centers)
		codes[i] = cluster
	
	codes = np.dstack(codes)[0]
	return codebooks, codes

# --------------------- Part 2 ---------------------

# return the distance for list sort
def return_distance(dis_list):
	return dis_list[1]

# get location in ADs from location_index
def get_location(ADs, location_index):
	return [ADs[i][location_index[i]][0] for i in range(len(location_index))]

# get distance in ADs from location_index
def get_distance(ADs, location_index):
	return sum([ADs[i][location_index[i]][1] for i in range(len(location_index))])

# 对比找到距离query最近的几个center
# 这些center中包含的向量数量大于等于T
# 输出所有包含在内的向量
# 对每一个quert执行以上操作之后整合为一个列表
# codebooks.shape(2, 256, 64), codes.shape(768, 2)
def query(queries, codebooks, codes, T):
	
	candidates = []
	
	_, M = queries.shape
	P, K, _ = codebooks.shape
	# print(M, P, K) get 128 2 256
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
			query_part = query_parts[i]
			AD_part = []
			for j in range(K):
				AD_part.append([j, L1_dis(query_part, codebooks[i][j])])
			
			AD_part.sort(key = return_distance)
			ADs.append(AD_part)
		
		# use location_index to make the current location index in each AD
		location_index = [0 for i in range(P)]
		
		# store information like [[location_index, distance]] in location_index_stack
		location_index_stack = [[location_index, get_distance(ADs, location_index)]]
		
		# add candidate points of first location
		first_location = get_location(ADs, location_index)
		for i in range(len(codes)):
			if (first_location == codes[i]).all():
				candidate_points.add(i)
		
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
					temp_distance = get_distance(ADs, temp_index)
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
			# find points match the location
			temp_location = get_location(ADs, location_index_stack[0][0])
			for i in range(len(codes)):
				if (temp_location == codes[i]).all():
					candidate_points.add(i)
		# end this query
		candidates.append(candidate_points)
	
	return candidates
		
	



#########################################################################

# 测试部分
import pickle
import time

# How to run your implementation for Part 1
with open('./toy_example/Data_File', 'rb') as f:
	Data_File = pickle.load(f, encoding = 'bytes')
with open('./toy_example/Centroids_File', 'rb') as f:
	Centroids_File = pickle.load(f, encoding = 'bytes')
#print(Data_File.shape)
#print(Centroids_File.shape)
data = Data_File
centroids = Centroids_File
start = time.time()
codebooks, codes = pq(data, P=2, init_centroids=centroids, max_iter = 20)
end = time.time()
time_cost_1 = end - start

print("第一阶段用时：",time_cost_1,"秒")
#print(codebooks.shape)
#print(codes.shape)

# How to run your implementation for Part 2
with open('./toy_example/Query_File', 'rb') as f:
	Query_File = pickle.load(f, encoding = 'bytes')
queries = Query_File
start = time.time()
# queries is [[9. 9. 9. ... 9. 9. 9.]]
candidates = query(queries, codebooks, codes, T=10)
end = time.time()
time_cost_2 = end - start

# output for part 2.
print("第二阶段用时：",time_cost_2,"秒")
print("最后输出为",candidates)
print("期望输出为","[{3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}]")

