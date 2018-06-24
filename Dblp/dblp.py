import json
data = json.load(open('dblp.json'))
# keys = list(data.keys())

# data_categories = data[keys[0]]
# data_papers = data[keys[1]]

# data_categories = data[keys[1]]
# data_papers = data[keys[0]]

data_categories = data['categories']
data_papers = data['data']

print(1)

AI = data_categories['artificial_intelligence']
DB = data_categories['databases']
DM = data_categories['big_data']
IR = data_categories['information']

uniounRa = list(set(DM) | set(IR) | set(AI) | set(DB) )

matrix = []
for i in range(len(data_papers)):
	if data_papers[i]['venue'] in uniounRa:
			matrix.append([data_papers[i]['venue'], data_papers[i]['authors'].split(",")])

print(2)
authors_dict = {}
for i in range(len(matrix)):
	for j in range(len(matrix[i][1])):
		if matrix[i][1][j] in authors_dict:
			authors_dict[matrix[i][1][j]] += 1
		else:
			authors_dict[matrix[i][1][j]] = 1

authors_keys = list(authors_dict.keys())

author_List= []
for key, value in authors_dict.items():
    author_List.append([key, value])

sorted_papers = sorted(author_List, key = lambda x: int(x[1])  , reverse=True)

new_author = {}
for i in range(5000):
	new_author[sorted_papers[i][0]] = [ sorted_papers[i][1] , i]

import numpy as np
adjmatrix = np.zeros((5000,5000), dtype=int)

ramatrix = np.zeros( (5000,4), dtype = int )

for i in range(len(matrix)):
	temp = np.zeros(( len(matrix[i][1]),), dtype=int)
	for j in range(len(matrix[i][1])):
		if matrix[i][1][j] in new_author:
			temp[j] = new_author[matrix[i][1][j]][1]
		else:
			temp[j] = -1
	temp = temp[temp >= 0]
	for j in range(len(temp)):
		for k in range(len(temp)):
			adjmatrix[temp[j]][temp[k]]=1
		if matrix[i][0] in AI:
			ramatrix[temp[j]][0]+=1
		if matrix[i][0] in DB:
			ramatrix[temp[j]][1]+=1
		if matrix[i][0] in DM:
			ramatrix[temp[j]][2]+=1
		if matrix[i][0] in IR:
			ramatrix[temp[j]][3]+=1


attra = np.zeros((5000,), dtype=int)

for i in range(5000):
	attra[i] = np.argmax(ramatrix[i])



for i in range(5000):
	adjmatrix[i][i]=0

attpr = np.zeros((5000,), dtype=int)
for i in range(5000):
	attpr[i] = sorted_papers[i][1]


highlyprolific = np.where((attpr >= 20))
prolific = np.where((attpr >= 10) & (attpr <20))
lowprolific = np.where((attpr > 0) & (attpr <10))

attpr[highlyprolific] = 0
attpr[prolific] = 1
attpr[lowprolific] = 2


links = adjmatrix.astype(np.float64)
attributes = np.column_stack((attpr,attra))
print(3)

#cluster
sigma=1
k=9
c=0.15
l=20

#transition_probability_matrix and weights
n = links.shape[0]
attrs = int(attributes.shape[1])
w = np.ones(attrs+1, dtype = float)

#calculating the number of diff types of sub attr in a particular attribute
no_of_attr_individual = np.zeros(attrs, dtype = float)
no_of_attr_cumulative = np.zeros(attrs, dtype = float)
for i in range (attrs):
  no_of_attr_individual[i]=np.unique(attributes[:,i]).shape[0]
  if i==0:
    no_of_attr_cumulative[i]= no_of_attr_individual[i]
  else:  
    no_of_attr_cumulative[i]= no_of_attr_cumulative[i-1]+no_of_attr_individual[i]
  
transition_pm = np.zeros( (n+int(no_of_attr_cumulative[attrs-1]), n+int(no_of_attr_cumulative[attrs-1])) , dtype=float)

print(4)
def caltransition(w):
  #trasition matrix Pa
  #filling matrix of vertex to vertex
  print(4.1)
  sigma_w = np.sum(w) - w[0]
  for i in range(n):
    total_neighbours = np.sum(links[i])
    indices = np.where(links[i]==1)
    transition_pm[i][indices] = w[0]/(total_neighbours*w[0] + sigma_w)
    
    for j in range(attrs):
        if j == 0:
          transition_pm[i][n+int(attributes[i][j]) ] = w[j+1]/(total_neighbours*w[0] + sigma_w)
        else:
          transition_pm[i][n+int(attributes[i][j]) + int(no_of_attr_cumulative[j-1]) ] = w[j+1]/(total_neighbours*w[0] + sigma_w)
  print(4.2)
  # filling matrix from attribute vertex to vertex
  for i in range( attrs ):
    temp_neighbours_attr = np.zeros( int(no_of_attr_individual[i]) )
    num_attr = int(no_of_attr_individual[i])
    for j in range(num_attr):
      temp_neighbours_attr[j] = np.where(attributes[:,i] == j)[0].size 
      if i == 0:
        transition_pm[n+j][np.where(attributes[:,i] == j)] = 1./temp_neighbours_attr[j]
      else:
        transition_pm[n+j+int(no_of_attr_cumulative[i-1])][np.where(attributes[:,i] == j)] = 1./temp_neighbours_attr[j]
  #random walk
  print(4.3)
  random_walk=np.zeros( (n+int(no_of_attr_cumulative[attrs-1]), n+int(no_of_attr_cumulative[attrs-1])) , dtype=float)
  for i in range(1,l+1):
      random_walk+=c*((1-c)**i)*np.linalg.matrix_power(transition_pm , i)
  random_walk=random_walk[:n,:n]
  return random_walk

random_walk = caltransition(w)


print(5)


influence_function =  1- np.exp(-np.square(random_walk)/(2*(sigma**2)))
density_function=np.sum(influence_function,axis=0)
# doubt on adding at axis
centroid=np.argsort(-density_function, kind = 'mergesort')[:k] #centroid has indices of top k in desc


curr_cluster = np.zeros(n , dtype='int')
past_cluster = np.zeros(n , dtype='int')
# cluster_averages = np.empty((k,n))

unstable = True
maxiter = 20
loop = 0

while (unstable and (loop<maxiter)) :
  print(6)
  for i in range(n):
    curr_cluster[i] = np.argmax(random_walk[i,centroid])
  if np.array_equal(curr_cluster,past_cluster):
  	# print('yes', maxiter)
  	unstable=False
  	# print((loop<maxiter))
  	# print(unstable)
  	# print('yes', loop)
  past_cluster = np.array(curr_cluster)
  print(7)
  for i in range(k):
    which_cluster=np.where(curr_cluster==i)[0]
    if which_cluster.size!=0:
      cluster_average=np.mean(random_walk[which_cluster,:],axis=0)
      cluster_avgpoint_dist = np.zeros(which_cluster.size)
      for j,node in enumerate(which_cluster):
        cluster_avgpoint_dist[j] = np.linalg.norm(random_walk[node] - cluster_average)
      centroid[i] = which_cluster[np.argmin(cluster_avgpoint_dist)]
  print(8)
  dw = np.zeros(attrs, dtype = float)
  for i in range(k):
    which_cluster=np.where(curr_cluster==i)[0]
    if(which_cluster.size != 0):
      for j in range(attrs):
        dw[j]+= np.sum(attributes[centroid[i]][j] == attributes[which_cluster][:,j:j+1])
  sdw = np.sum(dw)
  for i in range(attrs):
    w[i+1] = (w[i+1] + (dw[i]/sdw) )/2
  random_walk = caltransition(w)
  print(9)
  obj = 0.
  for i in range(k):
      which_cluster = np.where(curr_cluster==i)[0]
      if which_cluster.size != 0:
          obj += np.mean(random_walk[which_cluster , which_cluster])
      print(which_cluster.size)
  loop+=1
  print ('Obj: ',obj)
  print('Weights: ',w)

print(10)
Esum = 0
Cesum = 0

for i in range(n):
    for j in range(n):
        if(links[i][j] == 1):
            Esum += 1
            if(curr_cluster[i] == curr_cluster[j]):
                Cesum +=1
dens = (Cesum/Esum)

print('Desity: ', dens)

import math
def entropy(i, j):
  entsum = 0
  p = np.zeros(int(no_of_attr_individual[i]), dtype = np.float64 )
  pd = len(np.where(curr_cluster == j)[0])
  for kl in range(int(no_of_attr_individual[i])):
    p[kl] = len(np.where(attributes[np.where(curr_cluster==j)[0]][:,i:i+1] ==kl  )[0])
    if p[kl] != 0:
      entsum += -(p[kl]/pd) * ( math.log( (p[kl]/pd),2 ) )
  return entsum

entropysum = 0
for i in range(attrs):
  easum = 0 
  for j in range(k):
    easum += len(np.where(curr_cluster == j)[0])/n * entropy(i,j)
    print(easum)
  entropysum += easum * (w[i+1]/(w.sum() - w[0]))
print('Entropy: ', entropysum)

# SUmanth
def entropy(i,j):
  temp=0
  which_cluster = np.where(curr_cluster==j)[0]
  for t in range(int(no_of_attr_individual[i])):
    p=0
    for x in range(which_cluster.size):
      if(attributes[which_cluster[x]][i]==t):
        p+=1
    if(p!=0):
      temp+=(p/which_cluster.size)*(math.log((p/which_cluster.size),2)  )
  temp=temp*-1
  return temp

esum=0
for i in range(attrs):
  temp=0
  for j in range(k):
    temp+=len((np.where(curr_cluster==j)[0]))*entropy(i,j)
  esum+=w[i+1]*temp
esum=esum/((w.sum()-w[0])*n)  


print(esum)






