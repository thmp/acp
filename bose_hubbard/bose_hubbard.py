#!/usr/bin/python
import numpy as np
import scipy as sp
import time
import math
import scipy.sparse as sparse
import scipy.sparse.linalg as splg
from bisect import bisect_left
import matplotlib.pyplot as plt

class Timer: 
	def __init__(self):
		self.start = time.time()
		self.tround = self.start

	def round(self):
		interval = time.time() - self.tround
		self.tround = time.time()
		return interval

	def stop(self):
		return time.time() - self.start

print "Start basis construction"

timer = Timer()

M = 5 # site number 
N = 5 # atom number

dim = math.factorial(N+M-1) / math.factorial(N) / math.factorial(M-1)

print dim, "dimension"

basis = np.zeros((dim, M))
basis[0,:] = np.zeros((M))
basis[0,0] = N
k = 0

while (basis[k,M-1] != N):

	# find first non-zero entry
	atZero = True
	for i in range(M-2,-1,-1):
		if (basis[k,i] == 0 and atZero):
			basis[k+1,i] = 0
		elif (basis[k,i] != 0 and atZero):
			kP1 = i+1
			basis[k+1,i] = basis[k,i] - 1
			basis[k+1,kP1] = N - basis[k+1,i]
			atZero = False
		elif (basis[k,i] != 0):
			basis[k+1,i] = basis[k,i]
			basis[k+1,kP1] -= basis[k+1,i]

	k += 1

print k, "loops"
print timer.round(), "seconds"

print "Start basis tagging"

#w = [math.sqrt(100*i + 3) for i in range(M)]
#def T(v):
#	global w
#	return np.dot(w,v)
def T(v):
	return hash(v.tostring())

tags = []

for i in range(dim):
	tags.append( ( i, T(basis[i,:]) ) )

print timer.round(), "seconds"

print "Sort basis"

tags.sort(key=lambda r: r[1])
keys = [r[1] for r in tags]

print timer.round(), "seconds"

#lookup = np.array((0,0,0,0,11,0,0,0,0,0,0))

def lookup(ptag):
	global tags
	global keys
	tag= tags[bisect_left(keys, ptag)]
	return tag[0]

lookupa = np.array([1 for i in range(M)])
print "Test sorting, look for", lookupa
tag = T(lookupa)
i = lookup(tag)
#i = tags[bisect_left(keys, tag)]
print "index:", i

print "Calculate sparse Hamiltonian matrix"

H = sparse.dok_matrix((dim,dim))

# take j to be neighbor to the right
def adagger_a(v,i):
	v = v.copy()
	if i < v.shape[0]-1:
		j = i+1
	else:
		j = 0
	# check if we can annihilate
	if (v[i] > 0):
		pref = v[i]
		v[i] -= 1
		v[j] += 1
		pref *= v[j]
		pref = math.sqrt(pref)
		return (pref, lookup(T(v)))
	else: 
		return (0, -1)

def adagger_a_spdm(v,i,j):
	v = v.copy()
	# check if we can annihilate
	if v[j] > 0:
		pref = v[j]
		v[j] -= 1
		v[i] += 1
		pref = math.sqrt(v[i]*pref)
		return (pref, lookup(T(v)))
	else: 
		return (0, -1)

def a_adagger(v,i):
	v = v.copy()
	if i < v.shape[0]-1:
		j = i+1
	else:
		j = 0
	# check if we can annihilate
	if (v[j] > 0):
		pref = v[j]
		v[j] -= 1
		v[i] += 1
		pref = math.sqrt(v[i]*pref)
		return (pref, lookup(T(v)))
	else: 
		return (0, -1)

J = 1.
U = 1.

# loop through all the basis vectors
for k in range(dim): # dimension
	#if (k % 1000 == 0):
		#print "at dim", k
	for i in range(M): # site number 
		# apply hopping procedure
		ham_1 = adagger_a(basis[k,:], i)
		ham_2 = a_adagger(basis[k,:], i)
		if ham_1[1] != -1:
			H[ham_1[1], k] += -J*ham_1[0]
		if ham_2[1] != -1:
			H[ham_2[1], k] += -J*ham_2[0]
		H[k,k] += U/2.*basis[k,i]*(basis[k,i]-1)

print timer.round(), "seconds"

print H.toarray()

print "Calculate Lanczos eigenvalues and eigenvectors"

w, v = splg.eigsh(H,k=6,which='SA') # lowest eigenvalue

print w
print v[:,0]

print "Calculate SPDM single particle density matrix"

spdm = np.zeros((M,M))

for i in range(M):
	for j in range(M):
		# sum through basis
		for k in range(dim):
			pref, k_new = adagger_a_spdm(basis[k,:], i, j)
			if k_new > 0:
				spdm[i,j] += v[0,k_new]*v[0,k]*pref

print spdm
print spdm.trace()


print timer.round(), "seconds"

#plt.figure()
#plt.spy(H, markersize=1)
#plt.show()