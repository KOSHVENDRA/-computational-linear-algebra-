#SVD decomposition using built-in numpy function

import numpy as np
import time

A=np.array([[0,1,1],[0,1,0],[1,1,0],[0,1,0],[1,0,1]])

start_time=time.time()
U,S,V =np.linalg.svd(A)     # s.t. A=U*S*V (*represents matrix multoplication)
end_time=time.time()

time_taken=end_time-start_time

print("using numpy built in function")
print("time taken in running the built-in numpy function ",time_taken)
print("singular value decomposition of A:" ,S)
print("matrix U ",U)
print("matrix V",V)


# SVD decomposition of the given matrix A by coding


print('by coding the algorithm ')
   # this function gives the U and V matices of svd decomposition of A , this code is specific to the matrices of dimension as that of A

def svd(x):                              
    l,v=np.linalg.eigh(np.dot(np.transpose(A),A))
    v=np.transpose(V)
    v[[0,2],:]=v[[2,0],:]
    print("matrix V=",v)
    m,u=np.linalg.eigh(np.dot(A,np.transpose(A)))
    u[:,[1,3]]=u[:,[3,1]]
    u[:,[0,4]]=u[:,[4,0]]
    print("matrix U=",u)
    
    singval=[]                          # for writing singular values of A
    for i in range(3):
           k=np.sqrt(l[i])
           singval.append(k)
    print("the singular values for the matrix A are",singval)
           

A=np.array([[0,1,1],[0,1,0],[1,1,0],[0,1,0],[1,0,1]])

start_time=time.time()
svd(A)
end_time=time.time()

time_taken=end_time-start_time
print("time taken in the code execution for SVD calculation :", time_taken)

