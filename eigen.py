# finding eigenvalue by coding ,using q,r decomposition
import numpy as np

A= np.array([[5,-2],[-2,8]])

for i in range(20):
    q,r =np.linalg.qr(A)
    A=np.dot(r,q)

eign1=[]
for i in range(2):
    k=A[i,i]
    eign1.append(k)

print("eigenvalues are :",eign1)

#finding eigenvalues using numpy function

eign2=np.linalg.eigh(A)
print("eigenvalues using numpy functions:",eign2)


# power method for getting dominant eigenvalue and corresponding eigenvector of a given matrix A

A=np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])  #given matrix
k=np.dot(A,A)                              #square of given matrix
m=np.dot(k,A)                              # cube of given matrix
x=np.array([1,0,0])                        # x and y are arbitrary vector 
y=np.array([1,0,0])

for i in range(500):                       # iteration limit
    x=np.dot(A,x)
    w=np.dot(A,x)
    q=np.dot(A,w)
    l=(np.dot(q,y))/(np.dot(w,y))
    p=(np.dot(w,y))/(np.dot(x,y))
    if (np.absolute(l-p))/l < 0.01:        # putting accuracy limit
        break
z=0                                        # for normalising the eigenvector
for i in range(3):
    z=z+ (w[i]*w[i])

print("dominant eigenvalue :",l)
print("dominant eigenvector :",w /(np.sqrt(z)))    # normlised eigenvector

