#solving set of linear equations ,AX=B
import numpy as np

A=np.array([[0.2,0.1,1,1,0],[0.1,4,-1,1,-1],[1,-1,60,0,-2],[1,1,0,8,4],[0,-1,-2,4,700]])
B=np.array([1,2,3,4,5])

#initial guess of solution
X_i=np.array([0.0,0.0,0.0,0.0,0.0])
#known solution
X_f=np.array([7.859713071,0.422926408,-0.073592239,-0.540643016,0.010626163])
I=np.identity(5)
# matrix with diagonal elements of A
diagA=A*I
# forming inverse matrix of 'diagA'
k=np.zeros((5,5))
for l in range(5):
    k[l,l]=1/diagA[l,l]
    
# code for iteration in jacobi method
for j in range(150):
    X_i=np.dot((-np.dot(A,X_i)+B+np.dot(diagA,X_i)),k)
    if np.amax(np.absolute(X_i-X_f))<0.01:
        break

print("number of iterations in jacobi method",j)
print("solution :",X_i)


# Gauss Seidel Method
X_i=np.array([0.0,0.0,0.0,0.0,0.0])          # initial guess
for k in range(500):               # iteration limit
    for i in range(5):             # loking for ith element of solution array
        r=B[i]
        for j in range(5):         
            if j!=i:
                r=r-A[i,j]*X_i[j]
        X_i[i]=r/A[i,i]                      # updating the solution array
    if np.amax(np.absolute(X_i-X_f))<0.01:
        break
    
print('number of iterations in Gauss Seidel method:',k)
print('solution',X_i)


# Relaxation method

X_i=np.array([0.0,0.0,0.0,0.0,0.0])
w=1.25
for k in range(500):
    for i in range(5):
        s=B[i]
        for j in range(5):
            s=s-A[i,j]*X_i[j]
        X_i[i]=((s/A[i,i])*w)+X_i[i]
    if np.amax(np.absolute(X_i-X_f))<0.01:
        break
print("number of iterations in relaxation method",k)
print('solution :',X_i)
        
