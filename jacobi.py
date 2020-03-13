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

X_i=np.array([0.0,0.0,0.0,0.0,0.0])           # initial guess of solution
w=1.25
for k in range(500):                          # iteration limit       
    for i in range(5):
        s=B[i]
        for j in range(5):
            s=s-A[i,j]*X_i[j]
        X_i[i]=((s/A[i,i])*w)+X_i[i]          # updating the solution array
    if np.amax(np.absolute(X_i-X_f))<0.01:    # precision limit
        break
print("number of iterations in relaxation method",k)
print('solution :',X_i)
        

# conjugate gradient method

X_i =np.array([0.0,0.0,0.0,0.0,0.0])        # initial guess to start with

# p's are the set of conjugate vector with respect to matrix A ,which forms the basis for the solution vector to be expanded on 

r_0 = np.subtract(B,np.dot(A,X_i))      # initial residual vector
p_0 = r_0

for i in range(500):
    x=np.dot(np.transpose(r_0),r_0)
    y=np.dot(np.transpose(p_0),np.dot(A,p_0))
    c=x/y                            # expansion coefficient of solution vector in basis of P
    X_i=X_i+c*p_0                             # updating solution vector
    r_0=np.subtract(r_0,(c*np.dot(A,p_0)))    # updating residual vector
    d = (np.dot(np.transpose(r_0),r_0)) / x
    p_0 = r_0 + d*p_0                         # updating P's

    if np.amax(np.absolute(X_i-X_f))<0.01:   # precision limit
        break
print ("number of iterations in relaxation method is:",i)
print ("the solution is :",X_i)
