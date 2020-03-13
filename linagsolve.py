#code to solve set of linear scalar equations: ax=b
import numpy as np

a=np.array([[1,0.67,0.33],[0.45,1,0.55],[0.67,0.33,1]])   
b=np.array([2,2,2])

#solving for x
x=np.linalg.solve(a,b)
print(x)
