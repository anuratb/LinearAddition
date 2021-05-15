import numpy as np
x = np.random.rand(2000,2)
print(x)
c = np.array([5,9])
y = np.array([np.dot(c,z) for z in x])
y = np.reshape(y,(2000,1))
print(y)
w = np.zeros((2))
yhat = np.array([np.dot(w,z) for z in x])
yhat = np.reshape(yhat,(2000,1))
def find_inaccuracy(y,yhat):
    n = y.size
    cnt = 0.0
    for i in range(n):
        cnt+=((abs(y[i]-yhat[i])*100.0)/y[i])
    cnt/=n
    return cnt
def loss(y,yhat):
    n = y.length
    cnt = 0.0
    for i in range(n):
        cnt+=((abs(y[i]-yhat[i]))**2)
    cnt/=n
    return cnt
iterations = 2501
N = 2000
learning_rate = 0.0001
for j in range(iterations):
    dLbydw =  np.sum(x*(y-yhat),axis=0)/2
    w = w + learning_rate*dLbydw    
    yhat = np.array([np.dot(w,z) for z in x])
    yhat = np.reshape(yhat,(2000,1)) 
    if(j%500==499):
        print("Iteration",j+1)
        print(w)
        print(str(y[0])+" "+str(yhat[0]))  
        print("Accuracy" + str(abs(100-find_inaccuracy(y,yhat)))+"%")   
