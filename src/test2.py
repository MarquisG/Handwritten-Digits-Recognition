import scipy.io as sio
import numpy as np
import scipy.optimize as sc
import copy
np.set_printoptions(nanstr='NaN')
np.set_printoptions(threshold=10000000)
np.set_printoptions(linewidth=999999)

def sigmoidGradient(z):
    return np.multiply(sigmoid(z),(1-sigmoid(z)));

def vectorize(v1, v2):
    return np.append(np.ravel(v1), np.ravel(v2))

def backProp(p, num_input, num_hidden, num_labels, X, yvalue, l=1):

    Theta1 = np.reshape(p[:num_hidden*(num_input+1)], (num_hidden,-1))
    Theta2 = np.reshape(p[num_hidden*(num_input+1):], (num_labels,-1))
    m = len(X)
    delta1 = 0
    delta2 = 0
    for t in range(m):

        a1 = np.matrix(np.append([1],X[t],axis=0)).transpose()
        z2 = Theta1*a1
        a2 = np.append(np.ones(shape=(1,z2.shape[1])), sigmoid(z2),axis=0)
        z3 = Theta2*a2
        a3 = sigmoid(z3)
        w = np.zeros((num_labels,1))
        w[int(yvalue[t])-1] = 1
        d3 = (a3-w)
        d2 = np.multiply(Theta2[:,1:].transpose()*d3, sigmoidGradient(z2))
        delta1 += d2*a1.transpose()
        delta2 += d3*a2.transpose()
        
    
    Theta1_grad = (1/m)*delta1 + (l/m)*np.append(np.zeros(shape=(Theta1.shape[0],1)), Theta1[:,1:], axis=1);
    Theta2_grad = (1/m)*delta2 + (l/m)*np.append(np.zeros(shape=(Theta2.shape[0],1)), Theta2[:,1:], axis=1);
    answer = vectorize(Theta1_grad, Theta2_grad)
    return answer


def J(theta, num_input, num_hidden, num_lables,X, yvalue, l=1):
    Theta1 = np.reshape(theta[:num_hidden*(num_input+1)], (num_hidden,-1))
    Theta2 = np.reshape(theta[num_hidden*(num_input+1):], (num_lables,-1))
    m = len(X)
    X = np.append(np.ones(shape=(X.shape[0],1)),X,axis=1)
    J = 0
    for i in range(m):
        x = np.matrix(X[i])
        w = np.zeros((10,1))
        w[int(yvalue[i])-1] = 1
        hx = sigmoid(Theta2*np.append([[1]], sigmoid(Theta1*x.transpose()), axis=0))
        J += sum(-w.transpose()*np.log(hx)-(1-w).transpose()*np.log(1-hx))
    J = J/m
    J += (l/(2*m))*(sum(sum(Theta1[:,1:]**2)) + sum(sum(Theta2[:,1:]**2)))    
    return float(J)

def calculateGrad(p):
    return backProp(p, 900, 25, 10, Xtrain,ytrain, la)
    
def calculateJ(p):
    return J(p, 900, 25, 10, Xtrain,ytrain, la)


def sigmoid(z):
    return 1/(1+np.power(np.e,-z))

def splitData(X, y):
    size1 = X.shape[0] * 0.6
    size2 = X.shape[0] * 0.2
    Xtrain = np.zeros((size1,X.shape[1]))
    Xcv = np.zeros((size2,X.shape[1]))
    Xtest = np.zeros((size2,X.shape[1]))
    ytrain = np.zeros((size1,1))
    ycv = np.zeros((size2,1))
    ytest = np.zeros((size2,1))
    for i, v in enumerate(np.random.permutation(len(y))):      
        try:
            Xtrain[i] = X[v]
            ytrain[i] = y[v]
        except:
            try:
                Xcv[i-size1] = X[v]
                ycv[i-size1] = y[v]
            except:
                Xtest[i-(size1+size2)] = X[v]
                ytest[i-(size1+size2)] = y[v]
    return Xtrain, Xcv, Xtest, ytrain, ycv, ytest

def randomInitialization(i, epsilon=0.12):
    return np.random.rand(i,1)*2*epsilon-epsilon
            
            
mat_contents = sio.loadmat('newX.mat')
X = mat_contents['X']
mat_contents = sio.loadmat('newy.mat')
y = mat_contents['y']
num_hidden = 25
num_input = 900
num_lables = 10

Xt, Xcv, Xtest, yt, ycv, ytest = splitData(X,y)
Xtrain, ytrain = Xt, yt
la = 0.001
rndInit = randomInitialization(num_hidden*901+10*(num_hidden+1))

print("\n=================================================")
print("Learning curves\n")
print("lambda=",la)
print("# of training samples | Training error | Cross validation error")
for i in range(Xtrain.shape[0]//50):
    Xtrain = Xt[0:i*50+1,:]
    ytrain = yt[0:i*50+1,:]
    answer =  sc.fmin_cg(calculateJ, rndInit, calculateGrad, maxiter=90,  disp=False)
    J1 = J(answer, 900, 25, 10, Xtrain,ytrain, 0)
    J2 = J(answer, 900, 25, 10, Xcv,ycv, 0)
    print(i*50, J1, J2)
print("The program will terminate now")
input()
