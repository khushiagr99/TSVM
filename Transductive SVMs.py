
import scipy.io
import pandas as pd
import numpy as np
import sklearn
import csv
import cvxopt
import random
import matplotlib.pyplot as plot

# Load data
data = pd.read_csv('Aust.csv', sep=',',header=None)
data = np.array(data)

# Separate X(features) and Y(labels)
no_of_features = np.shape(data)[1]-1
X = data[:,0:no_of_features]
Y = data[:,no_of_features]

# Split training and test data (70% for training and 30% for test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

[n, d] = X_train.shape
#Setting the noise parameter 
noise = 0.1

#Switching labels according to noise
len_y = y_train.shape[0]
for i in range(int(noise*len_y)):
    r = random.randint(0,len_y-1)
    if(y_train[r]==1):
        y_train[r] = -1
    else:
        y_train[r] = 1

# no. of labelled samples
L = np.shape(X_train)[0]
# no. of unlabelled samples
U = np.shape(X_test)[0]


# SVM classifier
from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# initial weights (according to SVM classifer)
w0 = np.matmul(np.transpose(clf.support_vectors_),clf.dual_coef_.transpose())
b0 = clf.intercept_


# For TSVM, train the data as well as predict labels for unlabelled data

# Concatenate train and test data
XX = np.concatenate((X_train, X_test), axis = 0)

# Assign label = -2 for unlabelled data
yy = np.concatenate((y_train, -2*np.ones(U)), axis = None)
XX = XX.transpose()


#TSVM-dual form

#The following fucntion solves the following convex quadratic programming task:
#   min 0.5*x'*H*x + f'*x 
#    x
#
#   subject to a'*x = b 
#              LB(i) <= x(i) <= UB(i) for all i=1:n
#
#It then updates the weight and bias term until convergence.

def train_linear_transductive_svm(X,y,C1,C2,w0,b0):
    s = -0.2
    unlabeled = np.where(y==-2)
    pos = np.where(y==1)
    neg = np.where(y==-1)
    npos = len(pos[0])
    nneg = len(neg[0])
    XX = np.concatenate((X[:,pos], X[:,neg]), axis = 2)
    XX = np.concatenate((XX, X[:,unlabeled]), axis = 2)
    XX = np.concatenate((XX, X[:,unlabeled]), axis = 2)
    yy = np.concatenate((np.ones(npos), -np.ones(nneg)), axis = None)
    yy = np.concatenate((yy, np.ones(U)), axis = None)
    yy = np.concatenate((yy, -np.ones(U)), axis = None)
    y0 = 1
    x0 = np.sum(np.squeeze(X[:,unlabeled]), axis = 1) / U
    XX = np.squeeze(XX)
    x0 = np.reshape(x0, (no_of_features,1))
    XX = np.concatenate((x0, XX), axis = 1)
    yy = np.concatenate((np.asarray(y0), yy), axis = None)
    f = yy
    f[0] = np.sum(yy[1:L]) / L
    yy = np.reshape(yy,(L+2*U+1,1))
    nn = XX.shape[1]
    beta = np.zeros(2*U)
    scores = np.multiply((np.matmul(XX.transpose(),w0) + b0),yy)
    ll = np.where(scores[L+1:]<s)
    ll = ll[0]
    ll = np.reshape(ll, (ll.shape[0], 1))
    for i in ll:
        beta[i] = C2
    b = b0
    
    for i in range(1,6):
        Aeq = np.ones(L+(2*U)+1)
        beq = 0
        
        #Finding lower bound and upper bound for the possible solution
        LB = np.concatenate((np.array([-1]), np.zeros(npos)), axis = None)
        LB = np.concatenate((LB, -C1*np.ones(nneg)), axis = None)
        LB = np.concatenate((LB, -beta[0:U]), axis = None)
        LB = np.concatenate((LB, beta[U:2*U] - C2), axis = None)
        UB = np.concatenate((np.array([1]), C1*np.ones(npos)), axis = None)
        UB = np.concatenate((UB, np.zeros(nneg)), axis = None)
        UB = np.concatenate((UB, C2-beta[0:U]), axis = None)
        UB = np.concatenate((UB, beta[U:2*U]), axis = None)
        yyy = np.ones(len(yy))
        
        X = XX.transpose()
        m,n = X.shape
        y = yy.reshape(-1,1) * 1.
        X_dash = y * X
        H = np.dot(X_dash , X_dash.T) * 1.
 
        #Converting parameters to the required form for cvxopt.solvers
        P = cvxopt.matrix(H)
        q = cvxopt.matrix(-f)
        A = cvxopt.matrix(np.reshape(Aeq,(L+2*U+1,1)).transpose()*1.)
        b = cvxopt.matrix(np.zeros(1))
        G = cvxopt.matrix(np.vstack((np.eye(m)*-1, np.eye(m))))
        h = cvxopt.matrix(np.hstack((-LB, UB)))
    
    
        sol = cvxopt.solvers.qp(P,q,G,h,A,b)
        new_alpha = np.array(sol['x'])
        
        #Updating weight and bias term
        w = ((y*new_alpha).T @ X).reshape(-1,1)
        S = (new_alpha > 1e-5).flatten()
        b = y[S]-np.dot(X[S], w)
        b = b[0]
    
        w = w.flatten()
        print('w',w)
        print('b',b)
    
            
        beta = np.zeros(2*U)
        scores = np.multiply((np.matmul(XX.transpose(),w) + b),yy)
        ll = np.where(scores[L+1:nn]<s)
        ll = ll[0]
        ll = np.reshape(ll, (ll.shape[0], 1))
        for i in ll:
            beta[i] = C2
    return [w,b,new_alpha]



def transductive_linear_svm_sg(XX,yy,w,b,C1,C2,beta,alphat):
    
    # No. of iterations
    T = 1700
    n = len(yy)
    unlabeled = XX[:,L:].transpose()
    mean_unlabeled = np.sum(unlabeled,axis = 0)/np.shape(unlabeled)[0]
    mu = np.concatenate((mean_unlabeled,np.asarray(1)), axis = None)
    gamma=np.sum(yy[0:L])/L
    norm_mu = np.dot((mu).transpose(),mu)
    d = np.shape(XX)[0]
    n = L+(2*U)
    cost = float('inf')
    tol = 10**(-5)
    
    for t in range(T):
        print('t',t)
        
        # Taking a random training example in each iteration
        ri = np.random.permutation(range(n))
        alpha = alphat/(t+1)
        for i in range(n):
            ii = ri[i]
            xi = XX[:,ii]
            yi = yy[ii]
            score = (np.matmul(w.transpose(),xi)+b)*yi
            
            # Gradients using Ramp Loss Function
            if ii < L and score < 1:
                gw = (-C1*yi*xi/n)+(beta[ii]*yi*xi/n)
                gb = (-C1*yi/n)+(beta[ii]*yi/n)
            elif ii < L and score >= 1:
                gw = (beta[ii]*yi*xi/n)
                gb = (beta[ii]*yi/n)
            elif ii >= L and score < 1:
                gw = ((-C2*yi*xi)+(beta[ii]*yi*xi))/n
                gb = ((-C2*yi)+(beta[ii]*yi))/n
            elif ii >= L and score >= 1:
                gw = (beta[ii]*yi*xi)/n
                gb = (beta[ii]*yi)/n
            gw = gw.reshape(no_of_features,1)
            
            # Updating weights
            w = w-alpha*(w/n+gw)
            b = b-alpha*gb
            
            # Projection onto the constrained space
            ww = np.concatenate((w,b),axis=None)          
            val = np.matmul(mu.transpose(),ww)
            Pww = (mu*(gamma-val)/norm_mu)+ww
            w = Pww[0:d]
            b = Pww[d]
            w = w.reshape(no_of_features,1)
        # Updated scores
        mm = np.matmul(XX.transpose(),w)
        ma = mm + b
        scores = np.multiply(ma.transpose(),yy.transpose())
        scores = scores.transpose()
        ll = np.where(scores<1)
        ll = ll[0]
        kk = np.where(ll <= L)
        kk = kk[0]
        sum_scores = 0
        for a in kk:
            sum_scores += 1 - scores[ll[a]]
        c1 = sum_scores*C1/n
        c2 = np.matmul(np.transpose(beta[0:L]),scores[0:L])/n
        kk = np.where(ll > L)
        kk = kk[0]
        sum_scores1 = 0
        for a in kk:
            sum_scores1 += 1 - scores[ll[a]]
        c3 = sum_scores1*C2/n
        c4 = np.matmul(np.transpose(beta[L:]),scores[L:])/n
        norm_w = np.matmul(np.transpose(w),w)/15
        
        # Updated cost
        costn = (0.5*norm_w + (c1+c2+c3+c4))
        print('cost',costn)
        if cost-costn < tol:
            break
        cost = costn
    w = w.reshape(no_of_features,1)
    return [w,b,cost]



#Proposed RTSVM

def train_linear_transductive_svm_sg_robust(X,y,C1,C2,w0,b0,alpha):
    tol = 0.001
    nIter = 5
    s = -0.2
    unlabeled = np.where(y==-2)
    pos = np.where(y==1)
    neg = np.where(y==-1)
    npos = len(pos[0])
    nneg = len(neg[0])
    
    # Assigning label = 1 and label = -1 for unlabelled samples
    XX = np.concatenate((X[:,pos], X[:,neg]), axis = 2)
    XX = np.concatenate((XX, X[:,unlabeled]), axis = 2)
    XX = np.concatenate((XX, X[:,unlabeled]), axis = 2)
    XX = np.squeeze(XX)
    yy = np.concatenate((np.ones(npos), -np.ones(nneg)), axis = 0)
    yy = np.concatenate((yy, np.ones(U)), axis = 0)
    yy = np.concatenate((yy, -np.ones(U)), axis = 0)
    nn = np.shape(XX)[1]
    beta = np.zeros(L+2*U)
    mm = np.matmul(XX.transpose(),w0)
    ma = mm + b0
    scores = np.multiply(ma.transpose(),yy.transpose())
    scores = scores.transpose()
    ll = np.where(scores<s)
    ll = ll[0]
    for a in ll:
        beta[a] = C1
    kk = np.where(ll > L)
    for b in kk:
        beta[ll[b]] = C2
    w = w0
    b = b0
    for i in range(5):
        wp = w
        bp = b
        func = transductive_linear_svm_sg(XX,yy,w,b,C1,C2,beta,alpha)
        w = func[0]
        b = func[1]
        cost = func[2]
        print(cost)
        norm_w_wp = np.matmul(np.transpose(w-wp),w-wp)/(no_of_features)
        norm_w_wp = norm_w_wp**(0.5)
        if norm_w_wp < tol:
            break
        beta = np.zeros(L+2*U)
        mm = np.matmul(XX.transpose(),w)
        ma = mm + b
        scores = np.multiply(ma.transpose(),yy.transpose())
        scores = scores.transpose()
        ll = np.where(scores<s)
        ll = ll[0]
        for a in ll:
            beta[a] = C1 
        kk = np.where(ll > L)
        for a in kk:
            beta[ll[a]] = C2
    return [w,b]



# TSVM using dual form
C1 = 0.0001
C2 = 0.00005
cccp = train_linear_transductive_svm(XX,yy,C1,C2,w0,b0)
w1 = cccp[0]
b1 = cccp[1]


# TSVM using Stochastic gradient method
C1 = 2.1
C2 = 0.1
alpha = 0.00001
rtsvm = train_linear_transductive_svm_sg_robust(XX,yy,C1,C2,w0,b0,alpha)
w2 = rtsvm[0]
b2 = rtsvm[1]


# Accuracies from different Methods

# SVM
scores1 = (np.matmul(XX.transpose(),w0) + b0)
y1 = np.sign(scores1)

counter = 0
for i in range(L,L+U):
    if(y1[i] == y_test[i-L]):
        counter+=1
print('SVM Accuracy = ',(counter/U)*100)

scores2 = (np.matmul(XX.transpose(),w1) + b1)
y2 = np.sign(scores2)

# TSVM using dual form
counter = 0
for i in range(L,L+U):
    if(y2[i] == y_test[i-L]):
        counter+=1
print('TSVM CCCP Accuracy = ',(counter/U)*100)

# TSVM using stochastic gradient method
scores3 = (np.matmul(XX.transpose(),w2) + b2)
y3 = np.sign(scores3)

counter = 0
for i in range(L,L+U):
    if(y3[i] == y_test[i-L]):
        counter+=1
print('RTSVM Accuracy = ',(counter/U)*100)