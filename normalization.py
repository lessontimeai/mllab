import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

nsamples = 100
x1 = np.random.randn(nsamples, 2) + np.array([-1 , 0])
x2 = np.random.randn(nsamples, 2) + np.array([1 , 0])
X = np.vstack([x1, x2])
Y = np.hstack([[0]*nsamples, [1]*nsamples])

plt.scatter(x1[:,0], x1[:,1])
plt.scatter(x2[:,0], x2[:,1])
plt.show()

lr = linear_model.LogisticRegression()
lr.fit(X,Y)

ypred = lr.predict(X)
plt.scatter(X[:,0],X[:,1], c=ypred)
plt.show()

lowerbound = np.array([-1, -1])
upperbound = np.array([2, 2])

Xnormalized = (X - lowerbound)/(upperbound-lowerbound)
yprednormalized = lr.predict(Xnormalized)
plt.scatter(Xnormalized[:,0], Xnormalized[:,1], c=yprednormalized);plt.show()

lr.fit(Xnormalized, Y)
yn = lr.predict(Xnormalized)
plt.scatter(Xnormalized[:,0], Xnormalized[:,1], c=yn)
lx = np.arange(-0.5,0.5,0.01)
ly = (-lx*lr.coef_[0][0] - lr.intercept_)/lr.coef_[0][1]
plt.scatter(lx, ly);plt.show()


xx, yy = np.meshgrid(np.arange(-5, 5), np.arange(-5, 5))
xxp = np.vstack([xx.flatten(),yy.flatten()]).T
yyp = lr.predict(xxp)
plt.scatter(xxp[:,0], xxp[:,1], c=yyp)
plt.show()
