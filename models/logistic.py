"""Fixed L2-penalized binary logistic regression."""
import math
import numpy as np

def sigmoid(a):
    a=np.asarray(a,float);out=np.empty_like(a);positive=a>=0
    out[positive]=1/(1+np.exp(-a[positive]))
    e=np.exp(a[~positive]);out[~positive]=e/(1+e)
    return out

def logistic_objective(x,y,theta,lam):
    a=np.column_stack([np.ones(len(x)),x]);eta=a@theta;p=sigmoid(eta)
    penalty=np.r_[0.,np.repeat(lam,x.shape[1])]
    value=float(np.mean(np.logaddexp(0,eta)-y*eta)+.5*np.sum(penalty*theta**2))
    gradient=a.T@(p-y)/len(y)+penalty*theta
    hessian=(a.T*(p*(1-p)))@a/len(y)+np.diag(penalty)
    return value,gradient,hessian

def fit_logistic(x,y,lam=.01):
    theta=np.zeros(x.shape[1]+1);theta[0]=math.log(y.mean()/(1-y.mean()))
    for j in range(201):
        value,g,h=logistic_objective(x,y,theta,lam)
        gn=float(max(abs(g)))
        if gn<=1e-9:
            return theta,dict(iterations=j,objective=value,gradient_inf=gn)
        if j==200:
            raise RuntimeError('Fixed Newton iterations exhausted')
        direction=-np.linalg.solve(h,g);descent=float(g@direction)
        if descent>=0:
            raise RuntimeError('Newton direction is not descent')
        step=1.
        for _ in range(60):
            candidate=theta+step*direction
            if logistic_objective(x,y,candidate,lam)[0]<=value+1e-4*step*descent:
                theta=candidate;break
            step*=.5
        else:
            raise RuntimeError('Fixed Newton line search exhausted')
