"""The two frozen course learners; no search, class weighting or warm start."""
import math
import numpy as np
from sklearn.linear_model import Ridge

MODELS = ['binary_log13','relative_ridge12']


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


def model_parameters(name,n):
    if name=='relative_ridge12':
        return dict(alpha=n*.01,fit_intercept=True,solver='cholesky')
    raise ValueError('Unrecognized frozen model')


def fit_predict(name,x,y,z):
    if name=='binary_log13':
        theta,info=fit_logistic(x,y)
        return sigmoid(theta[0]+z@theta[1:]),dict(info,theta=theta.tolist(),lambda_value=.01)
    params=model_parameters(name,len(x))
    model=Ridge(**params)
    model.fit(x,y)
    prediction=model.predict(z)
    info=dict(parameters=params)
    residual=model.predict(x)-y
    info['gradient_inf']=max(float(np.max(abs(x.T@residual/len(x)+.01*model.coef_))),abs(float(residual.mean())))
    if info['gradient_inf']>=1e-9:raise RuntimeError('Ridge numerical verification failed')
    if not np.isfinite(prediction).all():raise ValueError('Nonfinite predictions')
    return prediction,info
