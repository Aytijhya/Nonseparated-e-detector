from __future__ import annotations
import math
import numpy as np
from numba import njit, prange

@njit(cache=True)
def log_pi_weight(s:int)->float:
    return math.log(0.5)-math.log(s)-2.0*math.log(math.log(math.e*s))

@njit(cache=True)
def g_interval(n,total,total2,c,logh):
    if n<=1:
        return -np.inf,np.inf
    mean=total/n
    W=total2-total*total/n
    if W<0.0 and W>-1e-10:
        W=0.0
    c2=c*c
    logg0=0.5*(math.log(c2)-math.log(n+c2))
    loggmax=logg0+0.5*n*math.log((n+c2)/c2)
    if logh>=loggmax-1e-14:
        return -np.inf,np.inf
    q=math.exp(2.0*(logh-logg0)/n)
    den=n+c2-q*c2
    if den<=0.0:
        return -np.inf,np.inf
    rad=math.sqrt(max((n+c2)*W*(q-1.0)/(n*den),0.0))
    return mean-rad,mean+rad

@njit(cache=True)
def gmax_one(x,pfa,threshold,c,check_start):
    H=x.size
    pre=np.zeros(H+1)
    pre2=np.zeros(H+1)
    los=np.full(H,-np.inf)
    his=np.full(H,np.inf)
    for tt in range(H):
        pre[tt+1]=pre[tt]+x[tt]
        pre2[tt+1]=pre2[tt]+x[tt]*x[tt]
        glo=-np.inf
        ghi=np.inf
        for s in range(tt+1):
            n=tt-s+1
            logh=threshold if pfa==0 else threshold-log_pi_weight(s+1)
            lo,hi=g_interval(n,pre[tt+1]-pre[s],pre2[tt+1]-pre2[s],c,logh)
            if lo>los[s]: los[s]=lo
            if hi<his[s]: his[s]=hi
            if los[s]>glo: glo=los[s]
            if his[s]<ghi: ghi=his[s]
        if tt+1>check_start and glo>ghi:
            return tt+1
    return H+1

@njit(parallel=True,cache=True)
def gmax_stops(x,pfa,threshold,c,check_start):
    out=np.empty(x.shape[0],np.int64)
    for r in prange(x.shape[0]):
        out[r]=gmax_one(x[r],pfa,threshold,c,check_start)
    return out

def sample_gauss(rng,reps,horizon,change,a,u,b,v):
    out=np.empty((reps,horizon))
    if change:
        out[:,:change]=rng.normal(a,math.sqrt(u),(reps,change))
    if horizon>change:
        out[:,change:]=rng.normal(b,math.sqrt(v),(reps,horizon-change))
    return out

def gaussian_oracle_paths(x,theta=0.0,c=1.0,nullvar=1.0):
    """UI, reduced-G, and full-G predictive log e-processes."""
    reps,horizon=x.shape
    ui=np.empty((reps,horizon))
    gg=np.empty((reps,horizon))
    full=np.empty((reps,horizon))
    means=np.zeros(reps)
    m2=np.zeros(reps)
    pred_ui=np.zeros(reps)
    pred_full=np.zeros(reps)
    sums=np.zeros(reps)
    sumsq=np.zeros(reps)
    null_log=np.zeros(reps)
    for t in range(horizon):
        z=x[:,t]
        pm=np.zeros(reps) if t==0 else means.copy()
        pv=(2.0+m2)/(t+2.0)
        pred_ui += -0.5*np.log(pv)-(z-pm)**2/(2.0*pv)
        pred_full += -0.5*np.log(2.0*np.pi*pv)-(z-pm)**2/(2.0*pv)
        null_log += -0.5*math.log(2.0*math.pi*nullvar)-(z-theta)**2/(2.0*nullvar)
        n=t+1
        delta=z-means
        means += delta/n
        m2 += delta*(z-means)
        sums += z
        sumsq += z*z
        rss=np.maximum(sumsq-2.0*theta*sums+n*theta*theta,1e-300)
        ui[:,t]=pred_ui+0.5*n*(1.0+np.log(rss/n))
        S=sums-n*theta
        den=np.maximum((n+c*c)*rss-S*S,1e-300)
        gg[:,t]=0.5*(math.log(c*c)-math.log(n+c*c))+0.5*n*(np.log((n+c*c)*rss)-np.log(den))
        full[:,t]=pred_full-null_log
    return ui,gg,full

def first_crossing(logpaths,boundary):
    hit=logpaths>=boundary
    anyh=hit.any(axis=1)
    out=np.full(logpaths.shape[0],logpaths.shape[1]+1,dtype=int)
    out[anyh]=hit[anyh].argmax(axis=1)+1
    return out
