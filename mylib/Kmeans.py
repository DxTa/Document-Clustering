# kmeans.py using any of the 20-odd metrics in scipy.spatial.distance
# kmeanssample 2 pass, first sample sqrt(N)

from __future__ import division
import random
import numpy as np
from scipy.spatial.distance import cdist  # $scipy/spatial/distance.py
    # http://docs.scipy.org/doc/scipy/reference/spatial.html
    # Computes distance between each pair of the two collections of inputs.
    # A m_A by m_B distance matrix is returned. For each i and j,
    # the metric dist(u=XA[i], v=XB[j]) is computed and stored in the ij th entry.
from scipy.sparse import issparse  # $scipy/sparse/csr.py

__date__ = "modified version from denis in Munich. Germany"
    # X sparse, any cdist metric: real app ?
    # centers get dense rapidly, metrics in high dim hit distance whiteout
    # vs unsupervised / semi-supervised svm

#...............................................................................
def kmeans( X, centers, delta=.001, max_iter=10, metric="euclidean", p=2, verbose=1 ):
    """ centers, labels, distances = kmeans( X, initial centers ... )
    in:
        X N x dim  may be sparse
        centers k x dim: initial centers, e.g. random.sample( X, k )
        delta: relative error, iterate until the average distance to centers
            is within delta of the previous average distance
        max_iter: maximum iteration
        metric: any of the 20-odd in scipy.spatial.distance
            "chebyshev" = max, "cityblock" = L1, "minkowski" with p=
            or a function( Xvec, centervec ), e.g. Lqmetric below
        p: for minkowski metric -- local mod cdist for 0 < p < 1 too
        verbose: 0 silent, 2 prints running distances
    out:
        centers, k x dim
        labels: labels of each point
        distances, N
    """
    if not issparse(X):
        X = np.asanyarray(X)  # ?
    centers = centers.todense() if issparse(centers) \
        else centers.copy()
    N, dim = X.shape
    k, cdim = centers.shape
    if dim != cdim:
        raise ValueError( "kmeans: X %s and centers %s must have the same number of columns" % (
            X.shape, centers.shape ))
    if verbose:
        print "kmeans: X %s  centers %s  delta=%.2g  max_iter=%d  metric=%s" % (
            X.shape, centers.shape, delta, max_iter, metric)
    allx = np.arange(N) # array([0,1,2,...,N-1])
    prevdist = 0
    for jiter in range( 1, max_iter+1 ):
        D = cdist_sparse( X, centers, metric=metric, p=p )  # |X| x |centers|
        labels = D.argmin(axis=1)  # X -> nearest center
        distances = D[allx,labels] # return distance from each point to the nearest center
        avdist = distances.mean()  # median
        if verbose >= 2:
            print "kmeans: av |X - nearest center| = %.4g" % avdist
        if (1 - delta) * prevdist <= avdist <= prevdist or jiter == max_iter: # when avdist not change anymore or change a very tiny amount delta or iter is maximum
            break
        prevdist = avdist
        for jc in range(k):  # (1 pass in C)
            c = np.where( labels == jc )[0] # find all point belong to cluster #jc
            if len(c) > 0:
                centers[jc] = X[c].mean( axis=0 ) # updated coordinate of each center to the new value
    if verbose:
        print "kmeans: %d iterations  cluster sizes:" % jiter, np.bincount(labels)
    if verbose >= 2:
        r50 = np.zeros(k)
        r90 = np.zeros(k)
        for j in range(k):
            dist = distances[ labels == j ]
            if len(dist) > 0:
                r50[j], r90[j] = np.percentile( dist, (50, 90) )
        print "kmeans: cluster 50 % radius", r50.astype(int)
        print "kmeans: cluster 90 % radius", r90.astype(int)
            # scale L1 / dim, L2 / sqrt(dim) ?
    return centers, labels, distances

#...............................................................................
def kmeanssample( X, k, nsample=0, **kwargs ):
    """ 2-pass kmeans, fast for large N:
        1) kmeans a random sample of nsample ~ sqrt(N) from X
        2) full kmeans, starting from those centers
    """
        # merge w kmeans ? mttiw
        # v large N: sample N^1/2, N^1/2 of that
        # seed like sklearn ?
    N, dim = X.shape
    if nsample == 0:
        nsample = max( 2*np.sqrt(N), 10*k )
    Xsample = randomsample( X, int(nsample) )
    pass1centers = randomsample( X, int(k) )
    samplecenters = kmeans( Xsample, pass1centers, **kwargs )[0]
    return kmeans( X, samplecenters, **kwargs )

def cdist_sparse( X, Y, **kwargs ):
    """ -> |X| x |Y| cdist array, any cdist metric
        X or Y may be sparse -- best csr
    """
        # todense row at a time, v slow if both v sparse
    sxy = 2*issparse(X) + issparse(Y)
    if sxy == 0:
        return cdist( X, Y, **kwargs )
    d = np.empty( (X.shape[0], Y.shape[0]), np.float64 )
    if sxy == 2:
        for j, x in enumerate(X):
            d[j] = cdist( x.todense(), Y, **kwargs ) [0]
    elif sxy == 1:
        for k, y in enumerate(Y):
            d[:,k] = cdist( X, y.todense(), **kwargs ) [0]
    else:
        for j, x in enumerate(X):
            for k, y in enumerate(Y):
                d[j,k] = cdist( x.todense(), y.todense(), **kwargs ) [0]
    return d

def randomsample( X, n ):
    """ random.sample of the rows of X
        X may be sparse -- best csr
    """
    sampleix = random.sample( xrange( X.shape[0] ), int(n) )
    return X[sampleix]

def nearestcenters( X, centers, metric="euclidean", p=2 ):
    """ each X -> nearest center, any metric
            euclidean2 (~ withinss) is more sensitive to outliers,
            cityblock (manhattan, L1) less sensitive
    """
    D = cdist( X, centers, metric=metric, p=p )  # |X| x |centers|
    return D.argmin(axis=1)

def Lqmetric( x, y=None, q=.5 ):
    # yes a metric, may increase weight of near matches; see ...
    return (np.abs(x - y) ** q) .mean() if y is not None \
        else (np.abs(x) ** q) .mean()

#...............................................................................
class Kmeans:
    """ km = Kmeans( X, k= or centers=, ... )
        in: either initial centers= for kmeans
            or k= [nsample=] for kmeanssample
        out: km.centers, km.labels, km.distances
        iterator:
            for jcenter, J in km:
                clustercenter = centers[jcenter]
                J indexes e.g. X[J], classes[J]
    """
    def __init__( self, X, k=0, centers=None, nsample=0, **kwargs ):
        self.X = X
        if centers is None:
            self.centers, self.labels, self.distances = kmeanssample(
                X, k=k, nsample=nsample, **kwargs )
        else:
            self.centers, self.labels, self.distances = kmeans(
                X, centers, **kwargs )

    def __iter__(self):
        for jc in range(len(self.centers)):
            yield jc, (self.labels == jc)

#...............................................................................
