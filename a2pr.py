
def a2pr(A):
    '''Pitch and roll estimator for DTAG data.

    This is a simple non-iterative estimator with |pitch| constrained to <= 90
    degrees.  The p & r estimates give the least-square-norm error between A
    and the A-vector that would be measured at the estimated pitch and roll.

    Inputs:
    A is a nx3 acceleration matrix

    Outputs:
    p is the pitch estimate in radians
    r is the roll estimate in radians
    v is the 2-norm of the acceleration measurements
    '''
    import numpy

    if min(cat(A.shape[0], A.shape[1])) == 1:
        A = A.ravel().T

    v = numpy.sqrt(A**2 *  [1,1,1])

    # compute pitch and roll
    p = numpy.arcsin(A[:, 0] / v)
    r = float(numpy.arctan2(A[:, 1], A[:, 2]))

    return p, r, v
