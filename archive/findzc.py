
def findzc(x, thresh, t_max=None):
    '''
    Find cues to each zero-crossing in vector x.

    EXPERIMENTAL - SUBJECT TO CHANGE!!

    thresh: magnitude threshold for detecting a zero-crossing.
    t_max:  (optional) maximum duration in samples between threshold
            crossings.

    To be accepted as a zero-crossing, the signal must pass from below
    -thresh to above thresh, or vice versa, in no more than t_max samples.

    Returns
    -------
    K: nx3 matrix [Ks,Kf,S]
       where:
       * Ks contains the cue of the first threshold-crossing in samples
       * Kf contains the cue of the second threshold-crossing in samples
       * S contains the sign of each zero-crossing
         (1 = positive-going, -1 = negative-going).

    mark johnson
    markjohnson@st-andrews.ac.uk
    January 2008

    fixed a bug (failure to recognize some starting and ending
    zero-crossings), 11 july 2011, mj

    fixed another small bug (second column of K was 1 less than it should
    be), 15 sept. 2013, mj
    '''
    import numpy

    # find all positive and negative threshold crossings
    xtp = numpy.diff(x > thresh) * numpy.sign(numpy.diff(x))
    # negative to positive
    p_np = numpy.where(xtp > 0)[0] + 1
    # positive to negative
    p_pn = numpy.where(xtp < 0)[0]

    xtn = numpy.diff(x < -thresh) * -numpy.sign(numpy.diff(x))
    # positive to negative
    n_pn = numpy.where(xtn > 0)[0] + 1
    # negative to positive
    n_np = numpy.where(xtn < 0)[0]

    # Indices and sign where data crosses over/under threshold
    K = numpy.zeros((len(p_np) + len(n_pn), 3), dtype=int)

    if min(p_np) < min(n_pn):
        SIGN = 1
    else:
        SIGN = -1

    cnt = 0
    while True:

        if SIGN == 1:
            if p_np.size == 0:
                break

            try:
                kk = max(numpy.where(n_np <= p_np[0])[0])
                cnt = cnt + 1
                K[cnt, :] = numpy.hstack((n_np[kk], p_np[0], SIGN))
                n_np = n_np[kk + range(len(n_np))]
                n_pn = n_pn[n_pn > p_np[0]]
                p_np = p_np[1:]
            except:
                pass

            SIGN = - 1

        else:
            if n_pn.size == 0:
                break

            try:
                kk = max(numpy.where(p_pn <= n_pn[0])[0])
                cnt = cnt + 1
                K[cnt, :] = numpy.hstack((p_pn[kk], n_pn[0], SIGN))
                p_pn = p_pn[kk + range(len(p_pn))]
                p_np = p_np[p_np > n_pn[0]]
                n_pn = n_pn[1:]
            except:
                pass

            SIGN = 1

    K = K[0:cnt, :]

    # Data at first and second threshold crossing indexs
    X = numpy.vstack((x[K[:, 0]], x[K[:, 1]])).T

    # Difference in values from start to end of crossing
    x_diff = X[:, 1] - X[:, 0]
    x_norm = ((X[:, 1] * K[:, 0]) - (X[:, 0] * K[:, 1])) / x_diff
    K = numpy.vstack((K, x_norm))

    if t_max:
        k = numpy.where(K[:, 1] - K[:, 0] <= t_max)[0]
        K = K[k, :]

    return K
