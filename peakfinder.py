
def peakfinder(x0, sel=None, thresh=None, extrema=None):
    '''
    PEAKFINDER Noise tolerant fast peak finding algorithm

    Args
    ----
    x0:      A real vector from the maxima will be found (required)

    sel:     The amount above surrounding data for a peak to be identified
             (default = (max(x0)-min(x0))/4). Larger values mean the algorithm
             is more selective in finding peaks.

    thresh:  A threshold value which peaks must be larger than to be maxima or
             smaller than to be minima.

    extrema: 1 if maxima are desired, -1 if minima are desired
             (default = maxima, 1)

    Returns
    -------
    peakLoc: The indicies of the identified peaks in x0

    peakMag: The magnitude of the identified peaks

    [peakLoc] = peakfinder(x0) returns the indicies of local maxima that
        are at least 1/4 the range of the data above surrounding data.

    [peakLoc] = peakfinder(x0,sel) returns the indicies of local maxima
        that are at least sel above surrounding data.

    [peakLoc] = peakfinder(x0,sel,thresh) returns the indicies of local
        maxima that are at least sel above surrounding data and larger
        (smaller) than thresh if you are finding maxima (minima).

    [peakLoc] = peakfinder(x0,sel,thresh,extrema) returns the maxima of the
        data if extrema > 0 and the minima of the data if extrema < 0

    [peakLoc, peakMag] = peakfinder(x0,...) returns the indicies of the
        local maxima as well as the magnitudes of those maxima

    If called with no output the identified maxima will be plotted along
        with the input data.

    Note: If repeated values are found the first is identified as the peak

    Ex:
    t = 0:.0001:10;
    x = 12*sin(10*2*pi*t)-3*sin(.1*2*pi*t)+randn(1,numel(t));
    x(1250:1255) = max(x);
    peakfinder(x)

    Copyright Nathanael C. Yoder 2011 (nyoder@gmail.com)
    '''
    import numpy

    # Perform error checking and set defaults if not passed in
    error(nargchk(1, 4, nargin, 'struct'))
    error(nargoutchk(0, 2, nargout, 'struct'))

    s = x0.shape

    flipData = s[1] < s[2]
    len0 = numel(x0)

    if len0 != s[1] and len0 != s[2]:
        error('PEAKFINDER:Input', 'The input data must be a vecto')
    elif isempty(x0):
        varargout = cellarray([[], []])
        # TODO correct
        return varargout

    if ~(isreal(x0)):
        warning('PEAKFINDER:NotReal', 'Absolute value of data will be used')
        x0 = abs(x0)

    if sel == None:
        sel = (max(x0) - min(x0)) / 4
    elif ~(isnumeric(sel)) or ~(isreal(sel)):
        sel = (max(x0) - min(x0)) / 4
        warning('PEAKFINDER:InvalidSel',
                'The selectivity must be a real scalar.  A selectivity of '
                '%.4g will be used' % sel)
    elif numel(sel) > 1:
        warning( 'PEAKFINDER:InvalidSel',
                 'The selectivity must be a scalar. The first '
                 'selectivity value in the vector will be used.')
        sel = sel[1]

    if ~(isnumeric(thresh)) or ~(isreal(thresh)):
        thresh = matlabarray([])
        warning('PEAKFINDER:InvalidThreshold',
                'The threshold must be a real scalar. No threshold will be used.')

    # TODO change from iterable to scalar
    elif numel(thresh) > 1:
        thresh = thresh[1]
        warning('PEAKFINDER:InvalidThreshold',
                'The threshold must be a scalar.  The first threshold value in
                 the vector will be used.')

    if extrema == None:
        extrema = 1
    else:
        extrema = sign(extrema[1])
        if extrema == 0:
            error('PEAKFINDER:ZeroMaxima',
                  'Either 1 (for maxima) or -1 (for minima) must be input for extrema')

    x0 = extrema * x0.ravel()
    # TODO how to handle, should set to empty list as in .m?
    thresh = thresh * extrema

    dx0 = diff(x0)

    dx0[dx0 == 0] = -eps

    ind = numpy.where(dx0[:-1] * dx0[1:]) < 0)[0] + 1

    # Include endpoints in potential peaks and valleys
    x = numpy.hstack((x0[1], x0[ind], x0[end()]))
    ind = numpy.hstack(([1], [ind], [len0]))

    # x only has the peaks, valleys, and endpoints
    len_   = numel(x)
    minMag = min(x)
    if len_ > 2:
        # Set initial parameters for loop
        tempMag = copy(minMag)
        foundPeak = copy(false)
        leftMin = copy(minMag)

        # Calculate the sign of the derivative since we taked the first point
        #  on it does not neccessarily alternate like the rest.
        signDx = sign(diff(x[1:3]))
        if signDx[1] <= 0:
            ii = 0
            if signDx[1] == signDx[2]:
                x[2]   = []
                ind[2] = []
                len_   = len_ - 1
        else:
            ii = 1
            if signDx[1] == signDx[2]:
                x[1]   = []
                ind[1] = []
                len_   = len_ - 1

        # Preallocate max number of maxima
        maxPeaks = numpy.ceil(len_ / 2)
        peakLoc  = numpy.zeros((maxPeaks, 1))
        peakMag  = numpy.zeros((maxPeaks, 1))

        cInd = 1
        while ii < len_:
            ii = ii + 1

            # Reset peak finding if we had a peak and the next peak is bigger
            #   than the last or the left min was small enough to reset.
            if foundPeak:
                tempMag   = copy(minMag)
                foundPeak = copy(false)

            # Make sure we don't iterate past the length of our vector
            if ii == len_:
                break

            # Found new peak that was lager than temp mag and selectivity larger
            # than the minimum to its left.
            if x[ii] > tempMag and x[ii] > leftMin + sel:
                tempLoc = copy(ii)
                tempMag = x[ii]
            ii = ii + 1

            # Come down at least sel from peak
            if ~(foundPeak) and tempMag > sel + x[ii]:
                foundPeak = copy(true)
                leftMin   = x[ii]

                peakLoc[cInd] = tempLoc
                peakMag[cInd] = tempMag
                cInd = cInd + 1

            elif x[ii] < leftMin:
                leftMin = x[ii]

        # Check end point
        if (x[-1] > tempMag) and (x[-1] > leftMin + sel):
            peakLoc[cInd] = len_
            peakMag[cInd] = x[end()]
            cInd = cInd + 1

        elif ~foundPeak and (tempMag > minMag):
            peakLoc[cInd] = tempLoc
            peakMag[cInd] = tempMag
            cInd = cInd + 1

        # Create output
        peakInds = ind[peakLoc[1:cInd - 1]]
        peakMags = peakMag[1:cInd - 1]

    else:
        peakMags, xInd = max(x, nargout=2)
        if peakMags > minMag + sel:
            peakInds = ind[xInd]
        else:
            peakMags = matlabarray([])
            peakInds = matlabarray([])

    # Apply threshold value. Since always finding maxima it will always be
    # larger than the thresh.
    if ~(isempty(thresh)):
        m = peakMags > thresh
        peakInds = peakInds[m]
        peakMags = peakMags[m]

    # Rotate data if needed
    if flipData:
        peakMags = peakMags.T
        peakInds = peakInds.T

    # Change sign of data if was finding minima
    if extrema < 0:
        peakMags = - peakMags
        x0 = - x0

    # Plot if no output desired
    if nargout == 0:
        if isempty(peakInds):
            disp('No significant peaks found')
        else:
            figure
            plot(range( 1, len0), x0, '.-', peakInds, peakMags, 'ro',
                    'linewidth', 2)
    else:
        varargout = cellarray([peakInds, peakMags])
