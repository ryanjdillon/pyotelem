
def split_glides(dur, GL):
    '''Get start/stop indices of each `dur` lenth sub-glide for glides in GL

    Args
    ----
    dur: int
        desired duration of glides

    GL: numpy.ndarray, shape(n, 2)
        matrix containing the start time (first column) and end time
        (2nd column) of any glides.Times are in seconds.

    Attributes
    ----------
    gl_ind_diff: numpy.ndarray, shape(n,3)
        GL, with aditional column of diffence between the first two columns

    Returns
    -------
    sub_glides: numpy.ndarray, shape(n, 2)
        matrix containing the start time (first column) and end time(2nd
        column) of the generated subglides.All glides must have duration
        equal to the given dur value.Times are in seconds.

    Note
    ----
    `glinf` renamed `gl_ind_diff`
    `SUM` removed
    `SGL` renamed `sub_glides`

    Lucia Martina Martin Lopez (May 2016)
    lmml2@st-andrews.ac.uk
    '''

    # Split all glides in GL
    for i in range(len(GL)):
        # GL plus column for total duration of glide
        gl_ind_diff = numpy.vstack((GL.T, GL[:, 1] - GL[:, 0])).T

        # number of sub-glides = glide duration, divided by dur+1
        n_gl = (gl_ind_diff[i, 2] / (dur + 1))

        # Relative index positions from original glide start index
        v = numpy.arange(0, round(n_gl)) * 6

        # Split into sub glides if longer than duration
        if abs(gl_ind_diff[i, 2]) > dur:
            # Get start and end index positions for each sub-glide
            for k in range(numpy.round(n_gl)):

                # starting at original glide start...
                # sgl_start_ind: add index increments of dur+1 for next start idx
                next_start_ind = gl_ind_diff[i, 0] + v[k]
                # end_glide: add `dur` to that to get ending idx
                next_end_ind   = next_start_ind + dur

                # If first iteration, set equal to first set of indices
                if k == 0:
                    sgl_start_ind = next_start_ind
                    sgl_end_ind   = next_end_ind
                else:
                    # Concat 1D arrays together, shape (n,)
                    sgl_start_ind = numpy.hstack((sgl_start_ind, next_start_ind))
                    sgl_end_ind   = numpy.hstack((sgl_end_ind, next_end_ind))

            # Stack and transpose indices into shape (n, 2)
            gl_ind = numpy.vstack((sgl_start_ind, sgl_end_ind)).T

        # Set sub_glides to gl_ind on first iteration, the concat thereafter
        if i == 0:
            sub_glides = gl_ind
        else:
            sub_glides = numpy.vstack((sub_glides, gl_ind))

    return sub_glides
