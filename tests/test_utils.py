
def test_contiguous_regions():
    import numpy

    x = numpy.arange(-10,10, 0.01)
    y= -x**2 + x + 10
    condition = y > 0
    start, stop = contiguous_regions(condition)
    # Assert that both

    assert (round(y[start[0]])==0) & (round(y[stop[0]])==0)
