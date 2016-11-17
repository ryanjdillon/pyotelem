
def get_des_asc(T, pitch, fs_a, min_dive_def):
    '''Return indices for descent and ascent periods of dives in T

    3.1 quick separation of descent and ascent phases
    '''
    import numpy

    import utils

    # Descent and Ascent lists of indices
    DES = list()
    ASC = list()

    # Index positions of bad dives to remove from T
    bad_dives = list()

    print('Get descent and ascents from depths...')
    with click.progressbar(range(len(T))) as dive_bar:
        for dive in dive_bar:
            # get list of indices to select the whole dive
            # multiply by acc sampling rate to scale indices
            kk = range(int(fs_a * T[dive, 0]), int(fs_a * T[dive, 1]))

            # Find first point pitch is positive, last point pitch is negative
            # Convert pitch from radians to degrees
            try:
                end_des = utils.first_idx(numpy.rad2deg(pitch[kk]) > 0) + \
                                          (T[dive, 0] * fs_a)
                start_asc = utils.last_idx(numpy.rad2deg(pitch[kk]) < 0) + \
                                           (T[dive, 0] * fs_a)

                # selects the whole descent phase
                des = list(range(int(fs_a * T[dive, 0]), int(end_des)))

                # selects the whole ascent phase
                asc = list(range(int(start_asc), int(fs_a * T[dive, 1])))

                # Concatenate lists
                DES += des
                ASC += asc

            # If acc signal does not match depth movement, remove dive
            except IndexError:
                print('Empty pitch array, likely all positive/negative.')
                # remove invalid dive from summary table
                bad_dives.append(dive)
                continue

    T = numpy.delete(T, bad_dives, 0)

    return T, DES, ASC


