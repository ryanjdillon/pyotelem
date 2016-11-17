#!/usr/bin/env python3
'''
Body density estimation. Japan workshop May 2016

Python translation: Ryan J. Dillon
Matlab Author:      Lucia Martina Martin Lopez lmml2@st-andrews.ac.uk

Attributes
----------

tmax: maximum duration allowable for a fluke stroke in seconds, it can be set
      as 1/`stroke_f`
J:    magnitude threshold for detecting a fluke stroke in [m/s2]

NOTE variable name changes
--------------------------
`p` changed to `depths`
`FLl`    changed to `cutoff_low`
`FLlmag` changed to `cutoff_low_mag`
`FR`     changed to `stroke_f`
`FRn`    changed to `stroke_f_n`
`FL1`    changed to `cutoff_low`
           frequency at the negative peak in the power spectral density
           plot/`stroke_f`

'''
import click
import os


@click.command(help='Calculate dive statistics for body condition predictions')

@click.option('--root-path', prompt=True,
              default=lambda: os.environ.get('LLEO_ROOT_PATH',''),
              help='Directory acceleration data files')
@click.option('--data-path', prompt=True,
              default=lambda: os.environ.get('LLEO_DATA_PATH',''),
              help='Directory acceleration data files')
@click.option('--cal-path', prompt=True,
              default=lambda: os.environ.get('LLEO_CAL_PATH',''),
              help='Directory acceleration data files')
@click.option('--debug', default=True, type=bool, help='Return on debug output')


def run(root_path, data_path, cal_path, debug=False):
    '''Calculate body conditions summary statistics

    Args
    ----
    root_path: path containing experiment data directories
    data_path: data directory name
    cal_path:  data directory name containing cal.yaml file for accelerometer
               data calibration
    '''
    import numpy

    import finddives
    import utils_signal
    import utils_plot

    # 1 LOAD DATA
    #============
    A_g, dt_a, fs_a, depths, dt_d, fs_d = load_lleo(root_path, data_path, cal_path)
    pitch, roll, heading = calc_PRH(A_g)


    # 2_DEFINE DIVES
    #===============

    # define min_dive_def as the minimum depth at which to recognize a dive.
    # TODO assign correct value
    min_dive_def = 0.5 #00

    # delta depth cutoff for control = 0.15hz

    T = finddives.finddives(depths, fs_d, thresh=min_dive_def, surface=1, findall=True)

    # make a summary table describing the characteristics of each dive.
    #D = [start_time(s), end_time(s), dive_duration(s), max_depth(s), max_depth(m), ID(n)]
    D = create_dive_summary(T)


    # 3 Filter Acceleration signals
    #==============================

    # 3.1 Quick separation of descent and ascent phases
    T, DES, ASC = get_asc_des(T, pitch, fs_a)


    # 3.2 Sesparate low and high acceleration signals
    #------------------------------------------------

    # 3.2.1 select periods of analysis
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # TODO verify what was here

    # 3.2.2_Determine `stroke_f` fluking rate and cut-off frequency
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TODO remove debug here?
    if debug:
        cutoff = 1.5 # Hz
        stroke_f = 1.5 #Hz
    else:
        # plot PSD for the whole deployment, all descents and all ascent phases.
        utils_plot.plot_descent_ascent(A_g, DES, ASC)

        # calculate the power spectrum of the accelerometer data at the whale frame
        cutoff, stroke_f = get_stroke_freq(A_g, fs, nperseg=512, peak_thresh=0.25)

    # Seperate low and high frequency signals
    A_g_lf, A_g_hf = filter_acceleromter_low_high(A_g, fs_a, cutoff, order=5)

    # Calculate glides with dummy input params # TODO remove
    J = 0.1
    t_max = 1/stroke_f
    # TODO yeilds no KK indices with control data
    # axis around which rotations are to be analyzed
    n = 1
    GL, KK = get_stroke_glide_indices(A_g_hf[:, n], fs_a, n, J, t_max)


    # 3.2.4 calculate the smooth pitch from the low pass filter acceleration signal
    # to avoid incorporating signals above the stroking periods
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # TODO remove?
    #smooth_pitch, roll_lf = a2pr(A_g_lf[k, : ])
    pitch_lf, roll_lf, heading_lf = calc_PRH(A_g)

    utils_plot.plot_pitch_roll(pitch, roll, pitch_lf, roll_lf)



    #* # 4 DEFINE PRECISE DESCENT AND ASCENT PHASES
    #* #======================================================================
    #*
    #* Bottom = []

    #* Phase = numpy.array(range(len(depths)))
    #* Phase[:] = numpy.nan
    #*
    #*
    #* #TODO check matlab here
    #* isdive = numpy.zeros(depths.size, dtype=bool)
    #* for i in range(T.size[0]):
    #*     isdive[round(T[i, 0] * fs): round(T[i, 1] * fs)] = true
    #*
    #* p_dive   = depths
    #* p_nodive = depths

    #* p_dive[~isdive]  = nan
    #* p_nodive[isdive] = nan
    #*
    #* I = range(len(depths))
    #*
    #* figure(4)
    #* subplot1 = subplot(5, 1, range(4))
    #*
    #* # TODO PLOT
    #* plot(I, p_dive, 'g')
    #*
    #* # TODO PLOT
    #* plot(I, p_nodive, 'r')
    #*
    #* set(gca, 'ydir', 'rev', 'ylim', [min(depths), max(depths)])
    #*
    #* tit = title('Click within the last dive you want to use')
    #* legend('Dive', 'Not Dive', 'location', 'southeast')
    #*
    #* subplot2 = subplot(5, 1, 5)
    #*
    #* # TODO PLOT
    #* plot(I, pitch, 'g', I, pitch_lf, 'r')
    #* legend('pitch', 'pitch_lf')
    #*
    #* x = ginput(1)
    #*
    #* nn = numpy.where(T[:, 0] < x[0]/fs, 1, 'last')
    #*
    #* for dive in range(T.size[0]):
    #*     # it is selecting the whole dive
    #*     kk = range(round(fs * T[dive, 0]), round(fs * T[dive, 1]))
    #*
    #*     kkI = numpy.zeros(depths.size, dtype=bool)
    #*     kkI[kk] = true
    #*
    #*     # search for the first point after diving below min_dive_def at which pitch
    #*     # is positive
    #*     end_des = round(numpy.where((pitch_lf[kkI] & depths > (min_dive_def * .75)) \
    #*                          * 180 / pi) > 0, 1, 'first') + T[dive, 0] * fs)
    #*
    #*     # search for the last point beroe diving above min_dive_def at which the
    #*     # pitch is negative if you want to do it manually as some times there is a
    #*     # small ascent
    #*     start_asc = round(numpy.where((pitch_lf[kkI] & depths > (min_dive_def * .75)) \
    #*                            * 180 / pi) < 0, 1, 'last') + T[dive, 0] * fs)
    #*
    #*
    #*
    #*     # TODO PLOT (CLICK): where pitch angle first goes to zero & last goes to
    #*     #                    zero in the ascent
    #*
    #*     # phase during the descent and a small descent phase during the ascent.
    #*     #     figure
    #*     #     # plott plots sensor data against a time axis
    #*     #     ax(1)=subplot(211) plott(depths[kk],fs)
    #*     #     ax(2)=subplot(212) plott(pitch[kk]*180/pi,fs,0)
    #*     #     linkaxes(ax, 'x') # links x axes of the subplots for zoom/pan

    #*     #     # click on where the pitch angle first goes to zero in the descent and
    #*     #     # last goes to zero in the ascent
    #*     #     [x,y]=ginput(2)
    #*     #     des=round(x[1])/fs+T[dive,1]
    #*     #     asc=round(x[2))/fs+T[dive,1]
    #*
    #*     Phase(kk[kk < end_des]) = -1
    #*     Phase(kk[(kk < start_asc) & (kk > end_des)]] = 0
    #*     Phase(kk[kk > start_asc]] = 1
    #*
    #*     # Time in seconds at the start of bottom phase (end of descent)
    #*     Bottom[dive, 0] = (end_des) / fs
    #*
    #*     # Depth in m at the start of the bottom phase (end of descent phase)
    #*     Bottom[dive, 1] = depths[end_des]
    #*
    #*     # Time in seconds at the end of bottom phase (start of descent)
    #*     Bottom[dive, 2] = (start_asc) / fs
    #*
    #*     # Depth in m at the end of the bottom phase (start of descent phase)
    #*     Bottom[dive, 3] = depths[start_asc]
    #*
    #* p_asc = depths
    #* p_des = depths
    #*
    #* p_asc(Phase < 1 | isnan(Phase)) = nan
    #* p_des(Phase > -1 | isnan(Phase)) = nan
    #*
    #* # TODO PLOT
    #* plot(subplot1, p_asc, 'k')
    #*
    #* # TODO PLOT
    #* plot(subplot1, p_des, 'b')
    #*
    #* legend('Dive', 'Not Dive', 'Ascent', 'Descent', 'location', 'southeast')
    #*
    #* linkaxes([subplot1 subplot2], 'x')



    #* # 5 ESTIMATE SWIM SPEED
    #* #======================================================================
    #*
    #* # degree threshold above which speed can be estimated
    #* th_deg = 30
    #*
    #* SwimSp = inst_speed(depths, pitch_lf, fs, stroke_f, f, k, th_deg)
    #*
    #* # TODO PLOT
    #* figure(4)
    #* subplot2 = subplot(5, 1, 5)
    #*
    #* plot(k, SwimSp, 'g') ylim([0 max(SwimSp)])
    #* legend('speed')
    #*
    #* # links x axes of the subplots for zoom/pan
    #* linkaxes([subplot1 subplot2], 'x')



    #* # 6 ESTIMATE SEAWATER DENSITY AROUND THE TAGGED WHALES
    #* #======================================================================
    #*
    #* # Seawater density estimation
    #* DPT = DTS[:, 0]
    #* TMP = DTS[:, 1]
    #* SL  = DTS[:, 2]
    #*
    #* SWdensity, depCTD = SWdensityFromCTD(DPT, TMP, SL, D)
    #*
    #* Dsw = EstimateDsw(SWdensity, depCTD, depths)



    #* # 7 EXTRACT STROKES AND GLIDES
    #* #======================================================================
    #*
    #* # it can be done using the body rotations (pry) estimated using the
    #* # magnetometer method, or it can be done using the dorso-ventral axis of the
    #* # high-pass filtered acceleration signal.
    #*
    #* # For both methods, tmax and J need to be determined.
    #* #   * tmax is the maximum duration allowable for a fluke stroke in seconds,
    #* #     it can be set as 1/`stroke_f`
    #* #   * J is the magnitude threshold for detecting a fluke stroke in


    #* # 7.1 using the heave high pass filtered acceleration signal,(use n=3)
    #* #----------------------------------------------------------------------
    #*
    #* # units of J are in m/s2 set J and tmax [] until determined.
    #* Anlf, Ahf, GL, KK = Ahf_Anlf(Aw, fs_a, stroke_f, f, n, k, J=[], tmax=[])
    #*
    #*
    #* # TODO PLOT
    #* figure(5)
    #*
    #* subplot1 = subplot(1, 4, 1: 3)
    #* ax2 = plotyy(range(len(depths)), depths, range(len(pitch)), Ahf[: , 2])
    #*
    #* set(ax2(1), 'ydir', 'rev', 'ylim', [0, max(depths)])
    #* set(ax2(2), 'nextplot', 'add')
    #* plot(ax2(2), range(len(pitch)), Ahf[:, 0], 'm')
    #*
    #* maxy = max(max(Ahf[round(T[0, 0] * fs): round(T[nn, 1] * fs), [1,3])))
    #*
    #* set(ax2(2),
    #*     'ylim', [-2 * maxy 2 * maxy],
    #*     'ytick', round(-2 * maxy * 10) / 10: 0.1: 2 * maxy)
    #*
    #* legend('Depth', 'HPF acc z axis', 'HPF acc x axis')
    #* title('Zoom in to find appropriate thresholds for fluking, then enter it for J')
    #* linkaxes([ax1 ax2], 'x')  # li
    #*
    #* flukeAcc1A = Ahf[ASC, 0]  # hpf-x acc ascent
    #* flukeAcc1D = Ahf[DES, 0]  # hpf-x acc descent
    #* flukeAcc3A = Ahf[ASC, 2]  # hpf-z acc  ascent
    #* flukeAcc3D = Ahf[DES, 2]  # hpf-z acc  desscent
    #*
    #* # to look at heave, change to 3
    #* TOTAL = abs([flukeAcc1A flukeAcc1D])
    #* Y = buffer(TOTAL, 2 * round(1 / stroke_f * fs_a))
    #*
    #* # TODO PLOT: histogram of x acc
    #* subplot2 = subplot(2, 4, 4)
    #*
    #* hist(max(Y), 100)
    #* set(subplot2, 'xlim', [0 max(max(Y))])
    #* title('hpf-x acc')
    #*
    #* subplot3 = subplot(2, 4, 8)
    #* TOTAL = abs([flukeAcc3A flukeAcc3D])
    #* Y = buffer(TOTAL, 2 * round(1 / stroke_f * fs_a))
    #*
    #* # TODO PLOT: histogram of y acc
    #* hist(max(Y), 100, 'FaceColor', 'g') set(subplot3, 'xlim', [0 max(max(Y))])
    #* title('hpf-z acc')

    #* # Choose a value for J based on the histogram for:
    #* #   hpf-x, then when detecting glides in the next step use Ahf_Anlf
    #* #   function with n=1
    #*
    #* #   hpf-z then when detecting glides in the next step use Ahf_Anlf
    #* #   function with n=3
    #*
    #* J =  # in m/s2
    #* tmax = 1 / stroke_f  # in seconds
    #*
    #* Anlf, Ahf, GL, KK = Ahf_Anlf(Aw, fs_a, stroke_f, f, n, k, J, tmax)
    #*
    #* # in case you want to check the performance of both methods in the following
    #* # figure define glides and strokes obtained from the high pass filtered
    #* # acceleration as GLa and KKa respectively
    #*
    #* Anlf, Ahf, GLa, KKa = Ahf_Anlf(Aw, fs_a, stroke_f, f, n, k, J, tmax)



    #* # 8_MAKE 5SEC SUB-GLIDES
    #* #======================================================================
    #*
    #* dur = 5
    #* SGL = splitGL[dur, GL]
    #* SGLT = [SGL[:, 0], SGL[:, 1] - SGL[:, 0]]
    #*
    #* # check that all subglides have a duration of 5 seconds
    #* rangeSGLT = [min(SGLT[:, 1]), max(SGLT[:, 1])]
    #*
    #* gl_k = eventon[SGLT, t]
    #* p_gl = depths
    #* p_gl[gl_k == 0] = numpy.nan
    #*
    #* # TODO PLOT
    #* figure(7)
    #*
    #* h3 = plot(subplot1, t * fs, p_gl, 'm', 'Linewidth', 3)



    #* # 9 create summary table required for body density
    #* #======================================================================
    #*
    #* stroke_f    = 0.4
    #* f     = 0.4
    #* alpha = 25
    #* n     = 1
    #* k     = range(len(depths))
    #* J     = 2 / 180 * pi
    #* tmax  = 1 / stroke_f
    #*
    #* MagAcc, pry, Sa, GL, KK = magnet_rot_sa(Aw, Mw, fs_a, stroke_f, f, alpha, n,
    #*                                         k, J, tmax)
    #*
    #* heading_lf = m2h(MagAcc.Mnlf[k, :], pitch_lf, roll_lf)
    #* pitch_lf_deg = pitch_lf * 180 / pi



    #* # 10 CALCULATE GLIDE DESC/GLIDE ASC
    #* #======================================================================
    #*
    #* dur = 5
    #*
    #* # TODO make numpy record array?
    #* Glide = numpy.zeros(len(SGL), 24)
    #*
    #* for i in range(len(SGL)):
    #*     cue1 = SGL[i,0] * fs
    #*     cue2 = SGL[i,1] * fs
    #*
    #*     # sub-glide start point in seconds
    #*     Glide[i,0] = SGL[i,0]
    #*
    #*     # sub-glide end point in seconds
    #*     Glide[i,1] = SGL[i,1]
    #*
    #*     # sub-glide duration
    #*     Glide[i,2] = SGL[i,1] - SGL[i,0]
    #*
    #*     # mean depth(m)during sub-glide
    #*     Glide[i,3] = numpy.mean(depths[round(cue1): round(cue2)])
    #*
    #*     # total depth(m)change during sub-glide
    #*     Glide[i,4] = abs(depths[round(cue1)] - depths[round(cue2)])
    #*
    #*     # mean swim speed during the sub-glide, only given if pitch>30 degrees
    #*     Glide[i,5] = numpy.mean(SwimSp[round(cue1): round(cue2)])
    #*
    #*     # mean pitch during the sub-glide
    #*     Glide[i,6] = numpy.mean(pitch_lf_deg[round(cue1): round(cue2)])
    #*
    #*     # mean sin pitch during the sub-glide
    #*     Glide[i,7] = numpy.sin(numpy.mean(pitch_lf_deg[round(cue1): round(cue2)]))
    #*
    #*     # SD of pitch during the sub-glide
    #*     Glide[i,8] = numpy.std(pitch_lf_deg[round(cue1): round(cue2)]) * 180 / pi
    #*
    #*     # mean temperature during the sub-glide
    #*     Glide[i,9] = numpy.mean(temp[round(cue1): round(cue2)])
    #*
    #*     # mean seawater density (kg/m^3) during the sub-glide
    #*     Glide[i,10] = numpy.mean(Dsw[round(cue1): round(cue2)])
    #*
    #*     # TODO check matlab here
    #*     try:
    #*         xpoly = (round(cue1): round(cue2))
    #*         ypoly = SwimSp[round(cue1): round(cue2)]
    #*
    #*         B, BINT, R, RINT, STATS = regress(ypoly, [xpoly,
    #*                                                   numpy.ones(len(ypoly), 1)])
    #*
    #*         # mean acceleration during the sub-glide
    #*         Glide[i,11] = B[0]
    #*
    #*         # R2-value for the regression swim speed vs. time during the sub-glide
    #*         Glide[i,12] = STATS[0]
    #*
    #*         # SE of the gradient for the regression swim speed vs. time during the
    #*         # sub-glide
    #*         Glide[i,13] = STATS[3]
    #*
    #*     except:
    #*
    #*         # mean acceleration during the sub-glide
    #*         Glide[i,11] = numpy.nan
    #*
    #*         # R2-value for the regression swim speed vs. time during the sub-glide
    #*         Glide[i,12] = numpy.nan
    #*
    #*         # SE of the gradient for the regression swim speed vs. time during the
    #*         # sub-glide
    #*         Glide[i,13] = numpy.nan
    #*
    #*     sumphase = sum(phase[round(cue1): round(cue2)])
    #*     # TODO check what dimensions of sp should be
    #*     sp = numpy.nan
    #*     sp[sumphase < 0]  = -1
    #*     sp[sumphase == 0] = 0
    #*     sp[sumphase > 0]  = 1
    #*
    #*     # Dive phase:0 bottom, -1 descent, 1 ascent, NaN not dive phase
    #*     Glide[i,14] = sp
    #*
    #*     Dinf = D[numpy.where((D[: , 0]*fs < cue1) & (D[: , 1]*fs > cue2)), : ]
    #*
    #*     if isempty(Dinf):
    #*         Dinf = numpy.zeros(D.shape)
    #*         Dinf[:] = numpy.nan
    #*
    #*     # Dive number in which the sub-glide recorded
    #*     Glide[i,15] = Dinf[6]
    #*
    #*     # Maximum dive depth (m) of the dive
    #*     Glide[i,16] = Dinf[5]
    #*
    #*     # Dive duration (s) of the dive
    #*     Glide[i,17] = Dinf[2]
    #*
    #*     # Mean pitch(deg) calculated using circular statistics
    #*     Glide[i,18] = circ_mean(pitch_lf[round(cue1): round(cue2)])
    #*
    #*     # Measure of concentration (r) of pitch during the sub-glide (i.e. 0 for
    #*     # random direction, 1 for unidirectional)
    #*     Glide[i,19] = 1 - circ_var(pitch_lf[round(cue1): round(cue2)])
    #*
    #*     # Mean roll (deg) calculated using circular statistics
    #*     Glide[i,20] = circ_mean(roll_lf[round(cue1): round(cue2)])
    #*
    #*     # Measure of concentration (r) of roll during the sub-glide
    #*     Glide[i,21] = 1 - circ_var(roll_lf[round(cue1): round(cue2)])
    #*
    #*     # Mean heading (deg) calculated using circular statistics
    #*     Glide[i,22] = circ_mean(heading_lf[round(cue1): round(cue2)])
    #*
    #*     # Measure of concentration (r) of heading during the sub-glide
    #*     Glide[i,23] = 1 - circ_var(heading_lf[round(cue1): round(cue2)])
    #*
    #*
    #* # TODO Write to csv or whatever
    #* # csvwrite('WhaleID.csv', Glide)



    #* # 11 Calculate glide ratio
    #* #======================================================================
    #*
    #* # TODO G_ratio as numpy record array
    #*
    #* G_ratio = zeros(T.size[0], 10)
    #*
    #* for dive = 1:
    #*     T.size[0]:
    #*         # it is selecting the whole dive
    #*     kkdes = round(fs * T[dive, 0]): round(fs * Bottom[dive, 0])
    #*
    #*     # it is selecting the whole dive
    #*     kkas = round(fs * Bottom[dive, 2]): round(fs * T[dive, 1])
    #*
    #*     # total duration of the descet phase (s)
    #*     G_ratio[dive, 0] = len(kkdes) / fs
    #*
    #*     # total glide duration during the descet phase (s)
    #*     G_ratio[dive, 1] = len(numpy.where(SGtype[kkdes] == 0)) / fs
    #*
    #*     # glide ratio during the descet phase
    #*     G_ratio[dive, 2] = G_ratio[dive, 1] / G_ratio[dive, 0]
    #*
    #*     # mean pitch during the descet phase(degrees)
    #*     G_ratio[dive, 3] = numpy.mean(pitch_lf[kkdes] * 180 / pi)
    #*
    #*     # descent rate (m/s)
    #*     G_ratio[dive, 4] = Bottom[dive, 1] / G_ratio[dive, 0]
    #*
    #*     # total duration of the ascet phase (s)
    #*     G_ratio[dive, 5] = len(kkas) / fs
    #*
    #*     # total glide duration during the ascet phase (s)
    #*     G_ratio[dive, 6] = len(numpy.where(SGtype[kkas] == 0)) / fs
    #*
    #*     # glide ratio during the ascet phase
    #*     G_ratio[dive, 7] = G_ratio[dive, 6] / G_ratio[dive, 5]
    #*
    #*     # mean pitch during the ascet phase(degrees)
    #*     G_ratio[dive, 8] = numpy.mean(pitch_lf[kkas] * 180 / pi)
    #*
    #*     # ascent rate (m/s)
    #*     G_ratio[dive, 9] = Bottom[dive, 2] / G_ratio[dive, 5]



    # TODO REMOVE

    # 3.2.2_Determine `stroke_f` fluking rate and cut-off frequency
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #def plot_power_spec(ax, acc_ax, fs_a, title=''):
    #    S, f = speclev(acc_ax, 512, fs_a)
    #
    #    # TODO check that S.size[1] == 3
    #    for nn in range(3):
    #        # make 0.1 smaller if it is not finding the peak you expect
    #        peakloc, peakmag = peakfinder(S[: , nn], 0.1)
    #        peakloc[0] = []
    #        peakmag[0] = []
    #        smooth_S    = runmean(S[: , nn], 10)
    #        peakmag    = peakmag - smooth_S[peakloc]
    #        _, peak    = max(peakmag)
    #        peak       = peakloc[peak]
    #
    #        min_f, idx_f = min(S[:peak, nn])
    #        cutoff_low[stroke_f_n]   = f[idx_f]
    #
    #        plot(f[peak], S[peak, nn], 'go', 'markersize', 10, 'linewidth', 2)
    #        plot([f[idx_f], f[idx_f]],[min(S[:,nn]), min_f],'k--','linewidth',2)
    #
    #        text(f[idx_f], min(min(S[:,[1, 3]])),
    #             ['f = '.format(float(round(f[idx_f]*100)/100))],
    #             'horizontalalignment','right','color',
    #             cs[nn],'verticalalignment',va[nn])
    #
    #        stroke_f[stroke_f_n]     = f[peak]
    #        stroke_f_mag[stroke_f_n] = S[peak,nn]
    #
    #        stroke_f_n = stroke_f_n+1
    #
    #    _, b = max(stroke_f_mag[1: 2])
    #
    #    if b == 2:
    #        v1 = 'top'
    #        v2 = 'bottom'
    #    else:
    #        v1 = 'bottom'
    #        v2 = 'top'
    #
    #    text(stroke_f[1],
    #         stroke_f_mag[0] + 2 * sign( len(v1) - len(v2)),
    #         num2str( round( stroke_f[0] * 100) / 100),
    #         'verticalalignment', v1, 'horizontalalignment', 'center')
    #
    #    text(stroke_f[2],
    #         stroke_f_mag[1] - 2 * sign( len(v1) - len(v2)),
    #         num2str( round( stroke_f[1] * 100) / 100),
    #         'verticalalignment', v2, 'horizontalalignment', 'center')
    #
    #    b = plot(f, S[:, 0], 'b')
    #    r = plot(f, S[:, 2], 'r')
    #
    #    ax1 = gca
    #
    #    set(get(ax1, 'Xlabel'), 'String', [{'$\bf\ Frequency \hspace{1mm} (Hz) $'}],
    #        'interpreter', 'latex', 'FontSize', 8, 'FontName', 'Arial')
    #
    #    ys = get(ax1, 'ylim')
    #
    #    # to write the name of the pannel
    #    hp1 = text( 0.02, diff(ys) * .92 + min(ys), title, 'FontSize', 12,
    #            'FontWeight', 'bold', 'horizontalalignment', 'left')
    #
    #    legend([b, r], 'HPF acc x axis (surge)', 'HPF acc z axis (heave)')
    #
    #    return ax



    #* # 7.2 using the body rotations (pry) estimated using the
    #* # magnetometer method (use n=1)
    #* #----------------------------------------------------------------------
    #*
    #* alpha = 25
    #* cutoff_low_mag = cutoff_low
    #*
    #* MagAcc, pry, Sa, GL, KK = magnet_rot_sa(Aw, Mw, fs_a, stroke_f, f, alpha, 1,
    #*                                         k, J, tmax)
    #*
    #* # TODO PLOT: find fluke thresholds
    #* figure(6)
    #* subplot1 = subplot(1, 4, 1: 3)
    #*
    #* ax1 = plotyy(range(len(depths)), depths, range(len(pry[: , 0])), pry[: , 0])
    #* set(ax1(1), 'ydir', 'rev', 'ylim', [0, max(depths)])
    #*
    #* maxy = max(pry(round(T[0, 0] * fs): round(T[nn, 1] * fs)))
    #*
    #* set(ax1(2),
    #*     'ylim', [-2 * maxy 2 * maxy],
    #*     'ytick', round(-2 * maxy * 10) / 10: 0.1: 2 * maxy)
    #*
    #* set(ax1(2), 'nextplot', 'add')
    #* legend('Depth', 'rotations in y axis')
    #* title('Zoom in to find appropriate thresholds for fluking, then enter it for J')
    #*
    #*
    #*
    #* # body rotations ascent in radians
    #* flukePA = pry[ASC, 0]
    #* flukePD = pry[DES, 0]
    #*
    #* TOTAL = abs([flukePA flukePD])
    #* Y = buffer(TOTAL, 2 * round(1 / stroke_f * fs_a))
    #*
    #* # TODO PLOT: find fluke thresholds
    #* subplot2 = subplot(1, 4, 4)
    #*
    #* hist(max(Y), 100)
    #* set(subplot2, 'xlim', [0 max(max(Y))])
    #*
    #* # Choose a value for J based on the histogram for:
    #* #   pry(:,1), then when detecting glides in the next step use magnet_rot_sa
    #* #   function with n=1
    #*
    #* J =  # in radians
    #* tmax = 1 / stroke_f  # in seconds
    #*
    #* MagAcc, pry, Sa, GL, KK = magnet_rot_sa(Aw, Mw, fs_a, stroke_f, f, alpha, 1,
    #*                                         k, J,tmax)
    #*
    #* # KK = matrix of cues to zero crossings in seconds (1st column) and
    #* # zero-crossing directions (2nd column). +1 means a positive-going
    #* # zero-crossing. Times are in seconds.
    #* # this is already ensuring that all glides are longer than tmax/2
    #*
    #* # check glides duration and positive and negative zero crossings (KK) based
    #* # on selected J and tmax#
    #*
    #* # TODO PLOT
    #* figure(7)
    #* subplot1 = subplot(5, 1, 1: 4)
    #*
    #* plot(k, p_dive, 'g')
    #*
    #* # TODO PLOT
    #* plot(k, p_nodive, 'r', 'linewidth', 2)
    #* set(gca, 'ydir', 'rev', 'ylim', [min(depths) max(depths)])
    #*
    #* legend('Dive', 'Not Dive', 'location', 'southeast')
    #*
    #* # TODO PLOT
    #* plot(subplot1, p_asc, 'k', 'linewidth', 2)
    #* # TODO PLOT
    #* plot(subplot1, p_des, 'b', 'linewidth', 2)
    #*
    #* # TODO PLOT
    #* subplot2 = subplot(5, 1, 5)
    #* ax2 = plotyy(k, SwimSp, k, pry[:, 1])
    #* set(ax2(2), 'nextplot', 'add')
    #*
    #* # TODO PLOT
    #* # plot(ax2(2),KK(:,1]*fs,pry(round(KK(:,1]*fs),1),'r*')
    #* plot(ax2(2), KK[: , 1]*fs, 0, 'r*')
    #* set(ax2(1), 'ylim', [0 max(SwimSp)])
    #* set(ax2(2), 'ylim', [min(pry([ASC DES]]) max(pry([ASC DES]])])
    #* legend('speed', 'body rotations')
    #* linkaxes([subplot1 subplot2 ax2], 'x')  # links x axes
    #*
    #* GLdura            = GLa[: , 1]-GLa [: , 0]
    #* GLTa              = [GLa[:, 0], GLdura]
    #* GLdur             = GL[: , 1]-GL [: , 1]
    #* GLT               = [GL[:, 0], GLdur]
    #* t                 = range(len(pry[: , 0])-1)/fs
    #* gl_k              = eventon[GLT, t]
    #* gl_ka             = eventon[GLTa, t]
    #* p_gl              = depths
    #* p_gla             = depths
    #* p_gl[gl_k == 0]   = numpy.nan
    #* p_gla[gl_ka == 0] = numpy.nan
    #*
    #* # TODO PLOT
    #* # glides detected with the body rotations (pry)
    #* h3 = plot(subplot1, t * fs, p_gl, 'm:', 'Linewidth', 3)
    #*
    #* # TODO PLOT
    #* # glides detected with the hpf acc
    #* h4 = plot(subplot1, t * fs, p_gla, 'y:', 'Linewidth', 3)
    #*
    #* legend(subplot1, 'Bottom', 'Not Dive', 'Ascent', 'Descent', 'Glide', 'location',
    #*        'southeast')
    #*
    #* # SGtype indicates whether it is stroking (1) or gliding(0)
    #* gl_k1[gl_k == 0] = 1
    #* gl_k1[gl_k < 0] = 0
    #* SGtype = gl_k1


def load_lleo(root_path, data_path, cal_path):
    '''Load lleo data for calculating body condition'''
    import numpy
    import os

    from pylleo.pylleo import lleoio, lleocal

    # TODO auto set tag model, tag_id

    # load your prhfile
    # tag = ''  # eg., md13_134a

    # pitch roll and heading are in radians
    # Aw tri-axial accelerometer data at whale frame
    # Mw magnetometer data at whale frame

    #Aw, Mw = some_func()

    data_path = os.path.join(root_path, data_path)
    cal_yaml_path = os.path.join(root_path, cal_path, 'cal.yaml')

    sample_f  = 1
    tag_model = 'W190PD3GT'
    tag_id    = '34839'

    # TODO verify sensor ID of data matches ID of CAL

    meta = lleoio.read_meta(data_path, tag_model, tag_id)
    acc, depth, prop, temp = lleoio.read_data(meta, data_path, sample_f)

    # Truncate data based on dive depths
    cutoff_depth = 2

    import utils
    date_start = depth['datetimes'][utils.first_idx(depth['depth'] > 2)]
    date_end   = depth['datetimes'][utils.last_idx(depth['depth'] > 2)]
    A_trunc = acc[(acc['datetimes'] > date_start) & (acc['datetimes'] < date_end)]

    # Load and calibrate data acc data
    cal_dict = lleocal.read_cal(cal_yaml_path)
    Ax_g = lleocal.apply_poly(A_trunc, cal_dict, 'acceleration_x')
    Ay_g = lleocal.apply_poly(A_trunc, cal_dict, 'acceleration_y')
    Az_g = lleocal.apply_poly(A_trunc, cal_dict, 'acceleration_z')

    A_g = numpy.vstack([Ax_g, Ay_g, Az_g]).T


    # Turn depth values into float array
    depth_trunc = depth[(depth['datetimes'] > date_start) & \
                        (depth['datetimes'] < date_end)]

    depths = depth_trunc['depth'].values.astype(float)

    dt_a = float(meta['parameters']['acceleration_x']['Interval(Sec)'])
    fs_a = 1/dt_a

    dt_d = float(meta['parameters']['depth']['Interval(Sec)'])
    fs_d = 1/dt_d

    #dt = acc['datetimes'][1] - acc['datetimes'][0]
    #fs_a = 1/(dt.microseconds/1e6)


    return A_g, dt_a, fs_a, depths, dt_d, fs_d


def calc_PRH(A):
    from biotelem.acc import movement

    pitch = movement.pitch(A[:,0], A[:,1], A[:,2])
    roll  = movement.roll(A[:,0], A[:,1], A[:,2])
    yaw   = movement.yaw(A[:,0], A[:,1], A[:,2])

    #from dtag_toolbox_python.dtag2 import a2pr
    #pitch, roll, A_norm = a2pr.a2pr(A_g)

    return pitch, roll, yaw


def filter_acceleromter_low_high(A_g, fs_a, cutoff, order=5):
    '''Calculate low and high filtered tri-axial accelerometer data'''
    import numpy

    import utils_signal

    b_lo, a_lo, = utils_signal.butter_filter(cutoff, fs_a, order=order,
                                             btype='low')
    b_hi, a_hi, = utils_signal.butter_filter(cutoff, fs_a, order=order,
                                             btype='high')

    Ax_g_lf = utils_signal.butter_apply(b_lo, a_lo, A_g[:,0])
    Ay_g_lf = utils_signal.butter_apply(b_lo, a_lo, A_g[:,1])
    Az_g_lf = utils_signal.butter_apply(b_lo, a_lo, A_g[:,2])

    Ax_g_hf = utils_signal.butter_apply(b_hi, a_hi, A_g[:,0])
    Ay_g_hf = utils_signal.butter_apply(b_hi, a_hi, A_g[:,1])
    Az_g_hf = utils_signal.butter_apply(b_hi, a_hi, A_g[:,2])

    A_g_lf = numpy.vstack((Ax_g_lf, Ay_g_lf, Az_g_lf)).T
    A_g_hf = numpy.vstack((Ax_g_lf, Ay_g_lf, Az_g_lf)).T

    return A_g_lf, A_g_hf


def get_stroke_glide_indices(A_g_hf_n, fs_a, n, J=None, t_max=None):
    '''Get stroke and glide indices from high-pass accelerometer data

    Args
    ----
    A_g_hf_n: (1-D ndarray)
       whale frame triaxial accelerometer matrix at sampling rate fs_a.

    n: (int)
        fundamental axis of the acceleration signal.
        1 for accelerations along the x axis, longitudinal axis.
        2 for accelerations along the y axis, lateral axis.
        3 for accelerations along the z axis, dorso-ventral axis.

    J: (float)
        magnitude threshold for detecting a fluke stroke in m/s2.  If J is not
        given, fluke strokes will not be located but the rotations signal (pry)
        will be computed.

    t_max: (int)
        maximum duration allowable for a fluke stroke in seconds.  A fluke
        stroke is counted whenever there is a cyclic variation in the pitch
        deviation with peak-to-peak magnitude greater than +/-J and consistent
        with a fluke stroke duration of less than t_max seconds, e.g., for
        Mesoplodon choose t_max=4.

    Returns
    -------
    GL: (1-D ndarray)
        matrix containing the start time (first column) and end time (2nd
        column) of any glides (i.e., no zero crossings in t_max or more
        seconds).Times are in seconds.

    KK: (1-d ndarray)
        matrix of cues to zero crossings in seconds (1st column) and
        zero-crossing directions (2nd column). +1 means a positive-going
        zero-crossing. Times are in seconds.

    Note
    ----
    If no J or t_max is given, J=[], or t_max=[], GL and KK returned as None

    `K`  changed to `zc`
    `kk` changed to `col`

    '''
    import numpy

    import utils_signal

    # Check if input array is 1-D
    if A_g_hf_n.ndim > 1:
        raise IndexError('A_g_hf_n multidimensional: Glide index determination '
                         'requires 1-D acceleration array as input')

    # TODO remove?
    if (J == None) or (t_max == None):
        GL = None
        KK = None
        print( 'Cues for strokes(KK) and glides (GL) are not given as J and '
               't_max are not set')
    else:
        # Find cues to each zero-crossing in vector pry(:,n), rotations around
        # the n axis.
        zc = utils_signal.findzc(A_g_hf_n, J, (t_max* fs_a) / 2)

        # find glides - any interval between zeros crossings greater than tmax
        k = numpy.where(zc[1:, 0] - zc[0:-1, 1] > fs_a*t_max)[0]
        gl_k = numpy.vstack([zc[k, 0] - 1, zc[k + 1, 1] + 1]).T

        # Compute mean index position of glide, and typecast to int for indexing
        # Shorten the glides to only include sections with jerk < J
        gl_c = numpy.round(numpy.mean(gl_k, 1)).astype(int)
        gl_k = numpy.round(gl_k).astype(int)

        # TODO Remove if necessary
        # Lambda function: return 0 if array has no elements, else returns first element
        #get_1st_or_zero = lambda x: x[0] if x.size != 0 else 0
                #over_J1 = get_1st_or_zero(over_J1)
                #over_J2 = get_1st_or_zero(over_J2)

        for i in range(len(gl_c)):
            col = range(gl_c[i], gl_k[i, 0], - 1)
            test = numpy.where(numpy.isnan(A_g_hf_n[col]))[0]
            if test.size != 0:
                gl_c[i]   = numpy.nan
                gl_k[i,0] = numpy.nan
                gl_k[i,1] = numpy.nan
            else:
                over_J1 = numpy.where(abs(A_g_hf_n[col]) >= J)[0][0]

                gl_k[i,0] = gl_c[i] - over_J1 + 1

                col = range(gl_c[i], gl_k[i, 1])

                over_J2 = numpy.where(abs(A_g_hf_n[col]) >= J)[0][0]

                gl_k[i,1] = gl_c[i] + over_J2 - 1

        # convert sample numbers to times in seconds
        # TODO zc[:, 2] could not be sign by the 4th col in zero-crossing K
        KK = numpy.vstack((numpy.mean(zc[:, 0:1], 1) / fs_a, zc[:, 2])).T

        GL = gl_k / fs_a
        GL = GL[numpy.where(GL[:, 1] - GL[:, 0] > t_max / 2)[0], :]

    return GL, KK


def get_stroke_freq(x, fs_a, nperseg=512, peak_thresh=0.25):
    # TODO nperseg 128 without smoothing
    '''Determine stroke frequency to use as a cutoff for filtering

    Args
    ----
    x: (nx3 ndarray)
        tri-axial accelerometer data
    fs_a: (float)
        sampling frequency (i.e. number of samples per second)
    nperseg: (int)
        length of each segment (i.e. number of samples per frq. band in PSD
        calculation. Default to 512 (scipy.signal.welch() default is 256)
    peak_thresh: (float)
        PSD power level threshold. Only peaks over this threshold are returned.

    Returns
    -------
    cutoff: (float)
        cutoff frequency of signal (Hz) to be used for low/high-pass filtering
    stroke_f: (float)
        frequency of dominant wavelength in signal

    Notes
    -----
    During all descents and ascents phases where mainly steady swimming occurs.
    When calculated for the whole dive it may be difficult to differenciate the
    peak at which stroking rate occurs as there is other kind of movements than
    only steady swimming

    Here the power spectra is calculated of the longitudinal and dorso-ventral
    accelerometer signals during descents and ascents to determine the dominant
    stroke frequency for each animal in each phase

    This numpyer samples per f segment 512 and a sampling rate of fs_a.

    Output: S is the amount of power in each particular frequency (f)
    '''
    # TODO double check what peak_thresh should be from .m code

    def automatic_freq(x, fs_a, nperseg, peak_thresh):
        '''Find peak of FFT PSD and set cutoff and stroke freq by this'''
        import utils_signal
        # TODO remove this routine? from original code

        f_welch, S, _, _ = utils_signal.calc_PSD_welch(x, fs_a, nperseg)

        smooth_S = utils_signal.runmean(S, 10)

        # TODO check that arguments are same as dtag2 peakfinder
        peak_loc, peak_mag = utils_signal.peakfinder(S, sel=None, thresh=peak_thresh)

        peak_mag = peak_mag - smooth_S[peak_loc]
        peak_idx = peak_loc[peak_mag == max(peak_mag)]

        min_f    = numpy.min(S[:peak_idx])
        idx_f    = numpy.argmin(S[:peak_idx])

        cutoff         = f_welch[idx_f]
        stroke_f       = f_welch[peak_idx]
        # TODO remove?
        #stroke_PSD_mag = S[peak_idx]

        #* # Calc constant multiplied by `stroke_f` to yield the cutoff frequency
        #* # `cutoff_low`. f_frac is a fraction of `stroke_f`.
        #* # or set based on a predetermined ratio f_frac = 0.4
        #* f_frac  = cutoff_low / stroke_f

        #* # TODO sort out normalization and glide stats
        #* # Separate low and high pass filtered signals using the parameters defined
        #* n = 1
        #* k = range(len(depths))
        #* # dummy variables
        #* J = 0.1
        #* tmax = 1 / stroke_f
        #* Anlf, Ahf, GL, KK = Ahf_Anlf(Aw, fs_a, stroke_f, f_frac, n, k, [], [])

        return cutoff, stroke_f


    def manual_freq(x, fs_a, nperseg, peak_thresh):
        '''Manually look at plot, then enter cutoff frequency for x,y,z'''
        import matplotlib.pyplot as plt

        import utils_signal

        f_welch, S, _, _ = utils_signal.calc_PSD_welch(x, fs_a, nperseg)

        peak_loc, peak_mag = utils_signal.peakfinder(S, sel=None, thresh=peak_thresh)
        peak_idx = peak_loc[peak_mag == max(peak_mag)]

        # Plot power specturm against frequency distribution
        # TODO add axes labels
        plt.plot(f_welch, S)
        plt.scatter(f_welch[peak_loc], S[peak_loc], label='peaks')
        plt.legend(loc='upper right')
        plt.show()

        # Get user input of cutoff frequency identified off plots
        cutoff = recursive_input('cutoff frequency', float)
        stroke_f = f_welch[peak_idx]
        # TODO remove?
        #stroke_PSD_mag = S[peak_idx]

        return cutoff, stroke_f

    stroke_axes = ['Ax', 'Az']

    cutoffs    = list()
    stroke_fqs = list()
    for i, key in enumerate(stroke_axes):

        cutoff, stroke_f =  manual_freq(x[:,i], fs_a, nperseg, peak_thresh)
        cutoffs.append(cutoff)
        stroke_fqs.append(stroke_f)

    return numpy.mean(cutoffs), numpy.mean(stroke_fqs)


def create_dive_summary(T):
    '''Create a numpy array with summary values of dives

    Args
    ----
    T: (ndarray)
      Dive table

    Returns
    -------
    D: (ndarray)
      table of dive summary files

    Notes
    -----
    D = numpy.zeros((n_dives)), dtype=dtypes)

    where:
    dtype = numpy.dtype([('start_time', int),  # start in sec since tag on time
                         ('end_time', int),    # end in sec since tag on time
                         ('duration', int),    # dive duration in sec
                         ('surface', int),     # post-dive surface duration in sec
                         ('max_time', int),    # time of deepest point
                         ('max_depth', float), # maximum dive depth of each dive
                         ('dive_id', int),     # dive ID number
                         ])
    '''
    import numpy

    n_dives = len(T[:, 0])

    D = numpy.zeros((n_dives, 7))

    # start in sec since tag on time
    D[:, 0] = T[:, 0]

    # end in sec since tag on time
    D[:, 1] = T[:, 1]

    # dive duration in sec
    D[:, 2] = T[:, 1] - T[:, 0]

    # post-dive surface duration in sec
    D[:, 3] = numpy.hstack((T[1:, 0] - T[0:-1, 1], [numpy.nan]))

    # time of deepest point
    D[:, 4] = T[:, 3]

    # maximum dive depth of each dive
    D[:, 5] = T[:, 2]

    # dive ID number
    D[:, 6] = range(n_dives)

    return D


def get_asc_des(T, pitch, fs_a):
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

    for dive in range(len(T)):
        print('Dive: {}'.format(dive))
        # get list of indices to select the whole dive
        # multiple by acc sampling rate to scale indices
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


if __name__ == '__main__':
    run()
