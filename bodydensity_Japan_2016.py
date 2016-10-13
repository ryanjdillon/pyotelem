'''
Body density estimation. Japan workshop May 2016

Lucia Martina Martin Lopez lmml2@st-andrews.ac.uk
'''
import numpy

# 1_LOAD DATA.
#==============================================================================

# load your prhfile
clear all
tag = ''  # eg., md13_134a
settagpath('prh', 'C:/tag/tag2/metadata/prh')
loadprh(tag)

# or as
[files, fileloc] = uigetfile('*.mat', 'select prh file', 'multiselect', 'on')
if ischar(files):
    files = {files}

nn = regexp(files{1}, 'prh')
tag = files{1}(1: nn - 1)
settagpath('prh', fileloc)
loadprh(tag)

# pitch roll and heading are in radians
# Aw and Mw are the tri-axial accelerometer and magnetometer data at the
# whale frame respectively


# 2_DEFINE DIVES and make a summary table describing the characteristics of
#==============================================================================

# each dive.
k = range(len(p))

# define min_dive_def as the minimum depth at which to recognize a dive.
min_dive_def = 300

T = finddives(p, fs, min_dive_def, [], 1)

# TODO make numpy record array

#[start_time(s) end_time(s) dive_duration(s) max_depth(s) max_depth(m) ID(n)]
D = []

# start time of each dive in seconds since the tag on time
D[:, 0] = T[: , 0]

# end time of each dive in seconds since the tag on time
D[:, 1] = T[: , 1]

# dive duration in seconds
D[:, 2] = T[: , 1] - T[: , 0]

# Post-dive surface duration in seconds
D[:, 3] = [T[1:-1, 0]-T[0:-2, 1], NaN]

# time of the deepest point of each dive
D[:, 4] = T[: , 3]

# maximum dive depth of each dive
D[:, 5] = T[: , 2]

# Dive ID number, starting from 1
# TODO check
D[:, 6] = 1*(0: T.size[0])


# 3 SEPARATE LOW AND HIGH ACCELERATION SIGNALS
#==============================================================================

# 3.1 QUICK SEPARATION OF DESCENT AND ASCENT PHASES
#------------------------------------------------------------------------------

BOTTOM = []
DES = []
ASC = []

for dive = range(T.size[0]):
    # TODO check indentation here

    # it is selecting the whole dive
    kk = round(fs * T[dive, 0]):round(fs * T[dive, 1])

    # search for the first point at which pitch is positive
    end_des = (numpy.where((pitch[kk] * 180 / pi) > 0, 1, 'first') + T[dive, 0] * fs)

    # search for the last point at which the pitch is negative
    start_asc = (numpy.where((pitch[kk] * 180 / pi) < 0, 1, 'last') + T[dive, 0] * fs)

    # Time in seconds at the start of bottom phase (end of descent)
    BOTTOM[:, 0] = (end_des)/fs

    # Time in seconds at the end of bottom phase (start of descent)
    BOTTOM[:, 2] = (start_asc)/fs

    # selects the whole descent phase
    des = round(fs * T[dive, 0]):
        round(end_des)

    # selects the whole ascent phase
    asc = round(start_asc): round(fs * T[dive, 1])

    DES = [DES, des]
    ASC = [ASC, asc]


# 3.2 SEPARATE LOW AND HIGH ACCELERATION SIGNALS
#------------------------------------------------------------------------------

# 3.2.1 select periods of analysis.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# during whole deployment
S, f = speclev(Aw[k, :], 512, fs)

# TODO PLOT: spec levels - f vs S
plot(f, S)

# during all descents and ascents phases where mainly steady swimming occurs.
# When calculated for the whole dive it may be difficult to differenciate the
# peak at which stroking rate occurs as there is other kind of movements than
# only steady swimming
S, f = speclev(Aw[DES, : ], 512, fs)
S, f = speclev(Aw[ASC, : ], 512, fs)

# here the power spectra is calculated of the longitudinal and dorso-ventral
# accelerometer signals during descents and ascents to determine the dominant
# stroke frequency for each animal in each phase with a fft of 512 and a
# sampling rate of fs.  Output: S is the amount of power in each particular
# frequency (f)


# 3.2.2_Determine FR fluking rate and cut-off frequency
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# calculate the power spectrum of the accelerometer data at the whale frame
# plot PSD for the whole deployment, all descents and all ascent phases.

FR     = numpy.zeros(6, 1)
FR[:]  = numpy.nan
FR_mag = FR
FRl    = numpy.zeros(6, 1)
FR1[:] = numpy.nan
FRlmag = FRl
FRn    = 0
cs     = 'bgr'
va     = {'top', '', 'bottom'}

S, f = speclev(Aw[k, : ], 512, fs), grid

# whole deployment
# TODO PLOT
figure(1)
clf
ax(1) = subplot(311)

# TODO check that S.size[1] == 3
for nn in range(3):
    # make 0.1 smaller if it is not finding the peak you expect
    peakloc, peakmag = peakfinder(S[: , nn], 0.1)
    peakloc[0] = []
    peakmag[0] = []
    smoothS    = runmean(S[: , nn], 10)
    peakmag    = peakmag - smoothS[peakloc]
    _, peak    = max(peakmag)
    peak       = peakloc[peak]

    plot(f[peak], S[peak, nn], 'go', 'markersize', 10, 'linewidth', 2)

    [minf, bf] = min(S[1: peak, nn])
    FRl[FRn]   = f[bf]

    # TODO removeL: plot
    # hold on
    # plot([f[bf] f[bf]],[min(S[:,nn]) minf],'k--','linewidth',2)
    # text(f[bf],min(min(S[:,[1 3]])),['f = ' num2str(round(f[bf]*100)/100)],'horizontalalignment','right','color',cs[nn],'verticalalignment',va{nn})
    # FR[FRn] = f[peak]
    # FR_mag[FRn] = S[peak,nn]
    # FRn = FRn+1

_, b = max(FR_mag[1: 2])

if b == 2:
    v1 = 'top'
    v2 = 'bottom'
else:
    v1 = 'bottom'
    v2 = 'top'

text(FR[1],
     FR_mag[0] + 2 * sign( len(v1) - len(v2)),
     num2str( round( FR[0] * 100) / 100),
     'verticalalignment', v1, 'horizontalalignment', 'center')

text(FR[2],
     FR_mag[1] - 2 * sign( len(v1) - len(v2)),
     num2str( round( FR[1] * 100) / 100),
     'verticalalignment', v2, 'horizontalalignment', 'center')

# TODO PLOT
b = plot(f, S[:, 0], 'b')
# TODO PLOT
r = plot(f, S[:, 2], 'r')

grid on
ax1 = gca

set(get(ax1, 'Xlabel'), 'String', [{'$\bf\ Frequency \hspace{1mm} (Hz) $'}],
        'interpreter', 'latex', 'FontSize', 8, 'FontName', 'Arial')

ys = get(ax1, 'ylim')

hp1 = text( 0.02, diff(ys) * .92 + min(ys), 'Whole deployment', 'FontSize', 12,
        'FontWeight', 'bold', 'horizontalalignment', 'left')  # to write the

# name of the pannel
legend([b r], 'HPF acc x axis (surge)', 'HPF acc z axis (heave)')

ax(2) = subplot(312)

S, f = speclev(Aw[DES, : ], 512, fs)
for nn in range(3):
    # make 0.1 smaller if it is not finding the peak you expect
    peakloc, peakmag = peakfinder(S[: , nn], 0.1)
    peakloc[0] = []
    peakmag[0] = []
    smoothS    = runmean(S[: , nn], 10)
    peakmag    = peakmag - smoothS[peakloc]
    _, peak    = max(peakmag)
    peak       = peakloc[peak]

    # TODO PLOT
    plot(f[peak], S[peak, nn], 'go', 'markersize', 10, 'linewidth', 2)
    [minf, bf] = min(S[0: peak, nn])
    FRl[FRn] = f[bf]

    # TODO PLOT
    plot([f[bf] f[bf]], [min(S[:, nn]) minf], 'k--', 'linewidth', 2)

    text(f[bf], min(min(S[:, [1,3]])),
         ['f = ' num2str(round(f[bf]*100)/100)],
         'horizontalalignment', 'right', 'color',
         cs[nn], 'verticalalignment', va{nn})

    FR[FRn]    = f[peak]
    FR_mag[FRn] = S[peak, nn]
    FRn += 1

_, b = max(FR_mag[0: 1])

if b == 2:
    v1 = 'top'
    v2 = 'bottom'
else:
    v1 = 'bottom'
    v2 = 'top'

text( FR[0], FR_mag[2] + 2 * sign( len(v1) - len(v2)), num2str( round( FR[2] *
    100) / 100), 'verticalalignment', v1, 'horizontalalignment', 'center')

text( FR[1], FR_mag[3] - 2 * sign( len(v1) - len(v2)), num2str( round( FR[3] *
    100) / 100), 'verticalalignment', v2, 'horizontalalignment', 'center')

# TODO PLOT
b = plot(f, S[:, 0], 'b') grid
# TODO PLOT
r = plot(f, S[:, 2], 'r')

ax1 = gca

set(get(ax1, 'Xlabel'), 'String', [{'$\bf\ Frequency \hspace{1mm} (Hz) $'}],
        'interpreter', 'latex', 'FontSize', 8, 'FontName', 'Arial')

ys = get(ax1, 'ylim')

# to write the name of the pannel
hp1 = text( 0.02, diff(ys) * .92 + min(ys), 'Descents', 'FontSize', 12,
        'FontWeight', 'bold', 'horizontalalignment', 'left')

legend([b r], 'HPF acc x axis (surge)', 'HPF acc z axis (heave)')

ax(3) = subplot(313)

S, f = speclev(Aw[ASC, : ], 512, fs)
# TODO check this for loop
for nn in range(3):
    peakloc, peakmag = peakfinder(S[: , nn], 0.1)
    peakloc[0] = [] peakmag[0] = []
    smoothS = runmean(S[: , nn], 10)
    peakmag = peakmag - smoothS(peakloc)
    _, peak = max(peakmag)
    peak = peakloc[peak]

    # TODO PLOT
    plot(f[peak], S[peak, nn], 'go', 'markersize', 10, 'linewidth', 2)

    minf, bf = min(S[0: peak, nn])
    FRl[FRn] = f[bf]

    # TODO PLOT
    plot([f[bf] f[bf]], [min(S[:, nn]) minf], 'k--', 'linewidth', 2)

    text(f[bf],
         min(min(S[:, [1 3]])),
         ['f = ' num2str(round(f[bf]*100)/100)],
         'horizontalalignment', 'right', 'color',
         cs[nn], 'verticalalignment', va{nn})

    FR[FRn]     = f[peak]
    FR_mag[FRn] = S[peak, nn]
    FRn += 1

_, b = max(FR_mag[2: 3])

if b == 2:
    v1 = 'top' v2 = 'bottom'
else:
    v1 = 'bottom' v2 = 'top'

text(FR[2],
     FR_mag[4] + 2 * sign( len(v1) - len(v2)),
     num2str( round( FR[4] * 100) / 100),
     'verticalalignment', v1, 'horizontalalignment', 'center')

text(FR[3],
     FR_mag[5] - 2 * sign( len(v1) - len(v2)),
     num2str( round( FR[5] * 100) / 100),
     'verticalalignment', v2, 'horizontalalignment', 'center')

# TODO PLOT
b = plot(f, S[:, 0], 'b')
r = plot(f, S[:, 2], 'r')

ax2 = gca

set(get(ax2, 'Xlabel'), 'String', [{'$\bf\ Frequency \hspace{1mm} (Hz) $'}],
        'interpreter', 'latex', 'FontSize', 8, 'FontName', 'Arial') ys =

# the name of the pannel
get(ax2, 'ylim') hp1 = text( 0.02, diff(ys) * .92 + min(ys), 'Ascents',
        'FontSize', 12, 'FontWeight', 'bold', 'horizontalalignment', 'left')

legend([b r], 'HPF acc x axis (surge)', 'HPF acc z axis (heave)')

# links x axes of the subplots for zoom/pan
linkaxes(ax, 'x')


# f = number that multiplied by the FR gives the cut-off frequency fl, of the
# low pass filter. f is a fraction of FR. You can set default value to 0.4 if
# not, otherwise set f as fl (frequency at the negative peak in the power
# spectral density plot)/FR.

# or set based on prior graph
FR = numpy.mean(FR)

# from last graph
f  = numpy.mean(FRl) / FR

# f=0.4  # or set based on a predetermined ratio
alpha = 25
n = 1
k = range(len(p))

# dummy variables
J = 0.1
tmax = 1 / FR


# 3.2.3 Separate low and high pass filtered acceleration signal using the
#       parameters defined earlier and the function Ahf_Alnf.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Anlf, Ahf, GL, KK = Ahf_Anlf(Aw, fs, FR, f, n, k, [], [])


# 3.2.4 calculate the smooth pitch from the low pass filter acceleration signal
# to avoid incorporating signals above the stroking periods
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

smoothpitch, smoothroll = a2pr(Anlf[k, : ])

# check the difference between pitch and smoothpitch
# TODO PLOT: pitch vs. smooth pitch

#figure(2) clf
#ax1 = subplot(2, 1, 1)
#plott(p, fs)
#ax2 = subplot(2, 1, 2)
#
#plot((1: len(p)) / fs, pitch * 180 / pi)
#hold on
#
#plot((1: len(p)) / fs, smoothpitch * 180 / pi, 'r')
#legend('pitch', 'smoothpitch')
#linkaxes([ax1 ax2], 'x')


# 4 DEFINE PRECISE DESCENT AND ASCENT PHASES
#==============================================================================

Bottom = []

Phase[range(len(p))] = numpy.nan


#TODO check matlab here
isdive = numpy.zeros(p.size, dtype=bool)
for i = range(T.size[0]):
    isdive[round(T[i, 0] * fs): round(T[i, 1] * fs)] = true

p_dive   = p
p_nodive = p
p_dive[~isdive]  = nan
p_nodive[isdive] = nan

I = range(len(p))

figure(4) clf
subplot1 = subplot(5, 1, 1: 4)

# TODO PLOT
plot(I, p_dive, 'g')

# TODO PLOT
plot(I, p_nodive, 'r')

set(gca, 'ydir', 'rev', 'ylim', [min(p) max(p)])

tit = title('Click within the last dive you want to use')
legend('Dive', 'Not Dive', 'location', 'southeast')

subplot2 = subplot(5, 1, 5)

# TODO PLOT
plot(I, pitch, 'g', I, smoothpitch, 'r')
legend('pitch', 'smoothpitch')

x = ginput(1)

nn = numpy.where(T[:, 0] < x[0]/fs, 1, 'last')

for dive in range(T.size[0]):
    # it is selecting the whole dive
    kk = round(fs * T[dive, 0]): round(fs * T[dive, 1])

    kkI = numpy.zeros(p.size, dtype=bool)
    kkI[kk] = true

    # search for the first point after diving below min_dive_def at which pitch
    # is positive
    end_des = round(numpy.where((smoothpitch(kkI & p > min_dive_def * .75)
                         * 180 / pi) > 0, 1, 'first') + T[dive, 0] * fs)

    # search for the last point beroe diving above min_dive_def at which the
    # pitch is negative if you want to do it manually as some times there is a
    # small ascent
    start_asc = round(numpy.where((smoothpitch[kkI & p > min_dive_def * .75)
                           * 180 / pi) < 0, 1, 'last') + T[dive, 0] * fs)

    # TODO PLOT (CLICK): where pitch angle first goes to zero & last goes to
    #                    zero in the ascent

    # phase during the descent and a small descent phase during the ascent.
    #     figure
    #     # plott plots sensor data against a time axis
    #     ax(1)=subplot(211) plott(p[kk],fs)
    #     ax(2)=subplot(212) plott(pitch[kk]*180/pi,fs,0)
    #     linkaxes(ax, 'x') # links x axes of the subplots for zoom/pan
    #     # click on where the pitch angle first goes to zero in the descent and
    #     # last goes to zero in the ascent
    #     [x,y]=ginput(2)
    #     des=round(x[1])/fs+T[dive,1]
    #     asc=round(x[2))/fs+T[dive,1]

    Phase(kk[kk < end_des]) = -1
    Phase(kk[(kk < start_asc) & (kk > end_des)]] = 0
    Phase(kk[kk > start_asc]] = 1

    # Time in seconds at the start of bottom phase (end of descent)
    Bottom[dive, 0] = (end_des) / fs

    # Depth in m at the start of the bottom phase (end of descent phase)
    Bottom[dive, 1] = p[end_des]

    # Time in seconds at the end of bottom phase (start of descent)
    Bottom[dive, 2] = (start_asc) / fs

    # Depth in m at the end of the bottom phase (start of descent phase)
    Bottom[dive, 3] = p[start_asc]

p_asc = p
p_des = p

p_asc(Phase < 1 | isnan(Phase)) = nan
p_des(Phase > -1 | isnan(Phase)) = nan

# TODO PLOT
plot(subplot1, p_asc, 'k')
# TODO PLOT
plot(subplot1, p_des, 'b')

legend('Dive', 'Not Dive', 'Ascent', 'Descent', 'location', 'southeast')

linkaxes([subplot1 subplot2], 'x')


# 5 ESTIMATE SWIM SPEED
#==============================================================================

# degree threshold above which speed can be estimated
th_deg = 30

SwimSp = inst_speed(p, smoothpitch, fs, FR, f, k, th_deg)

# TODO PLOT
figure(4)
subplot2 = subplot(5, 1, 5)

plot(k, SwimSp, 'g') ylim([0 max(SwimSp)])

legend('speed')

# links x axes of the subplots for zoom/pan
linkaxes([subplot1 subplot2], 'x')


# 6 ESTIMATE SEAWATER DESNSITY AROUND THE TAGGED WHALES
#==============================================================================

# Seawater density estimation
DPT = DTS[:, 0]
TMP = DTS[:, 1]
SL  = DTS[:, 2]

SWdensity, depCTD = SWdensityFromCTD(DPT, TMP, SL, D)

Dsw = EstimateDsw(SWdensity, depCTD, p)


# 7 EXTRACT STROKES AND GLIDES
#==============================================================================

# it can be done using the body rotations (pry) estimated using the
# magnetometer method, or it can be done using the dorso-ventral axis of the
# high-pass filtered acceleration signal Using whichever method, tmax and J
# need to be determined.  tmax is the maximum duration allowable for a fluke
# stroke in seconds, it can be set as 1/FR# J is the magnitude threshold for
# detecting a fluke stroke in


# 7.1 using the heave high pass filtered acceleration signal,(use n=3)
#------------------------------------------------------------------------------

# units of J are in m/s2 set J and tmax [] until determined.
Anlf, Ahf, GL, KK = Ahf_Anlf(Aw, fs, FR, f, n, k, J=[], tmax=[])


# TODO PLOT
figure(5) clf

subplot1 = subplot(1, 4, 1: 3)
ax2 = plotyy(range(len(p)), p, range(len(pitch)), Ahf[: , 2])

set(ax2(1), 'ydir', 'rev', 'ylim', [0, max(p)])
set(ax2(2), 'nextplot', 'add')
plot(ax2(2), range(len(pitch)), Ahf[:, 0], 'm')

maxy = max(max(Ahf[round(T[0, 0] * fs): round(T[nn, 1] * fs), [1,3])))

set(ax2(2),
    'ylim', [-2 * maxy 2 * maxy],
    'ytick', round(-2 * maxy * 10) / 10: 0.1: 2 * maxy)

legend('Depth', 'HPF acc z axis', 'HPF acc x axis')
title('Zoom in to find appropriate thresholds for fluking, then enter it for J')
linkaxes([ax1 ax2], 'x')  # li

flukeAcc1A = Ahf[ASC, 0]  # hpf-x acc ascent
flukeAcc1D = Ahf[DES, 0]  # hpf-x acc descent
flukeAcc3A = Ahf[ASC, 2]  # hpf-z acc  ascent
flukeAcc3D = Ahf[DES, 2]  # hpf-z acc  desscent

# to look at heave, change to 3
TOTAL = abs([flukeAcc1A flukeAcc1D])
Y = buffer(TOTAL, 2 * round(1 / FR * fs))

# TODO PLOT: histogram of x acc
subplot2 = subplot(2, 4, 4)

hist(max(Y), 100)
set(subplot2, 'xlim', [0 max(max(Y))])
title('hpf-x acc')

subplot3 = subplot(2, 4, 8)
TOTAL = abs([flukeAcc3A flukeAcc3D])
Y = buffer(TOTAL, 2 * round(1 / FR * fs))

# TODO PLOT: histogram of y acc
hist(max(Y), 100, 'FaceColor', 'g') set(subplot3, 'xlim', [0 max(max(Y))])
title('hpf-z acc')

# Choose a value for J based on the histogram for:
#   hpf-x, then when detecting glides in the next step use Ahf_Anlf
#   function with n=1

#   hpf-z then when detecting glides in the next step use Ahf_Anlf
#   function with n=3

J =  # in m/s2
tmax = 1 / FR  # in seconds

Anlf, Ahf, GL, KK = Ahf_Anlf(Aw, fs, FR, f, n, k, J, tmax)

# in case you want to check the performance of both methods in the following
# figure define glides and strokes obtained from the high pass filtered
# acceleration as GLa and KKa respectively
Anlf, Ahf, GLa, KKa = Ahf_Anlf(Aw, fs, FR, f, n, k, J, tmax)


# 7.2 using the body rotations (pry) estimated using the
# magnetometer method (use n=1)
#------------------------------------------------------------------------------

MagAcc, pry, Sa, GL, KK = magnet_rot_sa(Aw, Mw, fs, FR, f, alpha, 1, k, J,
                                        tmax)

# TODO PLOT: find fluke thresholds
figure(6) clf
subplot1 = subplot(1, 4, 1: 3)

ax1 = plotyy(range(len(p)), p, range(len(pry[: , 0])), pry[: , 0])
set(ax1(1), 'ydir', 'rev', 'ylim', [0, max(p)])

maxy = max(pry(round(T[0, 0] * fs): round(T[nn, 1] * fs)))

set(ax1(2),
    'ylim', [-2 * maxy 2 * maxy],
    'ytick', round(-2 * maxy * 10) / 10: 0.1: 2 * maxy)

set(ax1(2), 'nextplot', 'add')
legend('Depth', 'rotations in y axis')
title('Zoom in to find appropriate thresholds for fluking, then enter it for J')



# body rotations ascent in radians
flukePA = pry[ASC, 0]
flukePD = pry[DES, 0]

TOTAL = abs([flukePA flukePD])
Y = buffer(TOTAL, 2 * round(1 / FR * fs))

# TODO PLOT: find fluke thresholds
subplot2 = subplot(1, 4, 4)

hist(max(Y), 100)
set(subplot2, 'xlim', [0 max(max(Y))])

# Choose a value for J based on the histogram for:
#   pry(:,1), then when detecting glides in the next step use magnet_rot_sa
#   function with n=1

J =  # in radians
tmax = 1 / FR  # in seconds

MagAcc, pry, Sa, GL, KK = magnet_rot_sa(Aw, Mw, fs, FR, f, alpha, 1, k, J,
                                        tmax)

# KK = matrix of cues to zero crossings in seconds (1st column) and
# zero-crossing directions (2nd column). +1 means a positive-going
# zero-crossing. Times are in seconds.
# this is already ensuring that all glides are longer than tmax/2

# check glides duration and positive and negative zero crossings (KK) based
# on selected J and tmax#

# TODO PLOT
figure(7) clf
subplot1 = subplot(5, 1, 1: 4)

plot(k, p_dive, 'g')

# TODO PLOT
plot(k, p_nodive, 'r', 'linewidth', 2)
set(gca, 'ydir', 'rev', 'ylim', [min(p) max(p)])

legend('Dive', 'Not Dive', 'location', 'southeast')

# TODO PLOT
plot(subplot1, p_asc, 'k', 'linewidth', 2)
# TODO PLOT
plot(subplot1, p_des, 'b', 'linewidth', 2)

# TODO PLOT
subplot2 = subplot(5, 1, 5)
ax2 = plotyy(k, SwimSp, k, pry[:, 1])
set(ax2(2), 'nextplot', 'add')

# TODO PLOT
# plot(ax2(2),KK(:,1]*fs,pry(round(KK(:,1]*fs),1),'r*')
plot(ax2(2), KK[: , 1]*fs, 0, 'r*')
set(ax2(1), 'ylim', [0 max(SwimSp)])
set(ax2(2), 'ylim', [min(pry([ASC DES]]) max(pry([ASC DES]])])
legend('speed', 'body rotations')
linkaxes([subplot1 subplot2 ax2], 'x')  # links x axes

GLdura          = GLa[: , 1]-GLa [: , 0]
GLTa            = [GLa[:, 0], GLdura]
GLdur           = GL[: , 1]-GL [: , 1]
GLT             = [GL[:, 0], GLdur]
t               = range(len(pry[: , 0])-1)/fs
glk             = eventon[GLT, t]
glka            = eventon[GLTa, t]
pgl             = p
pgla            = p
pgl[glk == 0]   = numpy.nan
pgla[glka == 0] = numpy.nan

# TODO PLOT
# glides detected with the body rotations (pry)
h3 = plot(subplot1, t * fs, pgl, 'm:', 'Linewidth', 3)

# TODO PLOT
# glides detected with the hpf acc
h4 = plot(subplot1, t * fs, pgla, 'y:', 'Linewidth', 3)

legend(subplot1, 'Bottom', 'Not Dive', 'Ascent', 'Descent', 'Glide', 'location',
        'southeast')

# SGtype indicates whether it is stroking (1) or gliding(0)
glk1[glk == 0] = 1
glk1[glk < 0] = 0
SGtype = glk1


# 8_MAKE 5SEC SUB-GLIDES
#==============================================================================

dur = 5
SGL = splitGL[dur, GL]
SGLT = [SGL[:, 0], SGL[:, 1] - SGL[:, 0]]

# check that all subglides have a duration of 5 seconds
rangeSGLT = [min(SGLT[:, 1]), max(SGLT[:, 1])]

glk = eventon[SGLT, t]
pgl = p
pgl[glk == 0] = numpy.nan

# TODO PLOT
figure(7)

h3 = plot(subplot1, t * fs, pgl, 'm', 'Linewidth', 3)


# 9 create summary table required for body density
#==============================================================================

FR    = 0.4
f     = 0.4
alpha = 25
n     = 1
k     = range(len(p))
J     = 2 / 180 * pi
tmax  = 1 / FR

MagAcc, pry, Sa, GL, KK = magnet_rot_sa(Aw, Mw, fs, FR, f, alpha, n, k, J,
                                        tmax)
smoothhead = m2h(MagAcc.Mnlf[k, :], smoothpitch, smoothroll)
smoothpitchdeg = smoothpitch * 180 / pi


# 10 CALCULATE GLIDE DESC/GLIDE ASC
#==============================================================================

dur = 5
Glide = numpy.zeros(len(SGL), 24)
for i in range(len(SGL)):
    cue1 = SGL[i, 0] * fs
    cue2 = SGL[i, 1] * fs

    # sub-glide start point in seconds
    Glide[i, 0] = SGL[i, 0]

    # sub-glide end point in seconds
    Glide[i, 1] = SGL[i, 1]

    # sub-glide duration
    Glide[i, 2] = SGL[i, 1] - SGL[i, 0]

    # mean depth(m)during sub-glide
    Glide[i, 3] = numpy.mean(p[round(cue1): round(cue2)])

    # total depth(m)change during sub-glide
    Glide[i, 4] = abs(p[round(cue1)] - p[round(cue2)])

    # mean swim speed during the sub-glide, only given if pitch>30 degrees
    Glide[i, 5] = numpy.mean(SwimSp[round(cue1): round(cue2)])

    # mean pitch during the sub-glide
    Glide[i, 6] = numpy.mean(smoothpitchdeg[round(cue1): round(cue2)])

    # mean sin pitch during the sub-glide
    Glide[i, 7] = numpy.sin(numpy.mean(smoothpitchdeg[round(cue1): round(cue2)]))

    # SD of pitch during the sub-glide
    Glide[i, 8] = numpy.std(smoothpitchdeg[round(cue1): round(cue2)]) * 180 / pi

    # mean temperature during the sub-glide
    Glide[i, 9] = numpy.mean(temp[round(cue1): round(cue2)])

    # mean seawater density (kg/m^3) during the sub-glide
    Glide[i, 10] = numpy.mean(Dsw[round(cue1): round(cue2)])

    # TODO check matlab here
    try
    xpoly = (round(cue1): round(cue2))
    ypoly = SwimSp[round(cue1): round(cue2)]

    B, BINT, R, RINT, STATS = regress(ypoly, [xpoly, numpy.ones(len(ypoly), 1)])

    # mean acceleration during the sub-glide
    Glide[i, 11] = B[0]

    # R2-value for the regression swim speed vs. time during the sub-glide
    Glide[i, 12] = STATS[0]

    # SE of the gradient for the regression swim speed vs. time during the
    # sub-glide
    Glide[i, 13] = STATS[3]

    # catch  # TODO check matlab here

    # mean acceleration during the sub-glide
    Glide[i, 11] = numpy.nan

    # R2-value for the regression swim speed vs. time during the sub-glide
    Glide[i, 12] = numpy.nan

    # SE of the gradient for the regression swim speed vs. time during the
    # sub-glide
    Glide[i, 13] = numpy.nan

    sumphase = sum(phase[round(cue1): round(cue2)])
    # TODO check what dimensions of sp should be
    sp = numpy.nan
    sp[sumphase < 0]  = -1
    sp[sumphase == 0] = 0
    sp[sumphase > 0]  = 1

    # Dive phase:0 bottom, -1 descent, 1 ascent, NaN not dive phase
    Glide[i, 14] = sp

    Dinf = D[numpy.where((D[: , 0]*fs < cue1) & (D[: , 1]*fs > cue2)), : ]

    if isempty(Dinf):
        Dinf = numpy.zeros(D.size[0], D.size[1])
        Dinf[:] = numpy.nan

    # Dive number in which the sub-glide recorded
    Glide[i, 15] = Dinf[6]

    # Maximum dive depth (m) of the dive
    Glide[i, 16] = Dinf[5]

    # Dive duration (s) of the dive
    Glide[i, 17] = Dinf[2]

    # Mean pitch(deg) calculated using circular statistics
    Glide[i, 18] = circ_mean(smoothpitch[round(cue1): round(cue2)])

    # Measure of concentration (r) of pitch during the sub-glide (i.e. 0 for
    # random direction, 1 for unidirectional)
    Glide[i, 19] = 1 - circ_var(smoothpitch[round(cue1): round(cue2)])

    # Mean roll (deg) calculated using circular statistics
    Glide[i, 20] = circ_mean(smoothroll[round(cue1): round(cue2)])

    # Measure of concentration (r) of roll during the sub-glide
    Glide[i, 21] = 1 - circ_var(smoothroll[round(cue1): round(cue2)])

    # Mean heading (deg) calculated using circular statistics
    Glide[i, 22] = circ_mean(smoothhead[round(cue1): round(cue2)])

    # Measure of concentration (r) of heading during the sub-glide
    Glide[i, 23] = 1 - circ_var(smoothhead[round(cue1): round(cue2)])


csvwrite('WhaleID.csv', Glide)


# 11 Calculate glide ratio
#==============================================================================

G_ratio = zeros(T.size[0], 10)

for dive = 1:
    T.size[0]:
        # it is selecting the whole dive
    kkdes = round(fs * T[dive, 0]): round(fs * Bottom[dive, 0])

    # it is selecting the whole dive
    kkas = round(fs * Bottom[dive, 2]): round(fs * T[dive, 1])

    # total duration of the descet phase (s)
    G_ratio[dive, 0] = len(kkdes) / fs

    # total glide duration during the descet phase (s)
    G_ratio[dive, 1] = len(numpy.where(SGtype[kkdes] == 0)) / fs

    # glide ratio during the descet phase
    G_ratio[dive, 2] = G_ratio[dive, 1] / G_ratio[dive, 0]

    # mean pitch during the descet phase(degrees)
    G_ratio[dive, 3] = numpy.mean(smoothpitch[kkdes] * 180 / pi)

    # descent rate (m/s)
    G_ratio[dive, 4] = Bottom[dive, 1] / G_ratio[dive, 0]

    # total duration of the ascet phase (s)
    G_ratio[dive, 5] = len(kkas) / fs

    # total glide duration during the ascet phase (s)
    G_ratio[dive, 6] = len(numpy.where(SGtype[kkas] == 0)) / fs

    # glide ratio during the ascet phase
    G_ratio[dive, 7] = G_ratio[dive, 6] / G_ratio[dive, 5]

    # mean pitch during the ascet phase(degrees)
    G_ratio[dive, 8] = numpy.mean(smoothpitch[kkas] * 180 / pi)

    # ascent rate (m/s)
    G_ratio[dive, 9] = Bottom[dive, 2] / G_ratio[dive, 5]
