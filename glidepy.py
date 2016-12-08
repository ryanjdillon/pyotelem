#!/usr/bin/env python3
'''
Body density estimation. Japan workshop May 2016

Python implementation:  Ryan J. Dillon
Original Matlab Author: Lucia Martina Martin Lopez

Attributes
----------
dive_ind: numpy.ndarray, shape (n, 6)
    start_idx
    end_idx
    max_depth
    max_depth_idx
    mean_depth
    mean_compression
D: numpy.ndarray, shape (n, 7)
    start_time(s)
    end_time(s)
    dive_duration(s)
    max_depth(s)
    max_depth(m)
    ID(n)
des: numpy.ndarray, 1D array
asc: numpy.ndarray, 1D array
phase:
bottom:
cutoff:
stroke_f: float
A_g_lf: numpy.ndarray
A_g_hf: numpy.ndarray
ind: numpy.ndarray
    data indices to be analyzed
J:
    m/s2
t_max:
GL: ndarray
    indices of glide events in data
pitch_lf:
roll_lf:
heading_lf:
nn:
    index position in dive_ind of last dive to use for analysis
swim_speed:
DSP:
Dsw:
sgl_ind:
pitch_lf_deg
glide:
gl_ratio:
tmax: int
    maximum duration allowable for a fluke stroke in seconds, it can be set as
    1/`stroke_f`
J:
    magnitude threshold for detecting a fluke stroke in [m/s2]

NOTE variable name changes
--------------------------
`p`       renamed `depths`
`k`       renamed `ind`
`FLl`     renamed `cutoff_low`
`FLlmag`  renamed `cutoff_low_mag`
`FR`      renamed `stroke_f`
`FRn`     renamed `stroke_f_n`
`FL1`     renamed `cutoff_low`
            frequency at the negative peak in the power spectral density
            plot/`stroke_f`
`thdeg`   renamed `thresh_deg`
`SwimSp`  renamed `swim_speed`
`Glide`   renamed `glide`
`G_ratio` renamed `g_ratio`
`SGtype`  renamed `gl_mask`

'''
import click
import os

# TODO dump table output as .npy to experiment directory
#    * out_path param
# TODO save dive/nodive mask as .npy
# TODO sort out normalization and glide stats: get_stroke_freq()>automatic_freq()

# TODO finish implementation of `auto_select` in get_stroke_freq()

# TODO update dtag library, import from there instead of utils where applicable

# TODO rounding of index number should have thought out standard
#      http://stackoverflow.com/a/38228131/943773
# TODO account for Propeller/Speed data, Salinity (DTS)?

# TODO choose working indices, save to config


@click.command(help='Calculate dive statistics for body condition predictions')

@click.option('--data-root', prompt=True,
              default=lambda: os.environ.get('LLEO_DATA_ROOT',''),
              help='Parent directory with child directories for acceleration data files')
@click.option('--data-path', prompt=True,
              default=lambda: os.environ.get('LLEO_DATA_PATH',''),
              help='Child directory with data to be analyzed')
@click.option('--cal-path', prompt=True,
              default=lambda: os.environ.get('LLEO_CAL_PATH',''),
              help='Child directory with calibration YAML to be used')
@click.option('--analysis-root', prompt=True,
              default=lambda: os.environ.get('LLEO_ANALYSIS_ROOT',''),
              help='Parent directory with child directories for analysis output')
@click.option('--config-path', prompt=True,
              default=lambda: os.environ.get('LLEO_CONF_PATH',''),
              help='Child directory with analysis YAML to be used')
@click.option('--debug', default=True, type=bool, help='Return on debug output')


def run_lleo_glide_analysis(data_root, data_path, cal_path,
        analysis_root, config_path, debug=False):
    '''Run glide analysis with little leonarda data'''
    import os

    config_path = os.path.join(analysis_root, config_path)
    out_path    = os.path.join(analysis_root, data_path)

    # `load_lleo` linearly interpolated sensors to accelerometer
    A_g, depths, temperature, dt_a, fs_a = load_lleo(data_root, data_path,
                                                     cal_path, depth_thresh=2)

    glide, gl_ratio = glide_analysis(config_path, out_path, A_g, fs_a, depths,
                                     temperature, Mw=None,
                                     save=True, debug=False)

    return glide, gl_ratio


def glide_analysis(config_path, out_path, A_g, fs_a, depths, temperature,
        Mw=None, save=True, debug=False, show=True):
    '''Calculate body conditions summary statistics

    Args
    ----
    data_root: str
        path containing experiment data directories
    data_path: str
        data directory name
    cal_path: str
        data directory name containing cal.yaml file for accelerometer data
        calibration

    Returns
    -------
    GL
    '''
    import numpy

    import logger
    import utils
    import utils_dives
    import utils_glides
    import utils_plot
    import utils_prh
    import utils_signal
    import utils_seawater

    # Make out path if it does not exist
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    log = logger.Log(out_path, term_on=True, write_on=True)
    log.new_entry('Start body condition analysis')

    # 1 Analysis configuration and data
    #--------------------------------------------------------------------------
    # Load saved or default configuration parameters
    cfg, config_yaml_path = load_glide_config(config_path)
    log.new_entry('configuration file: {}'.format(config_yaml_path))

    log.new_entry('Calculate pitch, roll, heading')
    pitch, roll, heading = utils_prh.calc_PRH(A_g)

    # Data to analyze (first set `ind` to whole deployment
    ind = numpy.arange(len(depths))


    # 2 Define dives
    #--------------------------------------------------------------------------
    log.new_entry('Define dives')
    #dive_ind = utils_dives.finddives(depths, fs_a, cfg['min_dive_thresh'], surface=1,
    #                          findall=True)
    dive_ind, dive_mask = utils_dives.finddives2(depths, cfg['min_dive_thresh'])

    # make a summary table describing the characteristics of each dive.
    log.new_entry('Create dive summary table')
    D = utils_dives.create_dive_summary(dive_ind)


    # 3.1 Quick separation of descent and ascent phases
    #--------------------------------------------------------------------------
    # Get list of dives indices & direction, descents/ascents, phase/bottom indices
    log.new_entry('Get indices of descents, ascents, phase and bottom')
    des, asc = utils_dives.get_des_asc2(depths, dive_mask, pitch,
                                        cfg['cutoff'], fs_a, order=5)

    # plot all descents and all ascent phases
    utils_plot.plot_triaxial_descent_ascent(A_g, des, asc)


    # 3.2.1-2 Select periods of analysis
    #         Determine `stroke_f` fluking rate and cut-off frequency
    #--------------------------------------------------------------------------
    log.new_entry('Get stroke frequency')
    # calculate power spectrum of the accelerometer data at the whale frame
    cutoff, stroke_f, f = utils_glides.get_stroke_freq(A_g, fs_a,
                                                       cfg['f'],
                                                       cfg['nperseg'],
                                                       cfg['peak_thresh'])
    cfg['cutoff']   = cutoff
    cfg['stroke_f'] = stroke_f
    cfg['f']        = f

    # Calculate maximum duration of glides from stroke frequency
    cfg['tmax']  = 1 /cfg['stroke_f']  # seconds


    # 3.2.3 Separate low and high frequency signals
    #--------------------------------------------------------------------------
    log.new_entry('Separate accelerometry to high and low-pass signals')
    A_g_lf, A_g_hf = utils_signal.filter_accelerometer(A_g, fs_a,
                                                       cfg['cutoff'], order=5)

    utils_plot.plot_lf_hf(A_g[:,0], A_g_lf[:,0], A_g_hf[:,0], title='x axis')
    utils_plot.plot_lf_hf(A_g[:,1], A_g_lf[:,1], A_g_hf[:,1], title='y axis')
    utils_plot.plot_lf_hf(A_g[:,2], A_g_lf[:,2], A_g_hf[:,2], title='z axis')


    # 3.2.4 Calculate the smooth pitch from the low pass filter acceleration
    #       signal to avoid incorporating signals above the stroking periods
    #--------------------------------------------------------------------------
    log.new_entry('Calculate low-pass pitch, roll, heading')
    pitch_lf, roll_lf, heading_lf = utils_prh.calc_PRH(A_g_lf)


    # 4 Define precise descent and ascent phases
    #--------------------------------------------------------------------------
    # TODO remove pitch_roll plot?
    #cfg['nn'] = utils_dives.select_last_dive(depths, pitch, pitch_lf, dive_ind, fs_a)

    log.new_entry('Get precise indices of descents, ascents, phase and bottom')
    des, asc = utils_dives.get_des_asc2(depths, dive_mask, pitch_lf,
                                        cfg['cutoff'], fs_a, order=5)

    phase  = utils_dives.get_phase(depths, des, asc)
    bottom = utils_dives.get_bottom(depths, des, asc)

    utils_plot.plot_dives_pitch(depths, dive_mask, des, asc, pitch, pitch_lf)


    # 5 Estimate swim speed
    #--------------------------------------------------------------------------
    log.new_entry('Estimate swim speed')
    # TODO interpolate speed
    swim_speed = utils_signal.inst_speed(depths,
                                         pitch_lf,
                                         fs_a,
                                         cfg['stroke_f'],
                                         cfg['f'],
                                         ind,
                                         cfg['thresh_deg'])

    utils_plot.plot_swim_speed(swim_speed, ind)


    # 6 Estimate seawater desnity around the tagged animal
    #--------------------------------------------------------------------------
    log.new_entry('Estimate seawater density - NOTE: CURRENTLY RANDOM')
    Dsw = numpy.random.random(len(depths)) + 32.0
    #Dsw = utils_seawater.estimate_seawater_desnity(DTS, D)


    # 7.1 Extract strokes and glides using heave
    #     high-pass filtered (HPF) acceleration signal, n=3
    #--------------------------------------------------------------------------
    # Two methods for estimating stroke frequency `stroke_f`:
    # * from the body rotations (pry) using the magnetometer method
    # * from the dorso-ventral axis of the HPF acceleration signal.

    # For both methods, tmax and J need to be determined.

    # Choose a value for J based on a plot showing distribution of signals:
    #   hpf-x, when detecting glides in the next step use Ahf_Anlf() with n=1
    #   hpf-z when detecting glides in the next step use Ahf_Anlf() with n=3

    log.new_entry('Get fluke signal threshold')
    utils_plot.plot_hf_acc_histo(A_g_hf, fs_a, cfg['stroke_f'], des, asc)

    # Get user selection for J
    cfg['J'] = utils.recursive_input('J (fluke magnitude)', float)

    if Mw == None:
        # Get GL from dorso-ventral axis of the HPF acc signal
        GL = utils_glides.get_stroke_glide_indices(A_g_hf[:,cfg['n']],
                                                       fs_a,
                                                       cfg['n'],
                                                       cfg['J'],
                                                       cfg['t_max'])
    else:
        # Placeholder for magnetometer pry method implementation
        pass


    # check glides duration and positive and negative zero crossings based
    # on selected J and tmax#
    gl_mask = utils_glides.get_gl_mask(depths, fs_a, GL)

    # TODO calculate pry
    #utils_plot.plot_depth_at_glides(depths, sgl_dur, pry, fs_a, t)


    # 8 Make 5sec sub-glides
    #--------------------------------------------------------------------------
    log.new_entry('Make sub-glides, duration {}'.format(cfg['dur']))
    # TODO include this somewhere
    n_samples = len(depths)
    sgl_ind, sgl_mask = utils_glides.split_glides(n_samples, cfg['dur'], fs_a, GL)

    utils_plot.plot_glide_depths(depths, sgl_mask)


    # 9 create summary table required for body density
    #--------------------------------------------------------------------------
    log.new_entry('Create summary table for body density')
    if Mw:
        MagAcc, pry, Sa, GL, heading_lf, pitch_lf_deg = calc_mag_heading()
    # TODO check if correct
    else:
        pitch_lf_deg = numpy.rad2deg(pitch_lf)


    # 10 Calculate glide descent and ascent
    #--------------------------------------------------------------------------
    log.new_entry('Calculate glide descent and ascent')
    glide = utils_glides.calc_glide_des_asc(depths, fs_a, pitch_lf, roll_lf,
                                            heading_lf, swim_speed, D, phase,
                                            sgl_ind, pitch_lf_deg,
                                            temperature, Dsw)
    numpy.save(os.path.join(out_path, 'glide.npy'), glide)


    # 11 Calculate glide ratio
    #--------------------------------------------------------------------------
    log.new_entry('Calculate glide ratio')
    gl_ratio = utils_glides.calc_glide_ratios(dive_ind, des, asc, gl_mask,
                                              depths, pitch_lf)
    numpy.save(os.path.join(out_path, 'gl_ratio.npy'), glide)


    # SAVE CONFIG
    #--------------------------------------------------------------------------
    # Save parameter configuration to YAML files
    log.new_entry('Save config YAML file')
    save_config(cfg, config_yaml_path)

    return sgl_ind, glide, gl_ratio


def load_glide_config(config_path):
    '''Load analysis configuration defualts'''
    from collections import OrderedDict
    import datetime
    import os
    import numpy

    from rjdtools import yaml_tools

    config_yaml_path = os.path.join(config_path, 'glide_analysis_config.yaml')

    if os.path.isfile(config_yaml_path):
        cfg = yaml_tools.read_yaml(config_yaml_path)

    else:
        cfg = OrderedDict()

        # Record the last date modified & git version
        fmt = '%Y-%m-%d_%H%M%S'
        cfg['last_modified'] = datetime.datetime.now().strftime(fmt)
        #TODO add git version & other experiment info

        # Acceleromter/Magnotometer axis to analyze
        cfg['n'] = 1

        # Index position in dive_ind of last dive to use for analysis
        cfg['nn'] = -1

        # TODO dive threshold to identify what is active data from the tag
        cfg['depth_thresh'] = 2

        # Minimum depth at which to recognize a dive (2. Define dives)
        cfg['min_dive_thresh'] = 0.5

        # Number of samples per frequency segment in PSD calculation
        cfg['nperseg'] = 512

        # Minimumn power of frequency for identifying stroke_f (3. Get stroke_f)
        cfg['peak_thresh'] = 0.10

        # Frequency of stroking, determinded from PSD plot
        cfg['stroke_f'] = 0.4 # Hz

        # fraction of stroke_f to calculate cutoff frequency, Wn
        cfg['f'] = 0.4

        # Degree threshold above which speed can be estimated (5. Estimate speed)
        cfg['thresh_deg'] = 30

        # Duration of sub-glides (8. Split sub-glides, 10. Calc glide des/asc)
        cfg['dur'] = 5 # seconds

        # Minimum duration of sub-glides, `False` excludes sublides < dur seconds
        cfg['min_dur'] = False # seconds

        # Threshold frequency power for identifying the stroke frequency
        cfg['J'] = 2 / 180 * numpy.pi

        # Maximum length of stroke signal
        cfg['t_max'] = 1 / cfg['stroke_f'] # seconds

        # For magnetic pry routine
        cfg['alpha'] = 25

        # Save default config to file
        save_config(cfg, config_yaml_path)

    return cfg, config_yaml_path


def save_config(cfg, config_yaml_path):
    '''Update value of config dictionary'''
    from rjdtools import yaml_tools

    yaml_tools.write_yaml(cfg, config_yaml_path)

    return None


def update_config(cfg, key, value):
    cfg[key] = value
    save_config(cfg, config_yaml_path)
    return cfg


def load_lleo(data_root, data_path, cal_path, depth_thresh):
    '''Load lleo data for calculating body condition'''
    import numpy
    import os

    from pylleo.pylleo import lleoio, lleocal

    data_path = os.path.join(data_root, data_path)
    cal_yaml_path = os.path.join(data_root, cal_path, 'cal.yaml')

    sample_f  = 1

    # TODO auto set tag model, tag_id
    tag_model = 'W190PD3GT'
    tag_id    = '34839'

    # TODO verify sensor ID of data matches ID of CAL

    meta = lleoio.read_meta(data_path, tag_model, tag_id)
    data = lleoio.read_data(meta, data_path, sample_f)
    # TODO take note of linear interpolation in paper
    data.interpolate('linear', inplace=True)

    # Make index mask for values greater than `depth_thresh`
    # i.e. filter out data where tag/animal is not actively deployed
    start_idx = numpy.where(data['depth'] > depth_thresh)[0][0]
    end_idx = numpy.where(data['depth'] > depth_thresh)[0][-1]
    ind = (data['datetimes'] > data['datetimes'][start_idx]) & \
          (data['datetimes'] < data['datetimes'][end_idx])

    # Load and calibrate data acc data
    cal_dict = lleocal.read_cal(cal_yaml_path)
    Ax_g = lleocal.apply_poly(data[ind], cal_dict, 'acceleration_x')
    Ay_g = lleocal.apply_poly(data[ind], cal_dict, 'acceleration_y')
    Az_g = lleocal.apply_poly(data[ind], cal_dict, 'acceleration_z')

    A_g = numpy.vstack([Ax_g, Ay_g, Az_g]).T

    # Get original sampling rates of accelerometer and depth sensors
    dt_a = float(meta['parameters']['acceleration_x']['Interval(Sec)'])
    fs_a = 1/dt_a

    # With interpolation of other sensors above dt_a == dt_d
    #dt_d = float(meta['parameters']['depth']['Interval(Sec)'])
    #fs_d = 1/dt_d

    # Alternate method
    #dt = data['datetimes'][1] - data['datetimes'][0]
    #fs_a = 1/(dt.microseconds/1e6)

    depths = data['depth'][ind].values
    temperature = data['depth'][ind].values

    return A_g, depths, temperature, dt_a, fs_a


def make_paths():
    import os

    data_root = os.environ.get('LLEO_DATA_ROOT','')
    data_path = os.environ.get('LLEO_DATA_PATH','')
    cal_path = os.environ.get('LLEO_CAL_PATH','')
    analysis_root = os.environ.get('LLEO_ANALYSIS_ROOT','')
    config_path = os.environ.get('LLEO_CONF_PATH','')

    data_path = os.path.join(data_root, data_path)
    cal_path = os.path.join(data_root, cal_path)
    config_path = os.path.join(analysis_root, config_path)
    out_path = os.path.join(analysis_root, data_path)

    return data_path, cal_path, out_path, conf_path


#def calc_mag_heading():
#    '''Section 9'''
#    stroke_f = 0.4
#    f        = 0.4
#    alpha    = 25
#    n        = 1
#    ind      = range(len(depths))
#    J        = 2 / 180 * numpy.pi
#    tmax     = 1 / stroke_f
#
#    MagAcc, pry, Sa, GL = magnet_rot_sa(Aw, Mw, fs_a, stroke_f, f, alpha, n,
#                                            ind, J, tmax)
#
#    heading_lf = m2h(MagAcc.Mnlf[ind*fs_a, :], pitch_lf, roll_lf)
#    pitch_lf_deg = pitch_lf * 180 / numpy.pi
#
#    return MagAcc, pry, Sa, GL, heading_lf, pitch_lf_deg


if __name__ == '__main__':
    glide, gl_ratio = run_lleo_glide_analysis()
