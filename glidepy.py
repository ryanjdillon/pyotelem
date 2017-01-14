#!/usr/bin/env python3

'''
Body density estimation. Japan workshop May 2016

Python implementation:  Ryan J. Dillon
Original Matlab Author: Lucia Martina Martin Lopez

Attributes
----------
data: pandas.DataFrame
    contains calibrated sensor data and data processed within glide analysis
exp_ind: numpy.ndarray
    indices of data to be analyzed
dives: pandas.DataFrame
    start_idx
    stop_idx
    dive_dur
    depths_max
    depths_max_idx
    depths_mean
    compr_mean
cutoff: float
    cutoff frequency for separating low and high frequency signals
stroke_f: float
J:
    frequency of stroke signal in accelerometer data (m/s2)
t_max:
GL: ndarray
    start/stop indices of glide events in data
SGL:
    start/stop indices of subglide events in data
sgls: pandas.DataFrame
  Contains summary information for each subglide in data analyzed
glide_ratio: pandas.DataFrame
  Contains summary information of glide ratios for glides analyzed
tmax: int
    maximum duration allowable for a fluke stroke in seconds, it can be set as
    1/`stroke_f`
J:
    magnitude threshold for detecting a fluke stroke in [m/s2]
'''

# TODO sort out normalization and glide stats: get_stroke_freq()>automatic_freq()
# TODO put all masks into `data_masks` dataframe for simplicity
# TODO move unessecary plotting to own routine for switching on/off

import os
import click

@click.command(help='Calculate dive statistics for body condition predictions')

@click.option('--iopaths-filename', prompt=True, default='./iopaths.yaml',
              help='Path to iopaths.yaml')

@click.option('--debug', prompt=True, default=True, type=bool,
              help='Return on debug output')

def run_all(iopaths_filename, debug=False):
    from rjdtools import yaml_tools

    paths = yaml_tools.read_yaml(iopaths_filename)

    root_path  = paths['root']
    acc_path   = paths['acc']
    glide_path = paths['glide']

    # Use debug defaults from iopaths if debug arg `True`
    if debug is True:
        exp_path       = paths['debug']['exp_path']
        acc_cal_path   = paths['debug']['acc_cal_path']
        glide_cfg_path = paths['debug']['glide_cfg_path']

        print ('Debug activated, proccessing: {}'.format(exp_path))

        cfg, data, sgls, dives, glide_ratio = lleo_glide_analysis(root_path,
                                                                  acc_path,
                                                                  glide_path,
                                                                  exp_path,
                                                                  acc_cal_path,
                                                                  glide_cfg_path,
                                                                  plots=False,
                                                                  debug=True)
        return cfg, data, sgls, dives, glide_ratio

    # Process all experiments in accelerometer data path
    elif debug is False:

        # Process all experiments in `acc_path`
        for exp_path in os.listdir(os.path.join(root_path, acc_path)):

            # Only process directories
            if os.path.isdir(os.path.join(root_path, acc_path, exp_path)):

                print(exp_path)

                # Get correct calibration path given tag ID number
                acc_cal_path = paths['acc_cal'][int(exp_path.split('_')[2])]

                # Currently creating a new configuration for each exp
                glide_cfg_path = exp_path

                print('Processing: {}'.format(exp_path))

                # Run glide analysis
                lleo_glide_analysis(root_path, acc_path, glide_path, exp_path,
                                    acc_cal_path, glide_cfg_path)
        return None


def lleo_glide_analysis(root_path, acc_path, glide_path, exp_path,
        acc_cal_path, glide_cfg_path, plots=True, debug=False):
    '''Run glide analysis with little leonarda data'''
    import numpy
    import os

    # TODO copy yaml_tools to bodycondition, remove external import
    from rjdtools import yaml_tools
    import logger
    import utils_data
    import utils_plot

    # Input filenames
    cal_fname = 'cal.yaml'

    # Output filenames
    data_fname        = 'pydata_{}.p'.format(exp_path)
    cfg_fname         = 'glide_config.yaml'
    dives_fname       = 'glide_dives.p'
    sgl_fname         = 'glide_sgls.p'
    glide_ratio_fname = 'glide_ratios.p'
    sgls_mask_fname   = 'glide_sgls_mask.npy'
    data_masks_fname  = 'glide_data_masks.p'

    # Input and output directories
    acc_data_path = os.path.join(root_path, acc_path, exp_path)
    cal_yaml_path = os.path.join(root_path, acc_path, acc_cal_path, cal_fname)
    cfg_yaml_path = os.path.join(root_path, glide_path, glide_cfg_path, cfg_fname)
    out_path      = os.path.join(root_path, glide_path, exp_path)

    # Make out path if it does not exist
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # Create logger object for displaying/writing progress of current routine
    log = logger.Log(out_path, term_on=True, write_on=True)
    log.new_entry('Start little leonardo body condition analysis')

    # Load saved or default configuration parameters
    log.new_entry('Load configuration file')

    cfg = load_glide_config(cfg_yaml_path)
    log.new_entry('config file: {}'.format(os.path.split(cfg_yaml_path)[1]))

    # Load data - linearly interpolated sensors to accelerometer sensor
    data, dt_a, fs_a = load_lleo(acc_data_path, cal_yaml_path, min_depth=2)

    # Signal process data, calculate derivative data and find stroke freqencies
    cfg, data, dives, masks = process_sensor_data(log, cfg, data, fs_a,
                                                  Mw=None, plots=plots,
                                                  debug=debug)

    # Find glides, split into subglides, generate summary tables
    sgls, glide_ratio = process_glides(log, cfg, data, fs_a, dives, masks,
                                       plots=plots, debug=debug)

    # Filter data from config defaults
    pitch_lf_deg = numpy.rad2deg(data['p_lf'].values)

    exp_ind = numpy.where(masks['exp'])[0]
    masks['sgl'], sgls_mask = utils_data.filter_sgls(len(data),
                                                     exp_ind,
                                                     sgls,
                                                     cfg['pitch_thresh'],
                                                     cfg['min_depth'],
                                                     cfg['max_depth_delta'],
                                                     cfg['min_speed'],
                                                     cfg['max_speed'],
                                                     cfg['max_speed_delta'])

    # Plot filtered data
    utils_plot.plot_sgls(data['depth'].values, masks['sgl'], sgls, sgls_mask,
                         data['p_lf'].values, data['r_lf'].values,
                         data['h_lf'].values)

    # Save parameter configuration to YAML files
    log.new_entry('Save output')
    save_config(cfg, cfg_yaml_path)

    # Save data
    data.to_pickle(os.path.join(root_path, acc_path, exp_path, data_fname))
    sgls.to_pickle(os.path.join(out_path, sgl_fname))
    dives.to_pickle(os.path.join(out_path, dives_fname))
    glide_ratio.to_pickle(os.path.join(out_path,glide_ratio_fname))

    # Save masks
    masks.to_pickle(os.path.join(out_path, data_masks_fname))
    numpy.save(os.path.join(out_path, sgls_mask_fname), sgls_mask)

    return cfg, data, sgls, dives, glide_ratio


def process_sensor_data(log, cfg, data, fs_a, Mw=None, plots=True, debug=False):
    '''Calculate body conditions summary statistics'''
    import numpy
    import pandas

    import utils
    import utils_dives
    import utils_glides
    import utils_plot
    import utils_prh
    import utils_seawater
    import utils_signal


    # 1 Select indices for analysis
    #--------------------------------------------------------------------------
    log.new_entry('Select indices for analysis')

    if plots is True:
        # Plot accelerometer axes, depths, and propeller speed
        utils_plot.plot_triaxial_depths_speed(data)

    if debug is False:
        # Get indices user input - mask
        cfg['start_idx'] = utils.recursive_input('Analysis start index', int)
        cfg['stop_idx']  = utils.recursive_input('Analysis stop index', int)
    else:
        cfg['start_idx'] = 185231
        cfg['stop_idx'] = 446001

    # Creat dataframe for storing masks for various views of the data
    masks = pandas.DataFrame(index=range(len(data)), dtype=bool)

    # Create mask of values to be considered part of the analysis
    masks['exp'] = False
    masks['exp'][cfg['start_idx']:cfg['stop_idx']] = True

    # Create indices array `exp_ind` for analysis
    exp_ind = numpy.where(masks['exp'])[0]


    # 1.3 Calculate pitch, roll, and heading
    #--------------------------------------------------------------------------
    log.new_entry('Calculate pitch, roll, heading')
    data['p'], data['r'], data['h'] = utils_prh.calc_PRH(data['Ax_g'].values,
                                                         data['Ay_g'].values,
                                                         data['Az_g'].values)

    # 2 Define dives
    #--------------------------------------------------------------------------
    # TODO use min_dive_depth and min_analysis_depth?
    log.new_entry('Define dives')
    dives, masks['dive'] = utils_dives.finddives2(data['depth'].values,
                                              cfg['min_depth'])


    # 3.2.1 Determine `stroke_f` fluking rate and cut-off frequency
    #--------------------------------------------------------------------------
    log.new_entry('Get stroke frequency')
    # calculate power spectrum of the accelerometer data at the whale frame
    Ax_g = data['Ax_g'][masks['exp']].values
    Az_g = data['Az_g'][masks['exp']].values

    if debug is False:
        # TODO move to signal processing?
        cutoff, stroke_f, f = utils_glides.get_stroke_freq(Ax_g, Az_g, fs_a,
                                                           cfg['f'],
                                                           cfg['nperseg'],
                                                           cfg['peak_thresh'])
        # Store user input cutoff and stroke frequencies
        cfg['cutoff']   = cutoff
        cfg['stroke_f'] = stroke_f
        cfg['f']        = f

        # Calculate maximum duration of glides from stroke frequency
        cfg['tmax']  = 1 /cfg['stroke_f']  # seconds
    else:
        cutoff = 0.3
        cfg['cutoff'] = cutoff


    # 3.2.2 Separate low and high frequency signals
    #--------------------------------------------------------------------------
    log.new_entry('Separate accelerometry to high and low-pass signals')
    order = 5
    for btype, suffix in zip(['low', 'high'], ['lf', 'hf']):
        b, a, = utils_signal.butter_filter(cfg['cutoff'], fs_a, order=order,
                btype=btype)
        for param in ['Ax_g', 'Ay_g', 'Az_g']:
            key = '{}_{}'.format(param, suffix)
            data[key] = utils_signal.butter_apply(b, a, data[param].values)

    if plots is True:
        utils_plot.plot_lf_hf(data['Ax_g'][masks['exp']],
                              data['Ax_g_lf'][masks['exp']],
                              data['Ax_g_hf'][masks['exp']],
                              title='x axis')

        utils_plot.plot_lf_hf(data['Ay_g'][masks['exp']],
                              data['Ay_g_lf'][masks['exp']],
                              data['Ay_g_hf'][masks['exp']],
                              title='y axis')

        utils_plot.plot_lf_hf(data['Az_g'][masks['exp']],
                              data['Az_g_lf'][masks['exp']],
                              data['Az_g_hf'][masks['exp']],
                              title='z axis')


    # 3.2.3 Calculate the smooth pitch from the low pass filter acceleration
    #       signal to avoid incorporating signals above the stroking periods
    #--------------------------------------------------------------------------
    log.new_entry('Calculate low-pass pitch, roll, heading')
    prh_lf = utils_prh.calc_PRH(data['Ax_g_lf'].values, data['Ay_g_lf'].values,
                                data['Az_g_lf'].values,)

    data['p_lf'], data['r_lf'], data['h_lf'] = prh_lf


    # 4 Define precise descent and ascent phases
    #--------------------------------------------------------------------------
    log.new_entry('Get precise indices of descents, ascents, phase and bottom')
    masks['des'], masks['asc'] = utils_dives.get_des_asc2(data['depth'].values,
                                                  masks['dive'].values,
                                                  data['p_lf'].values,
                                                  cfg['cutoff'],
                                                  fs_a,
                                                  order=5)

    # Typecast des/asc columns to `bool` TODO better solution  later
    masks = masks.astype(bool)

    if plots is True:
        # plot all descents and all ascent phases
        utils_plot.plot_triaxial_descent_ascent(data['Ax_g'][masks['exp']],
                                                data['Az_g'][masks['exp']],
                                                masks['des'][masks['exp']],
                                                masks['asc'][masks['exp']])

        utils_plot.plot_dives_pitch(data['depth'][masks['exp']],
                                    masks['dive'][masks['exp']],
                                    masks['des'][masks['exp']],
                                    masks['asc'][masks['exp']],
                                    data['p'][masks['exp']],
                                    data['p_lf'][masks['exp']])


    # 5 Estimate swim speed
    #--------------------------------------------------------------------------
    log.new_entry('Estimate swim speed')

    #data['speed'] = utils_signal.estimate_speed(data['depth'].values,
    #                                            data['p_lf'].values,
    #                                            fs_a,
    #                                            fill_nans=True)
    # TODO config file?
    zero_level = 0.1
    theoretic_max = None #m/s
    data['speed'] = utils_prh.speed_from_acc_and_ref(data['Ax_g_lf'].values,
                                                     fs_a,
                                                     data['propeller'].values,
                                                     zero_level,
                                                     theoretic_max,
                                                     rm_neg=True)

    if plots is True:
        utils_plot.plot_swim_speed(exp_ind, data['speed'][masks['exp']].values)


    # 8 Estimate seawater desnity around the tagged animal
    #--------------------------------------------------------------------------
    log.new_entry('Estimate seawater density')

    salinity = 33.75
    salinities = numpy.asarray([salinity]*len(data))
    data['dsw'] = utils_seawater.sw_dens0(salinities, data['temperature'].values)


    # TODO split to own routine here

    # 6.1 Extract strokes and glides using heave
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

    if debug is False:
        # Plot PSD for J selection
        Ax_g_hf = data['Ax_g_hf'][masks['exp']].values
        Az_g_hf = data['Az_g_hf'][masks['exp']].values

        f_wx, Sx, Px, dpx = utils_signal.calc_PSD_welch(Ax_g_hf, fs_a, nperseg=512)
        f_wz, Sz, Pz, dpz = utils_signal.calc_PSD_welch(Az_g_hf, fs_a, nperseg=512)

        import matplotlib.pyplot as plt
        plt.plot(f_wx, Sx, label='hf-x PSD')
        plt.plot(f_wz, Sz, label='hf-z PSD')
        plt.legend(loc='upper right')
        plt.show()

        # TODO specify on which access the jerk is recorded/used
        # Get user selection for J
        cfg['J'] = utils.recursive_input('J (fluke magnitude)', float)
    else:
        cfg['J'] = 0.4

    return cfg, data, dives, masks


def process_glides(log, cfg, data, fs_a, dives, masks, Mw=None, plots=True,
        debug= False):
    import numpy

    import utils
    import utils_glides

    if Mw is None:
        # Get GL from dorso-ventral axis of the HPF acc signal
        GL = utils_glides.get_stroke_glide_indices(data['Az_g_hf'].values,
                                                   fs_a,
                                                   cfg['J'],
                                                   cfg['t_max'])
    elif Mw is not None:
        MagAcc, pry, Sa, GL, heading_lf, pitch_lf_deg = calc_mag_heading()

    # Get analysis indices from mask
    exp_ind = numpy.where(masks['exp'])[0]

    # check glides duration and positive and negative zero crossings based
    # on selected J and tmax#
    glide_mask = utils.mask_from_noncontiguous_indices(len(data), GL[:,0], GL[:,1])
    glide_ind = numpy.where(glide_mask)[0]


    # 7 Make 5sec sub-glides
    #--------------------------------------------------------------------------
    log.new_entry('Make sub-glides, duration {}'.format(cfg['dur']))

    SGL, sgl_mask = utils_glides.split_glides(len(data), cfg['dur'], fs_a, GL)


    # 9 Generate summary information table for subglides
    #--------------------------------------------------------------------------
    pitch_lf_deg = numpy.rad2deg(data['p_lf'].values)

    log.new_entry('Generate summary information table for subglides')
    sgls = utils_glides.calc_glide_des_asc(data['depth'].values,
                                             data['p_lf'].values,
                                             data['r_lf'].values,
                                             data['h_lf'].values,
                                             data['speed'].values,
                                             dives,
                                             SGL,
                                             pitch_lf_deg,
                                             data['temperature'].values,
                                             data['dsw'].values)

    # 10 Calculate glide ratio
    #--------------------------------------------------------------------------
    log.new_entry('Calculate sgl ratio')
    glide_ratio = utils_glides.calc_glide_ratios(dives, masks['des'].values,
                                                 masks['asc'].values,
                                                 glide_mask, data['depth'],
                                                 data['p_lf'])
    return sgls, glide_ratio


def load_glide_config(cfg_yaml_path):
    '''Load analysis configuration defualts'''
    from collections import OrderedDict
    import datetime
    import os
    import numpy

    from rjdtools import yaml_tools

    #if os.path.isfile(cfg_yaml_path):
    #    cfg = yaml_tools.read_yaml(cfg_yaml_path)

    #else:
    cfg = OrderedDict()

    # Record the last date modified & git version
    fmt = '%Y-%m-%d_%H%M%S'
    cfg['last_modified'] = datetime.datetime.now().strftime(fmt)

    #TODO add git version & other experiment info

    # Acceleromter/Magnotometer axis to analyze
    cfg['n'] = 0#1

    # Minimum depth at which to recognize a dive (2. Define dives)
    cfg['min_depth'] = 0.4

    # Maximum cummulative change in depth over a glide
    cfg['max_depth_delta'] = 8.0

    # Minimum mean speed of sublide
    cfg['min_speed'] = 0.3

    # Maximum mean speed of sublide
    cfg['max_speed'] = 10

    # Maximum cummulative change in speed over a glide
    cfg['max_speed_delta'] = 1.0

    # Number of samples per frequency segment in PSD calculation
    cfg['nperseg'] = 256

    # Minimumn power of frequency for identifying stroke_f (3. Get stroke_f)
    cfg['peak_thresh'] = 0.10

    # Frequency of stroking, determinded from PSD plot
    cfg['stroke_f'] = 0.4 # Hz

    # fraction of stroke_f to calculate cutoff frequency, Wn
    cfg['f'] = 0.4

    # Duration of sub-glides (8. Split sub-glides, 10. Calc glide des/asc)
    cfg['dur'] = 5 # seconds

    # Minimum duration of sub-glides, `False` excludes sublides < dur seconds
    cfg['min_dur'] = False # seconds

    # Threshold frequency power for identifying the stroke frequency
    cfg['J'] = 2 / 180 * numpy.pi

    # Maximum length of stroke signal
    cfg['t_max'] = 1 / cfg['stroke_f'] # seconds

    # Pitch angle (degrees) to consider sgls
    cfg['pitch_thresh'] = 30

    # For magnetic pry routine
    cfg['alpha'] = 25

    # Save default config to file
    save_config(cfg, cfg_yaml_path)

    return cfg


def save_config(cfg, cfg_yaml_path):
    '''Update value of config dictionary'''
    from rjdtools import yaml_tools

    yaml_tools.write_yaml(cfg, cfg_yaml_path)

    return None


def load_lleo(acc_data_path, cal_yaml_path, min_depth):
    '''Load lleo data for calculating body condition'''
    import numpy
    import os

    from pylleo.pylleo import lleoio, lleocal

    # Parse tag model and id from directory/experiment name
    experiment_id = os.path.split(acc_data_path)[1].replace('-','')
    tag_model = experiment_id.split('_')[1]
    tag_id = int(experiment_id.split('_')[2])

    sample_f  = 1

    # Load calibrate data
    cal_dict = lleocal.read_cal(cal_yaml_path)

    # Verify sensor ID of data matches ID of CAL
    print('exp', tag_model, tag_id)
    print('cal', cal_dict['tag_model'], cal_dict['tag_id'])
    if cal_dict['tag_id'] != tag_id:
        raise SystemError('Data `tag_id` does not match calibration `tag_id`')

    # Load meta data
    meta = lleoio.read_meta(acc_data_path, tag_model, tag_id)

    # Lod data
    data = lleoio.read_data(meta, acc_data_path, sample_f)

    # Linearly interpolate data
    data.interpolate('linear', inplace=True)

    # Apply calibration to data
    data['Ax_g'] = lleocal.apply_poly(data, cal_dict, 'acceleration_x')
    data['Ay_g'] = lleocal.apply_poly(data, cal_dict, 'acceleration_y')
    data['Az_g'] = lleocal.apply_poly(data, cal_dict, 'acceleration_z')

    # Get original sampling rates of accelerometer and depth sensors
    dt_a = float(meta['parameters']['acceleration_x']['Interval(Sec)'])
    fs_a = 1/dt_a

    return data, dt_a, fs_a#A_g, depths, speed, temperature,


if __name__ == '__main__':

    run_all()
