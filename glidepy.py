#!/usr/bin/env python3

'''
Body density estimation. Japan workshop May 2016

Python implementation:  Ryan J. Dillon
Original Matlab Author: Lucia Martina Martin Lopez

Attributes
----------
sensors: pandas.DataFrame
    contains calibrated sensor data and data processed within glide analysis
exp_ind: numpy.ndarray
    indices of `sensors` data to be analyzed
dives: pandas.DataFrame
    start_idx
    stop_idx
    dive_dur
    depths_max
    depths_max_idx
    depths_mean
    compr_mean
cutoff_frq: float
    cutoff frequency for separating low and high frequency signals
stroke_frq: float
    frequency at which maximum power is seen in accelerometer PSD
J:
    frequency of stroke signal in accelerometer data (m/s2)
t_max:
    
GL: ndarray
    start/stop indices of glide events in `sensors` data
SGL:
    start/stop indices of subglide events in `sensors` data
sgls: pandas.DataFrame
  Contains subglide summary information of `sensors` data
glide_ratio: pandas.DataFrame
  Contains glide ratio summary information of `sensors` data
t_max: int
    maximum duration allowable for a fluke stroke in seconds, it can be set as
    1/`stroke_frq`
J:
    magnitude threshold for detecting a fluke stroke in [m/s2]
'''
# TODO glide identification performed on z-axis, change?

# TODO CLARIFY `stroke_frq` vs `fluke rate` low pass vs high-pass signals

# TODO add experiment info: # dives, # subglides asc/des in cfg_filter.yaml
# TODO look into t_max / t_max yaml
# TODO use name convention in rest of repo: fname_, path_, etc.
# TODO copy yaml_tools to bodycondition, remove external import
# TODO GL and dives ind saved in masks? have routine that calcs dive info
# TODO move config glide, sgl, filter to default yaml files?
# TODO get rid of dives and glide_ratios
# TODO cleanup docstring(s)
# TODO sort out normalization and glide stats: get_stroke_freq()>automatic_freq()
# TODO put all masks into `acc_masks` dataframe for simplicity
# TODO move unessecary plotting to own routine for switching on/off

import os
import click

@click.command(help='Calculate dive statistics for body condition predictions')

@click.option('--path-cfg-paths', prompt=True, default='./cfg_paths.yaml',
              help='Path to cfg_paths.yaml')

@click.option('--debug', prompt=True, default=False, type=bool,
              help='Return on debug output')

@click.option('--plots', prompt=True, default=True, type=bool,
              help='Plot diagnostic plots')

def run_all(path_cfg_paths, debug=False, plots=True):
    from rjdtools import yaml_tools
    import utils

    cfg_paths = yaml_tools.read_yaml(path_cfg_paths)

    path_root  = cfg_paths['root']
    path_acc   = cfg_paths['acc']
    path_glide = cfg_paths['glide']

    # Use debug defaults
    if debug is True:

        path_exp       = cfg_paths['acc_cal'][34839]
        path_cal_acc   = cfg_paths['acc_cal'][34839]
        path_cfg_glide = cfg_paths['acc_cal'][34839]

        path_exps   = [path_exp,]
        process_ind = [0,]

    # Get user selection for data paths to process
    elif debug is False:

        path_exps = list()
        # Process all experiments in `path_acc`
        for path_exp in os.listdir(os.path.join(path_root, path_acc)):

            # Only process directories
            if os.path.isdir(os.path.join(path_root, path_acc, path_exp)):

                path_exps.append(path_exp)

        path_exps = sorted(path_exps)
        msg = 'Enter paths numbers to process:\n'
        process_ind = utils.get_dir_indices(msg, path_exps)

    # Process all experiments in accelerometer data path
    for i in process_ind:
        path_exp = path_exps[i]

        # Get correct calibration path given tag ID number
        tag_id = int(path_exp.split('_')[2])
        year   = int(path_exp[:4])
        month  = int(path_exp[4:6])
        path_cal_acc = cfg_paths['acc_cal'][tag_id][year][month]

        print('Tag calibration file path: {}'.format(path_cal_acc))

        # Currently creating a new configuration for each exp
        path_cfg_glide = path_exp

        print('Processing: {}'.format(path_exp))

        # Run glide analysis
        cfg_glide, sensors, sgls, dives, glide_ratio = lleo_glide_analysis(path_root,
                                                                  path_acc,
                                                                  path_glide,
                                                                  path_exp,
                                                                  path_cal_acc,
                                                                  plots=plots,
                                                                  debug=debug)
    return cfg_glide, sensors, sgls, dives, glide_ratio



def lleo_glide_analysis(path_root, path_acc, path_glide, path_exp,
        path_cal_acc, plots=True, debug=False):
    '''Run glide analysis with little leonarda data'''
    from collections import OrderedDict
    import numpy
    import os

    from rjdtools import yaml_tools
    import logger
    import utils
    import utils_smartmove
    import utils_plot

    # Input filenames
    fname_cal = 'cal.yaml'
    fname_cal_speed = 'speed_calibrations.csv'

    # Input paths
    path_data_acc  = os.path.join(path_root, path_acc, path_exp)
    file_cal_acc   = os.path.join(path_root, path_acc, path_cal_acc, fname_cal)
    file_cal_speed = os.path.join(path_root, path_acc, fname_cal_speed)
    # TODO review this implemenation later
    file_cfg_exp   = './cfg_experiments.yaml'

    # Output filenames
    fname_cfg_glide           = 'cfg_glide.yaml'
    fname_cfg_sgl             = 'cfg_sgl.yaml'
    fname_cfg_filt            = 'cfg_filter.yaml'

    fname_sensors             = 'pydata_{}.p'.format(path_exp)
    fname_dives               = 'data_dives.p'
    fname_sgls                = 'data_sgls.p'
    fname_glide_ratio         = 'data_ratios.p'
    fname_mask_sensors        = 'mask_sensors.p'
    fname_mask_sensors_glides = 'mask_sensors_glides.p'
    fname_mask_sensors_sgls   = 'mask_sensors_sgls.p'
    fname_mask_sensors_filt   = 'mask_sensors_filt.p'
    fname_mask_sgls           = 'mask_sgls.npy'
    fname_mask_sgls_filt      = 'mask_sgls_filt.p'



    # Setup configuration files
    cfg_glide = cfg_glide_params()
    cfg_sgl   = cfg_sgl_params()
    cfg_filt  = cfg_filt_params()

    # Output paths
    ignore = ['nperseg', 'peak_thresh', 'alpha', 'min_depth', 't_max, t_max']
    out_data  = os.path.join(path_root, path_acc, path_exp)
    os.makedirs(out_data, exist_ok=True)




    # Create logger object for displaying/writing progress of current routine
    path_log = os.path.join(path_root, path_glide)
    log = logger.Log(path_log, path_exp, term_on=True, write_on=False)



    # LOAD DATA
    #----------

    # linearly interpolated sensors to accelerometer sensor
    sensors, dt_a, fs_a = load_lleo(path_data_acc, file_cal_acc,
                                    file_cal_speed, min_depth=2)

    # Signal process data, calculate derived data and find stroke freqencies
    cfg_glide, sensors, dives, masks, exp_ind = process_sensor_data(log,
                                                           path_exp,
                                                           cfg_glide,
                                                           file_cfg_exp,
                                                           sensors,
                                                           fs_a,
                                                           Mw=None,
                                                           plots=plots,
                                                           debug=debug)
    # Save data
    sensors.to_pickle(os.path.join(out_data, fname_sensors))
    dives.to_pickle(os.path.join(out_data, fname_dives))
    masks.to_pickle(os.path.join(out_data, fname_mask_sensors))


    # TODO better way to save meta data from data processing? split routine?



    # Find Glides
    #------------
    # Find glides
    GL, masks['glides'], glide_ratio = process_glides(log,
                                                      cfg_glide,
                                                      sensors,
                                                      fs_a,
                                                      dives,
                                                      masks,
                                                      plots=plots,
                                                      debug=debug)

    # Save glide ratio dataframe
    dname_glide = utils.cat_keyvalues(cfg_glide, ignore)
    out_glide = os.path.join(path_root, path_glide, path_exp, dname_glide)
    os.makedirs(out_glide, exist_ok=True)
    glide_ratio.to_pickle(os.path.join(out_glide, fname_glide_ratio))

    # Save glide mask of sensor dataframe
    masks['glides'].to_pickle(os.path.join(out_glide, fname_mask_sensors_glides))

    # Save glide analysis configuration
    path_cfg_yaml = os.path.join(out_glide, fname_cfg_glide)
    save_config(cfg_glide, path_cfg_yaml, 'glides')


    # SPLIT GLIDES
    #-------------

    # Split into subglides, generate summary tables
    sgls, masks['sgls'] = process_sgls(log, cfg_sgl, sensors, fs_a, GL, dives)

    # Save subglide dataframe
    dname_sgl   = utils.cat_keyvalues(cfg_sgl, ignore)
    out_sgl   = os.path.join(out_glide, dname_sgl)
    os.makedirs(out_sgl, exist_ok=True)
    sgls.to_pickle(os.path.join(out_sgl, fname_sgls))

    # Save subglide mask of sensor dataframe
    masks['sgls'].to_pickle(os.path.join(out_sgl, fname_mask_sensors_sgls))

    # Save subglide analysis configuation
    save_config(cfg_sgl, os.path.join(out_sgl, fname_cfg_sgl), 'sgls')



    # FILTER SUBGLIDES
    #-----------------
    # Include duration in filtering if splitting is fast enough?
    # Filter subglides
    exp_ind = numpy.where(masks['exp'])[0]
    masks['filt_sgls'], sgls['mask'] = utils_smartmove.filter_sgls(len(sensors),
                                                   exp_ind,
                                                   sgls,
                                                   cfg_filt['pitch_thresh'],
                                                   cfg_filt['min_depth'],
                                                   cfg_filt['max_depth_delta'],
                                                   cfg_filt['min_speed'],
                                                   cfg_filt['max_speed'],
                                                   cfg_filt['max_speed_delta'])

    # Plot filtered data
    utils_plot.plot_sgls(sensors['depth'].values, masks['filt_sgls'], sgls,
                         sgls['mask'], sensors['p_lf'].values,
                         sensors['r_lf'].values, sensors['h_lf'].values)

    # Save filtered subglide mask of sensor dataframe
    dname_filt  = utils.cat_keyvalues(cfg_filt, ignore)
    out_filt  = os.path.join(out_sgl, dname_filt)
    os.makedirs(out_filt, exist_ok=True)
    masks['filt_sgls'].to_pickle(os.path.join(out_filt, fname_mask_sensors_filt))

    # Save filtered subglide mask of sgl dataframe
    # TODO find better solution later only using mask on whole sgls array
    sgls['mask'].to_pickle(os.path.join(out_filt, fname_mask_sgls_filt))

    # Save symlink to data and masks in filter directory
    create_symlink(os.path.join(out_data, fname_sensors),
                   os.path.join(out_filt, fname_sensors))
    create_symlink(os.path.join(out_data, fname_mask_sensors),
                   os.path.join(out_filt, fname_mask_sensors))
    create_symlink(os.path.join(out_glide, fname_mask_sensors_glides),
                   os.path.join(out_filt, fname_mask_sensors_glides))
    create_symlink(os.path.join(out_sgl, fname_mask_sensors_sgls),
                   os.path.join(out_filt, fname_mask_sensors_sgls))
    create_symlink(os.path.join(out_sgl, fname_sgls),
                   os.path.join(out_filt, fname_sgls))

    # Save filter analysis configuation
    cfg_all           = OrderedDict()
    cfg_all['glide']  = cfg_glide
    cfg_all['sgl']    = cfg_sgl
    cfg_all['filter'] = cfg_filt
    save_config(cfg_all, os.path.join(out_filt, fname_cfg_filt))

    return cfg_filt, sensors, sgls, dives, glide_ratio

# TODO move to utils
def create_symlink(src, dest):
    '''Failsafe creation of symlink if symlink already exists'''
    import os

    # Attempt to delete existing symlink
    try:
        os.remove(dest)
    except:
        pass

    os.symlink(src, dest)

    return None

def process_sensor_data(log, path_exp, cfg_glide, file_cfg_exp, sensors, fs_a,
        Mw=None, plots=True, debug=False):
    '''Calculate body conditions summary statistics'''
    from collections import OrderedDict
    import numpy
    import pandas

    import utils
    import utils_dives
    import utils_glides
    import utils_plot
    import utils_prh
    import utils_seawater
    import utils_signal

    from rjdtools import yaml_tools

    # TODO cleanup later
    exp_idxs = [None, None]
    try:
        cfg_exp = yaml_tools.read_yaml(file_cfg_exp)
    except:
        cfg_exp = OrderedDict()

    # 1 Select indices for analysis
    #--------------------------------------------------------------------------
    log.new_entry('Select indices for analysis')

    if path_exp in cfg_exp:
        exp_idxs[0] = cfg_exp[path_exp]['start_idx']
        exp_idxs[1] = cfg_exp[path_exp]['stop_idx']
    else:
        # Plot accelerometer axes, depths, and propeller speed
        utils_plot.plot_triaxial_depths_speed(sensors)

        # Get indices user input - mask
        exp_idxs[0] = utils.recursive_input('Analysis start index', int)
        exp_idxs[1] = utils.recursive_input('Analysis stop index', int)
        # TODO cleanup later
        cfg_exp[path_exp] = OrderedDict()
        cfg_exp[path_exp]['start_idx'] = exp_idxs[0]
        cfg_exp[path_exp]['stop_idx']  = exp_idxs[1]
        yaml_tools.write_yaml(cfg_exp, file_cfg_exp)

    # Creat dataframe for storing masks for various views of the data
    masks = pandas.DataFrame(index=range(len(sensors)), dtype=bool)

    # Create mask of values to be considered part of the analysis
    masks['exp'] = False
    masks['exp'][exp_idxs[0]:exp_idxs[1]] = True

    # Create indices array `exp_ind` for analysis
    exp_ind = numpy.where(masks['exp'])[0]


    # 1.3 Calculate pitch, roll, and heading
    #--------------------------------------------------------------------------
    log.new_entry('Calculate pitch, roll, heading')
    sensors['p'], sensors['r'], sensors['h'] = utils_prh.calc_PRH(sensors['Ax_g'].values,
                                                         sensors['Ay_g'].values,
                                                         sensors['Az_g'].values)

    # 2 Define dives
    #--------------------------------------------------------------------------
    # TODO use min_dive_depth and min_analysis_depth?
    log.new_entry('Define dives')
    dives, masks['dive'] = utils_dives.finddives2(sensors['depth'].values,
                                                  cfg_glide['min_depth'])


    # 3.2.1 Determine `stroke_frq` fluking rate and cut-off frequency
    #--------------------------------------------------------------------------
    log.new_entry('Get stroke frequency')
    # calculate power spectrum of the accelerometer data at the whale frame
    Ax_g = sensors['Ax_g'][masks['exp']].values
    Az_g = sensors['Az_g'][masks['exp']].values

    # NOTE change `auto_select` & `stroke_ratio` here to modify selectio method
    # TODO should perform initial lp/hp filter, then `stroke_f` comes from high-pass
    # should be OK other than t_max, these values are too high
    if debug is False:
        cutoff_frq, stroke_frq, stroke_ratio = utils_glides.get_stroke_freq(Ax_g,
                                                       Az_g,
                                                       fs_a,
                                                       cfg_glide['nperseg'],
                                                       cfg_glide['peak_thresh'],
                                                       auto_select=False,
                                                       stroke_ratio=None)
        # Store user input cutoff and stroke frequencies
        cfg_glide['cutoff_frq']   = cutoff_frq
        cfg_glide['stroke_frq']   = stroke_frq
        cfg_glide['stroke_ratio'] = stroke_ratio

        # Calculate maximum duration of glides from stroke frequency
        cfg_glide['t_max']  = 1 /cfg_glide['stroke_frq']  # seconds
    else:
        cutoff_frq = 0.3
        cfg_glide['cutoff_frq'] = cutoff_frq


    # 3.2.2 Separate low and high frequency signals
    #--------------------------------------------------------------------------
    log.new_entry('Separate accelerometry to high and low-pass signals')
    order = 5
    cutoff_str = str(cfg_glide['cutoff_frq'])
    for btype, suffix in zip(['low', 'high'], ['lf', 'hf']):
        b, a, = utils_signal.butter_filter(cfg_glide['cutoff_frq'], fs_a, order=order,
                btype=btype)
        for param in ['Ax_g', 'Ay_g', 'Az_g']:
            key = '{}_{}_{}'.format(param, suffix, cutoff_str)
            sensors[key] = utils_signal.butter_apply(b, a, sensors[param].values)

    # TODO set params for lf/hf to reduce clutter throughout this routine
    # TODO combine to single plot with shared axes
    if plots is True:
        utils_plot.plot_lf_hf(sensors['Ax_g'][masks['exp']],
                              sensors['Ax_g_lf_'+cutoff_str][masks['exp']],
                              sensors['Ax_g_hf_'+cutoff_str][masks['exp']],
                              title='x axis')

        utils_plot.plot_lf_hf(sensors['Ay_g'][masks['exp']],
                              sensors['Ay_g_lf_'+cutoff_str][masks['exp']],
                              sensors['Ay_g_hf_'+cutoff_str][masks['exp']],
                              title='y axis')

        utils_plot.plot_lf_hf(sensors['Az_g'][masks['exp']],
                              sensors['Az_g_lf_'+cutoff_str][masks['exp']],
                              sensors['Az_g_hf_'+cutoff_str][masks['exp']],
                              title='z axis')


    # 3.2.3 Calculate the smooth pitch from the low pass filter acceleration
    #       signal to avoid incorporating signals above the stroking periods
    #--------------------------------------------------------------------------
    log.new_entry('Calculate low-pass pitch, roll, heading')
    prh_lf = utils_prh.calc_PRH(sensors['Ax_g_lf_'+cutoff_str].values,
                                sensors['Ay_g_lf_'+cutoff_str].values,
                                sensors['Az_g_lf_'+cutoff_str].values,)

    sensors['p_lf'], sensors['r_lf'], sensors['h_lf'] = prh_lf


    # 4 Define precise descent and ascent phases
    #--------------------------------------------------------------------------
    log.new_entry('Get precise indices of descents, ascents, phase and bottom')
    masks['des'], masks['asc'] = utils_dives.get_des_asc2(sensors['depth'].values,
                                                  masks['dive'].values,
                                                  sensors['p_lf'].values,
                                                  cfg_glide['cutoff_frq'],
                                                  fs_a,
                                                  order=5)

    # Typecast des/asc columns to `bool` TODO better solution  later
    masks = masks.astype(bool)

    if plots is True:
        ## TODO remove, seems like a useless plot
        ## plot all descents and all ascent phases
        #utils_plot.plot_triaxial_descent_ascent(sensors['Ax_g'][masks['exp']],
        #                                        sensors['Az_g'][masks['exp']],
        #                                        masks['des'][masks['exp']],
        #                                        masks['asc'][masks['exp']])

        utils_plot.plot_dives_pitch(sensors['depth'][masks['exp']],
                                    masks['dive'][masks['exp']],
                                    masks['des'][masks['exp']],
                                    masks['asc'][masks['exp']],
                                    sensors['p'][masks['exp']],
                                    sensors['p_lf'][masks['exp']])


    # 5 Estimate swim speed
    #--------------------------------------------------------------------------
    #log.new_entry('Estimate swim speed')

    #sensors['speed'] = utils_signal.estimate_speed(sensors['depth'].values,
    #                                            sensors['p_lf'].values,
    #                                            fs_a,
    #                                            fill_nans=True)

    ## TODO config file?
    #zero_level = 0.1
    #theoretic_max = None #m/s
    #sensors['speed'] = utils_prh.speed_from_acc_and_ref(sensors['Ax_g_lf_'+cutoff_str].values,
    #                                                 fs_a,
    #                                                 sensors['propeller'].values,
    #                                                 zero_level,
    #                                                 theoretic_max,
    #                                                 rm_neg=True)
    #if plots is True:
    #    utils_plot.plot_swim_speed(exp_ind, sensors['speed'][masks['exp']].values)




    # 8 Estimate seawater density around the tagged animal
    #--------------------------------------------------------------------------
    log.new_entry('Estimate seawater density')

    salinity = 33.75
    salinities = numpy.asarray([salinity]*len(sensors))
    sensors['dsw'] = utils_seawater.sw_dens0(salinities, sensors['temperature'].values)


    # TODO split to own routine here

    # 6.1 Extract strokes and glides using heave
    #     high-pass filtered (HPF) acceleration signal, axis=3
    #--------------------------------------------------------------------------
    # Two methods for estimating stroke frequency `stroke_frq`:
    # * from the body rotations (pry) using the magnetometer method
    # * from the dorso-ventral axis of the HPF acceleration signal.

    # For both methods, t_max and J need to be determined.

    # Choose a value for J based on a plot showing distribution of signals:
    #   hpf-x, when detecting glides in the next step use Ahf_Anlf() with axis=0
    #   hpf-z when detecting glides in the next step use Ahf_Anlf() with axis=2

    log.new_entry('Get fluke signal threshold')

    # TODO Need to set J (signal threshold) here, user input should be the
    # power, not the frequency. Just use a standard plot of acceleration here?

    if debug is False:
        # Plot PSD for J selection
        Ax_g_hf = sensors['Ax_g_hf_'+cutoff_str][masks['exp']].values
        Az_g_hf = sensors['Az_g_hf_'+cutoff_str][masks['exp']].values

        f_wx, Sx, Px, dpx = utils_signal.calc_PSD_welch(Ax_g_hf, fs_a, nperseg=512)
        f_wz, Sz, Pz, dpz = utils_signal.calc_PSD_welch(Az_g_hf, fs_a, nperseg=512)

        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(f_wx, Sx, label='hf-x PSD')
        ax1.plot(f_wz, Sz, label='hf-z PSD')
        ax1.legend(loc='upper right')
        ax2.plot(sensors['datetimes'][masks['exp']], Ax_g_hf, label='hf-x')
        ax2.plot(sensors['datetimes'][masks['exp']], Az_g_hf, label='hf-z')
        ax2.legend(loc='upper right')

        fig.autofmt_xdate()
        plt.show()

        # Get user selection for J - select one for both axes
        cfg_glide['J'] = utils.recursive_input('J (fluke magnitude)', float)
    else:
        cfg_glide['J'] = 0.4

    return cfg_glide, sensors, dives, masks, exp_ind


def process_glides(log, cfg_glide, sensors, fs_a, dives, masks, Mw=None, plots=True,
        debug= False):
    import numpy

    import utils
    import utils_glides

    cutoff_str = str(cfg_glide['cutoff_frq'])

    # TODO t_max * fs_a in routine below, 16.0 in cfg, check Kagari
    if Mw is None:
        # Get GL from dorso-ventral axis of the HPF acc signal
        GL = utils_glides.get_stroke_glide_indices(sensors['Az_g_hf_'+cutoff_str].values,
                                                   fs_a,
                                                   cfg_glide['J'],
                                                   cfg_glide['t_max'])

    # TODO review once magnet_rot() routine finished
    elif Mw is not None:
        MagAcc, pry, Sa, GL, heading_lf, pitch_lf_deg = calc_mag_heading()

    # TODO
    # check glides duration and positive and negative zero crossings based
    # on selected J and t_max#


    # 10 Calculate glide ratio TODO keep?
    #--------------------------------------------------------------------------
    log.new_entry('Calculate glide ratio')
    glide_mask = utils.mask_from_noncontiguous_indices(len(sensors), GL[:,0], GL[:,1])

    glide_ratio = utils_glides.calc_glide_ratios(dives,
                                                 masks['des'].values,
                                                 masks['asc'].values,
                                                 glide_mask,
                                                 sensors['depth'],
                                                 sensors['p_lf'])

    return GL, glide_mask, glide_ratio


def process_sgls(log, cfg_sgl, sensors, fs_a, GL, dives):
    '''Split subglides and generate summary dataframe'''
    import numpy

    import utils_glides

    # 7 Make 5sec sub-glides
    #--------------------------------------------------------------------------
    log.new_entry('Make sub-glides, duration {}'.format(cfg_sgl['dur']))

    SGL, data_sgl_mask = utils_glides.split_glides(len(sensors),
                                                   cfg_sgl['dur'], fs_a, GL)


    # 9 Generate summary information table for subglides
    #--------------------------------------------------------------------------
    pitch_lf_deg = numpy.rad2deg(sensors['p_lf'].values)

    log.new_entry('Generate summary information table for subglides')
    sgls = utils_glides.calc_glide_des_asc(sensors['depth'].values,
                                           sensors['p_lf'].values,
                                           sensors['r_lf'].values,
                                           sensors['h_lf'].values,
                                           sensors['speed'].values,
                                           dives,
                                           SGL,
                                           pitch_lf_deg,
                                           sensors['temperature'].values,
                                           sensors['dsw'].values)
    return sgls, data_sgl_mask


def save_config(cfg_add, path_cfg_yaml, name='parameters'):
    '''Load analysis configuration defualts'''
    from collections import OrderedDict
    import datetime

    from rjdtools import yaml_tools
    import utils

    cfg = OrderedDict()

    # Record the last date modified & git version
    fmt = '%Y-%m-%d_%H%M%S'
    cfg['last_modified'] = datetime.datetime.now().strftime(fmt)

    # Get git hash and versions of dependencies
    # TODO the versions added by this should only be logged in release, or
    # maybe check local installed vs requirements versions
    cfg['versions'] = utils.get_versions('bodycondition')

    cfg[name] = cfg_add

    yaml_tools.write_yaml(cfg, path_cfg_yaml)

    return cfg


def cfg_glide_params():
    '''Add fields for glide analysis to config dictionary'''
    from collections import OrderedDict
    import numpy

    cfg_glide = OrderedDict()

    # TODO not currently used, useful with magnetometer data
    ## Acceleromter/Magnotometer axis to analyze
    #cfg_glide['axis'] = 0

    # Number of samples per frequency segment in PSD calculation
    cfg_glide['nperseg'] = 256

    # Threshold above which to find peaks in PSD
    cfg_glide['peak_thresh'] = 0.10

    # High/low pass cutoff frequency, determined from PSD plot
    cfg_glide['cutoff_frq'] = None

    # Frequency of stroking, determinded from PSD plot
    cfg_glide['stroke_frq'] = 0.4 # Hz

    # fraction of `stroke_frq` to calculate cutoff frequency (Wn)
    cfg_glide['stroke_ratio'] = 0.4

    # Maximum length of stroke signal
    cfg_glide['t_max'] = 1 / cfg_glide['stroke_frq'] # seconds

    # Minimumn frequency for identifying strokes (3. Get stroke_frq)
    cfg_glide['J'] = '{:.4f}'.format(2 / 180 * numpy.pi) # 0.0349065 Hz

    # For magnetic pry routine
    cfg_glide['alpha'] = 25

    # TODO redundant for filt_params?
    # Minimum depth at which to recognize a dive (2. Define dives)
    cfg_glide['min_depth'] = 0.4

    return cfg_glide


def cfg_sgl_params():
    '''Add fields for subglide analysis to config dictionary'''
    from collections import OrderedDict

    cfg_sgl = OrderedDict()

    # Duration of sub-glides (8. Split sub-glides, 10. Calc glide des/asc)
    cfg_sgl['dur'] = 2 # seconds

    ## TODO not used
    ## Minimum duration of sub-glides, `False` excludes sublides < dur seconds
    #cfg_sgl['min_dur'] = False # seconds

    return cfg_sgl


def cfg_filt_params():
    '''Add fields for filtering of subglides to config dictionary'''
    from collections import OrderedDict

    cfg_filt = OrderedDict()

    # Pitch angle (degrees) to consider sgls
    cfg_filt['pitch_thresh'] = 30

    # Minimum depth at which to recognize a dive (2. Define dives)
    cfg_filt['min_depth'] = 0.4

    # Maximum cummulative change in depth over a glide
    cfg_filt['max_depth_delta'] = 8.0

    # Minimum mean speed of sublide
    cfg_filt['min_speed'] = 0.3

    # Maximum mean speed of sublide
    cfg_filt['max_speed'] = 10

    # Maximum cummulative change in speed over a glide
    cfg_filt['max_speed_delta'] = 1.0

    return cfg_filt


def load_lleo(path_data_acc, file_cal_acc, file_cal_speed, min_depth):
    '''Load lleo data for calculating body condition'''
    import numpy
    import os

    from pylleo.pylleo import lleoio, lleocal

    import utils_prh
    import utils_plot
    from rjdtools import yaml_tools

    # Parse tag model and id from directory/experiment name
    experiment_id = os.path.split(path_data_acc)[1].replace('-','')
    tag_model = experiment_id.split('_')[1]
    tag_id = int(experiment_id.split('_')[2])

    # Load calibrate data
    cal_dict = yaml_tools.read_yaml(file_cal_acc)

    # Verify sensor ID of data matches ID of CAL
    # TODO add tag_id to pylleo cal, must enter manually now
    if cal_dict['tag_id'] != tag_id:
        raise SystemError('Data `tag_id` does not match calibration `tag_id`')

    # Load meta data
    meta = lleoio.read_meta(path_data_acc, tag_model, tag_id)

    # Load data
    sample_f  = 1
    sensors = lleoio.read_data(meta, path_data_acc, sample_f, overwrite=False)

    # Apply calibration to data
    sensors['Ax_g'] = lleocal.apply_poly(sensors, cal_dict, 'acceleration_x')
    sensors['Ay_g'] = lleocal.apply_poly(sensors, cal_dict, 'acceleration_y')
    sensors['Az_g'] = lleocal.apply_poly(sensors, cal_dict, 'acceleration_z')

    # Calibrate propeller measurements to speed m/s^2
    m_speed = utils_prh.speed_calibration_average(file_cal_speed)
    sensors['speed'] = m_speed*sensors['propeller']

    # Linearly interpolate data
    sensors.interpolate('linear', inplace=True)

    # TODO remove, leave diagnostic plot in parent routine?
    exp_ind = range(len(sensors))
    utils_plot.plot_swim_speed(exp_ind, sensors['speed'].values)

    # Get original sampling rates of accelerometer and depth sensors
    dt_a = float(meta['parameters']['acceleration_x']['Interval(Sec)'])
    fs_a = 1/dt_a

    return sensors, dt_a, fs_a#A_g, depths, speed, temperature,


if __name__ == '__main__':

    run_all()
