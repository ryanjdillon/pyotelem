'''
Text taken from the SMRU R script. Will update to correspond with current
implementation. - Ryan

Jags model script for the lowest DIC model in "Body density and diving gas
volume of the northern bottlenose animal (Hyperoodon ampullatus)" 2016.
Journal of Experimental Biology

Required input data for each 5s glide segment:

a          = acceleration
dsw        = sea water density
exps_id    = id number for each animal
mean_speed = mean swim speed
depth      = mean depth
exps_id    = id number for each dive
sin_pitch  = sine of pitch angle
tau        = measured precision (1/variance) = 1/(SE+0.001)^2
             where SE is the standard error of acceleration and small
             increment (0.001) ensures infinite values

Inter-dive and inter-individual variance of Gamma distributions
(v_air_var, CdAm_var, bdensity_var) were set a uniform prior over large range
[1e-06, 200] on the scale of the standard deviation, as recommended in Gelmans
(2006).

24k iterations
3 parallel jobs, similar to multiple chains
remaining posterier samples downsampled by factor of 36

Convergence was assessed for each parameter using trace history and
Brooks-Gelman_rubin diagnostic plots (Brooks and Gelman, 1998), DIC for
model selection.

References
----------
* Gelman, A. (2006) "Prior distributions for variance parameters in
  hierarchical models (comment on article by Browne and Draper). Bayesian
  analysis 1.3: 515-534.

PyMC3 Model Implementation: Ryan J. Dillon
R MCMC Model Author: Saana Isojunno si66@st-andrews.ac.uk

exps
  exp_id
  animal_id

sgls
  exp_id
  dive_id
  mean_speed
  mean_swdensity
  mean_sin_pitch
  mean_a

dives
  exp_id
  dive_id
'''

# NOTE pymc3 tutorial
# https://people.duke.edu/~ccc14/sta-663/PyMC3.html

# NOTE pymc3 distributions https://pymc-devs.github.io/pymc3/api.html

# T(,) in JAGS represents a truncation, or bounded distribution
# see following for implementation in PyMC3:
# http://stackoverflow.com/a/32654901/943773

# TODO improve design to remove loops, use `shape=` for dist defs
# http://stackoverflow.com/a/25052556/943773

# TODO, pass command to create/compile that just runs all data for server
# scripts

def run_mcmc_all(root_path, glide_path, mcmc_path, manual_selection=True,
        debug=False):
    import datetime
    import os

    import utils_data

    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    n_init = 10000
    init   = 'advi_map'
    n_iter = 50000
    njobs  = None
    trace_name = '{}_mcmc_iter{}_njobs{}'.format(now, n_iter, njobs)

    trace_path = os.path.join(mcmc_path, trace_name)
    os.makedirs(os.path.join(root_path, trace_path), exist_ok=True)

    # Load data
    sgl_cols = ['exp_id', 'dive_id', 'mean_speed', 'mean_depth',
                'mean_sin_pitch', 'mean_swdensity', 'mean_a', 'SE_speed_vs_time']
    exps, sgls, dives = utils_data.create_mcmc_inputs(root_path, glide_path,
                                                      trace_path, sgl_cols,
                                                      manual_selection=manual_selection)
    if debug:
        n_samples = 25
        print('Debug on. Running {}/{} subglides'.format(n_samples, len(sgls)))
        sgls = sgls.sample(n=n_samples)

    # TODO fix this trace name stuff
    trace_name = os.path.join(root_path, trace_path)
    # Run model
    results = run_mcmc(exps, sgls, dives, trace_name, n_iter=n_iter, init=init,
                       n_init=n_init, njobs=njobs)

    return results


def theanotype(a):
    import theano
    import numpy

    x = theano.shared(numpy.asarray(a, theano.config.floatX), borrow=True)
    #x = x.flatten()
    #TODO remove
    #x = theano.tensor.stack(theano.tensor.cast(x, 'float32'), axis=0)
    #x = theano.tensor.cast(x, 'float32')

    return x

def run_mcmc(exps, sgls, dives, trace_name, n_iter, init=None, n_init=None,
        njobs=None):


    def drag_with_speed(CdAm, dsw, mean_speed):
        '''Calculate drag on swim speed, Term 1 of hydrodynamic equation'''
        import theano.tensor as T
        dsw = theanotype(dsw)
        mean_speed = theanotype(mean_speed)
        return -0.5 * CdAm / 1e6 * dsw * mean_speed**2


    def non_gas_body_density(bdensity, depth, dsw, sin_pitch, compr, atm_pa, g):
        '''Calculate non-gas body density, Term 2 of hydrodynamic equation'''
        import theano.tensor as T
        depth = theanotype(depth)
        dsw = theanotype(dsw)
        sin_pitch = theanotype(sin_pitch)

        #term = (1 + 0.1*depth) * atm_pa * 1e-9
        #term2 = 1 - compr * term
        bdensity_tissue = bdensity / (1 - compr * (1 + 0.1*depth) * atm_pa * 1e-9)

        return (dsw / bdensity_tissue - 1) * g * sin_pitch


    def gas_per_unit_mass(vair, depth, dsw, sin_pitch, p_air, g):
        '''Calculate gas per unit mass, Term 3 of hydrodynamic equation'''
        import theano.tensor as T
        depth = theanotype(depth)
        dsw = theanotype(dsw)
        sin_pitch = theanotype(sin_pitch)

        term3 =  vair / 1e6 * g * sin_pitch * (dsw - p_air * \
                 (1 + 0.1*depth)) * 1 / (1 + 0.1*depth)
        return term3


    from collections import defaultdict
    import matplotlib.pyplot as plt
    import numpy
    import pymc3
    import seaborn as sns

    # CONSTANTS
    g      = 9.80665    # Gravitational acceleration m/s^2
    p_air  = 1.225      # Air density at sea level, kg/m3
    atm_pa = 101325     # conversion from pressure in atmospheres into Pascals

    # Term 1: Effect of drag on swim speed
    term1 = numpy.zeros(len(sgls), dtype=object)

    # Term 2: Non-gas body tissue density
    term2 = numpy.zeros(len(sgls), dtype=object)

    # Term 3: Gas per unit mass
    term3 = numpy.zeros(len(sgls), dtype=object)


    with pymc3.Model() as model:

        # Extend Normal distriubtion class to truncate dist. to lower/upper
        bounded_normal = pymc3.Bound(pymc3.Normal, lower=5, upper=20)

        # Extend Gamma distriubtion class to truncate dist. to lower/upper
        # Paramteres of the Gamma distribution were set priors following:
        # shape (alpha) parameter = (mean^2)/variance
        # rate (beta) parameter = mean/variance
        bounded_gamma = pymc3.Bound(pymc3.Gamma, lower=1e-6)

        # GLOBAL PARAMETERS

        # Compressibility factor (x10^-9 Pa-1) - Non-Informative
        compr = pymc3.Uniform(name='$Compressibility$', lower=0.3, upper=0.7)

        # Individual-average drag term (Cd*A*m^-1; x10^-6 m^2 kg-1)
        # Precision 1/variance; SD=2 => precision = 1/2^2 = 0.25
        CdAm_g          = bounded_normal('$CdAm_{global}$', mu=10, sd=0.25)
        CdAm_g_SD       = pymc3.Uniform('$\sigma_{CdAm$}', 1e-06, 200)
        CdAm_g_var      = CdAm_g_SD**2
        CdAm_g_var.name = '$\sigma_{CdAm}^{2}$'

        # Individual-average body density (kg m-3)
        # bottlenose whale - 800, 1200
        bd_g = pymc3.Uniform('$BodyDesnity_{global}$', 800, 1200)
        bd_g_SD  = pymc3.Uniform('$\sigma_{BodyDensity}$', 1e-06, 200)
        bd_g_var = bd_g_SD**2
        bd_g_var.name = '$\sigma_{BodyDensity}^{2}$'

        # Mass-specific volume of air (average across dives) (ml kg-1)
        vair_g = pymc3.Uniform(name='$Vair_{global}$', lower=0.01, upper=100)
        vair_g_SD  = pymc3.Uniform(name='$\sigma_{Vair}$', lower=1e-6, upper=100)
        vair_g_var = vair_g_SD**2
        vair_g_var.name = '$\sigma_{Vair}^{2}$'

        ## INDIVIDUAL-SPECIFIC PARAMETERS

        # Individual Drag
        CdAm_indv_shape = (CdAm_g ** 2) / CdAm_g_var
        CdAm_indv_shape.name = r'$CdAm\alpha$'

        CdAm_indv_rate  = CdAm_g / CdAm_g_var
        CdAm_indv_rate.name = r'$CdAm\beta$'

        CdAm_name = 'CdAm_indv'
        CdAm_indv = bounded_gamma(CdAm_name, alpha=CdAm_indv_shape,
                                 beta=CdAm_indv_rate)

        # Individual Body density
        bd_indv_shape = (bd_g ** 2) / bd_g_var
        bd_indv_shape.name = r'$BodyDensity\alpha$'

        bd_indv_rate  = bd_g / bd_g_var
        bd_indv_rate.name = r'$BodyDensity\beta$'

        bd_name = 'BodyDensity_indv'
        bd_indv = bounded_gamma(bd_name, alpha=bd_indv_shape, beta=bd_indv_rate)


        # DIVE SPECIFIC PARAMETERS
        vair_dive_shape = (vair_g ** 2) / vair_g_var
        vair_dive_shape.name = r'$Vair\alpha$'

        vair_dive_rate  = vair_g / vair_g_var
        vair_dive_rate.name = r'$Vair\beta$'

        vair_name = 'Vair_dive'
        vair_dive = bounded_gamma(vair_name, alpha=vair_dive_shape,
                                  beta=vair_dive_rate)



        # Model for hydrodynamic performance
        # Loop across subglides

        # Calculate term 1
        term1 = drag_with_speed(CdAm_indv,
                                sgls['mean_swdensity'].values,
                                sgls['mean_speed'].values)
        #term1.name = 'term1'

        # Calculate term 2
        term2 = non_gas_body_density(bd_indv,
                                     sgls['mean_depth'].values,
                                     sgls['mean_swdensity'].values,
                                     sgls['mean_sin_pitch'].values,
                                     compr,
                                     atm_pa,
                                     g)
        #term2.name = 'term2'

        # Calculate term 3
        term3 = gas_per_unit_mass(vair_dive,
                                  sgls['mean_depth'].values,
                                  sgls['mean_swdensity'].values,
                                  sgls['mean_sin_pitch'].values,
                                  p_air, g)
        #term3.name = 'term3'

        # Modelled acceleration
        a_mu = term1 + term2 + term3
        a_mu.name = '$a\mu$'

        # Fitting modelled acceleration `sgls['a_mu']` to observed
        # acceleration data `sgls['mean_a']` assumes observed values follow
        # a normal distribution with the measured precision
        # 'tau'=1/variance (i.e. 1/(SE+001)**2)

        # TODO perhaps perform a polyfit, for better residuals/tau
        a_tau = 1/((sgls['SE_speed_vs_time'].values+0.001)**2)
        a_tau = theanotype(a_tau)

        a_name = '$a$'
        a = pymc3.Normal(a_name, a_mu, a_tau, testval=1)



        # Define which vars are to be sampled from
        tracevars = [compr, CdAm_g, bd_g, vair_g, a,
                    CdAm_indv, bd_indv, vair_dive]

        extra_global = [CdAm_g_var, CdAm_g_SD,
                        bd_g_var, bd_g_SD,
                        vair_g_var, vair_g_SD]

        extra_indv  = [CdAm_indv_shape, CdAm_indv_rate,
                      bd_indv_shape, bd_indv_rate]

        extra_dive = [vair_dive_shape, vair_dive_rate]

        extra_sgl  = [a_mu,]

        tracevars = tracevars + extra_global + extra_indv + extra_dive + extra_sgl
        # Collect var names
        #TODO remove [[tracevars.append(v) for v in a] for a in add_vars]
        varnames = [v.name for v in tracevars]

        # Get counts and print
        n_traced = len(tracevars)
        print('traced:         {}'.format(n_traced))

        #n_global = len(extra_global)
        #n_indv    = numpy.sum([len(a) for a in extra_indv])
        #n_dive   = numpy.sum([len(a) for a in extra_dive])
        #n_sgl   = numpy.sum([len(a) for a in extra_sgl])
        #print('extra globals:  {}'.format(n_global))
        #print('extra ind:      {}'.format(n_indv))
        #print('extra dive:     {}'.format(n_dive))
        #print('extra sgl:      {}'.format(n_sgl))
        #print('total extra:    {}'.format(sum([n_global, n_indv, n_dive, n_sgl])))

        # Create backend for storing trace output
        backend = pymc3.backends.text.Text(trace_name, vars=tracevars)

        # ADVI
        #v_params = pymc3.variational.advi(n=n_init)
        #trace = pymc3.variational.sample_vp(v_params, draws=n_iter)

        # Metropolis
        start = pymc3.find_MAP()
        step = pymc3.Metropolis()
        trace = pymc3.sample(draws=n_iter, step=step, start=start, init=init,
        n_init=n_init, trace=backend)

        # NUTS
        #start = pymc3.find_MAP()
        #step = pymc3.NUTS()
        #trace = pymc3.sample(draws=n_iter, step=step, start=start, init=init,
        #                     n_init=n_init, trace=backend)

        pymc3.summary(trace, varnames=varnames[:5])

        # first 12k retained for "burn-in"
        pymc3.traceplot(trace, varnames=varnames[:5])
        plt.show()

        return exps, dives, sgls, trace


if __name__ == '__main__':

    from rjdtools import yaml_tools

    paths = yaml_tools.read_yaml('./cfg_paths.yaml')

    root_path  = paths['root']
    glide_path = paths['glide']
    mcmc_path  = paths['mcmc']

    run_mcmc_all(root_path, glide_path, mcmc_path, debug=False)
