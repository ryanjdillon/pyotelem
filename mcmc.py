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


def run_mcmc_all(root_path, glide_path, mcmc_path):
    import datetime
    import os

    import utils_data

    now = datetime.datetime.now().strftime('%Y-%m%-%d_%H%M%S')
    iterations = 1000
    njobs = 2
    trace_name = '{}_mcmc_iter{}_njobs{}'.format(now, iterations, njobs)
    trace_name = os.path.join(root_path, mcmc_path, trace_name)
    trace_path = os.path.join(mcmc_path, trace_name)

    # Load data
    exps, sgls, dives = utils_data.create_mcmc_inputs(root_path, glide_path,
                                                      trace_path)
    # Run model
    results = run_mcmc(exps, sgls, dives, trace_name)

    return None


def run_mcmc(exps, sgls, dives, trace_name, iterations, njobs):


    def drag_with_speed(CdAm, dsw, mean_speed):
        '''Calculate drag on swim speed, Term 1 of hydrodynamic equation'''
        return -0.5 * CdAm / 1e6 * dsw * mean_speed**2


    def non_gas_body_density(bdensity, depth, dsw, sin_pitch, compr, atm_pa, g):
        '''Calculate non-gas body density, Term 2 of hydrodynamic equation'''

        bdensity_tissue = bdensity / (1 - compr * (1 + 0.1*depth) * atm_pa * 1e-9)

        return (dsw / bdensity_tissue - 1) * g * sin_pitch


    def gas_per_unit_mass(vair, depth, dsw, sin_pitch, p_air, g):
        '''Calculate gas per unit mass, Term 3 of hydrodynamic equation'''
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
        CdAm_g = bounded_normal('$CdAm_{global}$', mu=10, sd=0.25)
        CdAm_g_SD  = pymc3.Uniform('$\sigma_{CdAm$}', 1e-06, 200)
        CdAm_g_var = CdAm_g_SD**2
        CdAm_g_var.name = '$\sigma_{CdAm}^{2}$'

        # Individual-average body density (kg m-3)
        # bottlenose whale - 800, 1200
        bd_g = pymc3.Uniform('$BodyDesnity_{global}$', 800, 1200)
        bd_g_SD  = pymc3.Uniform('$\sigma_{BodyDensity}$', 1e-06, 200)
        bd_g_var = bd_g_SD**2
        bd_g_var.name = '$\sigma_{BodyDensity}^{2}$'

        # Mass-specific volume of air (average across dives) (ml kg-1)
        vair_g = pymc3.Uniform(name='Vair_{global}', lower=0.01, upper=100)
        vair_g_SD  = pymc3.Uniform(name='$\sigma_{Vair}$', lower=1e-6, upper=200)
        vair_g_var = vair_g_SD**2
        vair_g_var.name = '$\sigma_{Vair}^{2}$'

        # INDIVIDUAL-SPECIFIC PARAMETERS
        CdAm_ind       = numpy.zeros(len(exps), dtype=object)
        CdAm_ind_shape = numpy.zeros(len(exps), dtype=object)
        CdAm_ind_rate  = numpy.zeros(len(exps), dtype=object)

        bd_ind       = numpy.zeros(len(exps), dtype=object)
        bd_ind_shape = numpy.zeros(len(exps), dtype=object)
        bd_ind_rate  = numpy.zeros(len(exps), dtype=object)
        for e in range(len(exps)):
            # Drag term
            CdAm_ind_shape[e] = (CdAm_g ** 2) / CdAm_g_var
            CdAm_ind_shape[e].name = r'$CdAm\alpha_{exp'+str(e)+'}$'
            CdAm_ind_rate[e]  = CdAm_g / CdAm_g_var
            CdAm_ind_rate[e].name = r'$CdAm\beta_{exp'+str(e)+'}$'

            CdAm_name = 'CdAm_{exp'+str(e)+'}'
            CdAm_ind[e] = bounded_gamma(CdAm_name, alpha=CdAm_ind_shape[e],
                                        beta=CdAm_ind_rate[e])
            # Body density
            bd_ind_shape[e] = (bd_g ** 2) / bd_g_var
            bd_ind_shape[e].name = r'$BodyDensity\alpha_{exp'+str(e)+'}$'
            bd_ind_rate[e]  = bd_g / bd_g_var
            bd_ind_rate[e].name = r'$BodyDensity\beta_{exp'+str(e)+'}$'

            bd_name = 'BodyDensity_{exp'+str(e)+'}'
            bd_ind[e] = bounded_gamma(bd_name, alpha=bd_ind_shape[e],
                                      beta=bd_ind_rate[e])

        # DIVE SPECIFIC PARAMETERS
        vair_dive       = numpy.zeros(len(dives), dtype=object)
        vair_dive_shape = numpy.zeros(len(dives), dtype=object)
        vair_dive_rate  = numpy.zeros(len(dives), dtype=object)
        for d in dives.index:
            vair_dive_shape[d] = (vair_g ** 2) / vair_g_var
            vair_dive_shape[e].name = r'$Vair\alpha_{dive'+str(d)+'}$'
            vair_dive_rate[d]  = vair_g / vair_g_var
            vair_dive_rate[e].name = r'$Vair\beta_{dive'+str(d)+'}$'

            vair_name = 'Vair_dive_{dive'+str(d)+'}'
            vair_dive[d] = bounded_gamma(vair_name, alpha=vair_dive_shape[d],
                                        beta=vair_dive_rate[d])

        # Model for hydrodynamic performance
        # Loop across subglides
        a    = numpy.zeros(len(sgls), dtype=object)
        a_mu = numpy.zeros(len(sgls), dtype=object)
        for j in range(len(sgls)):
            exp_idx  = numpy.argmax(exps['exp_id'] == sgls['exp_id'].iloc[j])
            dive_idx = numpy.argmax(dives['dive_id'] == sgls['dive_id'].iloc[j])

            # Calculate term 1
            term1[j] = drag_with_speed(CdAm_ind[exp_idx],
                                       sgls['mean_swdensity'].iloc[j],
                                       sgls['mean_speed'].iloc[j])
            term1[j].name = 'term1_{sgl'+str(j)+'}'

            # Calculate term 2
            term2[j] = non_gas_body_density(bd_ind[exp_idx],
                                            sgls['mean_depth'].iloc[j],
                                            sgls['mean_swdensity'].iloc[j],
                                            sgls['mean_sin_pitch'].iloc[j],
                                            compr, atm_pa, g)
            term2[j].name = 'term2_{sgl'+str(j)+'}'

            # Calculate term 3
            term3[j] = gas_per_unit_mass(vair_dive[dive_idx],
                                         sgls['mean_depth'].iloc[j],
                                         sgls['mean_swdensity'].iloc[j],
                                         sgls['mean_sin_pitch'].iloc[j],
                                         p_air, g)
            term3[j].name = 'term3_{sgl'+str(j)+'}'

            # Modelled acceleration
            a_mu[j] = term1[j] + term2[j] + term3[j]
            a_mu[j].name = '$Acc\mu_{sgl'+str(j)+'}$'

            # Fitting modelled acceleration `sgls['a_mu']` to observed
            # acceleration data `sgls['mean_a']` assumes observed values follow
            # a normal distribution with the measured precision
            # 'tau'=1/variance (i.e. 1/(SE+001)**2)

            # TODO perhaps perform a polyfit, for better residuals/tau
            a_tau = 1/((sgls['SE_speed_vs_time'].iloc[j]+0.001)**2)

            a_name = 'a_{sgl'+str(j)+'}'
            a[j] = pymc3.Normal(a_name, a_mu[j], a_tau, testval=1)
            a[j].name = '$Acc_{sgl'+str(j)+'}$'


        # Append each var in ndarrays of MCMC vars to list `tracevars`
        tracevars = [compr, CdAm_g, CdAm_g_var, bd_g, bd_g_var, vair_g, vair_g_var]
        #not_included = [CdAm_g_SD, bd_g_SD, vair_g_SD,]
        add_vars = [CdAm_ind, bd_ind, vair_dive, a]
        #not_included_nd =  [CdAm_ind_shape, CdAm_ind_rate, bd_ind_shape,
        #                    bd_ind_rate, vair_dive_shape, vair_dive_rate, a_mu]

        [[tracevars.append(v) for v in a] for a in add_vars]
        varnames = [v.name for v in tracevars]
        print(varnames)

        # Create backend for storing trace output
        backend = pymc3.backends.text.Text(trace_name, vars=tracevars)

        # 24k iterations
        # 3 parallel jobs, similar to multiple chains
        # remaining posterier samples downsampled by factor of 36
        trace = pymc3.sample(iterations, njobs=njobs, trace=backend)

        pymc3.summary(trace, varnames=varnames[:5])

        # first 12k retained for "burn-in"
        pymc3.traceplot(trace, varnames=varnames[:5])
        plt.show()

        return exps, dives, sgls, trace


if __name__ == '__main__':

    from rjdtools import yaml_tools

    paths = yaml_tools.read_yaml('iopaths.yaml')

    root_path  = paths['root']
    glide_path = paths['glide']
    mcmc_path  = paths['mcmc']

    run_mcmc_all(root_path, glide_path, mcmc_path)
