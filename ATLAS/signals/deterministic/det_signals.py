import jax.numpy as jnp
from ATLAS.signals.deterministic import utils



def create_gw_antenna_pattern(gwtheta, gwphi, psr_pos):
    """
    Create pulsar antenna pattern functions as defined
    in Ellis, Siemens, and Creighton (2012).

    Parameters
    ----------
    gwtheta : float
        Polar angle sky location of CW source.
    gwphi : float
        Azimuthal angle sky location of CW source.
    psr_pos : array
        (npsrs, 3) shaped array where npsrs is the number of pulsars in the array.
        These are the Cartesian unit vectors denoting the position of each pulsar.
    
    Returns
    -------
    (fplus, fcross, cosMu) : tuple
        fplus and fcross are the plus and cross antenna pattern functions, respectively,
        and are each arrays of shape (npsrs,) where npsrs are the number of pulsars
        in the array. cosMu is an array of shape (npsrs,) and the cosine of the angle
        between each pulsar and the GW source.
    """

    # use definition from Sesana et al 2010 and Ellis et al 2012
    sgwphi = jnp.sin(gwphi)
    cgwphi = jnp.cos(gwphi)
    sgwtheta = jnp.sin(gwtheta)
    cgwtheta = jnp.cos(gwtheta)

    # this looks dumb, but it plays nice with JAX and batches across pulsars
    mdotpos = sgwphi * psr_pos[:, 0] - cgwphi * psr_pos[:, 1]
    ndotpos = -cgwtheta * cgwphi * psr_pos[:, 0] - cgwtheta * sgwphi * psr_pos[:, 1] \
                + sgwtheta * psr_pos[:, 2]
    omhatdotpos = -sgwtheta * cgwphi * psr_pos[:, 0] - sgwtheta * sgwphi * psr_pos[:, 1] \
                    -cgwtheta * psr_pos[:, 2]

    fplus = 0.5 * (mdotpos ** 2 - ndotpos ** 2) / (1 + omhatdotpos)
    fcross = (mdotpos * ndotpos) / (1 + omhatdotpos)
    cosMu = -omhatdotpos

    return fplus, fcross, cosMu



def cw_delay_evolve_float64(toas, psr_pos, source_params, psr_phases, psr_dists):
    """
    Get the delays across pulsars induced by an evolving continuous gravitational
    wave from an individual supermassive black hole binary (including pulsar term)
    as in Ellis et. al 2012, 2013. This function is NOT float32 compatible and is
    used primarily to test the accuracy of the float32 version.
    
    Parameters
    ----------
    toas : array
        (npsrs, ntoas) shaped array where npsrs and ntoas are the number of pulsars
        and number of toas per pulsar respectively. Note these 'toas' need not be
        the actual observed TOAs of the array, and in implementation they are not.
        In deterministic models, the TOAs are a set of evenly spaced uniform TOAs
        used in the FFT. Then the Fourier design matrix maps the output of the
        FFT to the actual observed TOAs.
    psr_pos : array
        (npsrs, 3) shaped array where npsrs is the number of pulsars in the array.
        These are the Cartesian unit vectors pointing to each pulsar.
    source_params : array
        Array of shape (8,) storing parameters of the CW source. We use the ordering:
        log10(chirp mass [solar mass]), log10(frequency [Hz]), cosine(inclination angle),
        polarization angle, log10(characteristic strain), cosine(polar angle of sky location),
        azimuthal angle of sky location, initial phase.
    psr_phases : array
        Array of shape (npsrs,) where npsrs is the number of pulsars in the array.
        The array stores the phase of the CW at each pulsar.
    psr_dists : array
        Array of shape (npsrs,) where npsrs is the number of pulsars in the array.
        The array stores the distance to each pulsar [kpc].

    Returns
    -------
    res : array
        Array of same shape as 'toas' input. These are the delays in the timing residuals
        induced by the continuous wave in units of [ns].
    """
    # unpack parameters
    log10_mc, log10_fgw, cos_inc, psi, log10_h, cos_gwtheta, gwphi, phase0 = source_params
    p_phases = psr_phases
    pdists = psr_dists

    # convert units to time [s]
    mc = 10 ** log10_mc * utils.Tsun
    fgw = 10 ** log10_fgw
    gwtheta = jnp.arccos(cos_gwtheta)
    inc = jnp.arccos(cos_inc)
    p_dists = pdists * utils.kpc / utils.c
    dist = 2 * mc ** (5 / 3) * (jnp.pi * fgw) ** (2 / 3) / 10**log10_h

    # get antenna pattern funcs and cosMu
    # write function to get pos from theta,phi
    fplus, fcross, cosMu = create_gw_antenna_pattern(gwtheta, gwphi, psr_pos)

    # get pulsar time
    toas_copy = (toas - utils.tref)
    tp = toas_copy - (p_dists*(1-cosMu))[:, None]

    # orbital frequency
    w0 = jnp.pi * fgw
    phase0 = phase0 / 2.0  # convert GW to orbital phase

    # calculate time dependent frequency at earth and pulsar
    mc53 = mc**(5./3.)
    w083 = w0**(8./3.)
    fac1 = 256./5. * mc53 * w083
    omega = w0 * (1. - fac1 * toas_copy)**(-3./8.)
    omega_p = w0 * (1. - fac1 * tp)**(-3./8.)
    omega_p0 = (w0 * (1. + fac1 * p_dists*(1-cosMu))**(-3./8.))[:, None]

    # calculate time dependent phase
    phase = phase0 + 1./32./mc53 * (w0**(-5./3.) - omega**(-5./3.))

    phase_p = (phase0 + p_phases[:, None]
                + 1./32./mc53 * (omega_p0**(-5./3.) - omega_p**(-5./3.)))

    # define time dependent coefficients
    inc_factor = -0.5 * (3. + jnp.cos(2. * inc))
    At = jnp.sin(2. * phase) * inc_factor
    Bt = 2. * jnp.cos(2. * phase) * cos_inc
    At_p = jnp.sin(2. * phase_p) * inc_factor
    Bt_p = 2. * jnp.cos(2. * phase_p) * cos_inc

    # now define time dependent amplitudes
    alpha = mc**(5./3.)/(dist*omega**(1./3.))
    alpha_p = mc**(5./3.)/(dist*omega_p**(1./3.))

    # define rplus and rcross
    c2psi = jnp.cos(2. * psi)
    s2psi = jnp.sin(2. * psi)
    rplus = alpha*(-At*c2psi+Bt*s2psi)
    rcross = alpha*(At*s2psi+Bt*c2psi)
    rplus_p = alpha_p*(-At_p*c2psi+Bt_p*s2psi)
    rcross_p = alpha_p*(At_p*s2psi+Bt_p*c2psi)

    # residuals
    res = fplus[:, None] * (rplus_p - rplus) + fcross[:, None] * (rcross_p - rcross)
    return res   # (Np, Nsparse)

