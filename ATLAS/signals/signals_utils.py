
from Atlas.utils import jit
from Atlas.utils import F_YEAR_HZ

import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jax.random as jrandom


# Model utilities---------------------------------------------------------------

def get_harmonic_frequencies(nfreqs, tspan):
    """Get the lowest `nfreqs` harmonic frequencies from 1/`tspan` to nfreqs/`tspan`.

    This function returns the lowest `nfreqs` harmonic frequencies based on the
    provided timespan. The frequencies are evenly spaced between 1/`tspan` and
    nfreqs/`tspan`.

    Parameters
    ----------
    nfreqs : int
        The number of harmonic frequencies to generate.
    tspan : float
        The timespan of the data.

    Returns
    -------
    array
        A jax array of frequencies (in Hz) [nfreqs].
    """
    return jnp.arange(1, nfreqs + 1) / tspan


def get_fourier_design_matrix(toas, fgw, microseconds=False):
    """Get the Fourier design matrix for the given TOAs and frequencies.

    The Fourier design matrix is constructed such that each row corresponds to a TOA
    and each column corresponds to a sine or cosine basis function at the specified
    frequencies. The resulting matrix has dimensions (n_toas, 2 * n_freqs), where
    the columns alternate between sine and cosine functions for each frequency.

    Optionally, the unit of the coefficients can be changed from seconds to
    microseconds.

    Parameters
    ----------
    toas : array
        The toas (in seconds) [n_toas].
    fgw : array
        The gravitational wave frequencies for which to compute the design matrix [n_freqs].
    microseconds : bool
        If True, the design matrix coefficients will need to be in units of microseconds
        instead of seconds. By default False.

    Returns
    -------
    array
        The design matrix [n_toas, 2*n_freqs].
    """
    scale = 1e6 if microseconds else 1.0
    
    F = jnp.zeros(( len(toas), 2*len(fgw) ))
    # Every other column is sine, then cosine. This is vectorization bit is equivalent to: 
    # F_ij = sin(2*pi*toas_i*fgw_j) for even j, F_ij = cos(2*pi*toas_i*fgw_j) for odd j
    inp = 2*jnp.pi*toas[:,None]*fgw[None,:]
    F = F.at[:,0::2].set(scale * jnp.sin(inp))
    F = F.at[:,1::2].set(scale * jnp.cos(inp))
    return F # (n_toas, 2*n_freqs)
    

def get_power_law_psd(fgw, log10_A, gamma=13./3., modes=1):
    """Create a power-law Power Spectral Density (PSD) [s^2].

    Given the non-repeating set of gravitational wave frequencies, this function
    will calculate the power-law PSD using the given log amplitude and spectral index.
    Setting modes>1 will add duplicate frequencies to the PSD. (i.e. if you have sine and
    cosine modes for each frequency, set modes=2).

    This uses the equation: S(f) = A**2 / (12 * pi**2 * f**3) * (f/f_year)**(-gamma) * df
    where A is the amplitude, gamma is the spectral index, f is the frequency, and df 
    is the frequency bin width. S(f) has units of s^2 instead of s^3 since S(f) is
    the PSD per frequency bin.

    Parameters
    ----------
    fgw : array
        The (non-repeating) gravitational wave frequencies. [n_freqs]
    log10_A : float
        The log_10 amplitude of the power-law. Defaults to 0.0.
    gamma : float
        The spectral index. Defaults to 13./3..
    modes : int
        The number of modes to use. Defaults to 1.

    Returns
    -------
    array
        The power-law PSD at the given frequencies. [modes*n_freqs]
    """
    # Calculate frequency bin widths. Bin 0 starts at f=0 and is up to f[0]
    df = jnp.diff(fgw, prepend=jnp.array([0]))

    # Calculate the powerlaw (PSD)
    pl = 10**(2*log10_A) * (1/(12 * jnp.pi**2 * F_YEAR_HZ**3)) * (fgw/F_YEAR_HZ)**(-gamma) * df

    if modes>1:
        return jnp.repeat(pl,modes)
    else:
        return pl
    

# Sigma matrix utilities--------------------------------------------------------
# Sigma is the full fourier coefficient covariance matrix for all pulsars, which is 
# a matrix of size (n_psr*n_mode, n_psr*n_mode). The following functions are used 
# to easily convert between different representations to make it easier.

def blockPsrs2sigma(blocks):
    """Convert a pulsar block diagonal matrix of (npsr, nmode, nmode) to (npsr*nmode, npsr*nmode).

    This function takes in a block diagonal matrix where each block represents the
    fourier coefficient covariance for a single pulsar (nmode x nmode)
    and converts it into a full covariance matrix for all pulsars with final 
    dimensions (npsr*nmode, npsr*nmode). The ordering of the output matrix is such that
    the first nmode rows/columns correspond to the first pulsar, the next nmode
    rows/columns correspond to the second pulsar, and so on.

    i.e. (p1m1, p1m2, ..., p2m1, p2m2, ...) where pXmY corresponds to the Y'th mode 
    of pulsar X.

    This function can be undone using `sigma2blockPsrs()`. If doing matrix vector
    products with the resulting matrix, it is best to convert vectors using the
    `blockVec2sigmaVec()` and `sigmaVec2blockVec()` to ensure the element ordering
    is the same.

    Parameters
    ----------
    blocks : array
        A block diagonal matrix of shape (npsr, nmode, nmode)

    Returns
    -------
    array
        A full covariance matrix of shape (npsr*nmode, npsr*nmode)
    """
    npsr, nmode, _ = blocks.shape
    sigma = jnp.zeros((npsr*nmode, npsr*nmode))
    for i in range(npsr):
        idx = slice(i*nmode, (i+1)*nmode, 1)
        sigma = sigma.at[idx, idx].set(blocks[i])
    return sigma


def sigma2blockPsrs(sigma, npsrs):
    """Get the pulsar block diagonal matrix (npsr, nmode, nmode) from (npsr*nmode, npsr*nmode).

    This function gets the pulsar block diagonal component from a full covariance matrix.
    This grabs each (nmode, nmode) block corresponding to each pulsar and returns an 
    array with shape (npsrs, nmode, nmode). Note that this function discards any 
    off-diagonal blocks, so it is not always invertible. The ordering of the input 
    matrix is such that the first nmode rows/columns correspond to the first pulsar, 
    the next nmode rows/columns correspond to the second pulsar, and so on.

    i.e. (p1m1, p1m2, ..., p2m1, p2m2, ...) where pXmY corresponds to the Y'th mode 
    of pulsar X.

    The inverse of this function is `blockPsrs2sigma()`. If doing matrix vector 
    products with the input matrix, it is best to convert vectors using the 
    `blockVec2sigmaVec()` and `sigmaVec2blockVec()` to ensure the element ordering 
    is the same.

    Parameters
    ----------
    sigma : array
        A full covariance matrix of shape (npsr*nmode, npsr*nmode)
    npsrs : int
         The number of pulsars. This is needed to determine the size of each block.

    Returns
    -------
    array
        A block diagonal matrix of shape (npsr, nmode, nmode)
    """
    nmode = sigma.shape[0] // npsrs
    blocks = jnp.zeros((npsrs, nmode, nmode))
    for i in range(npsrs):
        idx = slice(i*nmode, (i+1)*nmode, 1)
        blocks = blocks.at[i].set(sigma[idx, idx])
    return blocks


def blockModes2sigma(blocks):
    """Convert a mode block diagonal matrix of (nmode, npsr, npsr) to (npsr*nmode, npsr*nmode).

    This function takes in a block diagonal matrix where each block represents the
    fourier coefficient covariance for a single mode across all pulsars (npsr x npsr) 
    and converts it into a full covariance matrix for all pulsars with final
    dimensions (npsr*nmode, npsr*nmode). The ordering of the output matrix is such that
    the first nmode rows/columns correspond to the first pulsar, the next nmode
    rows/columns correspond to the second pulsar, and so on.

    i.e. (p1m1, p1m2, ..., p2m1, p2m2, ...) where pXmY corresponds to the Y'th mode
    of pulsar X.

    This function can be undone using `sigma2blockModes()`. If doing matrix vector
    products with the resulting matrix, it is best to convert vectors using the
    `blockVec2sigmaVec()` and `sigmaVec2blockVec()` to ensure the element ordering is the same.

    Parameters
    ----------
    blocks : array
        A block diagonal matrix of shape (nmode, npsr, npsr)

    Returns
    -------
    array
        A full covariance matrix of shape (npsr*nmode, npsr*nmode)
    """
    nmode, npsr, _ = blocks.shape
    sigma = jnp.zeros((npsr*nmode, npsr*nmode))
    for i in range(nmode):
        idx = slice(i, npsr*nmode, nmode)
        sigma = sigma.at[idx, idx].set(blocks[i])
    return sigma


def sigma2blockModes(sigma, nmode):
    """Get the mode block diagonal matrix (nmode, npsr, npsr) from (npsr*nmode, npsr*nmode).

    This function gets the mode block diagonal component from a full covariance matrix.
    This grabs each (npsr, npsr) block corresponding to each mode and returns an 
    array with shape (nmode, npsr, npsr). Note that this function discards any 
    off-diagonal blocks, so it is not always invertible. The ordering of the input
    matrix is such that the first nmode rows/columns correspond to the first pulsar,
    the next nmode rows/columns correspond to the second pulsar, and so on.
    i.e. (p1m1, p1m2, ..., p2m1, p2m2, ...) where pXmY corresponds to the Y'th mode
    of pulsar X.

    The inverse of this function is `blockModes2sigma()`. If doing matrix vector
    products with the input matrix, it is best to convert vectors using the
    `blockVec2sigmaVec()` and `sigmaVec2blockVec()` to ensure the element ordering is the same.

    Parameters
    ----------
    sigma : array
        A full covariance matrix of shape (npsr*nmode, npsr*nmode)
    nmode : int
         The number of modes. This is needed to determine the size of each block.

    Returns
    -------
    array
        A block diagonal matrix of shape (nmode, npsr, npsr)
    """
    npsr = sigma.shape[0] // nmode
    blocks = jnp.zeros((nmode, npsr, npsr))
    for i in range(nmode):
        idx = slice(i, npsr*nmode, nmode)
        blocks = blocks.at[i].set(sigma[idx, idx])
    return blocks


def blockVec2sigmaVec(blocks):
    """Convert a block vector of (npsr, nmode) to (npsr*nmode).

    This function takes in a block vector where each block represents the fourier
    coefficient vector for each pulsar (npsr, nmode) and converts it into a full 
    single vector. The ordering of the output vector is such that the first nmode 
    entries correspond to the first pulsar, the next nmode entries correspond to 
    the second pulsar, and so on. 
    i.e. (p1m1, p1m2, ..., p2m1, p2m2, ...) where pXmY corresponds to the Y'th mode
    of pulsar X.

    This function can be undone using `sigmaVec2blockVec()`. Matrix vector products
    with the resulting vector should be done using the `blockPsrs2sigma()` and
    `blockModes2sigma()` functions to ensure the element ordering is the same.

    Parameters
    ----------
    blocks : array
        A block vector of shape (npsr, nmode)

    Returns
    -------
    array
        A single vector of shape (npsr*nmode)
    """
    npsr, nmode = blocks.shape
    vec = jnp.zeros(npsr*nmode)
    for i in range(npsr):
        idx = slice(i*nmode, (i+1)*nmode, 1)
        vec = vec.at[idx].set(blocks[i,:])
    return vec


def sigmaVec2blockVec(svec, npsr):
    """Get the block vector (npsr, nmode) from (npsr*nmode).

    This function gets the block vector component from a full vector. This grabs
    each (nmode) block corresponding to each pulsar and returns an array with shape
    (npsr, nmode). The ordering of the input vector is such that the first nmode 
    entries correspond to the first pulsar, the next nmode entries correspond to
    the second pulsar, and so on.
    i.e. (p1m1, p1m2, ..., p2m1, p2m2, ...) where pXmY corresponds to the Y'th mode of
    pulsar X.

    The inverse of this function is `blockVec2sigmaVec()`. Matrix vector products
    with the input vector should be done using the `blockPsrs2sigma()` and
    `blockModes2sigma()` functions to ensure the element ordering is the same.

    Parameters
    ----------
    svec : array
        A single vector of shape (npsr*nmode)
    npsr : int
         The number of pulsars. This is needed to determine the size of each block.

    Returns
    -------
    array
        A block vector of shape (npsr, nmode)
    """
    nmode = svec.shape[0] // npsr
    block_vec = jnp.zeros((npsr, nmode))
    for i in range(npsr):
        idx = slice(i*nmode, (i+1)*nmode, 1)
        block_vec = block_vec.at[i,:].set(svec[idx])
    return block_vec

