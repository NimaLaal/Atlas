
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jax.random as jrandom

# Most useful functions for ORF calculations------------------------------------

def get_orf_matrix(pos, orf):
    """Get the ORF matrix for an ordered array of pulsar positions and a given ORF

    This function computes the ORF matrix [npsr, npsr] for an ordered array of 
    pulsar positions [npsr, 3] and a given ORF function. The `orf` input can be 
    either a string specifying the ORF name (see `get_orf_function`) or an orf 
    function which takes a pair of angular separations as input and returns their
    ORF value. The output of this function will be the ORF matrix in the same
    order as the input pulsar positions [npsr, npsr]. 

    Parameters
    ----------
    pos : array
        An array of shape [npsr, 3] representing the positions of the pulsars.
    orf : str or callable
        The ORF function to use. Can be a string specifying the ORF name 
        (see `get_orf_function`) or a custom ORF function.

    Returns
    -------
    array
        The ORF matrix of shape [npsr, npsr].
    """
    # orf could be a string or a function
    orf = get_orf_function(orf) if isinstance(orf, str) else orf

    a,b = jnp.triu_indices(pos.shape[0], k=0) # Get all unique pairs [npair], [npair]
    xi = get_angular_separation(pos[a,:], pos[b,:]) # [npair]
    pair_orf = orf(xi) # [npair]
    orf_matrix = pairs2matrix(pair_orf) # [npsr, npsr]

    return orf_matrix


def get_orf_function(name):
    """Get the overlap reduction function (ORF) for a given ORF name.

    This function returns the appropriate ORF function based on the input name.
    The supported ORF functions are:
    - 'crn', 'curn' or 'commonrednoise' for the common-uncorrelated red noise ORF
    - 'hd' or 'hellingsdowns' for the Hellings-Downs ORF
    - 'dp' or 'dipole' for the dipole ORF
    - 'mp' or 'monopole' for the monopole ORF

    All ORF functions take the angular separation(s) in radians as input and return
    the corresponding ORF value(s).

    Parameters
    ----------
    name : str
        The name of the ORF function to use. See description for supported names.
    Returns
    -------
    callable
        The ORF function corresponding to the given name. See description for supported functions.

    Raises
    ------
    ValueError
        If the input name is not recognized.
    """
    if name in ['crn', 'curn', 'commonrednoise']:
        return orf_commonUncorrelatedRedNoise
    
    if name in ['hd', 'hellingsdowns']:
        return orf_hellingsDowns
    
    if name in ['dp', 'dipole']:
        return orf_dipole
    
    if name in ['mp', 'monopole']:
        return orf_monopole
    
    raise ValueError(f"Unrecognized ORF name: {name}")


# Conversions between pairs and full ORF matrices-------------------------------

def pairs2matrix(M):
    """Convert a pairs representation to a matrix representation.

    This function takes `M` of shape (npairs) and converts it to a full 
    covariance matrix representation of shape (npsr, npsr).  This is
    the inverse operation of `matrix2pairs`.
    
    NOTE: `M` is assumed to be in the order given by jnp.triu_indices with k=0.
    NOTE: `M` must include the diagonal pulsar pairs (i.e., autocorrelations).

    Parameters
    ----------
    M : jnp.ndarray
        The pairs representation of shape (npairs).

    Returns
    -------
    jnp.ndarray
        The matrix representation of shape (npsr, npsr).
    """
    # M has (npairs) -> return Mp with (npsr, npsr)
    npair = M.shape[0] # number of pairs
    npsr = int(jnp.sqrt(0.25 + 2*npair) - 0.5) # Smaller root of quadratic formula

    Mprime = jnp.zeros((npsr, npsr))
    a,b = jnp.triu_indices(npsr, k=0)
    Mprime = Mprime.at[a, b].set(M)
    Mprime = Mprime.at[b, a].set(M)

    return Mprime


def matrix2pairs(M):
    """Convert a matrix representation to a pairs representation.

    This function takes a matrix representation of shape (npsr, npsr) and 
    converts it to a pairs representation of shape (npairs). This is the
    inverse operation of `pairs2matrix`. The order of the pairs is given
    by jnp.triu_indices with k=0.

    Parameters
    ----------
    M : jnp.ndarray
        The matrix representation of shape (npsr, npsr).

    Returns
    -------
    jnp.ndarray
        The pairs representation of shape (npairs).
    """
    # M has shape (npsr, npsr) -> return Mp with (npairs)
    npsr = M.shape[1]
    npair = npsr*(npsr+1)//2

    a,b = jnp.triu_indices(npsr, k=0)
    Mprime = M[a, b] # Shape (npairs)

    return Mprime


# Overlap reduction function utilities------------------------------------------

def get_angular_separation(pos1, pos2):
    """Get the angular separation between two sets of normalized pulsar positions.

    This function computes the pulsar separation angle between every pair of pulsar
    positions in `pos1` and `pos2`. The pulsar positions need to be given in 
    unit-normalized Cartesian coordinates with shape (npair, 3) for both inputs.
    `pos1` and `pos2` can also be given as (3,). The output will be the separation
    angle between these positions in radians.

    Parameters
    ----------
    pos1 : array-like
        First set of pulsar positions in Cartesian coordinates with shape (3) or (n_pos, 3).
    pos2 : array-like
        Second set of pulsar positions in Cartesian coordinates with shape (3) or (n_pos

    Returns
    -------
    float or array
        The angular separation(s) in radians between the input positions.
    """

    pos1 = pos1[None, :] if len(pos1.shape)==1 else pos1 # (n,3) 
    pos2 = pos2[None, :] if len(pos2.shape)==1 else pos2 # (n,3)

    # Find identical positions and set them to zero separation (i.e., xi=0)
    identical = jnp.all(pos1 == pos2, axis=1) # (n,)

    # pos1 and pos2 are (n,3) arrays of unit vectors
    dot = jnp.sum(pos1*pos2, axis=1) # (n,) 

    # Get angular separation
    xi = jnp.arccos(dot) # (n,)
    xi = xi.at[identical].set(0.0) # Set identical positions to zero separation

    return xi


# ORF functions ----------------------------------------------------------------

def orf_commonUncorrelatedRedNoise(xi):
    """Compute the common red noise overlap reduction function.

    This function computes the common red noise overlap reduction function for
    1 or multiple angular separations. The common red noise ORF is simply
    1 for all pairs where the pulsars are the same (i.e., zero separation) and 
    0 for all other pairs.

    Parameters
    ----------
    xi : float or array
        Angular separation(s) in radians.

    Returns
    -------
    float or array
        The common red noise overlap reduction function value(s).
    """
    xi = jnp.asarray(xi) # (n)
    ret = jnp.zeros_like(xi) # (n)

    ret = ret.at[xi==0].set(1.0) # Set same pulsar value to 1

    return ret if ret.shape[0] > 1 else ret[0]


def orf_hellingsDowns(xi):
    """Compute the Hellings-Downs overlap reduction function.

    This function computes the Hellings-Downs overlap reduction function for
    1 or multiple angular separations.

    NOTE: Separations of 0 radians are set to 1 (i.e., HD(0)=1). 

    Parameters
    ----------
    xi : float or array
        Angular separation(s) in radians.

    Returns
    -------
    float or array
        The Hellings-Downs overlap reduction function value(s).
    """
    xi = jnp.asarray(xi) # (n)
    ret = jnp.zeros_like(xi) # (n)

    # Non-zero angular separations
    d = (1-jnp.cos(xi[xi!=0])) / 2
    ret = ret.at[xi!=0].set((1/2) - (d/2) * ((1/2) - 3*jnp.log(d))) # HD formula

    # Zero angular separations
    ret = ret.at[xi == 0].set(1.0) # Set same pulsar value to 1

    return ret if ret.shape[0] > 1 else ret[0]


def orf_dipole(xi):
    """Compute the dipole overlap reduction function.

    This function computes the dipole overlap reduction function for
    1 or multiple angular separations.

    NOTE: Separations of 0 radians are set to 1 (i.e., DP(0)=1).

    Parameters
    ----------
    xi : float or array
        Angular separation(s) in radians.

    Returns
    -------
    float or array
        The dipole overlap reduction function value(s).
    """
    xi = jnp.asarray(xi) # (n)
    ret = jnp.zeros_like(xi) # (n)

    # Non-zero angular separations
    ret = ret.at[xi!=0].set(jnp.cos(xi[xi!=0])) # Dipole formula

    # Zero angular separations
    ret = ret.at[xi == 0].set(1.0) # Set same pulsar value to 1

    return ret if ret.shape[0] > 1 else ret[0]


def orf_monopole(xi):
    """Compute the monopole overlap reduction function.

    This function computes the monopole overlap reduction function for
    1 or multiple angular separations.

    NOTE: The monopole ORF is constant and equal to 1 for all separations.
    (i.e. do you really need this function?)

    Parameters
    ----------
    xi : float or array
        Angular separation(s) in radians.

    Returns
    -------
    float or array
        The monopole overlap reduction function value(s).
    """
    xi = jnp.asarray(xi) # (n)
    ret = jnp.ones_like(xi) # (n)

    return ret if ret.shape[0] > 1 else ret[0]


