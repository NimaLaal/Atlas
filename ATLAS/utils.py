
import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jax.random as jrandom
import jax
import inspect
import ATLAS.psd_functions as psd_functions

# This file contains utility functions that are used across the entire codebase.
# Utility functions which are used withiin models are instead located in 
# signals.signals_utils.py to lower the total number of functions in each file.

# Constants---------------------------------------------------------------------
DAY_SEC = 86400.0  # Number of seconds in a day
YEAR_SEC = (365.24 * 24 * 3600)  # Number of seconds in a year
F_YEAR_HZ = 1.0 / YEAR_SEC  # Frequency of 1/year in Hz

C_MPS = 299792458.0  # Speed of light in m/s

PC_M = 3.085677581491367e+16  # Parsec in meters
KPC_M = PC_M * 1e3  # Kiloparsec in meters
MPC_M = PC_M * 1e6  # Megaparsec in meters

T_SUN_SEC = 4.9254909476412675e-06 # Solar mass time in seconds

# Pulsar utilities--------------------------------------------------------------

def get_pulsar_timespan(psr):
    """Get the total timespan of a pulsar or pulsar array.

    This function computes the total timespan covered by the observations of a pulsar
    or a list of pulsars. If a list of pulsars is provided, it calculates the overall
    timespan from the earliest to the latest time of arrival (TOA) across all pulsars.

    This function assumes that the `toas` attribute for each pulsar is in seconds.

    Parameters
    ----------
    psr : Pulsar or list
        A pulsar object or a list of pulsar objects.

    Returns
    -------
    float
        The timespan in the specified unit.
    """
    if isinstance(psr, (list, tuple)):
        tmin = jnp.min(jnp.array([jnp.min(p.toas) for p in psr])) # Do not trust that the first toa is the min!
        tmax = jnp.max(jnp.array([jnp.max(p.toas) for p in psr]))
    else:
        tmin = jnp.min(psr.toas)
        tmax = jnp.max(psr.toas)
    ret = float(tmax - tmin)
    return ret


# Matrix utilities--------------------------------------------------------------
def stabilize_covariance(cov, eig_thresh=1e-10):
    """Stabilize a covariance matrix by setting a minimum eigenvalue

    This function takes a covariance matrix, or batch of covariance matrices, 
    and sets the eigenvalues below a certain threshold to that threshold. This 
    is useful for numerical stabilization for things like inversion. This function
    also works for batches of covariance matrices like [..., N, N].

    NOTE: This function is not JIT compiled

    Parameters
    ----------
    cov : array
        The input covariance matrix or batch of covariance matrices. [..., N, N]
    eig_thresh : float
        The threshold for eigenvalues, by default 1e-10

    Returns
    -------
    jax array
        The stabilized covariance matrix or batch of covariance matrices. [..., N, N]
    """
    N = cov.shape[-1]
    idx = jnp.arange(N)
    dims = cov.shape[:-2]

    # Compute eigenvalues and eigenvectors
    e_vals, e_vec = jnp.linalg.eigh(cov) # [dims, N], [dims, N, N]
    # Determine the threshold for eigenvalues
    e_thr = eig_thresh * jnp.max(e_vals, axis=-1, keepdims=True) # [dims, 1]
    # Compute new eigenvalues by thresholding
    e_vals = jnp.where(e_vals < e_thr, e_thr, e_vals) # [dims, N]
    # Construct the diagonal matrix of eigenvalues
    e_mat = jnp.zeros((*dims, N, N))
    e_mat = e_mat.at[..., idx, idx].set(e_vals) # [dims, N, N]
    # Reconstruct the covariance matrix with the modified eigenvalues
    new_cov = e_vec @ e_mat @ jnp.linalg.inv(e_vec) # [dims, N, N]
    return new_cov


def jagged2padded(jagged, pad_value=-1):
    """Convert a list of jagged arrays to a padded array along with a boolean mask.

    This function takes a list of jagged arrays and converts it into a single
    padded numpy array which has dimensions (n_array, max_length) where max_length
    is the length of the longest array in the input list. The shorter arrays are
    padded with the value specified by `pad_value`. Additionally, a boolean mask 
    of the same shape where True indicates valid entries (i.e. not padding).

    NOTE: This function cannot be jitted as it uses numpy operations!

    Parameters
    ----------
    jagged : list of arrays
        A list of jagged arrays (arrays of different lengths).
    pad_value : int
        The value to use for padding shorter arrays, by default -1

    Returns
    -------
    tuple
        A tuple containing:
        - padded : array
            The padded array with shape (n_array, max_length).
        - mask : array
            A boolean mask indicating valid entries (True for valid, False for padding).
    """
    final_shape = (len(jagged), max(len(arr) for arr in jagged))
    dtype = jagged[0].dtype
    padded = np.full(final_shape, pad_value, dtype=dtype)
    for i, arr in enumerate(jagged):
        padded[i, :len(arr)] = arr

    mask = padded != pad_value

    return padded, mask


def padded2jagged(padded, mask):
    """Convert a padded array back to a list of jagged arrays.

    This function takes a padded array and a boolean mask and converts it back
    to a list of jagged arrays, where each array corresponds to a row in the
    padded array, with padding removed.

    NOTE: This function cannot be jitted as it uses numpy operations!

    Parameters
    ----------
    padded : array
         The padded array to be converted back to jagged format. [n_array, max_length]
    mask : array
        The boolean mask indicating valid entries. [n_array, max_length]

    Returns
    -------
    list of arrays
        The list of jax arrays.
    """
    jagged = []
    for i in range(len(padded)):
        ent = padded[i][mask[i]]
        jagged.append(ent)

    return jagged


def diagonalize(x):
    """Transform a vector or batch of vectors into a diagonal matrix.

    This function takes a vector of shape (..., N) and returns a diagonal matrix
    of shape (..., N, N) where the diagonal elements are the elements of the input vector.

    Parameters
    ----------
    x : array
        Input array of shape (..., N).

    Returns
    -------
    array
        Diagonal matrix of shape (..., N, N) with the input vector on the diagonal.
    """
    return x[..., None] * jnp.eye(x.shape[-1])


def is_diagonal(x):
    """Check if a matrix or batch of matrices is diagonal.

    This function checks whether the input matrix (or batch of matrices) is diagonal
    by comparing the sum of all elements with the sum of the diagonal elements. If
    the difference is zero, the matrix is considered diagonal.

    Parameters
    ----------
    x : array
        Input array of shape (..., N, N).

    Returns
    -------
    bool
        True if the matrix (or all matrices in the batch) is diagonal, False otherwise.
    """
    diff = (jnp.sum(x) - jnp.sum(jnp.diagonal(x, axis1=-2, axis2=-1)))
    return diff == 0


# Decorators for JIT compilation with proper metadata preservation--------------

def jit(func=None, static_argnums=None):
    """A wrapper/decorator to JIT compile functions while preserving metadata.

    This decorator can be used to JIT compile a function or method while
    preserving the original function's metadata using functools.wraps. It can be
    used in two ways:
    1. As a simple decorator: `@jit` or `@jit(static_argnums=...)`
    2. As a wrapper with arguments: `jit(func, static_argnums=...)`

    The `static_argnums` parameter allows you to specify which arguments should
    be treated as static (i.e., not traced by JAX) when JIT compiling the function.
    For class methods, you typically want to include `0` in `static_argnums` to treat
    `self` as static. 

    Parameters
    ----------
    func : callable, optional
        The function to be wrapped and JIT compiled, by default None
    static_argnums : tuple of ints, optional
        The indices of the arguments to be treated as static, by default None

    Returns
    -------
    callable
        The JIT compiled version of the input function with preserved metadata.
    """
    from jax import jit as jax_jit
    from functools import wraps

    # Check if static_argnums is provided, if so, ensure it's a tuple
    if static_argnums is not None:
        if isinstance(static_argnums, int):
            # If it's a single integer, convert it to a tuple
            static_argnums = (static_argnums,)
        else:
            # Assume its already an iterable of integers, convert to tuple
            static_argnums = tuple(static_argnums)


    def decorator(f):
        # Jit the function with the specified static arguments
        jit_func = jax_jit(f, static_argnums=static_argnums)

        @wraps(f)
        def wrapper(*args, **kwargs):
            return jit_func(*args, **kwargs)

        # functools.wraps hides jax's ahead-of-time API, so keep a handle on the
        # jitted callable itself. Needed for .lower()/.compile() -- and hence for
        # memory_analysis(), which is the only way to see an executable's size.
        wrapper.jitted = jit_func

        return wrapper
    
    if func is None:
        return decorator # Return the decorator if no function is provided
    else:
        return decorator(func) # Otherwise, apply the decorator to the function
    

def jit_method(func):
    """A wrapper to JIT compile class methods with static `self` while preserving metadata.

    This decorator is specifically designed for class methods. It JIT compiles the 
    method while treating `self` as a static argument. This is identical to
    using `@jit(static_argnums=0)` on a class method.

    Parameters
    ----------
    func : callable
        The class method to be wrapped and JIT compiled.

    Returns
    -------
    callable
        The JIT compiled version of the input class method with preserved metadata.
    """
    return jit(func, static_argnums=0)


# Random number stuff-----------------------------------------------------------

def get_PRNGKey():
    """Get a JAX PRNGKey based on the current time in nanoseconds.

    This function generates a JAX PRNGKey using random.randint to seed. This 
    function is intended for use in non-signal code and should not be used when 
    reproducibility is a concern.

    NOTE: Do not use this function for rng-generation in the signal models. Instead
    a random key will be supplied for you and you should use jax.random.split to 
    generate any extras you need. 

    Returns
    -------
    jax.random.PRNGKey
         A JAX PRNGKey generated from the current time in nanoseconds.
    """
    import random
    seed = random.randint(0, 2**32 - 1) 
    return jrandom.PRNGKey(seed)