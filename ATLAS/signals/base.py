
from Atlas.utils import jit, jit_method
from Atlas.signals import signals_utils as sutils
from Atlas.signals import orf_functions as orf_funcs

import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jax.random as jrandom

#-------------------------------------------------------------------------------
# This file contains the base class for signals. All signals should inherit from 
# this class. This class is not meant to be instantiated directly and serves mostly 
# to define the interface for the required methods and to use isinstance checks. 
# See the docstring for the Signal_Base class for more details.
# 
# If you want a copy-paste template for a new signal, there is a commented-out 
# version with limited comments at the bottom of this file.
#-------------------------------------------------------------------------------

class Signal_Base:
    """The base class for all signals. DO NOT INSTANTIATE THIS CLASS DIRECTLY.

    This base class constructor is not meant to be called directly. Instead
    it serves as a template for how to structure child class constructors.
    All child classes should abide by the following requirements:
    1) The constructor should have the optional `data` as the last argument.
    2) If `data=None` the constructor should only do a simple initialization.
    3) If `data!=None` the constructor should fully initialize the signal.
    4) All attributes should remain static after initialization!
    5) The class should have the following required attributes:
        - `initialized`: whether the signal is initialized with data (bool)
        - `name`: name of the signal (str)
        - `init_params`: a dictionary containing the parameters used for initialization (except data)
        - `parameter_names`: list of parameter names (length n_parameters)
        - `n_parameters`: number of parameters (scalar)
        - `parameter_range`: range for each parameter, shape (n_parameters, 2)
        - `allow_posterior_draw`: whether to allow posterior draws for this signal (bool)
        - `sampling_methods`: list of tuples (method_name, weight) for sampling this signal
    6) The class should have the following required methods:
        - `get_helpers(self, N_list, reff)`: get any helper matrices needed
        - `get_delta_t(self, helpers, params, key)`: get the delta t for this signal
        - `ln_likelihood(self, helpers, params)`: get the log-likelihood for this signal
        - `ln_prior(self, params)`: get the log-prior for this signal
        - `prior_draw(self, key)`: draw a set of parameters from the prior
    7) Optionally, if `allow_posterior_draw` is True, the class should also have:
        - `posterior_draw(self, helpers, params, key)`: draw a set of parameters from the posterior

    If simple initialization is done (i.e. data=None), the `initialized` attribute 
    should be set to False and the `name` and `init_params` attributes should be set. 
    The `init_params` attribute should be a dictionary containing the parameters 
    used for initialization (except data). Such that the signal can be 
    re-initialized with data later with:
    - `signal(**signal.init_params, data=new_data)`

    If full initialization is done (i.e. data!=None), the `initialized` attribute
    should be set to True and all required attributes for the signal to function
    properly within the global fit should be set. 

    Do not override the _reinitialize method!

    Attributes
    ----------
    name : str
        Name of the signal.
    init_params : dict
        A dictionary containing the parameters used for initialization (except data).
    parameter_names : list
        List of parameter names (length n_parameters).
    n_parameters : int
        Number of parameters (scalar).
    parameter_range : array
        Range for each parameter, shape (n_parameters, 2).
    allow_posterior_draw : bool
        Whether to allow posterior draws for this signal.
    sampling_method : str
        The sampling method to use for the signal. Should be one of the allowed methods
        in SAMPLING_METHODS from Atlas.samplers.CaneToadRacing.
    initialized : bool
        Whether the signal is initialized with data.
    """
    def __init__(self, name='base_signal', sampling_method='hmc', data=None):
        """Constructor for the base signal class. DO NOT INSTANTIATE DIRECTLY.

        This base class constructor is not meant to be called directly. Instead
        it serves as a template for how to structure child class constructors. 
        All child classes should abide by the following requirements:
        1) The constructor should have the optional `data` as the last argument.
        2) If `data=None` the constructor should only do a simple initialization.
        3) If `data!=None` the constructor should fully initialize the signal.

        Simple initialization should set the `initialized` attribute to False and set
        the `name` and `init_params` attributes. The `init_params` attribute should be a
        dictionary containing the parameters used for initialization (except data).
        Such that the signal can be re-initialized with data later with:
        - `signal(**signal.init_params, data=new_data)`
        
        Full initialization should set the `initialized` attribute to True and set all
        required attributes for the signal to function properly within the global fit.
        Importantly, all attributes should remain static after initialization!
        The required attributes for the signal to function properly within the global fit are:
        - `parameter_names`: list of parameter names (length n_parameters)
        - `n_parameters`: number of parameters (scalar)
        - `parameter_range`: range for each parameter, shape (n_parameters, 2)
        - `allow_posterior_draw`: whether to allow posterior draws for this signal (bool)
        - `sampling_method`: the sampling method to use for the signal
        Other attributes can be added as needed.

        Parameters
        ----------
        name : str
            Name of the signal, by default 'base_signal'
        sampling_method : str, optional
            The sampling method to use for the signal. Should be one of the allowed methods
            in SAMPLING_METHODS from Atlas.samplers.CaneToadRacing.
            Default = 'hmc'
        data : Atlas.data.PTA_Data, optional
            Data for initializing the signal. None by default. (simple initialization)

        Raises
        ------
        NotImplementedError
            If this constructor is called directly instead of through a child class.
        """
        raise NotImplementedError("This is a base class. Do not instantiate directly.")
        # Example of what a child signal class should do in its __init__ method:
        # Simple initialization-------------------------------------------------
        self.initialized = False # Will be False if data is not provided.
        self.name = name
        self.init_params = {'name': name, 'sampling_method': sampling_method}

        if data is None: # Guard clause to initialize without data.
            return
        
        # Data is present, we can fully initialize the signal-------------------
        # Required attributes:      parameter_names, n_parameters,
        # parameter_range, allow_posterior_draw, sampling_methods, initialized
        self.parameter_names = [] # List of parameter names
        self.n_parameters = 0 # Number of parameters
        self.parameter_range = jnp.zeros((0,2)) # Range for each parameter, shape (n_parameters, 2)
        self.allow_posterior_draw = False # Whether to allow posterior draws for this signal
        # Brain will check if sampling_method is valid.
        self.sampling_method = sampling_method # The sampling method to use for the signal

        # Helper attributes needed for the rest of the class--------------------
        # These will depend on the signal, but could include things like:
        self.psr_toas = data.toas 
        self.npsrs = data.npsrs
        self.npairs = data.npairs
        self.tspan = data.pta_tspan
        self.freqs = sutils.get_harmonic_frequencies(10, self.tspan) # Example F matrices

        # Hidden attributes if needed-------------------------------------------
        # These are attributes that the signal class can use internally, but are
        # maybe not that helpful for users to see or use directly.
        # Examples like pre-computed matrix products or niche attributes
        # If you aren't sure, you probably don't need any hidden attributes.
        self._hidden_attribute = None

        # Done! Set initialized!
        self.initialized = True


    # Helper methods------------------------------------------------------------
    # These are methods that the signal class and are not required for proper
    # functioning within the global fit. They can be public or private
    def get_phi_diag(self, params):
        """EX: Get the diagonal of the fourier coefficient covariance matrix (phi)

        DO NOT CALL THIS METHOD DIRECTLY. This is just an example.
        An example of a helper method a child class may have. This example method
        computes the diagonal of the fourier coefficient covariance matrix (phi)
        for a power-law signal.

        Parameters
        ----------
        params : array
             An array containing the current set of parameters for this signal. 
             The order should be the same as self.parameter_names.

        Returns
        -------
        phi_diag : array
             An array containing the diagonal of the fourier coefficient covariance matrix (phi).

        Raises
        ------
        NotImplementedError
            If this method is called directly instead of through a child class.
        """
        raise NotImplementedError("This is a base class. Do not call this method directly.")
        # This method is an example of a helper method a child class may have. 
        # It computes the diagonal of the fourier coefficient covariance matrix (phi) 
        # for a power-law signal. Note that the parameter order in `params` should 
        # match the order in `self.parameter_names`.
        log10_A, gamma = params[0], params[1] # Unpack parameters
        phi_diag = sutils.get_power_law_psd(self.freqs, log10_A=log10_A, gamma=gamma, modes=2)
        return phi_diag # [nmode]
    

    # Required methods----------------------------------------------------------
    # These are methods that the global fit will call. They must be implemented
    # for the signal to work properly. They should have the EXACT signatures shown
    @jit_method
    def get_helpers(self, N_list, reff):
        """Get any helper matrices needed for the rest of the methods.

        DO NOT CALL THIS METHOD DIRECTLY. This is just an example. 
        This method should return any helper matrices needed for the rest of the methods. 
        The global fit will call this method once before sampling this signal block.

        examples of helper matrices could be T.T N^{-1} T or T.T N^{-1} r for each pulsar

        Parameters
        ----------
        N_list : list of Atlas.nMatrix.base.Base_TOA_cov
            A list of noise covariance matrix objects for each pulsar.
        reff : list of jnp.ndarray
            A list of effective residuals for each pulsar.

        Returns
        -------
        helpers : tuple
             A tuple containing any helper matrices needed for the rest of the methods.

        Raises
        ------
        NotImplementedError
            If this method is called directly instead of through a child class.
        """
        raise NotImplementedError("This is a base class. Do not call this method directly.")
        # This method should return any helper matrices needed for the rest of the methods.
        # The global fit will call this method once before each block of sampling this
        # signal. N_list is a list of noise covariance matrix objects (which have a
        # N.solve(left, right) method). reff is a list of effective residuals for each pulsar
        # For example: Suppose you want T.T N^{-1} T and T.T N^{-1} r for each pulsar.
        TNT = jnp.zeros((self.npsrs, self.nmodes, self.nmodes))
        TNr = jnp.zeros((self.npsrs, self.nmodes))
        for i in range(self.npsrs):
            T,N,r = T_list[i], N_list[i], reff[i]
            TNT = TNT.at[i,:,:].set(N.solve(T, T))
            TNr = TNr.at[i,:].set(N.solve(T, r)[:,0])
        helpers = (TNT, TNr)
        return helpers
    

    @jit_method
    def get_delta_t(self, helpers, params, key):
        """Get the delta t for this signal.

        DO NOT CALL THIS METHOD DIRECTLY. This is just an example.
        This method should return a list of arrays, where each array is the delta t for that
        pulsar. The global fit will call this method at the end of sampling this signal block.

        Parameters
        ----------
        helpers : tuple
            The outputs from get_helpers(). This can be a tuple of any helper matrices needed.
        params : array
            An array containing the current set of parameters for this signal. 
            The order should be the same as self.parameter_names.
        key : jax.random.PRNGKey
            A jax random key that can be used for any randomness needed. If no randomness
            is needed, you can ignore the key.

        Returns
        -------
        delta_t : list of arrays
            A list of arrays, where each array is the delta t for that pulsar.

        Raises
        ------
        NotImplementedError
            If this method is called directly instead of through a child class.
        """
        raise NotImplementedError("This is a base class. Do not call this method directly.")
        # This method should return a list of arrays, where each array is the
        # delta t for that pulsar. The global fit will call this method at the end
        # of sampling this signal block. helpers are the outputs from get_helpers()
        # params are the current set of parameters for this signal. key is a jax
        # random key that can be used for any randomness needed. If no randomness
        # is needed, you can ignore the key.
        delta_t = None # List of arrays [npsr_toas]
        return delta_t
    
    
    @jit_method
    def ln_likelihood(self, helpers, params):
        """Get the log-likelihood for this signal.

        DO NOT CALL THIS METHOD DIRECTLY. This is just an example.
        This method should return the log-likelihood for this signal. The global fit
        will call this method at each step of sampling this signal block.

        Parameters
        ----------
        helpers : tuple
            The outputs from get_helpers(). This can be a tuple of any helper matrices needed.
        params : array
            An array containing the current set of parameters for this signal.
            The order should be the same as self.parameter_names.

        Returns
        -------
        lnlike : float
            The log-likelihood for this signal given the current set of parameters and helpers.

        Raises
        ------
        NotImplementedError
            If this method is called directly instead of through a child class.
        """
        raise NotImplementedError("This is a base class. Do not call this method directly.")
        # This method should return the log-likelihood. The global fit will call this 
        # method at each step of sampling this signal block. helpers are the outputs
        # from get_helpers() and params are the current set of parameters for this signal.
        lnlike = None # scalar
        return lnlike
    
    
    @jit_method
    def ln_prior(self, params):
        """Get the log-prior for this signal.

        DO NOT CALL THIS METHOD DIRECTLY. This is just an example.
        This method should return the log-prior for this signal. The global fit will call
        this method at each step of sampling this signal block.

        Parameters
        ----------
        params : array
            An array containing the current set of parameters for this signal.
            The order should be the same as self.parameter_names.

        Returns
        -------
        lnprior : float
            The log-prior for this signal given the current set of parameters.

        Raises
        ------
        NotImplementedError
            If this method is called directly instead of through a child class.
        """
        raise NotImplementedError("This is a base class. Do not call this method directly.")
        # This method should return the log-prior. The global fit will call this 
        # method at each step of sampling this signal block. params are the current
        # set of parameters for this signal.
        lnprior = None # scalar
        return lnprior
    
    
    @jit_method
    def prior_draw(self, key):
        """Get a set of parameters drawn from the prior.

        DO NOT CALL THIS METHOD DIRECTLY. This is just an example.
        This method should return a set of parameters drawn from the prior. The global fit
        will call this method when it needs to draw a new set of parameters from 
        the prior distribution.

        Parameters
        ----------
        key : jax.random.PRNGKey
            A jax random key that can be used for any randomness needed.

        Returns
        -------
        params : array
            An array containing a set of parameters drawn from the prior. 
            The order should be the same as self.parameter_names.

        Raises
        ------
        NotImplementedError
            If this method is called directly instead of through a child class.
        """
        raise NotImplementedError("This is a base class. Do not call this method directly.")
        # This method should return a set of parameters drawn from the prior. The
        # global fit will call this method when it needs to draw a new set of
        # parameters from the prior distribution. key is a jax random key that can
        # be used for randomness.
        params = None # [n_parameters]
        return params
    

    # Posterior drawing method (if allow_posterior_draw)------------------------
    @jit_method
    def posterior_draw(self, helpers, params, key):
        """Get a set of parameters drawn from the posterior.

        DO NOT CALL THIS METHOD DIRECTLY. This is just an example.
        This is an optional method which is only needed if `allow_posterior_draw` 
        is True. This method should return a set of parameters drawn from the 
        posterior. The global fit will call this method when it needs to draw a 
        new set of parameters from the posterior distribution. 

        Parameters
        ----------
        helpers : tuple
            The outputs from get_helpers(). This can be a tuple of any helper matrices needed.
        params : array
            An array containing the current set of parameters for this signal.
            The order should be the same as self.parameter_names.
        key : jax.random.PRNGKey
            A jax random key that can be used for any randomness needed.

        Returns
        -------
        params : array
            An array containing a new set of parameters drawn from the posterior. 
            The order should be the same as self.parameter_names.

        Raises
        ------
        NotImplementedError
            If this method is called directly instead of through a child class.
        """
        raise NotImplementedError("This is a base class. Do not call this method directly.")
        # This method should return a set of parameters drawn from the posterior
        # if the posterior form is known. This method only needs to be implemented
        # if allow_posterior_draw is True (i.e. when the posterior form is known). 
        # The global fit will call this method when it needs to draw a new set of 
        # parameters from the posterior distribution. helpers are the outputs from 
        # get_helpers() and params are the current set of parameters for this signal. 
        # key is a jax random key that can be used for randomness.
        new_params = None # [n_parameters]
        return new_params
    
    # Reinitialization method (do not override)---------------------------------
    def _reinitialize(self, data):
        """Re-initialize the signal with new data.

        This method is used to re-initialize the signal with a set of pta_data. 
        self must be static after re-initialization. 

        This method simply calls the constructor again with init_params and the new data.
        - `signal(**signal.init_params, data=new_data)`

        This method is why `init_params` is required!

        Parameters
        ----------
        data : Atlas.data.PTA_Data
            The new data to initialize the signal with.
        """
        # Re-initialize the signal with data! (Use init_params)
        self.__init__(**self.init_params, data=data) 
        

    # User-friendly methods (not required to override)--------------------------
    def __str__(self):
        """A user-friendly string representation of signal objects.

        This method provides a user-friendly string representation of signal objects. 
        It will indicate whether the signal is initialized or not, and will show 
        the name and init_params. Child classes do not need to override this method,
        but can if they want to.

        Returns
        -------
        str
            A user-friendly string representation of the signal object, including its name,
            and init_params.
        """
        if self.initialized:
            s = f'Initialized Signal: {self.name}, {self.init_params=}'
        else:
            s = f'Uninitialized Signal: {self.name}, {self.init_params=}'
        return s
    


# Copy-paste template for a new signal class------------------------------------
'''
from Atlas.signals.base import Signal_Base

class Signal(Signal_Base):
    """_Quick Summary_

    _Extended summary_

    Attributes
    ----------
    name : str
        Name of the signal.
    init_params : dict
        A dictionary containing the parameters used for initialization (except data).
    parameter_names : list
        List of parameter names (length n_parameters).
    n_parameters : int
        Number of parameters (scalar).
    parameter_range : array
        Range for each parameter, shape (n_parameters, 2).
    allow_posterior_draw : bool
        Whether to allow posterior draws for this signal.
    sampling_method : str
        The sampling method to use for the signal.
    initialized : bool
        Whether the signal is initialized with data.
    """
    def __init__(self, name='base_signal', sampling_method='hmc', data=None):
        """_Quick Summary_

        _Extended summary_

        Parameters
        ----------
        name : str
            Name of the signal, by default 'base_signal'
        sampling_method : str, optional
            The sampling method to use for the signal. Should be one of the allowed methods
            in SAMPLING_METHODS from Atlas.samplers.CaneToadRacing.
            Default = 'hmc'
        data : Atlas.data.PTA_Data, optional
            Data for initializing the signal. None by default. (simple initialization)
        """
        # Simple initialization-------------------------------------------------
        self.initialized = False # Will be False if data is not provided.
        self.name = name
        self.init_params = {'name': name, 'sampling_method': sampling_method}

        if data is None: # Guard clause to initialize without data.
            return
        
        # Data is present, we can fully initialize the signal-------------------
        # Required attributes:      parameter_names, n_parameters,
        # parameter_range, allow_posterior_draw, sampling_method, initialized
        self.parameter_names = [] # List of parameter names
        self.n_parameters = 0 # Number of parameters
        self.parameter_range = jnp.zeros((0,2)) # Range for each parameter, shape (n_parameters, 2)
        self.allow_posterior_draw = False # Whether to allow posterior draws for this signal
        self.sampling_method = sampling_method # The sampling method to use for the signal

        # Helper attributes needed for the rest of the class--------------------
        # Needed attributes but not required by the global fit.

        # Hidden attributes if needed-------------------------------------------
        # These are attributes that the signal class can use internally, but are
        # maybe not that helpful for users to see or use directly.

        # Done! Set initialized!
        self.initialized = True


    # Helper methods------------------------------------------------------------
    # These are methods that the signal class and are not required for proper
    # functioning within the global fit. They can be public or private

    # Required methods----------------------------------------------------------
    # These are methods that the global fit will call. They must be implemented
    # for the signal to work properly. They should have the EXACT signatures shown
    @jit_method
    def get_helpers(self, N_list, reff):
        """Get any helper matrices needed for the rest of the methods.

        _Extended summary of things this method returns_

        Parameters
        ----------
        N_list : list of Atlas.nMatrix.base.Base_TOA_cov
            A list of noise covariance matrix objects for each pulsar.
        reff : list of jnp.ndarray
            A list of effective residuals for each pulsar.

        Returns
        -------
        helpers : tuple
             A tuple containing any helper matrices needed for the rest of the methods.
        """
        # This method should return any helper matrices needed for the rest of the methods.
        # The global fit will call this method once before each block of sampling this
        # signal. N_list is a list of noise covariance matrix objects (which have a
        # N.solve(left, right) method). reff is a list of effective residuals for each pulsar
        helpers = None
        return helpers

    @jit_method
    def get_delta_t(self, helpers, params, key):
        """Get the delta t for this signal.

        _Extended summary of how this method is implemented_
        _Note that this method MUST return a list of arrays of delta t for each pulsar_

        Parameters
        ----------
        helpers : tuple
            The outputs from get_helpers(). This can be a tuple of any helper matrices needed.
        params : array
            An array containing the current set of parameters for this signal. 
            The order should be the same as self.parameter_names.
        key : jax.random.PRNGKey
            A jax random key that can be used for any randomness needed. If no randomness
            is needed, you can ignore the key.

        Returns
        -------
        delta_t : list of arrays
            A list of arrays, where each array is the delta t for that pulsar.
        """
        # This method should return a list of arrays, where each array is the
        # delta t for that pulsar. The global fit will call this method at the end
        # of sampling this signal block. helpers are the outputs from get_helpers()
        # If no randomness is needed, you can ignore the key.
        delta_t = None # List of arrays [npsr_toas]
        return delta_t
    
    
    @jit_method
    def ln_likelihood(self, helpers, params):
        """Get the log-likelihood for this signal.

        _Extended summary of how this method is implemented_
        _Note that this method MUST return a scalar log-likelihood_

        Parameters
        ----------
        helpers : tuple
            The outputs from get_helpers(). This can be a tuple of any helper matrices needed.
        params : array
            An array containing the current set of parameters for this signal.
            The order should be the same as self.parameter_names.

        Returns
        -------
        lnlike : float
            The log-likelihood for this signal given the current set of parameters and helpers.

        """
        # This method should return the log-likelihood. The global fit will call this 
        # method at each step of sampling this signal block.
        lnlike = None # scalar
        return lnlike
    
    
    @jit_method
    def ln_prior(self, params):
        """Get the log-prior for this signal.

        _Extended summary of how this method is implemented_
        _Note that this method MUST return a scalar log-prior_

        Parameters
        ----------
        params : array
            An array containing the current set of parameters for this signal.
            The order should be the same as self.parameter_names.

        Returns
        -------
        lnprior : float
            The log-prior for this signal given the current set of parameters.
        """
        raise NotImplementedError("This is a base class. Do not call this method directly.")
        # This method should return the log-prior. The global fit will call this 
        # method at each step of sampling this signal block.
        lnprior = None # scalar
        return lnprior
    
    
    @jit_method
    def prior_draw(self, key):
        """Get a set of parameters drawn from the prior.

        _Extended summary of how this method is implemented_
        _Note that this method MUST return an array of parameters_

        Parameters
        ----------
        key : jax.random.PRNGKey
            A jax random key that can be used for any randomness needed.

        Returns
        -------
        params : array
            An array containing a set of parameters drawn from the prior. 
            The order should be the same as self.parameter_names.
        """
        # This method should return a set of parameters drawn from the prior. The
        # global fit will call this method when it needs to draw a new set of
        # parameters from the prior distribution.
        params = None # [n_parameters]
        return params
    

    # Posterior drawing method (if allow_posterior_draw)------------------------
    @jit_method
    def posterior_draw(self, helpers, params, key):
        """Get a set of parameters drawn from the posterior.

        _Extended summary of how this method is implemented_
        _Note that this method MUST return an array of parameters_
        _This method only needs to be implemented if `allow_posterior_draw` is 
        True (i.e. when the posterior form is known)_ 

        Parameters
        ----------
        helpers : tuple
            The outputs from get_helpers(). This can be a tuple of any helper matrices needed.
        params : array
            An array containing the current set of parameters for this signal.
            The order should be the same as self.parameter_names.
        key : jax.random.PRNGKey
            A jax random key that can be used for any randomness needed.

        Returns
        -------
        params : array
            An array containing a new set of parameters drawn from the posterior. 
            The order should be the same as self.parameter_names.

        Raises
        ------
        NotImplementedError
            If this method is called directly instead of through a child class.
        """
        # This method should return a set of parameters drawn from the posterior
        # if the posterior form is known. This method only needs to be implemented
        # if allow_posterior_draw is True (i.e. when the posterior form is known). 
        new_params = None # [n_parameters]
        return new_params

'''