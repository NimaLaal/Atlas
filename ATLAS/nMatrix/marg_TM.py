
from Atlas.nMatrix.base import Base_TOA_cov
from Atlas.nMatrix.TM import NoBackendWhiteCov, WhiteCov
from Atlas.utils import jagged2padded, stabilize_covariance_matrix
from Atlas.utils import jit, jit_method

import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from functools import partial

import jax
from jax.tree_util import register_pytree_node_class

# ------------------------------------------------------------------------------
# These TOA_covariance objects marginalize over the linearized-timing model 
# coefficients. This helps remove the effects of the timing model from GW analysis.
# ------------------------------------------------------------------------------

# Enabling timing model SVD for more stable marginalization. 
USE_TM_SVD = True

@register_pytree_node_class
class Marg_NoBackendWhiteCov(NoBackendWhiteCov):
    """A timing-model-marginalized covariance matrix without EFAC, EQUAD, or ECORR.

    This class represents a simple timing-model marginalized TOA covariance matrix.
    This class is helpful when dealing with simulated datasets or whitened data such
    that they do not include EFAC, EQUAD, or ECORR components. The noise covariance matrix
    is solved using the Woodbury matrix identity. This uses the Fix_TM_TOA_cov_simple 
    class to compute the N^{-1} part of the solution, and then applies the timing 
    model marginalization using the Woodbury identity. 
    
    This class inherits from the Fix_TM_TOA_cov_simple parent class and overrides 
    the solve method.

    Attributes
    ----------
    psr_name : str
        The name of the pulsar this covariance matrix corresponds to.
    ntoas : int
        The number of TOAs for this pulsar.
    toaerrs : array
        The TOA errors for this pulsar. [ntoas]
    nvec : array
        An array of the diagonal elements of the covariance matrix (the white noise variances). [ntoas]
    Mmat : array
        The design matrix for the timing model. [ntoas, nparams]
    """
    def __init__(self, psr):
        """The constructor for the simple timing-model-marginalized covariance matrix.

        This constructor initializes the simple timing-model-marginalized covariance matrix. 
        It takes a pulsar object and extracts the TOA errors to construct the diagonal
        covariance matrix, and also extracts the design matrix for the timing model.

        Parameters
        ----------
        psr : object
            The pulsar object to construct the covariance matrix for.
        """
        super().__init__(psr=psr)
        if USE_TM_SVD:
            self.Mmat = _timing_model_svd(psr.Mmat)
        else:
            self.Mmat = psr.Mmat


    @jit_method
    def solve(self, left, right):
        """Solve the linear system left^T D^{-1} right. 

        This method solves the linear equation (left).T @ D^{-1} @ right, where
        D is the timing-model marginalized covariance matrix.

        NOTE: the left matrix will be transposed for you and should be [N_toa, N]

        Parameters
        ----------
        left : array-like
            The left-hand side of the equation. [Ntoa, n]
        right : array-like
            The right-hand side of the equation. [Ntoa, m]

        Returns
        -------
        array-like
            The result of the linear equation. [n, m] or [m]
        """
        parent = super()
        # Solve L^T N^{-1} R - L^T N^{-1} M (M^T N^{-1} M)^{-1} M^T N^{-1} R
        # Term1 = L^T N^{-1} R
        term1 = parent.solve(self, left, right)

        # Term2 = L^T N^{-1} M (M^T N^{-1} M)^{-1} M^T N^{-1} R
        MNM = parent.solve(self, self.Mmat, self.Mmat)
        LNM = parent.solve(self, left, self.Mmat)
        MNR = parent.solve(self, self.Mmat, right)

        if not USE_TM_SVD:
            # The covariance matrix MNM can be ill-conditioned, we should stabilize it
            MNM = stabilize_covariance_matrix(MNM, n=1e-6)
            
        cf = jsl.cho_factor(MNM)
        term2 = LNM @ jsl.cho_solve(cf, MNR)

        return term1 - term2
    

    # Jax pytree methods -------------------------------------------------------
    def tree_flatten(self):
        """Tree flatten method for JAX pytrees.

        This method defines how to flatten the TOA_Covariance object into
        its tracable components (children) and auxiliary components (aux_data) 
        for JAX pytrees.

        Returns
        -------
        tuple
            A tuple containing:
            - children: a tuple of the tracable attributes.
            - aux_data: a tuple of the auxiliary attributes.
        """
        # Children are dynamic or array attributes.
        children = (self.toaerrs, self.nvec, self.Mmat) # Static arrays
        # Aux data is static attributes not used for computations
        aux_data = (self.psr_name, self.ntoas)
        return children, aux_data
    

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """The tree unflatten method for JAX pytrees.

        This method defines how to reconstruct the TOA_Covariance object from
        its children and aux_data for JAX pytrees.

        Parameters
        ----------
        aux_data : tuple
            The static attributes that were stored in aux_data during flattening.
        children : tuple
            The tracable attributes that were stored in children during flattening.

        Returns
        -------
        Marg_TM_TOA_cov_simple
            The reconstructed Marg_TM_TOA_cov_simple object.
        """
        # Reconstruct the object from children and aux data
        (psr_name, ntoas) = aux_data
        (toaerrs, nvec, Mmat) = children

        obj = cls.__new__(cls) # Create an uninitialized instance of the class
        
        # Static attributes ----------------------------------------------------
        obj.psr_name = psr_name
        obj.ntoas = ntoas
        obj.toaerrs = toaerrs

        obj.nvec = nvec
        obj.Mmat = Mmat

        return obj


@register_pytree_node_class
class Marg_WhiteCov(WhiteCov):
    """A timing-model-marginalized white noise covariance matrix with EFAC, EQUAD, and ECORR.

    This model includes EFAC, EQUAD, or ECORR parameters. The noise covariance matrix
    is solved using the Woodbury matrix identity. This uses the Fix_TM_TOA_cov_full 
    class to compute the N^{-1} part of the solution, and then applies the timing 
    model marginalization using the Woodbury identity. 

    This class inherits from the Fix_TM_TOA_cov_full parent class and overrides 
    the solve method to compute the timing-model-marginalized solution.

    Attributes
    ----------
    psr_name : str
        The name of the pulsar this covariance matrix corresponds to.
    ntoas : int
        The number of TOAs for this pulsar.
    nepochs : int
        The number of epochs for this pulsar.
    toaerrs : array
        The TOA errors for this pulsar. [ntoas]
    backends : list
        A list of unique backend names for this pulsar.
    B : array
        An array of backend indices for each TOA. [ntoas]
    U_pad : array   
        A padded array of TOA indices for each epoch. [nepochs, max_epoch_size]
    U_mask : array
        A boolean mask indicating which elements of U_pad are valid. [nepochs, max_epoch_size]
    V : array
        An array of epoch indices for each TOA. [ntoas]
    nvec : array
        An array of the diagonal elements of the covariance matrix (the white noise variances). [ntoas]
    jvec : array
        An array of the ECORR values for each epoch. [nepochs]
    Mmat : array
        The design matrix for the timing model. [ntoas, nparams]
    """
    def __init__(self, 
                 psr, 
                 params=None, 
                lower_efac = 0.01, 
                upper_efac = 10,
                sigma_efac = 0.25,
                center_efac = 1.,
                lower_log10equad = -9.,
                upper_log10equad = -5.,
                lower_log10ecorr = -9.,
                upper_log10ecorr = -5.):
        """Initialize the TOA_MargCovariance_full class.

        This class represents a full white noise model for a pulsar with timing
        model marginalization. This model includes EFAC, EQUAD, or ECORR parameters. 

        Parameters
        ----------
        psr : Pulsar object
            The pulsar object to create the noise model for.
        params : dict, optional
            A dictionary of noise parameters to initialize the model with, by default None
        """
        # Super will initialize: 
        # psr_name, ntoas, nepochs, toaerrs, backends, B, U_pad, U_mask, V, nvec, jvec
        super().__init__(psr = psr, 
                        params=params,
                        lower_efac = lower_efac, 
                        upper_efac = upper_efac,
                        sigma_efac = sigma_efac,
                        center_efac = center_efac,
                        lower_log10equad = lower_log10equad,
                        upper_log10equad = upper_log10equad,
                        lower_log10ecorr = lower_log10ecorr,
                        upper_log10ecorr = upper_log10ecorr
                        )
        if USE_TM_SVD:
            self.Mmat = _timing_model_svd(psr.Mmat)
        else:
            self.Mmat = psr.Mmat
            
        self.Mprior = self.Mmat.shape[1] * jnp.log(1e40)
        
    @jit
    def solve(self, helpers, left, right):
        """Solve a linear equation (left).T @ D^{-1} @ right.

        This method solves the linear equation (left).T @ D^{-1} @ right, where
        D is the timing-model marginalized covariance matrix.

        NOTE: the left matrix will be transposed for you and should be [N_toa, N]

        Parameters
        ----------
        helpers : tuple
            Tuple ``(nvec, jvec)`` containing the diagonal and ECORR covariance components.
        Fmat : array [total_ntoas, N_basis]
        left : array-like
            The left-hand side of the equation. [Ntoa, n]
        right : array-like
            The right-hand side of the equation. [Ntoa, m]

        Returns
        -------
        array-like
            The result of the linear equation. [n, m] or [m]
        """
        parent = super()
        # Solve L^T N^{-1} R - L^T N^{-1} M (M^T N^{-1} M)^{-1} M^T N^{-1} R
        # Term1 = L^T N^{-1} R
        term1 = parent.solve(helpers, left, right)

        # Term2 = L^T N^{-1} M (M^T N^{-1} M)^{-1} M^T N^{-1} R
        MNM = parent.solve(helpers, self.Mmat, self.Mmat)
        LNM = parent.solve(helpers, left, self.Mmat)
        MNR = parent.solve(helpers, self.Mmat, right)

        # The covariance matrix MNM can be ill-conditioned, we should stabilize it
        if not USE_TM_SVD:
            MNM = stabilize_covariance_matrix(MNM, n=1e-6)
        cf = jsl.cho_factor(MNM)
        term2 = LNM @ jsl.cho_solve(cf, MNR)

        return term1 - term2

    @partial(jax.jit, static_argnums=(0, ))
    def solve_and_get_logdet(self, helpers, left, right):
        """Solve a linear equation (left).T @ D^{-1} @ right.

        This method solves the linear equation (left).T @ D^{-1} @ right, where
        D is the timing-model marginalized covariance matrix.

        NOTE: the left matrix will be transposed for you and should be [N_toa, N]

        Parameters
        ----------
        helpers : tuple
            Tuple ``(nvec, jvec)`` containing the diagonal and ECORR covariance components.
        Fmat : array [total_ntoas, N_basis]
        left : array-like
            The left-hand side of the equation. [Ntoa, n]
        right : array-like
            The right-hand side of the equation. [Ntoa, m]

        Returns
        -------
        array-like
            The result of the linear equation. [n, m] or [m]
        """
        parent = super()
        # Solve L^T N^{-1} R - L^T N^{-1} M (M^T N^{-1} M)^{-1} M^T N^{-1} R
        # Term1 = L^T N^{-1} R
        term1 = parent.solve(helpers, left, right)

        # Term2 = L^T N^{-1} M (M^T N^{-1} M)^{-1} M^T N^{-1} R
        MNM = parent.solve(helpers, self.Mmat, self.Mmat)
        LNM = parent.solve(helpers, left, self.Mmat)
        MNR = parent.solve(helpers, self.Mmat, right)

        # The covariance matrix MNM can be ill-conditioned, we should stabilize it
        if not USE_TM_SVD:
            MNM = stabilize_covariance_matrix(MNM, n=1e-6)
        cf = jsl.cho_factor(MNM)
        term2 = LNM @ jsl.cho_solve(cf, MNR)

        solve_result = term1 - term2
        
        logdet_D = parent.logdet(helpers) + 2 * jnp.sum(jnp.log(cf[0].diagonal())) + self.Mprior 

        return solve_result, logdet_D
        
    # Jax pytree methods -------------------------------------------------------
    def tree_flatten(self):
        """Tree flatten method for JAX pytrees.

        This method defines how to flatten the TOA_Covariance object into
        its tracable components (children) and auxiliary components (aux_data) 
        for JAX pytrees.

        Returns
        -------
        tuple
            A tuple containing:
            - children: a tuple of the tracable attributes.
            - aux_data: a tuple of the auxiliary attributes.
        """
        # Children are dynamic, traceable attributes, and arrays.
        children = (self.nvec, self.jvec, # Dynamic attributes
                    self.toaerrs, self.B, self.U_pad, self.U_mask, self.V, self.Mmat) # Static arrays
        # Aux data is static attributes not used for computations
        aux_data = (self.psr_name, self.ntoas, self.nepochs, self.backends)
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """The tree unflatten method for JAX pytrees.

        This method defines how to reconstruct the TOA_Covariance object from
        its children and aux_data for JAX pytrees.

        Parameters
        ----------
        aux_data : tuple
            The static attributes that were stored in aux_data during flattening.
        children : tuple
            The tracable attributes that were stored in children during flattening.

        Returns
        -------
        Marg_TM_TOA_cov_full
            The reconstructed Marg_TM_TOA_cov_full object.
        """
        # Reconstruct the object from children and aux data
        (psr_name, ntoas, nepochs, backends) = aux_data
        (nvec, jvec, toaerrs, B, U_pad, U_mask, V, Mmat) = children

        obj = cls.__new__(cls) # Create an uninitialized instance of the class
        
        # Static attributes ----------------------------------------------------
        obj.psr_name = psr_name
        obj.ntoas = ntoas
        obj.nepochs = nepochs
        obj.toaerrs = toaerrs

        obj.backends = backends
        obj.B = B
        obj.U_pad = U_pad
        obj.U_mask = U_mask
        obj.V = V
        obj.Mmat = Mmat

        # Dynamic attributes (fixed sizes)--------------------------------------
        obj.nvec = nvec
        obj.jvec = jvec
        return obj

def _timing_model_svd(M):
    """Create an more stable basis for the timing model design matrix using SVD.

    This function is used to create a basis U which represents the timing model
    design matrix M and normalizes each of the basis vectors. This can be used
    as a more stable alternative to M in timing model marginalization. 

    This takes an SVD of the design matrix M, and returns only the left singular 
    vectors U as the new basis. This works since during the marginalization process,
    the singular values aren't important when integrating over the whole range of
    timing model coefficients. Likewise, the right singular vectors are only
    used to project into the original basis, which we do not need.

    Parameters
    ----------
    M : array
        The design matrix for the timing model. [ntoas, nparams]

    Returns
    -------
    array
        The left singular vectors of the design matrix M, which can be used as a more
        stable basis for timing model marginalization. [ntoas, nparams]
    """
    U,C,V = jnp.linalg.svd(M, full_matrices=False)
    # Return just the left singular vector.
    # The singular values are a weighting factor that isn't important when marginalizing.
    # the right singular vectors are used to project into the original basis, which isn't 
    # important for marginalization either. 
    return U

