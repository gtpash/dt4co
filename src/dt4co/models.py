import ufl
import dolfin as dl
import numpy as np

class RDTumorVarf:
    """Variational form for the Fisher-KPP equation. Spatially varying reaction-diffusion equation.
    du/dt = div(exp(m1)*grad(u)) + exp(m2)*u*(1 - u) + f
    """
    def __init__(self, dt:float, lumped:bool=False):
        """Constructor for the RDTumorVarf class.

        Args:
            dt (float): Time step.
            lumped (bool, optional): Whether or not to use mass lumping for the reaction term. Defaults to True.
        """
        self._dt = dt
        self.dt_inv = dl.Constant(1./dt)
        if lumped:
            self.dX = ufl.dx(scheme="vertex", metadata={"quadrature_degree":1, "representation":"quadrature"})
        else:
            self.dX = ufl.dx

    @property
    def dt(self):
        return self._dt
    
    def diffusion(self):
        D = ufl.exp(self.d)
        return D*ufl.inner(ufl.grad(self.u), ufl.grad(self.p))*ufl.dx
    
    def reaction(self):
        kappa = ufl.exp(self.k)
        return kappa*self.u*(dl.Constant(1.) - self.u)*self.p*self.dX

    def __call__(self, u, u_old, m, p, t):
        self.u = u
        self.p = p
        self.m = m
        self.d, self.k = ufl.split(self.m)
        
        return (u - u_old)*p*self.dt_inv*self.dX \
                + self.diffusion() \
                - self.reaction()


class mollifiedRDTumorVarf(RDTumorVarf):
    """Variational form for the Fisher-KPP equation. Spatially varying reaction-diffusion equation.
    du/dt = div(exp(m1)*grad(u)) + exp(m2)*u*(1 - u) + f
    """
    def __init__(self, dt:float, lumped:bool=False, nudge:float=1e-14, quad_degree=5):
        """Constructor.

        Args:
            dt (float): Time step [days].
            lumped (bool, optional): Whether or not to use mass lumping. Defaults to False.
            nudge (float, optional): Nudge parameter for mollification. Defaults to 1e-14.
            quad_degree (int, optional): Quadrature degree for integration. Defaults to 5.
        """
        super().__init__(dt, lumped)
        self.nudge = nudge
        self.dX = ufl.dx(metadata={"quadrature_degree": quad_degree})
        
    def diffusion(self):
        D = ufl.exp(self.d)
        return D*ufl.inner(ufl.grad(self.u), ufl.grad(self.p))*self.dX
        
    def reaction(self):
        """Mollified reaction term.
        """
        kappa = ufl.exp(self.k)
        # moll = ufl.max_value(self.u, dl.Constant(0.))
        moll = ( self.u + ufl.sqrt(self.u**2 + dl.Constant(self.nudge)) ) / 2
        return kappa*moll*(dl.Constant(1.) - self.u)*self.p*self.dX

    def __call__(self, u, u_old, m, p, t):
        self.u = u
        self.p = p
        self.m = m
        self.d, self.k = ufl.split(self.m)
        
        return (u - u_old)*p*self.dt_inv*self.dX \
                + self.diffusion() \
                - self.reaction()


class PWRDTumorVarf:
    """Variational form for the Fisher-KPP equation. Spatially varying reaction-diffusion equation.
    du/dt = div(exp(m_d)*grad(u)) + exp(m_k)*u*(1 - u)
    
    where m_d = mg + mw, defined on gray and white matter, respectively.
    The diffusion term is tied to the underlying tissue (piece-wise).
    """
    def __init__(self, dt:float, chi_gm: dl.Function):
        """Constructor

        Args:
            dt (float): Time step.
            chi_gm (dl.Function): DG0 indicator function for tissue type (1 for gray-matter, 0 for white-matter).
        """
        self._dt = dt
        self.dt_inv = dl.Constant(1./dt)
        
        self.chi_gm = chi_gm    # DG0 indicator function for tissue type (1 for gray-matter, 0 for white-matter)
        self.dX = ufl.dx

    @property
    def dt(self):
        return self._dt
    
    def diffusion(self):
        D = ufl.exp(self.dg)*self.chi_gm + ufl.exp(self.dw)*(dl.Constant(1.) - self.chi_gm)
        return D*ufl.inner(ufl.grad(self.u), ufl.grad(self.p))*self.dX
    
    def reaction(self):
        kappa = ufl.exp(self.k)
        return kappa*self.u*(dl.Constant(1.) - self.u)*self.p*self.dX
        
    def __call__(self, u, u_old, m, p, t):
        self.u = u
        self.p = p
        self.m = m
        self.t = t
        self.dg, self.dw, self.k = ufl.split(self.m)
        
        # be careful with the signs, we are in residual form LHS = 0
        return (u - u_old)*p*self.dt_inv*self.dX \
                + self.diffusion() \
                - self.reaction() \


class mollifiedPWRDTumorVarf(PWRDTumorVarf):
    """Variational form for the Fisher-KPP equation. Spatially varying reaction-diffusion equation.
    du/dt = div(exp(m_d)*grad(u)) + exp(m_k)*u*(1 - u) + radio + chemo
    
    where m_d = mg + mw, defined on gray and white matter, respectively.
    The diffusion term is tied to the underlying tissue (piece-wise).
    
    The reaction term is mollified to avoid spurious oscillations.
    """
    def __init__(self, dt:float, chi_gm: dl.Function, nudge:float=1e-14, quad_degree=5):
        """Constructor

        Args:
            dt (float): Time step.
            rtmodel (radioModel): Radiotherapy model.
            ctmodel (chemoModel): Chemotherapy model.
            chi_gm (dl.Function): DG0 indicator function for tissue type (1 for gray-matter, 0 for white-matter).
            nudge (float, optional): Nudge parameter for mollification. Defaults to 1e-14.
            quad_degree (int, optional): Quadrature degree for integration. Defaults to 5.
        """
        super().__init__(dt, chi_gm)
        
        self.nudge = nudge
        self.dX = ufl.dx(metadata={"quadrature_degree": quad_degree})
    
    def reaction(self):
        """Mollified reaction term.
        """
        kappa = ufl.exp(self.k)
        # moll = ufl.max_value(self.u, dl.Constant(0.))
        moll = ( self.u + ufl.sqrt(self.u**2 + dl.Constant(self.nudge)) ) / 2
        return kappa*moll*(dl.Constant(1.) - self.u)*self.p*self.dX

    def __call__(self, u, u_old, m, p, t):
        self.u = u
        self.p = p
        self.m = m
        self.t = t
        self.dg, self.dw, self.k = ufl.split(self.m)
        
        # be careful with the signs, we are in residual form LHS = 0
        return (u - u_old)*p*self.dt_inv*self.dX \
                + self.diffusion() \
                - self.reaction() \


class naiveRTxModel:
    """Class to implement the naive radiotherapy model.
    """
    def __init__(self, tx_days:np.array, tx_doses:np.array, alpha: float=3., alpha_beta_ratio: float=10.):
        self.tx_days = tx_days        # therapy days
        self.tx_doses = tx_doses      # therapy doses
        
        assert len(self.tx_days) == len(self.tx_doses), "Number of therapy days and doses must match."
        
        self.alpha = alpha                      # radiosensitivity parameter alpha
        self.beta = alpha / alpha_beta_ratio    # radiosensitivity parameter beta
        
    def get_tx_factor(self, cur_t: float) -> dl.Constant:
        """Check if the current time is a radiotherapy time.
        If it is, returns the radiotherapy factor.
        
        This assumes an instantaneous effect.
        """
        if (cur_t in self.tx_days):
            # Therapy is applied at this time.
            idx = np.where(self.tx_days == cur_t)[0][0]
            dose = self.tx_doses[idx]
            dose_factor = 1. - np.exp(-self.alpha*dose - self.beta*dose**2)
            # dose_factor = self.alpha*dose + self.beta*dose**2  # Bashkirtseva (2021) model
            return dl.Constant(dose_factor)
        else:
            return dl.Constant(0.)  # no therapy. Radiotherapy effect is zero.
        
    def get_rt_sf(self, cur_t: float) -> dl.Constant:
        """Check if the current time is a radiotherapy time.
        If it is, returns the radiotherapy surviving fraction.
        
        This assumes an instantaneous effect.
        """
        if (cur_t in self.tx_days):
            # Therapy is applied at this time.
            idx = np.where(self.tx_days == cur_t)[0][0]
            dose = self.tx_doses[idx]
            dose_factor = np.exp(-self.alpha*dose - self.beta*dose**2)
            return dl.Constant(dose_factor)
        else:
            return dl.Constant(1.)  # no therapy. Surviving fraction is 1.
        
        
class naiveRTxTumorVarf:
    """Variational form for the Fisher-KPP equation. Spatially varying reaction-diffusion equation.
    du/dt = div(exp(m1)*grad(u)) + exp(m2)*u*(1 - u) + f
    """
    def __init__(self, dt:float, txmodel):
        """Constructor for the RDTumorVarf class.

        Args:
            dt (float): Time step.
        """
        self._dt = dt
        self.dt_inv = dl.Constant(1./dt)
        self.txmodel = txmodel

    @property
    def dt(self):
        return self._dt
    
    def diffusion(self):
        D = ufl.exp(self.d)
        return D*ufl.inner(ufl.grad(self.u), ufl.grad(self.p))*ufl.dx
    
    def reaction(self):
        kappa = ufl.exp(self.k)
        return kappa*self.u*(dl.Constant(1.) - self.u)*self.p*ufl.dx

    def __call__(self, u, u_old, m, p, t):
        self.u = u
        self.p = p
        self.m = m
        self.d, self.k = ufl.split(self.m)
        
        return (u - u_old)*p*self.dt_inv*ufl.dx \
                + self.diffusion() \
                - self.reaction()
    


class radioModel:
    """Class to compute the radiotherapy effect.
    """
    def __init__(self, tx_days:np.array, tx_doses:np.array, alpha: float=3., alpha_beta_ratio: float=10., m_width: float=0.25):
        self.tx_days = tx_days        # therapy days
        self.tx_doses = tx_doses      # therapy doses
        
        assert len(self.tx_days) == len(self.tx_doses), "Number of therapy days and doses must match."
        
        self.alpha = alpha                      # radiosensitivity parameter alpha
        self.beta = alpha / alpha_beta_ratio    # radiosensitivity parameter beta
        self.m_width = m_width                  # mollifier width

    def get_tx_factor(self, cur_t: float) -> dl.Constant:
            """Check if the current time is a radiotherapy time.
            If it is, returns the radiotherapy factor.
            
            This assumes an instantaneous effect.
            """
            if (cur_t in self.tx_days):
                # Therapy is applied at this time.
                idx = np.where(self.tx_days == cur_t)[0][0]  # get the index of the therapy
                dose = self.tx_doses[idx]
                
                # compute LQ model effect
                dose_factor = 1. - np.exp(-self.alpha*dose - self.beta*dose**2)
                return dl.Constant(dose_factor)
            else:
                # no therapy.
                return dl.Constant(0.)
            
            
    def get_mollified_tx_factor(self, cur_t) -> dl.Constant:
            """Check if the current time is a radiotherapy time.
            """
            raise NotImplementedError("Mollified (in time) effects not yet implemented.")
            if np.any(cur_t <= self.tx_days) and np.any(cur_t >= self.tx_days):
                # time is in therapy window
                
                idx = (np.abs(cur_t - self.tx_days)).argmin()  # for getting the dosage
                toff = cur_t - self.tx_days[idx]
                
                if (toff <= self.m_width):
                    # The time is within the (mollified) therapy window.
                    return True
                
                if np.any(cur_t in self.tx_timeline):
                # probably some kind of np.isclose?
                    # todo: check mollifier mollifier here
                    # todo: should do this with sim_times?
                    return True
            else:
                return False


class chemoModel:
    """Class to compute the chemotherapy effect.
    """
    def __init__(self, tx_days:np.array, ct_effect:float, beta:float):
        self.tx_days = tx_days        # therapy days
        self.ct_effect = ct_effect    # chemotherapy effect (surviving fraction)
        self.beta = beta              # clearance rate

    def get_tx_factor(self, cur_t: float) -> dl.Constant:
            """Apply the decaying chemotherapy effect.
            """
            time_since_applied = cur_t - self.tx_days
            
            if np.any(time_since_applied > 0):
                active_time = time_since_applied[time_since_applied >= 0]
                ct_decay = np.exp(-self.beta * active_time)  # clearance term
                ct_factor = (1. - self.ct_effect) * np.sum(ct_decay)  # 1 - surviving fraction
                return dl.Constant(ct_factor)
            else:
                return dl.Constant(0.)
            
    def get_mollified_tx_factor(self, cur_t) -> dl.Constant:
            """Check if the current time is a radiotherapy time.
            """
            raise NotImplementedError("Mollified effects not yet implemented.")


class RDchemoTumorVarf:
    """Variational form for the Fisher-KPP equation. Spatially varying reaction-diffusion equation.
    du/dt = div(exp(m1)*grad(u)) + exp(m2)*u*(1 - u) + Tx
    """
    def __init__(self, dt:float, txmodel: chemoModel):
        """Constructor for the RDTumorVarf class.

        Args:
            dt (float): Time step.
            
        """
        self._dt = dt
        self.dt_inv = dl.Constant(1./dt)
        self.txmodel = txmodel

    @property
    def dt(self):
        return self._dt
    
    def diffusion(self):
        D = ufl.exp(self.d)
        return D*ufl.inner(ufl.grad(self.u), ufl.grad(self.p))*ufl.dx
    
    def reaction(self):
        kappa = ufl.exp(self.k)
        return kappa*self.u*(dl.Constant(1.) - self.u)*self.p*ufl.dx
    
    def chemo(self, t):
        """Return the radiotherapy effect.
        """
        cteffect = dl.Constant(self.txmodel.get_tx_factor(t))  # don't need the 1/dt because chemo is already decayed
        return self.dt_inv*cteffect*self.u*self.p*ufl.dx
        

    def __call__(self, u, u_old, m, p, t):
        self.u = u
        self.p = p
        self.m = m
        self.t = t
        self.d, self.k = ufl.split(self.m)
        
        # be careful with the signs, we are in residual form LHS = 0
        return (u - u_old)*p*self.dt_inv*ufl.dx \
                + self.diffusion() \
                - self.reaction() \
                + self.chemo(t)


class RDradioTumorVarf:
    """Variational form for the Fisher-KPP equation. Spatially varying reaction-diffusion equation.
    du/dt = div(exp(m1)*grad(u)) + exp(m2)*u*(1 - u) + Tx
    """
    def __init__(self, dt:float, txmodel: radioModel):
        """Constructor for the RDTumorVarf class.

        Args:
            dt (float): Time step.
            
        """
        self._dt = dt
        self.dt_inv = dl.Constant(1./dt)
        self.txmodel = txmodel

    @property
    def dt(self):
        return self._dt
    
    def diffusion(self):
        D = ufl.exp(self.d)
        return D*ufl.inner(ufl.grad(self.u), ufl.grad(self.p))*ufl.dx
    
    def reaction(self):
        kappa = ufl.exp(self.k)
        return kappa*self.u*(dl.Constant(1.) - self.u)*self.p*ufl.dx
    
    def tx(self, t):
        """Return the radiotherapy effect.
        """
        rteffect = dl.Constant(self.txmodel.get_tx_factor(t))
        return self.dt_inv*rteffect*self.u*self.p*ufl.dx
        

    def __call__(self, u, u_old, m, p, t):
        self.u = u
        self.p = p
        self.m = m
        self.t = t
        self.d, self.k = ufl.split(self.m)
        
        # be careful with the signs, we are in residual form LHS = 0
        return (u - u_old)*p*self.dt_inv*ufl.dx \
                + self.diffusion() \
                - self.reaction() \
                + self.tx(t)

                            
class RDTXTumorVarf:
    """Variational form for the Fisher-KPP equation. Spatially varying reaction-diffusion equation.
    du/dt = div(exp(m1)*grad(u)) + exp(m2)*u*(1 - u) + radio + chemo
    """
    def __init__(self, dt:float, rtmodel: radioModel, ctmodel: chemoModel):
        """Constructor

        Args:
            dt (float): Time step.
            rtmodel (radioModel): Radiotherapy model.
            ctmodel (chemoModel): Chemotherapy model.
        """
        self._dt = dt
        self.dt_inv = dl.Constant(1./dt)
        self.rtmodel = rtmodel  # radiotherapy model
        self.ctmodel = ctmodel  # chemotherapy model

    @property
    def dt(self):
        return self._dt
    
    def diffusion(self):
        D = ufl.exp(self.d)
        return D*ufl.inner(ufl.grad(self.u), ufl.grad(self.p))*ufl.dx
    
    def reaction(self):
        kappa = ufl.exp(self.k)
        return kappa*self.u*(dl.Constant(1.) - self.u)*self.p*ufl.dx
    
    def radio(self, t):
        """Return the radiotherapy effect.
        """
        rteffect = dl.Constant(self.rtmodel.get_tx_factor(t))
        return self.dt_inv*rteffect*self.u*self.p*ufl.dx
    
    def chemo(self, t):
        """Return the radiotherapy effect.
        """
        cteffect = dl.Constant(self.ctmodel.get_tx_factor(t))
        return self.dt_inv*cteffect*self.u*self.p*ufl.dx
        

    def __call__(self, u, u_old, m, p, t):
        self.u = u
        self.p = p
        self.m = m
        self.t = t
        self.d, self.k = ufl.split(self.m)
        
        # be careful with the signs, we are in residual form LHS = 0
        return (u - u_old)*p*self.dt_inv*ufl.dx \
                + self.diffusion() \
                - self.reaction() \
                + self.radio(t) \
                + self.chemo(t)


class mollifiedRDTXTumorVarf(RDTXTumorVarf):
    """Variational form for the Fisher-KPP equation. Spatially varying reaction-diffusion equation.
    du/dt = div(exp(m1)*grad(u)) + exp(m2)*u*(1 - u) + radio + chemo
    
    The reaction term is mollified to avoid spurious oscillations.
    """
    def __init__(self, dt:float, rtmodel: radioModel, ctmodel: chemoModel, nudge:float=1e-14, quad_degree=5):
        """Constructor

        Args:
            dt (float): Time step.
            rtmodel (radioModel): Radiotherapy model.
            ctmodel (chemoModel): Chemotherapy model.
            nudge (float, optional): Nudge parameter for mollification. Defaults to 1e-14.
            quad_degree (int, optional): Quadrature degree for integration. Defaults to 5.
        """
        super().__init__(dt, rtmodel, ctmodel)
        
        self.nudge = nudge
        self.dX = ufl.dx(metadata={"quadrature_degree": quad_degree})
    
    def diffusion(self):
        D = ufl.exp(self.d)
        return D*ufl.inner(ufl.grad(self.u), ufl.grad(self.p))*self.dX
    
    def reaction(self):
        """Mollified reaction term.
        """
        kappa = ufl.exp(self.k)
        # moll = ufl.max_value(self.u, dl.Constant(0.))
        moll = ( self.u + ufl.sqrt(self.u**2 + dl.Constant(self.nudge)) ) / 2
        return kappa*moll*(dl.Constant(1.) - self.u)*self.p*self.dX
    
    def radio(self, t):
        """Return the radiotherapy effect.
        """
        rteffect = dl.Constant(self.rtmodel.get_tx_factor(t))
        return self.dt_inv*rteffect*self.u*self.p*self.dX
    
    def chemo(self, t):
        """Return the radiotherapy effect.
        """
        cteffect = dl.Constant(self.ctmodel.get_tx_factor(t))
        return self.dt_inv*cteffect*self.u*self.p*self.dX
        

    def __call__(self, u, u_old, m, p, t):
        self.u = u
        self.p = p
        self.m = m
        self.t = t
        self.d, self.k = ufl.split(self.m)
        
        # be careful with the signs, we are in residual form LHS = 0
        return (u - u_old)*p*self.dt_inv*self.dX \
                + self.diffusion() \
                - self.reaction() \
                + self.radio(t) \
                + self.chemo(t)


class PWRDTXTumorVarf:
    """Variational form for the Fisher-KPP equation. Spatially varying reaction-diffusion equation.
    du/dt = div(exp(m_d)*grad(u)) + exp(m_k)*u*(1 - u) + radio + chemo
    
    where m_d = mg + mw, defined on gray and white matter, respectively.
    The diffusion term is tied to the underlying tissue (piece-wise).
    """
    def __init__(self, dt:float, rtmodel: radioModel, ctmodel: chemoModel, chi_gm: dl.Function):
        """Constructor

        Args:
            dt (float): Time step.
            rtmodel (radioModel): Radiotherapy model.
            ctmodel (chemoModel): Chemotherapy model.
            chi_gm (dl.Function): DG0 indicator function for tissue type (1 for gray-matter, 0 for white-matter).
        """
        self._dt = dt
        self.dt_inv = dl.Constant(1./dt)
        
        self.rtmodel = rtmodel  # radiotherapy model
        self.ctmodel = ctmodel  # chemotherapy model
        self.chi_gm = chi_gm    # DG0 indicator function for tissue type (1 for gray-matter, 0 for white-matter)
        self.dX = ufl.dx

    @property
    def dt(self):
        return self._dt
    
    def diffusion(self):
        D = ufl.exp(self.dg)*self.chi_gm + ufl.exp(self.dw)*(dl.Constant(1.) - self.chi_gm)
        return D*ufl.inner(ufl.grad(self.u), ufl.grad(self.p))*self.dX
    
    def reaction(self):
        kappa = ufl.exp(self.k)
        return kappa*self.u*(dl.Constant(1.) - self.u)*self.p*self.dX
    
    def radio(self, t):
        """Return the radiotherapy effect.
        """
        rteffect = dl.Constant(self.rtmodel.get_tx_factor(t))
        return self.dt_inv*rteffect*self.u*self.p*self.dX
    
    def chemo(self, t):
        """Return the radiotherapy effect.
        """
        cteffect = dl.Constant(self.ctmodel.get_tx_factor(t))
        return self.dt_inv*cteffect*self.u*self.p*self.dX
        

    def __call__(self, u, u_old, m, p, t):
        self.u = u
        self.p = p
        self.m = m
        self.t = t
        self.dg, self.dw, self.k = ufl.split(self.m)
        
        # be careful with the signs, we are in residual form LHS = 0
        return (u - u_old)*p*self.dt_inv*self.dX \
                + self.diffusion() \
                - self.reaction() \
                + self.radio(t) \
                + self.chemo(t)


class mollifiedPWRDTXTumorVarf(PWRDTXTumorVarf):
    """Variational form for the Fisher-KPP equation. Spatially varying reaction-diffusion equation.
    du/dt = div(exp(m_d)*grad(u)) + exp(m_k)*u*(1 - u) + radio + chemo
    
    where m_d = mg + mw, defined on gray and white matter, respectively.
    The diffusion term is tied to the underlying tissue (piece-wise).
    
    The reaction term is mollified to avoid spurious oscillations.
    """
    def __init__(self, dt:float, rtmodel: radioModel, ctmodel: chemoModel, chi_gm: dl.Function, nudge:float=1e-14, quad_degree=5):
        """Constructor

        Args:
            dt (float): Time step.
            rtmodel (radioModel): Radiotherapy model.
            ctmodel (chemoModel): Chemotherapy model.
            chi_gm (dl.Function): DG0 indicator function for tissue type (1 for gray-matter, 0 for white-matter).
            nudge (float, optional): Nudge parameter for mollification. Defaults to 1e-14.
            quad_degree (int, optional): Quadrature degree for integration. Defaults to 5.
        """
        super().__init__(dt, rtmodel, ctmodel, chi_gm)
        
        self.nudge = nudge
        self.dX = ufl.dx(metadata={"quadrature_degree": quad_degree})
    
    def reaction(self):
        """Mollified reaction term.
        """
        kappa = ufl.exp(self.k)
        # moll = ufl.max_value(self.u, dl.Constant(0.))
        moll = ( self.u + ufl.sqrt(self.u**2 + dl.Constant(self.nudge)) ) / 2
        return kappa*moll*(dl.Constant(1.) - self.u)*self.p*self.dX

    def __call__(self, u, u_old, m, p, t):
        self.u = u
        self.p = p
        self.m = m
        self.t = t
        self.dg, self.dw, self.k = ufl.split(self.m)
        
        # be careful with the signs, we are in residual form LHS = 0
        return (u - u_old)*p*self.dt_inv*self.dX \
                + self.diffusion() \
                - self.reaction() \
                + self.radio(t) \
                + self.chemo(t)


# ------------------------------------------------------------
# Centered models (parameter doesn't include mean).
# ------------------------------------------------------------
class CenteredPWRDTXTumorVarf:
    """Variational form for the Fisher-KPP equation. Spatially varying reaction-diffusion equation.
    du/dt = div(exp(m_d)*grad(u)) + exp(m_k)*u*(1 - u) + radio + chemo
    
    where m_d = mg + mw, defined on gray and white matter, respectively.
    The diffusion term is tied to the underlying tissue (piece-wise).
    """
    def __init__(self, dt:float, mdg0:float, mdw0:float, mk0:float, rtmodel: radioModel, ctmodel: chemoModel, chi_gm: dl.Function):
        """Constructor

        Args:
            dt (float): Time step.
            mdg0 (float): Mean of the diffusion parameter for gray matter.
            mdw0 (float): Mean of the diffusion parameter for white matter.
            mk0 (float): Mean of the reaction parameter.
            rtmodel (radioModel): Radiotherapy model.
            ctmodel (chemoModel): Chemotherapy model.
            chi_gm (dl.Function): DG0 indicator function for tissue type (1 for gray-matter, 0 for white-matter).
        """
        self._dt = dt
        self.dt_inv = dl.Constant(1./dt)
        
        self.mdg0 = mdg0
        self.mdw0 = mdw0
        self.mk0 = mk0
        
        self.rtmodel = rtmodel  # radiotherapy model
        self.ctmodel = ctmodel  # chemotherapy model
        self.chi_gm = chi_gm    # DG0 indicator function for tissue type (1 for gray-matter, 0 for white-matter)
        self.dX = ufl.dx

    @property
    def dt(self):
        return self._dt
    
    def diffusion(self):
        D = self.mdg0*ufl.exp(self.dg)*self.chi_gm + self.mdw0*ufl.exp(self.dw)*(dl.Constant(1.) - self.chi_gm)
        return D*ufl.inner(ufl.grad(self.u), ufl.grad(self.p))*self.dX
    
    def reaction(self):
        kappa = ufl.exp(self.k)*self.mk0
        return kappa*self.u*(dl.Constant(1.) - self.u)*self.p*self.dX
    
    def radio(self, t):
        """Return the radiotherapy effect.
        """
        rteffect = dl.Constant(self.rtmodel.get_tx_factor(t))
        return self.dt_inv*rteffect*self.u*self.p*self.dX
    
    def chemo(self, t):
        """Return the radiotherapy effect.
        """
        cteffect = dl.Constant(self.ctmodel.get_tx_factor(t))
        return self.dt_inv*cteffect*self.u*self.p*self.dX
        

    def __call__(self, u, u_old, m, p, t):
        self.u = u
        self.p = p
        self.m = m
        self.t = t
        self.dg, self.dw, self.k = ufl.split(self.m)
        
        # be careful with the signs, we are in residual form LHS = 0
        return (u - u_old)*p*self.dt_inv*self.dX \
                + self.diffusion() \
                - self.reaction() \
                + self.radio(t) \
                + self.chemo(t)


class mollifiedCenteredPWRDTXTumorVarf(CenteredPWRDTXTumorVarf):
    """Variational form for the Fisher-KPP equation. Spatially varying reaction-diffusion equation.
    du/dt = div(exp(m_d)*grad(u)) + exp(m_k)*u*(1 - u) + radio + chemo
    
    where m_d = mg + mw, defined on gray and white matter, respectively.
    The diffusion term is tied to the underlying tissue (piece-wise).
    
    The reaction term is mollified to avoid spurious oscillations.
    """
    def __init__(self, dt:float, mdg0:float, mdw0:float, mk0:float, rtmodel: radioModel, ctmodel: chemoModel, chi_gm: dl.Function, nudge:float=1e-14, quad_degree=5):
        """Constructor

        Args:
            dt (float): Time step.
            mdg0 (float): Mean of the diffusion parameter for gray matter.
            mdw0 (float): Mean of the diffusion parameter for white matter.
            mk0 (float): Mean of the reaction parameter.
            rtmodel (radioModel): Radiotherapy model.
            ctmodel (chemoModel): Chemotherapy model.
            chi_gm (dl.Function): DG0 indicator function for tissue type (1 for gray-matter, 0 for white-matter).
            nudge (float, optional): Nudge parameter for mollification. Defaults to 1e-14.
            quad_degree (int, optional): Quadrature degree for integration. Defaults to 5.
        """
        super().__init__(dt, mdg0, mdw0, mk0, rtmodel, ctmodel, chi_gm)
        
        self.nudge = nudge
        self.dX = ufl.dx(metadata={"quadrature_degree": quad_degree})
    
    def reaction(self):
        """Mollified reaction term.
        """
        kappa = ufl.exp(self.k)*ufl.exp(self.mk0)
        # moll = ufl.max_value(self.u, dl.Constant(0.))
        moll = ( self.u + ufl.sqrt(self.u**2 + dl.Constant(self.nudge)) ) / 2
        return kappa*moll*(dl.Constant(1.) - self.u)*self.p*self.dX

    def __call__(self, u, u_old, m, p, t):
        self.u = u
        self.p = p
        self.m = m
        self.t = t
        self.dg, self.dw, self.k = ufl.split(self.m)
        
        # be careful with the signs, we are in residual form LHS = 0
        return (u - u_old)*p*self.dt_inv*self.dX \
                + self.diffusion() \
                - self.reaction() \
                + self.radio(t) \
                + self.chemo(t)


class DiffusionTumorVarf:
    """Variational form for the Fisher-KPP equation. Spatially varying reaction-diffusion equation.
    du/dt = div(exp(m)*grad(u)) + k*u*(1 - u) + f
    """
    def __init__(self, dt:float, f, kappa:float, lumped:bool=True):
        """Constructor for the RDTumorVarf class.

        Args:
            dt (float): Time step.
            f : Forcing function. Constant or dolfin.Function.
            lumped (bool, optional): Whether or not to use mass lumping. Defaults to True.
            diff_only (bool, optional): Whether the diffusion parameter should be the modeled parameter. Defaults to True.
        """
        self._dt = dt
        self.dt_inv = dl.Constant(1./dt)
        self.f = f
        self.kappa = kappa
        
        if lumped:
            self.dX = ufl.dx(scheme="vertex", metadata={"quadrature_degree":1, "representation":"quadrature"})
        else:
            self.dX = ufl.dx

    @property
    def dt(self):
        return self._dt
    
    def diffusion(self):
        D = ufl.exp(self.m)
        return D*ufl.inner(ufl.grad(self.u), ufl.grad(self.p))*ufl.dx
    
    def reaction(self):
        return self.kappa*self.u*(dl.Constant(1.) - self.u)*self.p*self.dX

    def __call__(self, u, u_old, m, p, t):
        self.u = u
        self.p = p
        self.m = m
        
        return (u - u_old)*p*self.dt_inv*self.dX \
                - self.f*p*ufl.dx \
                + self.diffusion() \
                - self.reaction()


class ReactionTumorVarf:
    """Variational form for the Fisher-KPP equation. Spatially varying reaction-diffusion equation.
    du/dt = div(exp(m)*grad(u)) + k*u*(1 - u) + f
    """
    def __init__(self, dt:float, f, d:float, lumped:bool=True):
        """Constructor for the RDTumorVarf class.

        Args:
            dt (float): Time step.
            f : Forcing function. Constant or dolfin.Function.
            lumped (bool, optional): Whether or not to use mass lumping. Defaults to True.
            diff_only (bool, optional): Whether the diffusion parameter should be the modeled parameter. Defaults to True.
        """
        self._dt = dt
        self.dt_inv = dl.Constant(1./dt)
        self.f = f
        self.d = d
        
        if lumped:
            self.dX = ufl.dx(scheme="vertex", metadata={"quadrature_degree":1, "representation":"quadrature"})
        else:
            self.dX = ufl.dx

    @property
    def dt(self):
        return self._dt
    
    def diffusion(self):
        return self.d*ufl.inner(ufl.grad(self.u), ufl.grad(self.p))*ufl.dx
    
    def reaction(self):
        K = ufl.exp(self.m)
        return K*self.u*(dl.Constant(1.) - self.u)*self.p*self.dX

    def __call__(self, u, u_old, m, p, t):
        self.u = u
        self.p = p
        self.m = m
        
        return (u - u_old)*p*self.dt_inv*self.dX \
                - self.f*p*ufl.dx \
                + self.diffusion() \
                - self.reaction()


class LinearizedRDTumorVarf:
    """Variational form for the Fisher-KPP equation. Spatially varying reaction-diffusion equation.
    du/dt = div(D0*exp(D)*grad(u)) + kappa0*exp(kappa)*u*(1 - u) + f
    """
    def __init__(self, dt, f, lumped=True):
        self._dt = dt
        self.dt_inv = dl.Constant(1./dt)
        self.f = f
        
        if lumped:
            self.dX = ufl.dx(scheme="vertex", metadata={"quadrature_degree":1, "representation":"quadrature"})
        else:
            self.dX = ufl.dx

    @property
    def dt(self):
        return self._dt
    
    def diffusion(self):
        # D = self.md*ufl.exp(self.m.sub(0))
        D = ufl.exp(self.m.sub(0))
        return D*ufl.inner(ufl.grad(self.u), ufl.grad(self.p))*ufl.dx
    
    def reaction(self, u_old):
        # kappa = self.mk*ufl.exp(self.m.sub(1))
        kappa = ufl.exp(self.m.sub(1))
        return kappa*u_old*(dl.Constant(1.) - self.u)*self.p*ufl.dX

    def __call__(self, u, u_old, m, p, t):
        self.u = u
        self.p = p
        self.m = m
        
        return (u - u_old)*p*self.dt_inv*ufl.dX \
                - self.f*p*ufl.dx \
                + self.diffusion() \
                - self.reaction(u_old)


class HomogeneousMCRDTumorVarf:
    """Variational form for the mechanically-coupled reaction-diffusion tumor growth equation.
    # TODO: add more description.
    """
    def __init__(self, dt, f, mesh, lmbdaf, D0, gamma, Ew, Eg, nu, chi, dim=3):
        self._dt = dt
        self.dt_inv = dl.Constant(1./dt)
        self.f = f
        self.chi = chi
        self.mesh = mesh
        self.lmbdaf = lmbdaf
        self.D0 = D0
        self.gamma = gamma
        self.dim = dim
        
        # Material properties
        self.nu = nu
        self.mu = (Ew*self.chi + Eg*(1. - self.chi)) / (2*(1 + self.nu))  # shear modulus
        self.lmbda = (Ew*self.chi + Eg*(1. - self.chi))*self.nu / ((1 + self.nu)*(1 - 2*self.nu))  # first lame parameter

    @property
    def dt(self):
        return self._dt
    
    def diffusion(self, D):
        return D*ufl.inner(ufl.grad(self.u), ufl.grad(self.p))*ufl.dx
    
    def reaction(self):
        kappa = ufl.exp(self.m)
        return kappa*self.u*(dl.Constant(1.) - self.u)*self.p*ufl.dx

    def epsilon(self, d):
        # return 0.5*(ufl.nabla_grad(d) + ufl.nabla_grad(d).T)
        return ufl.sym(ufl.grad(d))
    
    def sigma(self, d):
        return 2*self.mu*self.epsilon(d) + self.lmbda*ufl.tr(self.epsilon(d))*ufl.Identity(self.dim)
    
    def stressDeviator(self, d):
        return self.sigma(d) - (1./3)*ufl.tr(self.sigma(d))*ufl.Identity(self.dim)
    
    def vonMises(self, d):
        return ufl.sqrt(3./2*ufl.inner(self.stressDeviator(d), self.stressDeviator(d)))
    
    def compute_diffusivity(self, u):
        V = dl.VectorFunctionSpace(self.mesh, "CG", 1)
        u_ = dl.TrialFunction(V)
        v_ = dl.TestFunction(V)
        
        def boundary(x, on_boundary):
            return on_boundary
        
        bc = dl.DirichletBC(V, dl.Constant((0., 0., 0.)), boundary)  # zero displacement on the boundary
        
        # bc = []  # no traction on the boundary
        
        # Solve for displacement.
        a = ufl.inner(self.sigma(u_), self.epsilon(v_))*ufl.dx  #todo: this can be preassembled
        L = ufl.dot(self.lmbdaf*ufl.nabla_grad(u), v_)*ufl.dx
        
        d = dl.Function(V)
        dl.solve(a == L, d, bc, solver_parameters={'linear_solver' : 'mumps'})  #todo: mumps is probably a bad idea in 3D
        
        # Compute Von Mises stress, return diffusivity.
        vm = self.vonMises(d)
        return self.D0 * ufl.exp(-self.gamma*vm)

    def __call__(self, u, u_old, m, p, t):
        self.u = u
        self.p = p
        self.m = m
        
        D = self.compute_diffusivity(u_old)
        
        return (u - u_old)*p*self.dt_inv*ufl.dx \
                - self.f*p*ufl.dx \
                + self.diffusion(D) \
                - self.reaction()


# todo: implement this class according to Umbe's suggestions
class HeterogeneousMCRDTumorVarf:
    pass

# -------------------------------------
class TDRealizationTracer(object):
    """Class to store realizations from a hIPPYlib MCMC run.
    """
    def __init__(self, n, tf, par_mv, state_mv):
        self.data = np.zeros((n,2))
        self.i = 0
        self.tf = tf
        self.par_mv = par_mv
        self.state_mv = state_mv
        
    def append(self,current, q):
        self.data[self.i, 0] = q
        self.data[self.i, 1] = current.cost
        
        # append the parameter and state to the respective multi-vectors
        self.par_mv[self.i].axpy(1., current.m)
        self.state_mv[self.i].axpy(1., current.u.view(self.tf))
        
        self.i+=1


# -------------------------------------
class gw_diffusion(dl.UserExpression):
    """Class implementing a two tissue heterogeneous diffusion (gray/white matter)
    """
    def __init__(self, tissues: dl.MeshFunction, k_g: float, k_w: float, **kwargs):
        """
        Args:
            tissues (dl.MeshFunction): mesh funciton specifying tissue types
            k_g (float): gray matter diffusion coefficient.
            k_w (float): white matter diffusion coefficient.
        """
        super().__init__(**kwargs)
        self.tissues = tissues
        self.k_g = k_g
        self.k_w = k_w

    def eval_cell(self, values, x, cell):
        # gray is tagged 1, white is tagged 2
        if self.tissues[cell.index] == 1:
            values[0] = self.k_g
        else:
            values[0] = self.k_w

    def value_shape(self):
        return ()


class tumor_(dl.UserExpression):
    """Class implementing a piecewise function.
    """
    def __init__(self, tissues: dl.MeshFunction, k_e: float, k_ne: float, **kwargs):
        """
        Args:
            tissues (dl.MeshFunction): mesh funciton specifying tissue types
            k_e (float): enhancing tumor coefficient.
            k_ne (float): non-enhancing tumor coefficient.
        """
        super().__init__(**kwargs)
        self.tissues = tissues
        self.k_e = k_e
        self.k_ne = k_ne

    def eval_cell(self, values, x, cell):
        # gray is tagged 1, white is tagged 2
        if self.tissues[cell.index] == 4:
            values[0] = self.k_e
        else:
            values[0] = self.k_ne

    def value_shape(self):
        return ()

