#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import random
import math
import copy
import time
import numpy as np
import pathlib


# In[2]:


import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import dolfin as dl
import hippylib as hl


# In[3]:


import scipy.stats as sstats
import scipy.io as sio
import scipy.integrate as sint
from scipy.integrate import solve_ivp
import scipy.stats as sstats
from scipy.linalg import expm


# In[4]:


# for plotting
import matplotlib
# matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set()
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=4, rc={"lines.linewidth": 4})
mpl.rcParams['lines.linewidth'] = 3
plt.rcParams['axes.unicode_minus'] = False


# In[5]:


# needed to read vtu files
import vtk
from vtk.util import numpy_support as VN


# In[6]:


# needed for sensitivity analysis
from SALib.analyze import morris
from SALib.analyze import sobol
import SALib.sample as sample
from SALib.sample import saltelli
from SALib.plotting.morris import horizontal_bar_plot, covariance_plot, sample_histograms


# In[7]:


dl.parameters["form_compiler"]["optimize"]     = True
dl.parameters["form_compiler"]["cpp_optimize"] = True
dl.parameters["form_compiler"]["representation"] = "uflacs"
dl.parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"


# In[8]:


# dl.PETScOptions.set("ksp_type", "gmres")
# # PETScOptions.set("ksp_rtol", 1e-12)
# # PETScOptions.set("pc_type", "ilu")
# #PETScOptions.set("snes_type", "ngmres")
# #PETScOptions.set("snes_rtol", 1e-8)
# dl.PETScOptions.set("pc_type", "bjacobi")
# dl.PETScOptions.set("sub_pc_type", "ilu")
# dl.PETScOptions.set("sub_pc_factor_levels", 10)
# #PETScOptions.set("pc_factor_mat_solver_type", "mumps")
# #PETScOptions.set("snes_atol", 1e-8)
# # PETScOptions.set("sub_pc_factor_shift_type", PETSc.POSITIVE_DEFINITE)
# # PETScOptions.set("ksp_error_if_not_converged")
# # dl.PETScOptions.set("error_on_nonconvergence", False)
dl.PETScOptions.set("ksp_monitor", False)


# # Diffusion-reaction HP MRI model with vascular domain

# In[9]:


from PDE_model_help import GenericReprBase, NutSource, NutSourceLeftEdge, QoIFn, IC_Tum, TC_Tum, IC_Error_Tum, TC_Error_Tum


# # In[10]:


# from PDE_model_help import Misfit, FullTracer, plot_prior


# In[15]:


STATE = 0
PARAMETER = 1

class PDEModel(GenericReprBase):
    'Implement tumor ode model'
    
    def __init__(self, Vh, dt = 0.01, \
                 theta0 = [0.2, 0.1, 0.05], \
                 tum_ic = 0.25, \
                 save = False, out_path = '/tmp/fwd_result/', \
                 sim_out_freq = 10, sim_max_out = 20, \
                 tol = 1.0e-12, max_iter = 100, verbosity = 1, print_freq = 1):
        
        # some params for solveFwd called by HippyLib
        self.use_approx = True # solve for errors to get exact solution
        self.which_est = 0 # if 0, use Q(u) - Q(u0), else use one of the estimates for QoI error
        
        ## Vh groups parameter space and function space
        self.Vh = Vh
        
        ## time and domain
        self.t = 0.
        self.tF = 1.
        self.L = 1.
        self.vol = 1.
        self.dt = dt
        self.Nsteps = int(self.tF / self.dt)
        
        ## output related parameters
        self.save = save
        self.out_path = out_path
        pathlib.Path(self.out_path).mkdir(parents=True, exist_ok=True)
        
        # store LF solutions
        self.lf_sim_out_path = None
        self.lf_sim_partial_out_path = 'lf_sim'
        
        self.sim_out_path = None
        self.cur_sim_out_path = None
        self.cur_sim_partial_out_path = 'cur_sim'
        self.cur_err_sim_partial_out_path = 'cur_err_sim'
        self.save_suf = ''
        self.file = None

        self.solveFwdBck_save_err = False
        
        self.sim_out_freq = sim_out_freq  # output 1 every 10 simulation
        self.sim_max_out = sim_max_out    # do not produce results for sims more than this
        self.sim_out_current = 0
        
        # verbosity 
        self.verbosity = verbosity
        self.sim_count = 0
        self.print_freq = print_freq
        
        ## parameters
        self.param_dim = Vh[PARAMETER].dim()
        # lam_p, lam_d, eps, C
        self.par_m = dl.Function(Vh[PARAMETER]) 
        self.par_lam_p = dl.Constant(math.log(0.5))
        self.par_lam_d = dl.Constant(math.log(0.1))
        self.par_C = dl.Constant(math.log(1.))
        self.par_eps = dl.Constant(math.log(0.01))
        
        # low-fidelity
        self.lf_param_dim = 3
        # lam_p, lam_d, D
        self.lf_par_real_vec = theta0
        self.lf_par_lam_p = dl.Constant(math.log(theta0[0]))
        self.lf_par_lam_d = dl.Constant(math.log(theta0[1]))
        self.lf_par_D = dl.Constant(math.log(theta0[2]))
        
        ## initial condition valus
        self.tum_ic = tum_ic
        self.adj_tum_ic = 1.
        
        ## nut source and QoI function
        #self.nut_src = NutSource(nut_s = 0.08)
        self.nut_src = NutSourceLeftEdge(a = 1.5)
        self.qoi_fn = QoIFn()
        
        ## mesh
        self.mesh = Vh[STATE].mesh()
        self.dx = dl.Measure('dx', domain=self.mesh)
        
        self.FE_space = dl.FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
        self.FS_tum = dl.FunctionSpace(self.mesh, "Lagrange", 1)
        
        ## state variables
        
        # solution
        self.u = dl.Function(self.FS_tum, name = 'u')
        self.p = dl.Function(self.FS_tum, name = 'p')
        
        # old solution
        self.u0 = dl.Function(self.FS_tum, name = 'u0')
        self.p0 = dl.Function(self.FS_tum, name = 'p0')

        # picard iteration solution
        self.uk = dl.Function(self.FS_tum, name = 'uk')
        
        # error
        self.eu = dl.Function(self.FS_tum, name = 'eu')
        self.ep = dl.Function(self.FS_tum, name = 'ep')
        
        # old error
        self.eu0 = dl.Function(self.FS_tum, name = 'eu0')
        self.ep0 = dl.Function(self.FS_tum, name = 'ep0')
        
        # low-fidelity solution
        self.uLF = dl.Function(self.FS_tum, name = 'uLF')
        self.pLF = dl.Function(self.FS_tum, name = 'pLF')
        self.u0LF = dl.Function(self.FS_tum, name = 'u0LF')
        self.p0LF = dl.Function(self.FS_tum, name = 'p0LF')
        
        # solution ic and adjoint tc
        self.u_ic = dl.Function(self.FS_tum, name = 'u_ic')
        self.u_ic_Fn = IC_Tum(tum_ic = self.tum_ic, degree = 2)
        self.u_ic.interpolate(self.u_ic_Fn)
        
        self.p_tc = dl.Function(self.FS_tum, name = 'p_tc')
        self.p_tc_Fn = TC_Tum(degree = 2)
        self.p_tc.interpolate(self.p_tc_Fn)
        
        # error ic and adjoint error tc
        self.eu_ic = dl.Function(self.FS_tum, name = 'eu_ic')
        self.eu_ic_Fn = IC_Error_Tum(tum_ic = self.tum_ic, degree = 2)
        self.eu_ic.interpolate(self.eu_ic_Fn)
        
        self.ep_tc = dl.Function(self.FS_tum, name = 'ep_tc')
        self.ep_tc_Fn = TC_Error_Tum(degree = 2)
        self.ep_tc.interpolate(self.ep_tc_Fn)
        
        ## test and trial functions
        
        # to compute integral of function
        self.u_help = dl.Function(self.FS_tum)
        self.u_help_trial = dl.TrialFunction(self.FS_tum)
        self.u_help_test = dl.TestFunction(self.FS_tum)
        
        # test function v, q, w
        self.v = dl.TestFunction(self.FS_tum)
        
        # for integral QoI calculation
        self.z = dl.assemble(dl.Constant(1.0)*self.u_help_test*dl.dx)
        self.M = dl.assemble(self.u_help_trial*self.u_help_test*dl.dx)
        
        ## residual forms for u, p, e, and E
        self.Fu = None        
        self.Fp = None
        self.Feu = None
        self.Fep = None
        
        self.Fu_LF = None        
        self.Fp_LF = None
        
        # generate low-fidelity data
        self.lf_qoi = None
        
        # qoi
        self.qoi_using_exact = 0
        self.qoi_using_approx = 0

        self.qoi_est_direct_using_exact = 0
        
        self.qoi_est1_using_exact = 0
        self.qoi_est2_using_exact = 0
        self.qoi_est3_using_exact = 0

        self.qoi_est_direct_using_approx = 0
        
        self.qoi_est1_using_approx = 0
        self.qoi_est2_using_approx = 0
        self.qoi_est3_using_approx = 0
        
        # to read vtu files
        self.DOF_tum = None
        self.read_key_tum = None
        self.setup_read_keys()
        
    def get_vtu_data(self, f, field):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(f)
        reader.Update()
        data = reader.GetOutput()
        
        # debug
        #print(f, field, data)
        
        return VN.vtk_to_numpy(data.GetPointData().GetArray(field))
        
    def setup_read_keys(self):
        
        # need dofmap to set the components of mixed space function
        # see: https://fenicsproject.discourse.group/t/manipulate-components-of-mixed-functions/1671/2
        self.DOF_tum = self.FS_tum.dofmap()
        
        # create test vector
        a = dl.Function(self.FS_tum, name = 'a')
        
        # write
        a.vector()[self.DOF_tum.dofs()] = self.DOF_tum.dofs()

        fa = dl.File('a.pvd')

        fa << (a, 0)
        
        # read the keys
        ar = dl.Function(self.FS_tum, name = 'ar')
        ar.vector().vec()[self.DOF_tum.dofs()] = self.get_vtu_data('a000000.vtu', 'a')

        self.read_key_tum = np.array(ar.vector().vec()[self.DOF_tum.dofs()], dtype = np.int32)
        
        # clean up
        rm_files = ['a.pvd', 'a{:06d}.vtu'.format(0)]
        for f in rm_files:
            os.remove(f)
        
    def reset_qoi(self):
        self.qoi_using_exact = 0
        self.qoi_using_approx = 0

        self.qoi_est_direct_using_exact = 0
        
        self.qoi_est1_using_exact = 0
        self.qoi_est2_using_exact = 0
        self.qoi_est3_using_exact = 0
        
        self.qoi_est_direct_using_approx = 0

        self.qoi_est1_using_approx = 0
        self.qoi_est2_using_approx = 0
        self.qoi_est3_using_approx = 0
        
        
    def indicator_fn(self, x, x1, x2):
        a = float((x1 - x) * (x2 - x) < 0.)
        #print(a, x, x1, x2)
        return a
    
    def pmsg(self, msg, lvl = 0):
        if lvl <= self.verbosity:
            print(msg, flush = True)
        
    def generate_state(self):
        """ Return a vector in the shape of the fwd model (QoI) output. """
        return np.zeros((1,1))

    def generate_pde_state(self):
        """ Return a list of vectors that correspons to the model state vectors. """
        return [dl.Function(self.Vh[STATE]) for i in range(self.nspecies)]

    def generate_parameter(self):
        """ Return a vector in the shape of the parameter. """
        return dl.Function(self.Vh[PARAMETER]).vector()

    def init_parameter(self, m):
        """ Initialize the parameter. """
        dummy = self.generate_parameter()
        m.init( dummy.mpi_comm(), dummy.local_range() )
        
    def set_parameters(self, param):
        """ Replace parameters with new parameters. """
        
        self.par_m.vector().zero()
        self.par_m.vector().axpy(1., param)
            
        (self.par_lam_p, self.par_lam_d, self.par_eps, self.par_C) = dl.split(self.par_m)
    
    def vector2Function(self, u_func, u_vec):
        """ Convert from vector representation to function representation. """
        u_func.vector().zero()
        u_func.vector().axpy(1., u_vec)
    
    def function2Vector(self, u_vec, u_func):
        """ Convert from vector representation to function representation. """
        u_vec.zero()
        u_vec.axpy(1., u_func.vector())
        
    def scalar2Function(self, u_func, u_scalar):
        """ Project scalar to a function. """
        u_func.vector().set(u_scalar)
        
    def scalar2Vector(self, u_vec, u_scalar):
        """ Project scalar to a vector. """
        u_vec.set(u_scalar)
    
    def assign_vectors(self, u1, u2):
        """ Copy values of u2 vector into u1. """
        u1.zero()
        u1.axpy(1, u2)
            
    def copy_function(self, u1, u2):
        u1.vector().zero()
        u1.vector().axpy(1., u2.vector())
        
    def add_function(self, u, u1, u2, a1 = 1., a2 = 1.):
        """ u = a1 * u1 + a2 * u2 """
        u.vector().zero()
        u.vector().axpy(a1, u1.vector())
        u.vector().axpy(a2, u2.vector())
        
    def err_to_sol(self):
        self.add_function(self.u, self.uLF, self.eu, 1., 1.)
    
    def err_to_sol_adj(self):
        self.add_function(self.p, self.pLF, self.ep, 1., 1.)
        
    def err0_to_sol(self):
        self.add_function(self.u0, self.u0LF, self.eu0, 1., 1.)
        
    def err0_to_sol_adj(self):
        self.add_function(self.p0, self.p0LF, self.ep0, 1., 1.)
        
    def sol_to_err(self):
        self.add_function(self.eu, self.u, self.uLF, 1., -1.)
        
    def sol_to_err_adj(self):
        self.add_function(self.ep, self.p, self.pLF, 1., -1.)
        
    def sol0_to_err(self):
        self.add_function(self.eu0, self.u0, self.u0LF, 1., -1.)
        
    def sol0_to_err_adj(self):
        self.add_function(self.ep0, self.p0, self.p0LF, 1., -1.)
            
    def error_norm(self, uk, u0):
        """ Compute Ldef form_B(self, u, v):
            ^2 norm of the difference uk - u0. """
        
        diff = uk.copy()
        diff.axpy(-1., u0)
        return math.sqrt(diff.inner(self.M*diff))

    def error_norm_function(self, uk, u0):
        """ Compute Ldef form_B(self, u, v):
            ^2 norm of the difference uk - u0. """
        
        diff = uk.copy()
        diff.vector().axpy(-1., u0.vector())
        return math.sqrt(diff.vector().inner(self.M*diff.vector()))

    
    def psi_fn(self, x):
        return dl.exp(self.par_C) * np.power(x * (1. - x), 2)
    
    def dpsi_fn(self, x):
        return dl.exp(self.par_C) * 2 * (x - 3 * x * x + 2 * x * x * x)
    
    def ddpsi_fn(self, x):
        return dl.exp(self.par_C) * 2 * (1 - 6 * x + 6 * x * x)
    
    def psi_fn_var(self, u):
        return dl.exp(self.par_C) * u * u * (1 - u) * (1 - u)
    
    def dpsi_fn_var(self, u):
        return 2 * dl.exp(self.par_C) * u * (1 - 3 * u + 2 * u * u)
    
    def ddpsi_fn_var(self, u):
        return 2 * dl.exp(self.par_C) * (1 - 6 * u + 6 * u * u)
    
    def dddpsi_fn_var(self, u):
        return 12 * dl.exp(self.par_C) * (1 + 2 * u)
    
    def psi_fn_var_convex(self, u):
        return 1.5 * dl.exp(self.par_C) * u * u
    
    def dpsi_fn_var_convex(self, u):
        return 3 * dl.exp(self.par_C) * u
    
    def ddpsi_fn_var_convex(self, u):
        return 3 * dl.exp(self.par_C)
    
    def psi_fn_var_concave(self, u):
        return dl.exp(self.par_C) * u * u * (u * u - 2 * u - 0.5)
    
    def dpsi_fn_var_concave(self, u):
        return dl.exp(self.par_C) * u * (4 * u * u - 6 * u - 1)
    
    def ddpsi_fn_var_concave(self, u):
        return dl.exp(self.par_C) * (12 * u * u - 12 * u - 1)
    
    def form_A(self, u, v):
        return    dl.exp(self.par_eps) * dl.inner(dl.grad(u), dl.grad(v)) \
                + dl.exp(self.par_lam_d) * u * v
    
    def form_N(self, u, v):
        self.nut_src.set_t(self.t)
        return    self.dpsi_fn_var(u) * v \
                - dl.exp(self.par_lam_p) * u * (1 - u) * v * self.nut_src
    
    def form_N_imp_exp(self, u, v, u0):
        self.nut_src.set_t(self.t)
        return    self.dpsi_fn_var_convex(u) * v + self.dpsi_fn_var_concave(u0) * v \
                - dl.exp(self.par_lam_p) * u * (1 - u0) * v * self.nut_src
    
    def form_N_der(self, u, v, p):
        self.nut_src.set_t(self.t)
        return    self.ddpsi_fn_var(u) * v * p \
                - dl.exp(self.par_lam_p) * (1 - 2 * u) * v * p * self.nut_src
    
    def form_N_dder(self, u, q, v, p):
        self.nut_src.set_t(self.t)
        return    self.dddpsi_fn_var(u) * q * v * p \
                + dl.exp(self.par_lam_p) * 2 * q * v * p * self.nut_src
    
    def form_F(self, v):
        return None

    def form_F_ic(self, u):
        return self.u_ic * u
    
    def form_Q(self, u):
        self.qoi_fn.set_t(self.t)
        
        return self.qoi_fn * u
    
    def form_Q_der(self, u, v):
        self.qoi_fn.set_t(self.t)
        
        return self.qoi_fn * v
    
    def form_Q_tc(self, u):
        return self.p_tc * u
    
    def form_Q_tc_der(self, u, v):
        return self.form_Q_tc(v)
    
    def form_B(self, u, v, u0 = None):
        A = self.form_A(u, v)
        if u0 is not None:
            N = self.form_N_imp_exp(u, v, u0)
        else:
            N = self.form_N(u, v)
        
        return A + N
    
    def form_B_der(self, u, v, p):
        A = self.form_A(v, p)
        N = self.form_N_der(u, v, p)
        
        return A + N
    
    def form_B_dder(self, u, q, v, p):
        return self.form_N_dder(u, q, v, p)
    
    def form_B_ic(self, u, v):
        return u * v
    
    def form_B_tc_used_by_adj(self, u, v):
        return u * v
    
    def form_B_dt(self, ui, vi, uii, vii):
        return (0.5 / self.dt) * (   (uii - ui) * (vii + vi) )
    
    def form_B_dt_used_by_adj(self, ui, vi, uii, vii):
        return -(0.5 / self.dt) * (   (vii - vi) * (uii + ui) )
    
    def form_R(self, u, v):
        # form_F is None
        return - self.form_B(u, v)
    
    def form_R_ic(self, u, v):
        return self.form_F_ic(v) - self.form_B_ic(u, v)
    
    def form_R_dt(self, ui, vi, uii, vii):
        return -self.form_B_dt(ui, vi, uii, vii)
    
    def form_R_bar_tc(self, u, v, p):
        return self.form_Q_tc_der(u, v) - self.form_B_tc_used_by_adj(v, p)
    
    def form_R_bar_dt(self, ui, vi, uii, vii):
        return -self.form_B_dt_used_by_adj(ui, vi, uii, vii)
    
    def form_R_bar(self, u, v, p):
        return self.form_Q_der(u, v) - self.form_B_der(u, v, p)
    
    def form_A_LF(self, u, v):
        return    dl.exp(self.lf_par_D) * dl.inner(dl.grad(u), dl.grad(v)) \
                + dl.exp(self.lf_par_lam_d) * u * v \
                - dl.exp(self.lf_par_lam_p) * u * v * self.nut_src
    
    def form_B_LF(self, u, v):
        return self.form_A_LF(u, v)
    
    def form_B_der_LF(self, u, v, p):
        return self.form_A_LF(v, p)
    
    def form_err_eu_RHS(self, uLF, v):
        return self.form_B_LF(uLF, v) - self.form_B(uLF, v)
    
    def form_err_ep_RHS(self, uLF, pLF, v):
        return self.form_B_der_LF(uLF, v, pLF) - self.form_B_der(uLF, v, pLF)
    
    def set_form_Fu(self):
        B = self.form_B(self.u, self.v, self.u0) # use energy splitting
        F = self.form_F(self.v) # it is None
        
        self.Fu = (self.u * self.v - self.u0 * self.v + self.dt * B) * dl.dx

    def set_form_Fu_picard(self):
        B = self.form_B(self.u, self.v, self.uk) # use energy splitting
        F = self.form_F(self.v) # it is None
        
        self.Fu = (self.u * self.v - self.u0 * self.v + self.dt * B) * dl.dx
        
    def set_form_Fp(self):
        B = self.form_B_der(self.u, self.v, self.p)
        Q = self.form_Q_der(self.u, self.v)
        
        self.Fp = (self.p * self.v - self.p0 * self.v + self.dt * B - self.dt * Q) * dl.dx
        
    def set_form_Feu(self):
        B = self.form_B_der(self.uLF, self.eu, self.v)
        F = self.form_err_eu_RHS(self.uLF, self.v)
        
        self.Feu = (self.eu * self.v - self.eu0 * self.v + self.dt * B - self.dt * F) * dl.dx
        
    def set_form_Fep(self):
        B = self.form_B_der(self.uLF, self.v, self.ep)
        F = self.form_err_ep_RHS(self.uLF, self.pLF, self.v)
        
        self.Fep = (self.ep * self.v - self.ep0 * self.v + self.dt * B - self.dt * F) * dl.dx
        
    def set_form_Feu_high_order(self):
        B = self.form_B_der(self.uLF, self.eu, self.v)
        F = self.form_err_eu_RHS(self.uLF, self.v)
        B2 = self.form_B_dder(self.uLF, self.eu, self.eu, self.v)
        
        self.Feu = (  self.eu * self.v - self.eu0 * self.v + self.dt * B \
                   + self.dt * 0.5 * B2 - self.dt * F) * dl.dx
        
    def set_form_Fep_high_order(self):
        B = self.form_B_der(self.uLF, self.v, self.ep)
        F = self.form_err_ep_RHS(self.uLF, self.pLF, self.v)
        B2 = self.form_B_dder(self.uLF, self.eu, self.v, self.ep)
        F2 = self.form_B_dder(self.uLF, self.eu, self.v, self.pLF)
        
        self.FE = (  self.ep * self.v - self.ep0 * self.v \
                   + self.dt * B + self.dt * B2 \
                   - self.dt * F + self.dt * F2) * dl.dx
        
    def set_form_Fu_LF(self):
        B = self.form_B_LF(self.uLF, self.v)
        F = None
        
        self.Fu_LF = (self.uLF * self.v - self.u0LF * self.v + self.dt * B) * dl.dx
        
    def set_form_Fp_LF(self):
        B = self.form_B_der_LF(self.uLF, self.v, self.pLF)
        Q = self.form_Q_der(self.uLF, self.v)
        
        self.Fp_LF = (self.pLF * self.v - self.p0LF * self.v + self.dt * B - self.dt * Q) * dl.dx
        
    def form_residual(self, u, v):
        # F is None
        return -self.form_B(u, v)
    
    def form_residual_adj(self, u, v, p):
        return self.form_Q_der(u, v) - self.form_B_der(u, v, p)
    
    def get_est1_contrib(self, state = 'regular'):
        if state == 'init':
            # get R_ic(uLF, pLF)
            R_ic_uLF_pLF = self.form_R_ic(self.uLF, self.pLF)
            # get R_ic(uLF, ep)
            R_ic_uLF_ep = self.form_R_ic(self.uLF, self.ep)
            # get R(uLF, pLF)
            R_uLF_pLF = self.form_R(self.uLF, self.pLF)
            # get R(uLF, ep)
            R_uLF_ep = self.form_R(self.uLF, self.ep)
            # get R_bar(uLF; eu, pLF)
            R_bar_uLF_eu_pLF = self.form_R_bar(self.uLF, self.eu, self.pLF)
            
            R_tot = (  R_ic_uLF_pLF + 0.5 * R_ic_uLF_ep \
                     + 0.5 * self.dt * ( R_uLF_pLF + 0.5 * R_uLF_ep + 0.5 * R_bar_uLF_eu_pLF) \
                    ) * dl.dx
            
            return dl.assemble(R_tot)
            
        elif state == 'final':
            # get R_bar_tc(uLF, eu)
            R_bar_tc_uLF_eu = self.form_R_bar_tc(self.u0LF, self.p0LF, self.eu0)
            # get R(uLF, pLF)
            R_uLF_pLF = self.form_R(self.u0LF, self.p0LF)
            # get R(uLF, ep)
            R_uLF_ep = self.form_R(self.u0LF, self.ep0)
            # get R_bar(uLF; eu, pLF)
            R_bar_uLF_eu_pLF = self.form_R_bar(self.u0LF, self.eu0, self.p0LF)
            
            R_tot = (  0.5 * R_bar_tc_uLF_eu \
                     + 0.5 * self.dt * ( R_uLF_pLF + 0.5 * R_uLF_ep + 0.5 * R_bar_uLF_eu_pLF) \
                    ) * dl.dx
            
            return dl.assemble(R_tot)
        
        elif state == 'dt':
            # get R_dt(uLF, pLF, u0LF, p0LF)
            R_dt_uLF_pLF = self.form_R_dt(self.uLF, self.pLF, self.u0LF, self.p0LF)
            # get R_dt(uLF, ep, u0LF, ep0)
            R_dt_uLF_ep = self.form_R_dt(self.uLF, self.ep, self.u0LF, self.ep0)
            # get R_bar_dt(eu, pLF, eu0, p0LF)
            R_bar_dt_eu_pLF = self.form_R_bar_dt(self.pLF, self.eu, self.p0LF, self.eu0)
            
            R_tot = self.dt * (  R_dt_uLF_pLF + 0.5 * R_dt_uLF_ep \
                               + 0.5 * R_bar_dt_eu_pLF \
                              ) * dl.dx
            
            return dl.assemble(R_tot)
        
        elif state == 'regular':
            # get R(uLF, pLF)
            R_uLF_pLF = self.form_R(self.uLF, self.pLF)
            # get R(uLF, ep)
            R_uLF_ep = self.form_R(self.uLF, self.ep)
            # get R_bar(uLF; eu, pLF)
            R_bar_uLF_eu_pLF = self.form_R_bar(self.uLF, self.eu, self.pLF)
            
            R_tot = self.dt * (  R_uLF_pLF + 0.5 * R_uLF_ep \
                               + 0.5 * R_bar_uLF_eu_pLF \
                              ) * dl.dx
            
            return dl.assemble(R_tot)
    
    def get_est3_contrib(self, state = 'regular'):        
        if state == 'init':
            # get R_ic(uLF, p)
            R_ic_uLF_p = self.form_R_ic(self.uLF, self.p)
            # get R(uLF, p)
            R_uLF_p = self.form_R(self.uLF, self.p)
            
            R_tot = (R_ic_uLF_p + 0.5 * self.dt * R_uLF_p) * dl.dx
            
            return dl.assemble(R_tot)
            
        elif state == 'final':
            # get R(uLF, p)
            R_uLF_p = self.form_R(self.u0LF, self.p0)
            
            R_tot = 0.5 * self.dt * R_uLF_p * dl.dx
            
            return dl.assemble(R_tot)
        
        elif state == 'dt':
            # get R_dt(uLF, p, u0LF, p0)
            R_dt_uLF_p = self.form_R_dt(self.uLF, self.p, self.u0LF, self.p0)
            
            R_tot = self.dt * R_dt_uLF_p * dl.dx
            
            return dl.assemble(R_tot)
        
        elif state == 'regular':
            # get R(uLF, p)
            R_uLF_p = self.form_R(self.uLF, self.p)
            
            R_tot = self.dt * R_uLF_p * dl.dx
            
            return dl.assemble(R_tot)
        
    def get_est2_contrib(self, state = 'regular'):
        if state == 'init':
            # get R_ic(uLF, p)
            R_ic_uLF_p = self.form_R_ic(self.uLF, self.p)
            # get R(uLF, p)
            R_uLF_p = self.form_R(self.uLF, self.p)
            # get Qdder --> it is zero
            # get Bdder_uLF_eu_eu_pLF
            Bdder_uLF_eu_eu_pLF = self.form_B_dder(self.uLF, self.eu, self.eu, self.pLF)
            # get Bdder_uLF_eu_eu_ep
            Bdder_uLF_eu_eu_ep = self.form_B_dder(self.uLF, self.eu, self.eu, self.ep)
            
            R_tot = ( R_ic_uLF_p \
                         + 0.5 * self.dt * (R_uLF_p + Bdder_uLF_eu_eu_pLF + 0.5 * Bdder_uLF_eu_eu_ep) \
                    ) * dl.dx
            
            return dl.assemble(R_tot)
            
        elif state == 'final':
            # get R(uLF, p)
            R_uLF_p = self.form_R(self.u0LF, self.p0)
            # get Qdder --> it is zero
            # get Bdder_uLF_eu_eu_pLF
            Bdder_uLF_eu_eu_pLF = self.form_B_dder(self.u0LF, self.eu0, self.eu0, self.p0LF)
            # get Bdder_uLF_eu_eu_ep
            Bdder_uLF_eu_eu_ep = self.form_B_dder(self.u0LF, self.eu0, self.eu0, self.ep0)
            
            R_tot = 0.5 * self.dt * (R_uLF_p + Bdder_uLF_eu_eu_pLF + 0.5 * Bdder_uLF_eu_eu_ep) * dl.dx
            
            return dl.assemble(R_tot)
        
        elif state == 'dt':
            # get R_dt(uLF, p, u0LF, p0)
            R_dt_uLF_p = self.form_R_dt(self.uLF, self.p, self.u0LF, self.p0)
            
            R_tot = self.dt * R_dt_uLF_p * dl.dx
            
            return dl.assemble(R_tot)
        
        elif state == 'regular':
            # get R(uLF, p)
            R_uLF_p = self.form_R(self.uLF, self.p)
            # get Qdder --> it is zero
            # get Bdder_uLF_eu_eu_pLF
            Bdder_uLF_eu_eu_pLF = self.form_B_dder(self.uLF, self.eu, self.eu, self.pLF)
            # get Bdder_uLF_eu_eu_ep
            Bdder_uLF_eu_eu_ep = self.form_B_dder(self.uLF, self.eu, self.eu, self.ep)
            
            R_tot = self.dt * (R_uLF_p + Bdder_uLF_eu_eu_pLF + 0.5 * Bdder_uLF_eu_eu_ep) * dl.dx
            
            return dl.assemble(R_tot)
    
    def set_save_files(self, file_path = None):
        """ Create output directory and files for saving state of model. """
        
        self.sim_out_path = self.out_path + str(self.sim_count) + '/'
        #if file_path is not None:
        #    self.sim_out_path = self.out_path + file_path + '/'
            
        pathlib.Path(self.sim_out_path).mkdir(parents=True, exist_ok=True)
            
    def get_avg(self):
        #return [self.u_TMu.vector().vec()[self.DOF_tum].inner(self.z), self.u_D.vector().inner(self.z)]
        return self.u.vector().inner(self.z)
    
    def get_avg_LF(self):
        #return [self.u_TMu.vector().vec()[self.DOF_tum].inner(self.z), self.u_D.vector().inner(self.z)]
        return self.uLF.vector().inner(self.z)
            
    def contrib_to_qoi(self, i, t, is_lf = False):
        if i == self.Nsteps:
            if is_lf:
                return self.get_avg_LF()
            else:
                return self.get_avg()
        else:
            I = self.qoi_fn.find_interval(t)
            if I >= 0:
                t1 = self.qoi_fn.tau[I]
                t2 = t1 + self.qoi_fn.dtau
                N1 = math.floor(t1 / self.dt)
                N2 = math.floor(t2 / self.dt)
                N = int(t / self.dt)
                fact = 0.5 * (self.dt / (t2 - t1))
                if is_lf:
                    avg_u = self.get_avg_LF()
                else:
                    avg_u = self.get_avg()
                
                if N == N1 or N == N2:
                    return fact * avg_u
                else:
                    return 2. * fact * avg_u
            else:
                return 0.
            
    def read_and_update_fwd_and_lf_fields(self, i, is_approx = False):
        
        if is_approx == False:
            # read the current forward solution
            self.u.vector().vec()[self.read_key_tum] \
                = self.get_vtu_data(self.cur_sim_out_path + 'u{:06d}.vtu'.format(i-1), 'u')

            # read the old forward solution
            self.u0.vector().vec()[self.read_key_tum] \
                = self.get_vtu_data(self.cur_sim_out_path + 'u{:06d}.vtu'.format(i), 'u')
        else:
            self.eu.vector().vec()[self.read_key_tum] \
                = self.get_vtu_data(self.cur_sim_out_path + 'eu{:06d}.vtu'.format(i-1), 'eu')
            
            self.eu0.vector().vec()[self.read_key_tum] \
                = self.get_vtu_data(self.cur_sim_out_path + 'eu{:06d}.vtu'.format(i), 'eu')

        # read low-fidelity
        # current
        self.uLF.vector().vec()[self.read_key_tum] \
                = self.get_vtu_data(self.lf_sim_out_path + 'uLF{:06d}.vtu'.format(i-1), 'uLF')
        
        # for p, we write in reverse order (so ith solution is found in 'N - i')
        self.pLF.vector().vec()[self.read_key_tum] \
                = self.get_vtu_data(self.lf_sim_out_path + 'pLF{:06d}.vtu'.format(self.Nsteps - (i-1)), 'pLF')
        
        # old
        self.u0LF.vector().vec()[self.read_key_tum] \
                = self.get_vtu_data(self.lf_sim_out_path + 'uLF{:06d}.vtu'.format(i), 'uLF')
        
        self.p0LF.vector().vec()[self.read_key_tum] \
                = self.get_vtu_data(self.lf_sim_out_path + 'pLF{:06d}.vtu'.format(self.Nsteps - i), 'pLF')

    def solveFwdBck(self, x, sol_fwd = True, sol_bck = False, picard = True):
        
        if self.sim_count%self.print_freq == 0:
            self.pmsg("Solve with parameters {}".format(np.exp(x[PARAMETER].get_local())), 1)
            
        # replace parameter with specified parameters
        self.set_parameters(x[PARAMETER]) 
        
        # create save files
        # save solution only if adjoint solve requires it
        save_cur_sol = sol_bck

        self.cur_sim_out_path = self.out_path + self.cur_sim_partial_out_path + '/'
        if save_cur_sol:
            pathlib.Path(self.cur_sim_out_path).mkdir(parents=True, exist_ok=True) 
            file_names = ['u', 'p']
            cur_files = ['' for i in range(len(file_names))]
            for i in range(len(file_names)):
                cur_files[i] = dl.File(self.cur_sim_out_path \
                                    + file_names[i] + '.pvd', 'compressed')
                
                if cur_files[i] is None:
                    raise Exception("Files for saving are not created.")
        
        file_names_key = {'u': 0, 'p': 1} if sol_bck else {'u': 0}
        file_names = ['u', 'p'] if sol_bck else ['u']
        if self.solveFwdBck_save_err:
            file_names_key = {'u': 0, 'p': 1, 'eu': 2, 'ep': 3} if sol_bck else {'u': 0, 'eu': 1}
            file_names = ['u', 'p', 'eu', 'ep'] if sol_bck else ['u', 'eu']  

        saved_state_current_sim = False
        if self.save and self.sim_out_current % self.sim_out_freq == 0 \
                     and self.sim_out_current < self.sim_max_out:

            saved_state_current_sim = True
            
            if self.save_suf == '':
                self.sim_out_path = self.out_path + self.cur_sim_partial_out_path + '/'
            else:
                self.sim_out_path = self.out_path + self.save_suf + '/'
            pathlib.Path(self.sim_out_path).mkdir(parents=True, exist_ok=True)
                
            self.files = ['' for i in range(len(file_names))]
            for i in range(len(file_names)):
                self.files[i] = dl.File(self.sim_out_path \
                                + file_names[i] + '.pvd', 'compressed')
                
                if self.files[i] is None:
                    raise Exception("Files for saving are not created.")

            # write parameter values to a file
            np.savetxt(self.sim_out_path + 'param.txt', np.exp(x[PARAMETER].get_local()), delimiter=', ')
                    
        
        # reset qoi
        self.qoi_using_exact = 0
        
        self.qoi_est1_using_exact = 0
        self.qoi_est2_using_exact = 0
        self.qoi_est3_using_exact = 0
                    
        ## solve forward problem
        if sol_fwd:
            self.pmsg('solving forward problem', 2)

            # reset solution to initial condition
            self.t = 0
            self.copy_function(self.u, self.u_ic)
            self.copy_function(self.u0, self.u_ic)

            if save_cur_sol:
                cur_files[0] << (self.u, self.t)

            if saved_state_current_sim:
                self.files[file_names_key['u']] << (self.u, self.t)
                if self.solveFwdBck_save_err:
                    self.sol_to_err()
                    self.files[file_names_key['eu']] << (self.eu, self.t)


            # loop over time steps
            for i in range(self.Nsteps):
                self.t = self.t + self.dt
                self.pmsg('time step = {}, t = {}'.format(i+1, self.t), 3)
                
                # update time in QoI and nut source
                self.qoi_fn.set_t(self.t)
                self.nut_src.set_t(self.t)

                # set old solution
                self.copy_function(self.u0, self.u)

                if picard == False:
                    self.pmsg('no picard iteration', 3)
                    # set forms
                    self.set_form_Fu()

                    # solve
                    dl.solve(self.Fu == 0, self.u)
                    #, solver_parameters={"newton_solver":{"relative_tolerance": 1e-6}})
                else:
                    self.pmsg('picard iteration', 3)
                    # picard iteration
                    # uk --> old iteration solution
                    # u --> current (k+1) iteration solution

                    # set forms
                    self.set_form_Fu_picard()
                    
                    pic_err = 1.e10
                    pic_err_tol = 1.e-10
                    pic_iter = 0
                    pic_iter_max = 100
                    while pic_err > pic_err_tol and pic_iter < pic_iter_max:
                        # fix old iter solution
                        self.copy_function(self.uk, self.u)
                        
                        # compute new iter solution
                        dl.solve(self.Fu == 0, self.u)
                        
                        # compute error
                        #pic_err = self.error_norm_function(self.u, self.uk)
                        pic_err = self.error_norm(self.u.vector(), self.uk.vector())

                        # increment the counter
                        pic_iter = pic_iter + 1

                    if pic_err < pic_err_tol:
                        self.pmsg('picard iteration converged with num iters = {} and error = {}'.format(pic_iter, pic_err), 3)
                    
                    if pic_iter >= pic_iter_max:
                        self.pmsg('picard iteration stopped as it reached max iterations = {}. Error = {}'.format(pic_iter_max, pic_err), 3)
                
                # add contribution to QoI
                self.qoi_using_exact = self.qoi_using_exact + self.contrib_to_qoi(i+1, self.t)

                # check if we need to save the file
                if save_cur_sol:
                    cur_files[0] << (self.u, self.t)

                if saved_state_current_sim:
                    if (i+1) % self.print_freq == 0:
                        self.files[file_names_key['u']] << (self.u, self.t)
                        if self.solveFwdBck_save_err:
                            self.sol_to_err()
                            self.files[file_names_key['eu']] << (self.eu, self.t)

            self.qoi_est_direct_using_exact = self.qoi_using_exact - self.lf_qoi
                    
        ## solve backward problem
        if sol_bck:
            self.pmsg('solving backward problem', 2)
            
            # convention
            # i  --> time step such that we seek solution at t_{i-1} given solution at t_i
            # a0 --> field at previous time, i.e., t_i
            # a  --> field at current time, i.e., t_{i-1}

            # reset solution to terminal condition
            self.t = self.tF
            self.copy_function(self.p, self.p_tc)
            self.copy_function(self.p0, self.p_tc)
            
            # output
            cur_files[1] << (self.p, self.t)

            if saved_state_current_sim:
                self.files[file_names_key['p']] << (self.p, self.t)
                if self.solveFwdBck_save_err:
                    self.sol_to_err_adj()
                    self.files[file_names_key['ep']] << (self.ep, self.t)

            # i --> current time step (seeking solution at time t_{i-1} given solution at t_i)
            for i in range(self.Nsteps, 0, -1):
                self.t = self.t - self.dt
                self.pmsg('time step = {}, t = {}'.format(i, self.t), 3)
                
                # update time in QoI and nut source
                self.qoi_fn.set_t(self.t)
                self.nut_src.set_t(self.t)

                # set old solution
                self.copy_function(self.p0, self.p)
                
                # read and update forward solution and low-fidelity solutions
                self.read_and_update_fwd_and_lf_fields(i)

                # set forms
                self.set_form_Fp()

                # solve 
                dl.solve(self.Fp == 0, self.p)
                
                # now that we have update adjoint solutions, we can compute the error fields
                # compute error fields
                self.sol_to_err()
                self.sol_to_err_adj()
                
                self.sol0_to_err()
                self.sol0_to_err_adj()

                # check if we need to save the file
                cur_files[1] << (self.p, self.t)

                if saved_state_current_sim:
                    if i % self.print_freq == 0:
                        self.files[file_names_key['p']] << (self.p, self.t)
                        if self.solveFwdBck_save_err:
                            self.files[file_names_key['ep']] << (self.ep, self.t)
                
                # add contribution from last time step
                if i == self.Nsteps:
                    self.qoi_est1_using_exact =   self.qoi_est1_using_exact \
                                                + self.get_est1_contrib('final')
                    
                    self.qoi_est2_using_exact =   self.qoi_est2_using_exact \
                                                + self.get_est2_contrib('final')
                    
                    self.qoi_est3_using_exact =   self.qoi_est3_using_exact \
                                                + self.get_est3_contrib('final')
                
                # add contribution from current time step
                if i > 1:
                    self.qoi_est1_using_exact =   self.qoi_est1_using_exact \
                                                + self.get_est1_contrib()
                    
                    self.qoi_est2_using_exact =   self.qoi_est2_using_exact \
                                                + self.get_est2_contrib()
                    
                    self.qoi_est3_using_exact =   self.qoi_est3_using_exact \
                                                + self.get_est3_contrib()
                    
                # add contribution from time derivative terms
                self.qoi_est1_using_exact =   self.qoi_est1_using_exact \
                                            + self.get_est1_contrib('dt')
                
                self.qoi_est2_using_exact =   self.qoi_est2_using_exact \
                                            + self.get_est2_contrib('dt')
                
                self.qoi_est3_using_exact =   self.qoi_est3_using_exact \
                                            + self.get_est3_contrib('dt')
                
                # add contribution from first time step
                if i == 1:
                    self.qoi_est1_using_exact =   self.qoi_est1_using_exact \
                                                + self.get_est1_contrib('init')
                    
                    self.qoi_est2_using_exact =   self.qoi_est2_using_exact \
                                                + self.get_est2_contrib('init')
                    
                    self.qoi_est3_using_exact =   self.qoi_est3_using_exact \
                                                + self.get_est3_contrib('init')
                
                        
    def solveFwdBckErr(self, x, sol_fwd = True, sol_bck = False, high_order = False):
        
        if self.sim_count%self.print_freq == 0:
            self.pmsg("Solve with parameters {}".format(np.exp(x[PARAMETER].get_local())), 1)
            
        # replace parameter with specified parameters
        self.set_parameters(x[PARAMETER]) 
        
        # create save files
        # save solution only if adjoint solve requires it
        save_cur_sol = sol_bck

        self.cur_sim_out_path = self.out_path + self.cur_err_sim_partial_out_path + '/'
        if high_order:
            self.cur_sim_out_path = self.out_path + self.cur_err_sim_partial_out_path + '_high_order/'

        if save_cur_sol:
            pathlib.Path(self.cur_sim_out_path).mkdir(parents=True, exist_ok=True) 
        
            file_names = ['u', 'p', 'eu', 'ep']
            cur_files = ['' for i in range(len(file_names))]
            for i in range(len(file_names)):
                cur_files[i] = dl.File(self.cur_sim_out_path \
                                    + file_names[i] + '.pvd', 'compressed')
                
                if cur_files[i] is None:
                    raise Exception("Files for saving are not created.")
            
        file_names_key = {'u': 0, 'p': 1, 'eu': 2, 'ep': 3} if sol_bck else {'u': 0, 'eu': 1}
        file_names = ['u', 'p', 'eu', 'ep'] if sol_bck else ['u', 'eu']

        saved_state_current_sim = False
        if self.save and self.sim_out_current % self.sim_out_freq == 0 \
                     and self.sim_out_current < self.sim_max_out:

            saved_state_current_sim = True
            
            ss = self.cur_err_sim_partial_out_path if self.save_suf == '' else self.save_suf + '_err'
            if high_order:
                ss = ss + '_high_order'
            ss = ss + '/'

            self.sim_out_path = self.out_path + ss
            
            if not os.path.isdir(self.sim_out_path):
                os.mkdir(self.sim_out_path)
                
            self.files = ['' for i in range(len(file_names))]
            for i in range(len(file_names)):
                self.files[i] = dl.File(self.sim_out_path \
                                + file_names[i] + '.pvd', 'compressed')
                
                if self.files[i] is None:
                    raise Exception("Files for saving are not created.")

            # write parameter values to a file
            np.savetxt(self.sim_out_path + 'param.txt', np.exp(x[PARAMETER].get_local()), delimiter=', ')
                    
        
        # reset qoi
        self.qoi_using_approx = 0
        
        self.qoi_est1_using_approx = 0
        self.qoi_est2_using_approx = 0
        self.qoi_est3_using_approx = 0
                    
        ## solve forward problem
        if sol_fwd:
            self.pmsg('solving forward problem for error e', 2)

            # reset solution to initial condition
            self.t = 0
            self.copy_function(self.eu, self.eu_ic)
            self.copy_function(self.eu0, self.eu_ic)
            
            # read low-fidelity
            i = 0
            self.u0LF.vector().vec()[self.read_key_tum] \
                    = self.get_vtu_data(self.lf_sim_out_path + 'uLF{:06d}.vtu'.format(i), 'uLF')
            
            # define high-fidelity solution as error + low-fidelity solution
            self.err_to_sol()
            
            # write initial solution
            if save_cur_sol:
                cur_files[0] << (self.u, self.t)
                cur_files[2] << (self.eu, self.t)

            if saved_state_current_sim:
                self.files[file_names_key['u']] << (self.u, self.t)
                self.files[file_names_key['eu']] << (self.eu, self.t)

            # loop over time steps
            for i in range(self.Nsteps):
                self.t = self.t + self.dt
                self.pmsg('time step = {}, t = {}'.format(i+1, self.t), 3)
                
                # update time in QoI and nut source
                self.qoi_fn.set_t(self.t)
                self.nut_src.set_t(self.t)

                # set old solution
                self.copy_function(self.eu0, self.eu)
                
                # set forms
                if high_order:
                    self.set_form_Feu_high_order()
                else:
                    self.set_form_Feu()

                # solve
                dl.solve(self.Feu == 0, self.eu)
                #, solver_parameters={"newton_solver":{"relative_tolerance": 1e-6}})
                
                # read low-fidelity
                self.uLF.vector().vec()[self.read_key_tum] \
                    = self.get_vtu_data(self.lf_sim_out_path + 'uLF{:06d}.vtu'.format(i+1), 'uLF')

                
                # high-fidelity solution
                self.err_to_sol()
                
                # add contribution to QoI
                self.qoi_using_approx = self.qoi_using_approx + self.contrib_to_qoi(i+1, self.t)

                # check if we need to save the file
                if save_cur_sol:
                    cur_files[0] << (self.u, self.t)
                    cur_files[2] << (self.eu, self.t)

                if saved_state_current_sim:
                    if (i+1) % self.print_freq == 0:
                        self.files[file_names_key['u']] << (self.u, self.t)
                        self.files[file_names_key['eu']] << (self.eu, self.t)

            self.qoi_est_direct_using_approx = self.qoi_using_approx - self.lf_qoi
                    
        ## solve backward problem
        if sol_bck:
            self.pmsg('solving backward problem', 2)
            
            # convention
            # i  --> time step such that we seek solution at t_{i-1} given solution at t_i
            # a0 --> field at previous time, i.e., t_i
            # a  --> field at current time, i.e., t_{i-1}

            # reset solution to terminal condition
            self.t = self.tF
            self.copy_function(self.ep, self.ep_tc)
            self.copy_function(self.ep0, self.ep_tc)
            
            # read and update forward solution and low-fidelity solutions
            self.read_and_update_fwd_and_lf_fields(self.Nsteps, True)
            
            # define high-fidelity solution as error + low-fidelity solution
            self.err_to_sol()
            self.err0_to_sol()
            
            self.err_to_sol_adj()
            self.err0_to_sol_adj()

            # write
            cur_files[1] << (self.p, self.t)
            cur_files[3] << (self.ep, self.t)

            if saved_state_current_sim:
                self.files[file_names_key['p']] << (self.p, self.t)
                self.files[file_names_key['ep']] << (self.ep, self.t)

            # loop over time steps
            for i in range(self.Nsteps, 0, -1):
                self.t = self.t - self.dt
                self.pmsg('time step = {}, t = {}'.format(i, self.t), 3)
                
                # update time in QoI and nut source
                self.qoi_fn.set_t(self.t)
                self.nut_src.set_t(self.t)

                # set old solution
                self.copy_function(self.ep0, self.ep)
                
                # read and update forward solution and low-fidelity solutions
                self.read_and_update_fwd_and_lf_fields(i, True)

                # set forms
                if high_order:
                    self.set_form_Fep_high_order()
                else:
                    self.set_form_Fep()

                # solve
                dl.solve(self.Fep == 0, self.ep)
                
                # now that we have update adjoint error, we can compute the forward fields
                # compute forward fields
                self.err_to_sol()
                self.err0_to_sol()

                self.err_to_sol_adj()
                self.err0_to_sol_adj()

                # check if we need to save the file
                cur_files[1] << (self.p, self.t)
                cur_files[3] << (self.ep, self.t)

                if saved_state_current_sim:
                    if i % self.print_freq == 0:
                        self.files[file_names_key['p']] << (self.p, self.t)
                        self.files[file_names_key['ep']] << (self.ep, self.t)
                
                # add contribution from last time step
                if i == self.Nsteps:
                    self.qoi_est1_using_approx =   self.qoi_est1_using_approx \
                                                + self.get_est1_contrib('final')
                    
                    self.qoi_est2_using_approx =   self.qoi_est2_using_approx \
                                                + self.get_est2_contrib('final')
                    
                    self.qoi_est3_using_approx =   self.qoi_est3_using_approx \
                                                + self.get_est3_contrib('final')
                
                # add contribution from current time step
                if i > 1:
                    self.qoi_est1_using_approx =   self.qoi_est1_using_approx \
                                                + self.get_est1_contrib()
                    
                    self.qoi_est2_using_approx =   self.qoi_est2_using_approx \
                                                + self.get_est2_contrib()
                    
                    self.qoi_est3_using_approx =   self.qoi_est3_using_approx \
                                                + self.get_est3_contrib()
                    
                # add contribution from time derivative terms
                self.qoi_est1_using_approx =   self.qoi_est1_using_approx \
                                            + self.get_est1_contrib('dt')
                
                self.qoi_est2_using_approx =   self.qoi_est2_using_approx \
                                            + self.get_est2_contrib('dt')
                
                self.qoi_est3_using_approx =   self.qoi_est3_using_approx \
                                            + self.get_est3_contrib('dt')
                
                # add contribution from first time step
                if i == 1:
                    self.qoi_est1_using_approx =   self.qoi_est1_using_approx \
                                                + self.get_est1_contrib('init')
                    
                    self.qoi_est2_using_approx =   self.qoi_est2_using_approx \
                                                + self.get_est2_contrib('init')
                    
                    self.qoi_est3_using_approx =   self.qoi_est3_using_approx \
                                                + self.get_est3_contrib('init')

    def solveFwdBckLF(self, sol_fwd = True, sol_bck = False, save_lf_visual = False):
        
        self.pmsg("Solve LF with parameters {}".format(self.lf_par_real_vec), 1)
            
        # create save files
        save_lf_sim = True
        self.lf_sim_out_path = self.out_path + self.lf_sim_partial_out_path + '/'
        pathlib.Path(self.lf_sim_out_path).mkdir(parents=True, exist_ok=True) 
        
        file_names = ['uLF', 'pLF'] if sol_bck else ['uLF']
        cur_files = ['' for i in range(len(file_names))]
        for i in range(len(file_names)):
            cur_files[i] = dl.File(self.lf_sim_out_path \
                                + file_names[i] + '.pvd', 'compressed')
            
            if cur_files[i] is None:
                raise Exception("Files for saving are not created.")

        file_names = ['uLF', 'pLF'] if sol_bck else ['uLF']
        if save_lf_visual:

            if self.save_suf == '':
                self.sim_out_path = self.out_path + 'lf/'
            else:
                self.sim_out_path = self.out_path + self.save_suf + '_lf/'
            pathlib.Path(self.sim_out_path).mkdir(parents=True, exist_ok=True)
                
            self.files = ['' for i in range(len(file_names))]
            for i in range(len(file_names)):
                self.files[i] = dl.File(self.sim_out_path \
                                + file_names[i] + '.pvd', 'compressed')
                
                if self.files[i] is None:
                    raise Exception("Files for saving are not created.")

            # write parameter values to a file
            np.savetxt(self.sim_out_path + 'param_lf.txt', self.lf_par_real_vec, delimiter=', ')
        
        # reset qoi
        self.lf_qoi = 0
                    
        ## solve forward problem
        if sol_fwd:
            self.pmsg('solving forward problem', 2)

            # reset solution to initial condition
            self.t = 0
            self.copy_function(self.uLF, self.u_ic)
            self.copy_function(self.u0LF, self.u_ic)

            if save_lf_sim:
                cur_files[0] << (self.uLF, self.t)

            if save_lf_visual:
                self.files[0] << (self.uLF, self.t)

            # loop over time steps
            for i in range(self.Nsteps):
                self.t = self.t + self.dt
                self.pmsg('time step = {}, t = {}'.format(i+1, self.t), 3)
                
                # update time in QoI and nut source
                self.qoi_fn.set_t(self.t)
                self.nut_src.set_t(self.t)

                # set old solution
                self.copy_function(self.u0LF, self.uLF)

                # set forms
                self.set_form_Fu_LF()

                # solve
                dl.solve(self.Fu_LF == 0, self.uLF)
                #, solver_parameters={"newton_solver":{"relative_tolerance": 1e-6}})
                
                # add contribution to QoI
                self.lf_qoi = self.lf_qoi + self.contrib_to_qoi(i+1, self.t, is_lf = True)

                # check if we need to save the file
                if save_lf_sim:
                    cur_files[0] << (self.uLF, self.t)

                if save_lf_visual:
                    if (i+1) % self.print_freq == 0:
                        self.files[0] << (self.uLF, self.t)
                    
        ## solve backward problem
        if sol_bck:
            self.pmsg('solving backward problem', 2)
            
            # convention
            # i  --> time step such that we seek solution at t_{i-1} given solution at t_i
            # a0 --> field at previous time, i.e., t_i
            # a  --> field at current time, i.e., t_{i-1}

            # reset solution to terminal condition
            self.t = self.tF
            self.copy_function(self.pLF, self.p_tc)
            self.copy_function(self.p0LF, self.p_tc)
            
            # output
            if save_lf_sim:
                cur_files[1] << (self.pLF, self.t)

            if save_lf_visual:
                self.files[1] << (self.pLF, self.t)

            # i --> current time step (seeking solution at time t_{i-1} given solution at t_i)
            for i in range(self.Nsteps, 0, -1):
                self.t = self.t - self.dt
                self.pmsg('time step = {}, t = {}'.format(i, self.t), 3)
                
                # update time in QoI and nut source
                self.qoi_fn.set_t(self.t)
                self.nut_src.set_t(self.t)

                # set old solution
                self.copy_function(self.p0LF, self.pLF)
                
                # we don't need LF solution uLF as the dual problem for low-fidelity does not
                # depend on uLF. This is because forward low-fidelity problem is linear
                #self.uLF.vector().vec()[self.read_key_tum] \
                #    = self.get_vtu_data(self.lf_sim_out_path + 'uLF{:06d}.vtu'.format(i), 'uLF')

                # set forms
                self.set_form_Fp_LF()

                # solve 
                dl.solve(self.Fp_LF == 0, self.pLF)
                
                # check if we need to save the file
                if save_lf_sim:
                    cur_files[1] << (self.pLF, self.t)

                if save_lf_visual:
                    if i % self.print_freq == 0:
                        self.files[1] << (self.pLF, self.t)
    
    def solveFwd(self, out, x, save_suf = 'mcmc'):
        """ 
        This function is called by HippyLib during MCMC simulatin. 
        Return the model output which in this case is the difference in LF and HF QOIs. 
        """
        if save_suf == 'mcmc':
            self.verbosity = 1
            self.save_suf = save_suf + '_{}'.format(self.sim_count)
        else:
            self.save_suf = save_suf

        
        # need adjoints
        solve_adj = self.which_est != 0
        
        if self.use_approx:
            self.solveFwdBckErr(x, True, solve_adj)
            
            if self.which_est == 0:
                out[0] = np.array([self.qoi_est_direct_using_approx])[:, None]
            elif self.which_est == 1:
                out[0] = np.array([self.qoi_est1_using_approx])[:, None]
            elif self.which_est == 2:
                out[0] = np.array([self.qoi_est2_using_approx])[:, None]
            elif self.which_est == 3:
                out[0] = np.array([self.qoi_est3_using_approx])[:, None]
        else:
            self.solveFwdBck(x, True, solve_adj)
            
            if self.which_est == 0:
                out[0] = np.array([self.qoi_est_direct_using_exact])[:, None]
            elif self.which_est == 1:
                out[0] = np.array([self.qoi_est1_using_exact])[:, None]
            elif self.which_est == 2:
                out[0] = np.array([self.qoi_est2_using_exact])[:, None]
            elif self.which_est == 3:
                out[0] = np.array([self.qoi_est3_using_exact])[:, None]
                
        if self.sim_count%self.print_freq == 0 and self.verbosity > 0:
            print('  Count: ', self.sim_count, ', QoI: ', out[0])

        # increment the counter for number of times the forward model is solved
        self.sim_count += 1


