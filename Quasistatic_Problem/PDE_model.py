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


# In[9]:


from PDE_model_help import GenericReprBase, FwdModelSource, FwdBoundary


# # In[10]:


# from PDE_model_help import Misfit, FullTracer, plot_prior


# In[15]:



STATE = 0
PARAMETER = 1
        
class PDEModel(GenericReprBase):
    def __init__(self, Vh, save = False, \
                 theta0 = [0.25], \
                 out_path = './fwd_result/', sim_out_freq = 10, sim_max_out = 20,\
                 tol = 1.0e-12, max_iter = 100, verbosity = 1, print_freq = 1):

        # some params for solveFwd called by HippyLib
        self.use_approx = True # solve for errors to get exact solution
        self.which_est = 0 # if 0, use Q(u) - Q(u0), else use one of the estimates for QoI error
        
        ## Vh groups parameter space and function space
        self.Vh = Vh 
        self.nspecies = 1
        
        ## output related parameters
        self.save = save
        self.out_path = out_path
        pathlib.Path(self.out_path).mkdir(parents=True, exist_ok=True)
        self.file = None
        
        self.sim_out_freq = sim_out_freq  # output 1 every 10 simulation
        self.sim_max_out = sim_max_out    # do not produce results for sims more than this
        self.sim_out_current = 0
        
        # verbosity 
        self.verbosity = verbosity
        self.sim_count = 0
        self.print_freq = print_freq
        
        ## parameters
        self.param_dim = Vh[PARAMETER].dim()
        self.m = dl.Function(Vh[PARAMETER])
        self.k = dl.Constant(math.log(0.25))
        self.alpha = dl.Constant(math.log(10))
        
        ## LF parameters
        self.theta0 = theta0
        self.k0 = dl.Constant(math.log(theta0[0]))
        
        # rhs function
        self.f = FwdModelSource()
        
        ## state variables
        self.u = self.generate_pde_state()
        self.p = self.generate_pde_state()
        
        self.u0 = self.generate_pde_state()
        self.p0 = self.generate_pde_state()
        
        self.eu = self.generate_pde_state()
        self.ep = self.generate_pde_state()
        
        ## to restore to initial state
        self.u_ic = dl.interpolate(dl.Constant(0), Vh[STATE])
        
        ## test and trial functions
        self.help = dl.Function(Vh[STATE])
        self.v = dl.TestFunction(self.Vh[STATE])
        self.u_trial = dl.TrialFunction(self.Vh[STATE])
        
        # for integral QoI calculation
        self.z = dl.assemble(dl.Constant(1.0)*self.v*dl.dx)
        
        ## mesh
        self.mesh = Vh[STATE].mesh()
        self.dx = dl.Measure('dx', domain=self.mesh)
        
        ## bc
        self.u_bc_val = dl.Constant(0.)
        self.u_bc = dl.DirichletBC(Vh[STATE], self.u_bc_val, FwdBoundary)
        
        ## 
        self.F = [None]*2
        self.F0 = [None]*2
        self.FE = [None]*2
        
        self.qoi_err_est = [None]*2
        
        self.set_varf_forms()
        
        # generate low-fidelity data
        self.lf_qoi = None
        
        # flag which specifies if we have solved the low-fidelity problem
        self.is_lf_solved = False
        
        # qoi
        self.qoi_using_exact = 0
        self.qoi_using_approx = 0
        
        self.qoi_est_using_exact = [0, 0]
        
        self.qoi_est_using_approx = [0, 0]
        
    def reset_qoi(self):
        self.qoi_using_exact = 0
        self.qoi_using_approx = 0
        
        self.qoi_est_direct_using_exact = 0
        self.qoi_est_using_exact = [0, 0]
        
        self.qoi_est_direct_using_approx = 0
        self.qoi_est_using_approx = [0, 0]
        
    def pmsg(self, msg, lvl = 0):
        if lvl <= self.verbosity:
            print(msg, flush = True)
            
    def generate_state(self):
        """ Return a vector in the shape of the fwd model (QoI) output. """
        return np.zeros((1,1))

    def generate_pde_state(self):
        """ Return a list of vectors that correspons to the model state vectors. """
        return dl.Function(self.Vh[STATE])

    def generate_parameter(self):
        """ Return a vector in the shape of the parameter. """
        return dl.Function(self.Vh[PARAMETER]).vector()

    def init_parameter(self, m):
        """ Initialize the parameter. """
        dummy = self.generate_parameter()
        m.init( dummy.mpi_comm(), dummy.local_range() )
        
    def set_parameters(self, param):
        """ Replace parameters with new parameters. """
        
        self.m.vector().zero()
        self.m.vector().axpy(1., param)
            
        (self.k, self.alpha) = dl.split(self.m)
        
        # since parameter is updated, we need to update the variational forms
        self.set_varf_forms()
    
    def vector2Function(self, u_func, u_vec):
        """ Convert from vector representation to function representation. """
        u_func.vector().zero()
        u_func.vector().axpy(1., u_vec)
    
    def function2Vector(self, u_vec, u_func):
        """ Convert from vector representation to function representation. """
        u_vec.zero()
        u_vec.axpy(1., u_func.vector())
    
    def assign_vectors(self, u1, u2):
        """ Copy values of u2 vector into u1. """
        for i in range(self.nspecies):
            u1[i].zero()
            u1[i].axpy(1, u2[i])
            
    def copy_function(self, u1, u2):
        u1.vector().zero()
        u1.vector().axpy(1., u2.vector())
        
    def add_function(self, u, u1, u2, a1 = 1., a2 = 1.):
        """ u = a1 * u1 + a2 * u2 """
        u.vector().zero()
        u.vector().axpy(a1, u1.vector())
        u.vector().axpy(a2, u2.vector())
        
    def err_to_sol(self):
        self.add_function(self.u, self.u0, self.eu, 1., 1.)
        
    def err_to_sol_adj(self):
        self.add_function(self.p, self.p0, self.ep, 1., 1.)        
        
    def sol_to_err(self):
        self.add_function(self.eu, self.u, self.u0, 1., -1.)
        
    def sol_to_err_adj(self):
        self.add_function(self.ep, self.p, self.p0, 1., -1.)
            
    def error_norm(self, u_k, u_0):
        """ Compute L^2 norm of the difference u_k - u_0. """
        
        error = 0.0
        for i in range(self.nspecies):
            diff = u_k[i].copy()
            diff.axpy(-1., u_0[i])
            error +=math.sqrt(diff.inner(self.M*diff))

        return error
    
    def get_avg(self, u):
        return u.vector().inner(self.z)
    
    def form_Q(self, u):
        return u
    
    def form_Q_der(self, u, v):
        return v

    def form_F(self, v):
        return self.f * v
    
    def form_B(self, u, v):
        return   dl.exp(self.k) * (1 + u * u) * dl.inner(dl.grad(u), dl.grad(v)) \
               + dl.exp(self.alpha) * u * v
    
    def form_B_der(self, u, v, p):
        return   dl.exp(self.k) * (1 + u * u) * dl.inner(dl.grad(v), dl.grad(p)) \
               + 2 * dl.exp(self.k) * u * v * dl.inner(dl.grad(u), dl.grad(p)) \
               + dl.exp(self.alpha) * v * p
    
    def form_B_dder(self, u, q, v, p):
        return   2 * dl.exp(self.k) * u * q * dl.inner(dl.grad(v), dl.grad(p)) \
               + 2 * dl.exp(self.k) * u * v * dl.inner(dl.grad(q), dl.grad(p)) \
               + 2 * dl.exp(self.k) * q * v * dl.inner(dl.grad(u), dl.grad(p))
    
    def form_R(self, u, v):
        return self.form_F(v) - self.form_B(u, v)
    
    def form_R_bar(self, u, v, p):
        return self.form_Q_der(u, v) - self.form_B_der(u, v, p)
    
    def form_B_LF(self, u, v):
        return   dl.exp(self.k0) * dl.inner(dl.grad(u), dl.grad(v)) 
    
    def form_B_der_LF(self, u, v, p):
        return self.form_B_LF(v, p)
    
    def form_R_LF(self, u, v):
        return self.form_F(v) - self.form_B_LF(u, v)
    
    def form_R_bar_LF(self, u, v, p):
        return self.form_Q_der(u, v) - self.form_B_der_LF(u, v, p)
            
    def set_varf_forms(self):
        """ Set variational form. """
        
        # linear model
        self.F0[0] = self.form_R_LF(self.u0, self.v) * dl.dx
        self.F0[1] = self.form_R_bar_LF(self.u0, self.v, self.p0) * dl.dx
        
        # nonlinear model
        self.F[0] = self.form_R(self.u, self.v) * dl.dx
        self.F[1] = self.form_R_bar(self.u, self.v, self.p) * dl.dx
        
        # approximate equation for errors
        self.FE[0] = (self.form_B_der(self.u0, self.eu, self.v) - self.form_R(self.u0, self.v)) * dl.dx
        self.FE[1] = (   self.form_B_der(self.u0, self.v, self.ep) \
                         - self.form_R_bar(self.u0, self.v, self.p0)
                       ) * dl.dx
        
        # error estimator forms
        self.qoi_err_est[0] = self.form_R(self.u0, self.p) * dl.dx
        self.qoi_err_est[1] = (   self.form_R(self.u0, self.p) \
                                + self.form_B_dder(self.u0, self.eu, self.eu, self.p0) \
                                + 0.5 * self.form_B_dder(self.u0, self.eu, self.eu, self.ep) \
                              ) * dl.dx
    def solve_fwd_bck_lf(self):
        
        # solve only if needed
        if self.is_lf_solved:
            return None
            
        if self.sim_count%self.print_freq == 0:
            self.pmsg("Solve LF problem with parameters {}".format(self.theta0), 1)
            
        # set variational form
        self.set_varf_forms()
        
        # re-init solution fields
        self.copy_function(self.u0, self.u_ic)
        self.copy_function(self.p0, self.u_ic)
        
        # solve
        dl.solve(self.F0[0] == 0, self.u0, self.u_bc, \
                         solver_parameters={"newton_solver":{"relative_tolerance": 1e-6}})
        
        dl.solve(self.F0[1] == 0, self.p0, self.u_bc, \
                         solver_parameters={"newton_solver":{"relative_tolerance": 1e-6}})
        
        # update flag
        self.is_lf_solved =  True        
        
        # qoi
        self.lf_qoi = self.get_avg(self.u0)
    
    def solve_fwd_bck(self, x):
        
        if self.sim_count%self.print_freq == 0:
            self.pmsg("Solve with parameters {}".format(np.exp(x[PARAMETER].get_local())), 1)
            
        # replace parameter with specified parameters
        self.set_parameters(x[PARAMETER]) 
        
        # set variational form
        self.set_varf_forms()
        
        # if LF is not solved, solve it now
        if self.is_lf_solved == False:
            self.solve_fwd_bck_lf()
        
        # re-init solution fields
        self.copy_function(self.u, self.u_ic)
        self.copy_function(self.p, self.u_ic)
        
        # solve
        dl.solve(self.F[0] == 0, self.u, self.u_bc, \
                         solver_parameters={"newton_solver":{"relative_tolerance": 1e-6}})
        
        dl.solve(self.F[1] == 0, self.p, self.u_bc, \
                         solver_parameters={"newton_solver":{"relative_tolerance": 1e-6}})
        
        # update error fields
        self.sol_to_err()
        self.sol_to_err_adj()
        
        # qoi and error in qoi
        self.qoi_using_exact = self.get_avg(self.u)
        self.qoi_est_direct_using_exact = self.qoi_using_exact - self.lf_qoi
        self.qoi_est_using_exact[0] = dl.assemble(self.qoi_err_est[0])
        self.qoi_est_using_exact[1] = dl.assemble(self.qoi_err_est[1])
        
    def solve_err(self, x):
        
        if self.sim_count%self.print_freq == 0:
            self.pmsg("Solve for errors with parameters {}".format(np.exp(x[PARAMETER].get_local())), 1)
            
        # replace parameter with specified parameters
        self.set_parameters(x[PARAMETER]) 
        
        # set variational form
        self.set_varf_forms()
        
        # if LF is not solved, solve it now
        if self.is_lf_solved == False:
            self.solve_fwd_bck_lf()
        
        # re-init solution fields
        self.copy_function(self.eu, self.u_ic)
        self.copy_function(self.ep, self.u_ic)
        
        # solve
        dl.solve(self.FE[0] == 0, self.eu, self.u_bc, \
                         solver_parameters={"newton_solver":{"relative_tolerance": 1e-6}})
        
        dl.solve(self.FE[1] == 0, self.ep, self.u_bc, \
                         solver_parameters={"newton_solver":{"relative_tolerance": 1e-6}})
        
        # update error fields
        self.err_to_sol()
        self.err_to_sol_adj()
        
        # qoi and error in qoi
        self.qoi_using_approx = self.get_avg(self.u)
        self.qoi_est_direct_using_approx = self.qoi_using_approx - self.lf_qoi
        self.qoi_est_using_approx[0] = dl.assemble(self.qoi_err_est[0])
        self.qoi_est_using_approx[1] = dl.assemble(self.qoi_err_est[1])
        

    def set_save_files(self, file_path = None):
        """ Create output directory and files for saving state of model. """
        
        out_path = self.out_path + str(self.sim_count) + '/'
        if file_path is not None:
            out_path = self.out_path + file_path + '/'
            
        pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
            
        self.file= [None]*6
        names = ["u", "p", "u_lf", "p_lf", "eu", "ep"]
        for i in range(6):
            self.file[i] = dl.File(out_path + names[i] + '.pvd', "compressed")
        
        if self.file is None:
            raise Exception("Files for saving are not created.")
            
    def save_state(self):
        """ Save the state of pde system to file. """
        
        # write states to file
        usave = [self.u, self.p, self.u0, self.p0, self.eu, self.ep]
        for i in range(6):
            self.file[i] << (usave[i], 0)            
        
    def solveFwd(self, out, x):        
        """ 
        This function is called by HippyLib during MCMC simulatin. 
        Return the model output which in this case is the difference in LF and HF QOIs. 
        """
        self.save = False
        self.verbosity = 1
        
        if self.use_approx:
            self.solve_err(x)
            
            if self.which_est == 0:
                out[0] = np.array([self.qoi_est_direct_using_approx])[:, None]
            elif self.which_est == 1:
                out[0] = np.array([self.qoi_est_using_approx[0]])[:, None]
            elif self.which_est == 2:
                out[0] = np.array([self.qoi_est_using_approx[0]])[:, None]
        else:
            self.solve_fwd_bck(x)
            
            if self.which_est == 0:
                out[0] = np.array([self.qoi_est_direct_using_exact])[:, None]
            elif self.which_est == 1:
                out[0] = np.array([self.qoi_est_using_exact[0]])[:, None]
            elif self.which_est == 2:
                out[0] = np.array([self.qoi_est_using_exact[0]])[:, None]
                
        if self.sim_count%self.print_freq == 0 and self.verbosity > 0:
            print('  Count: ', self.sim_count, ', QoI: ', out[0])
            
        # check if we need to save the pde fields
        saved_state_current_sim = False
        if self.save and self.sim_out_current % self.sim_out_freq == 0 \
                     and self.sim_out_current < self.sim_max_out:

            saved_state_current_sim = True
            self.set_save_files()

            # actually save files
            self.save_state()

        # increment the counter for number of times the forward model is solved
        self.sim_count += 1
        if saved_state_current_sim:
            self.sim_out_current += 1
