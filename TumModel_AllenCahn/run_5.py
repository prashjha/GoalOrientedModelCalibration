import sys
import os
import random
import math
import copy
import time
import numpy as np
import pathlib

import argparse

import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import dolfin as dl
import hippylib as hl

import scipy.stats as sstats
import scipy.io as sio
import scipy.integrate as sint
from scipy.integrate import solve_ivp
import scipy.stats as sstats
from scipy.linalg import expm

# needed to read vtu files
import vtk
from vtk.util import numpy_support as VN

dl.parameters["form_compiler"]["optimize"]     = True
dl.parameters["form_compiler"]["cpp_optimize"] = True
dl.parameters["form_compiler"]["representation"] = "uflacs"
dl.parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

# import
from PDE_model_help import GenericReprBase, NutSource, NutSourceLeftEdge, QoIFn, IC_Tum, TC_Tum, IC_Error_Tum, TC_Error_Tum
from PDE_model_help import Misfit, FullTracer, plot_prior
from PDE_model import PDEModel

# output to a file
#file_path = 'randomfile.txt'
#sys.stdout = open(file_path, "w")


STATE = 0
PARAMETER = 1

run_dir = 'run_5/'
theta0 = [0.2, 0.1, 0.05]
dt = 0.005
use_approx = True
which_est = 0
noise_variance = np.power(0.0001, 2)

num_samples = 2000
pcn_s = 0.4
p_mode = np.array([0.5, 0.1, 0.01, 1.])
p_sigma = np.array([0.4, 0.4, 0.4, 0.4])
out_path = '/tmp/TumModel_AllenCahn/' + run_dir

mcmc_path = run_dir # results in chain_<id> folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' Goaloriented')

    # AMAL: start
    parser.add_argument('--chain_id',
                        default=1,
                        type=int,
                        help="ID of MCMC chain")
    
    args = parser.parse_args()
    print('Arguments passed: ')
    print(args)
    try:
        dl.set_log_active(False)
        #dl.set_log_level(40)
    except:
        pass

    # mcmc chain
    chain_id = args.chain_id

    # set paths
    out_path_chain = out_path + 'chain_%d' % (chain_id)
    pathlib.Path(out_path_chain).mkdir(parents=True, exist_ok=True)
    mcmc_path_chain = mcmc_path + 'chain_%d/' % (chain_id)
    pathlib.Path(mcmc_path_chain).mkdir(parents=True, exist_ok=True)


    # load mesh
    mesh = dl.RectangleMesh(dl.Point(0, 0), dl.Point(1,1), 50, 50)

    # FE space
    FE_polynomial = 1
    Vu = dl.FunctionSpace(mesh, "Lagrange", FE_polynomial)

    # Define elements: P1 and real number
    P1  = dl.FiniteElement("Lagrange", mesh.ufl_cell(), 2)
    R   = dl.FiniteElement("R", mesh.ufl_cell(), 0)

    # parameter space
    # lam_p, lam_d, eps, C
    param_dim = 4
    Vh_PARAMETER = dl.VectorFunctionSpace(mesh, "R", degree=0, dim=param_dim)

    # Define the state space (solution space) together with parameter space
    Vh = [Vu, Vh_PARAMETER, None]

    print_freq = 10

    # create pde problem
    pde = PDEModel(Vh, dt = dt, \
                   theta0 = theta0, \
                   save = False, out_path = out_path_chain)
    pde.print_freq = print_freq

    pde.use_approx = use_approx
    pde.which_est = which_est

    # create misfit function
    total_u = np.zeros((1,1))
    misfit = Misfit(total_u)
    misfit.set_noise_variance([noise_variance], no_scale = True)

    # set prior distribution (mode from the validation data in the paper)
    p_mean = np.log(p_mode) + p_sigma**2
    mean = pde.generate_parameter()
    mean.set_local(p_mean)

    # prior distribution (lognormal)
    prior = hl.GaussianRealPrior(Vh[PARAMETER], np.diag(p_sigma**2), mean=mean)

    # plot prior
    # plot_prior(prior)

    # create hippylib model (consisting of forward problem, prior distribution, and misfit function)
    # Forward problem: must provide 'solveFwd()' method
    # misfit: must provide 'cost()' method
    model = hl.Model(pde, prior, misfit)
    
    # preconditioned Crank-Nicolson (pCN) sampler
    kernel = hl.pCNKernel(model)
    kernel.parameters["s"] = pcn_s
    chain = hl.MCMC(kernel)
    
    # save data
    np.save(mcmc_path_chain + 'data.npy', misfit.data)
    
    tracer = FullTracer(Vh, mcmc_path_chain, print = True, print_freq = print_freq)

    chain.parameters["number_of_samples"]     = num_samples
    chain.parameters["burn_in"]               = 0
    chain.parameters["print_progress"]        = 10
    chain.parameters["print_level"]           = -1
    
    for idx in range(chain_id-1):
        chain.consume_random() # exhaust randoms used in previous chains
        
    noise = dl.Vector()
    prior.init_vector(noise,"noise")
    hl.parRandom.normal(1., noise)
    m0 = dl.Vector()
    prior.init_vector(m0, 0)
    prior.sample(noise,m0)

    ## solve linear model
    print("\n\nGetting low-fidelity solution\n")
    #pde.verbosity = 3
    pde.solveFwdBckLF(True, True)

    ## compute error using exact solutions
    print("\n\nCompute error at one parameter sample\n")
    x = [None, pde.generate_parameter(), None]
    #p_test = p_mode #np.array([1, 0.1, 0.05, 1.])
    #m_test = np.log(p_test)
    m_test = dl.Vector()
    prior.sample(noise, m_test)
    print('test parameter: {}'.format(m_test))
    x[PARAMETER].set_local(m_test)

    ## solve nonlinear model
    print("\n\nSolve original forward and dual problem\n")
    pde.solveFwdBck(x, True, True)

    ## solve approximate error to compute QoI
    print("\n\nSolve for approximate errors\n")
    pde.solveFwdBckErr(x, True, True)

    #print
    print('\n *** comparing various estimates ***\n')
    print('\nQ(u0) = ', pde.lf_qoi)

    print('\n\nQoI using exact\n')
    print('Q(u) = ', pde.qoi_using_exact)
    print('\nError Q(u) - Q(u0) = ', pde.qoi_using_exact - pde.lf_qoi)
    print('\nEstimate 1 Q(u) - Q(u0) = ', pde.qoi_est1_using_exact)
    print('\nEstimate 2 Q(u) - Q(u0) = ', pde.qoi_est2_using_exact)
    print('\nEstimate 3 Q(u) - Q(u0) = ', pde.qoi_est3_using_exact)

    print('\n\nQoI using approximate\n')
    print('\nQ(u) = ', pde.qoi_using_approx)
    print('\nError Q(u) - Q(u0) = ', pde.qoi_using_approx - pde.lf_qoi)
    print('\nEstimate 1 Q(u) - Q(u0) = ', pde.qoi_est1_using_approx)
    print('\nEstimate 2 Q(u) - Q(u0) = ', pde.qoi_est2_using_approx)
    print('\nEstimate 3 Q(u) - Q(u0) = ', pde.qoi_est3_using_approx)

    ## run MCMC
    print('\n\n *** Running MCMC simulation ***\n\n')
    n_accept = chain.run(m0, qoi = None, tracer = tracer)
    print("Number accepted = {0}".format(n_accept))
    tracer.save()
