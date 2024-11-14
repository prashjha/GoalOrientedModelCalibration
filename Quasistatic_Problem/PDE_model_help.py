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
# import petsc4py
# petsc4py.init(sys.argv)
# from petsc4py import PETSc
import dolfin as dl
import hippylib as hl
import scipy.stats as sstats


# In[2]:


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

# needed for sensitivity analysis
from SALib.analyze import morris
from SALib.analyze import sobol
import SALib.sample as sample
from SALib.sample import saltelli
from SALib.plotting.morris import horizontal_bar_plot, covariance_plot, sample_histograms


# In[7]:


# dl.parameters["form_compiler"]["optimize"]     = True
# dl.parameters["form_compiler"]["cpp_optimize"] = True
# dl.parameters["form_compiler"]["representation"] = "uflacs"
# dl.parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"


# # Diffusion-reaction HP MRI model with vascular domain

# In[3]:


class GenericReprBase:
    
    def get_info(self,delim = '\n '):
        name = type(self).__name__
        
        vars_list = []
        for key, value in vars(self).items():
            if isinstance(value, np.ndarray):
                value = value.flatten()
            vars_list.append(f'{key}={value!r}')
            
        vars_str = delim.join(vars_list)
        return f'{name}({vars_str})'

    def __repr__(self,delim = '\n '):
        return self.get_info(delim)


# In[ ]:

# In[29]:


class FwdModelSource(dl.UserExpression):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
  
    def eval(self, values, x):
        n = 4*np.pi
        values[0] = 10*np.power(np.cos(n*x[0])*np.cos(n*x[1]), 2)
        
def FwdBoundary(x, on_boundary):
    return on_boundary 


# In[31]:

STATE = 0
PARAMETER = 1


# In[6]:


class Misfit:
    def __init__(self, data):
        
        self.noise_variance = None
        self.data = data

    def set_noise_variance(self, percentages, no_scale = False):
        """Setting the noise variance based on the percentages         (numpy array of 2 values) at one standard deviation"""
        if no_scale:
            self.noise_variance = percentages
        else:
            if isinstance(percentages, list) == False:
                self.noise_variance = np.array([(self.data[0]*percentages)**2])
            else:
                if len(percentages) != len(self.data):
                    raise IndexError("The percentages shape does not match the data shape")

                self.noise_variance = []
                for i in range(len(self.data)):
                    self.noise_variance.append((self.data[i]*percentages[i])**2)
                self.noise_variance = np.array(self.noise_variance)
        
    def cost(self, x):
        
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        
        if not x[STATE].shape == self.data.shape:
            raise IndexError("The state output is of shape ", x[STATE].shape,                             ", while data is of shape ", self.data.shape, ".")
            
        return np.sum(np.divide(np.power(self.data - x[STATE], 2), 2*self.noise_variance))


# In[7]:


class FullTracer:
    def __init__(self, Vh, dir_name, save = True, print = True, print_freq = 1):
        self.posterior = None
        self.qoi = None
        self.acceptance_rate = None
        self.cost = None
        self._dir_name = dir_name
        self._time = 0
        self._parameter = None
        self._param_dim = Vh[PARAMETER].dim()
        self.accepted = 0
        self.rejected = 0
        self._save = save
        self._print = print
        self._sample_count = 0
        self._print_freq = print_freq
        
    def append(self, current, q):
        
        if self._parameter is None:
            self._parameter = current.m.get_local()
            acceptance_rate = 0.0
        elif not np.array_equal(self._parameter, current.m.get_local()):
            self.accepted += 1
            acceptance_rate = self.accepted/float(self.accepted + self.rejected+1)
            self.write(current, acceptance_rate)
        else:
            self.rejected += 1
            acceptance_rate = self.accepted/float(self.accepted + self.rejected+1)
        if self._print:
            if self._sample_count % self._print_freq == 0:
                print("Acceptance ratio: ", acceptance_rate*100, "%")
        
        self._sample_count += 1
    
    def write(self, current, acceptance_rate):
        self._parameter = current.m.get_local()
        m_vector = current.m.get_local().reshape((1,self._param_dim))
        q_vector = current.u.reshape((1, 1))
        ar_vector = np.array([acceptance_rate])
        cost_vector = np.array([current.cost])
        if self.posterior is None:
            self.posterior = m_vector
        else:
            self.posterior = np.append(self.posterior, m_vector, axis = 0)
        if self.qoi is None:
            self.qoi = q_vector
        else:
            self.qoi = np.append(self.qoi, q_vector, axis = 0)
        if self.acceptance_rate is None:
            self.acceptance_rate = ar_vector
        else:
            self.acceptance_rate = np.append(self.acceptance_rate, ar_vector)
        if self.cost is None:
            self.cost = cost_vector
        else:
            self.cost = np.append(self.cost, cost_vector)
        if self._save and self.accepted % 10 == 0:
            self.save()
            
    def save(self):
        np.save(self._dir_name +'/param_samples.npy', self.posterior)
        np.save(self._dir_name +'/qoi.npy', self.qoi)
        np.save(self._dir_name +'/acceptance_rate.npy', self.acceptance_rate)
        np.save(self._dir_name +'/cost.npy', self.cost)

def plot_prior(prior, titles_str = None):
    noise = dl.Vector()
    prior.init_vector(noise,"noise")
    hl.parRandom.normal(1., noise)
    m = dl.Vector()
    prior.init_vector(m, 0)
    a = m.get_local()
    param_dim = len(a)
    n_samples = 50000
    samples = np.empty((n_samples, param_dim))
    for i in range(n_samples):
        hl.parRandom.normal(1., noise)
        prior.sample(noise,m)
        samples[i,:] = m.get_local()

    real_mean = [np.mean(np.exp(samples[:, i])) for i in range(param_dim)]
    ln_mean = [np.mean(samples[:, i]) for i in range(param_dim)]

    # plot
    titles = None 
    titles_reald = None 
    if titles_str is None:
        titles = [r'$\ln(\theta_{}$)'.format(i+1) for i in range(param_dim)]
        titles_real = [r'$\theta_{}$'.format(i+1) for i in range(param_dim)]
    else:
        titles_real = titles_str
        titles = [r'$\ln({})$'.format(i) for i in titles_real]

    fig, axs = plt.subplots(nrows=2, ncols=param_dim, figsize=(30, 16))
    fs = 14
    sample_ids = np.arange(samples.shape[0])
    idx = 0
    for col in range (param_dim):
        row = 0
        ax = sns.distplot(samples[:, idx], hist=False, norm_hist = True, ax=axs[row, col])
        
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        vline_x = ln_mean[idx]
        text_x = vline_x + 0.1 * (xlims[1] - xlims[0])
        text_y = 0.5 * (ylims[0] + ylims[1])
        
        #text2_y = 0.2 * (ylims[1] - ylims[0])
        #text2_x = vline_x - 0.2 * (xlims[1] - xlims[0])

        axs[row, col].axvline(x=vline_x, color='r')
        axs[row, col].text(text_x, text_y, \
                           titles[idx] + '=\n' + '%6.4f' % (vline_x), \
                           color = 'r', va = 'center')
        #axs[row, col].text(text2_x, text2_y, '%5.3e' % (vline_x))

        #axs[row, col].plot(sample_ids, samples[:, idx], '*-')
        #axs[col].set_title(titles[idx])
        axs[row, col].set_xlabel(titles[idx])
        axs[row, col].set_ylabel(r'$\rho($'+titles[idx]+r'$)$')
        
        row = 1
        ax = sns.distplot(np.exp(samples[:, idx]), hist=False, norm_hist = True, ax=axs[row, col])

        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        vline_x = real_mean[idx]
        text_x = vline_x + 0.1 * (xlims[1] - xlims[0])
        text_y = 0.5 * (ylims[0] + ylims[1])
        
        #text2_y = 0.2 * (ylims[1] - ylims[0])
        #text2_x = vline_x - 0.2 * (xlims[1] - xlims[0])

        axs[row, col].axvline(x=vline_x, color='r')
        axs[row, col].text(text_x, text_y, \
                           titles_real[idx] + '=\n' + '%6.4f' % (vline_x), \
                           color = 'r', va = 'center')
        #axs[row, col].text(text2_x, text2_y, '%5.3e' % (vline_x))

        #axs[row, col].plot(sample_ids, np.exp(samples[:, idx]), '*-')
        #axs[col].set_title(titles[idx])
        axs[row, col].set_xlabel(titles_real[idx])
        axs[row, col].set_ylabel(r'$\rho($'+titles_real[idx]+r'$)$')
        
        idx +=1
        
    #plt.xlabel('samples')
    plt.tight_layout()

def run_and_compare_approximations(m_test, pde, save_suf = ''):
    x = [None, pde.generate_parameter(), None]
    x[PARAMETER].set_local(m_test)

    msg_tot = ''

    ss = '_' + save_suf if save_suf != '' else ''

    if save_suf != '':
        msg = '\n\nsave_suf = {}\n\n'.format(save_suf)
        msg_tot = msg_tot + msg 
        print(msg)

    msg = "\n\n Test parameters = {}\n\n".format(np.exp(x[PARAMETER].get_local()))
    msg_tot = msg_tot + msg 
    print(msg)

    ## solve nonlinear model
    msg = "\n\nSolve original forward and dual problem\n"
    msg_tot = msg_tot + msg 
    print(msg)
    pde.solve_fwd_bck(x)

    ## save
    pde.set_save_files('exact' + ss)
    pde.save_state()

    ## solve approximate error to compute QoI
    msg = "\n\nSolve for approximate errors\n"
    msg_tot = msg_tot + msg 
    print(msg)
    pde.solve_err(x)

    ## save
    pde.set_save_files('err' + ss)
    pde.save_state()

    #print
    msg = '\n *** comparing various estimates ***\n'
    msg_tot = msg_tot + msg 
    print(msg)

    msg = '\nQ(u0) = {}'.format(pde.lf_qoi)
    msg_tot = msg_tot + msg 
    print(msg)

    msg = '\n\nQoI using exact\n'
    msg_tot = msg_tot + msg 
    print(msg)

    msg = 'Q(u) = {}'.format(pde.qoi_using_exact)
    msg_tot = msg_tot + msg 
    print(msg)

    msg = '\nError Q(u) - Q(u0) = {}, Percent error = {}'.format(pde.qoi_est_direct_using_exact, \
                                        100 * pde.qoi_est_direct_using_exact / pde.lf_qoi)
    msg_tot = msg_tot + msg 
    print(msg)

    msg = '\nEstimate 1 Q(u) - Q(u0) = {}, Percent error = {}'.format(pde.qoi_est_using_exact[0], \
                                        100 * pde.qoi_est_using_exact[0] / pde.lf_qoi)
    msg_tot = msg_tot + msg 
    print(msg)

    msg = '\nEstimate 2 Q(u) - Q(u0) = {}, Percent error = {}'.format(pde.qoi_est_using_exact[1], \
                                        100 * pde.qoi_est_using_exact[1] / pde.lf_qoi)
    msg_tot = msg_tot + msg 
    print(msg)

    msg = '\n\nQoI using approximate\n'
    msg_tot = msg_tot + msg 
    print(msg)

    msg = '\nQ(u) = {}'.format(pde.qoi_using_approx)
    msg_tot = msg_tot + msg 
    print(msg)

    msg = '\nError Q(u) - Q(u0) = {}, Percent error = {}'.format(pde.qoi_est_direct_using_approx, \
                                        100 * pde.qoi_est_direct_using_approx / pde.lf_qoi)
    msg_tot = msg_tot + msg 
    print(msg)

    msg = '\nEstimate 1 Q(u) - Q(u0) = {}, Percent error = {}'.format(pde.qoi_est_using_approx[0], \
                                        100 * pde.qoi_est_using_approx[0] / pde.lf_qoi)
    msg_tot = msg_tot + msg 
    print(msg)

    msg = '\nEstimate 2 Q(u) - Q(u0) = {}, Percent error = {}\n\n'.format(pde.qoi_est_using_approx[1], \
                                        100 * pde.qoi_est_using_approx[1] / pde.lf_qoi)
    msg_tot = msg_tot + msg 
    print(msg)


    return msg_tot

