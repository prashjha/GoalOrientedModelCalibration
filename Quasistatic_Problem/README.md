# Quasistatic problem (section 4.1 in Jha and Oden JCP 2022)

## plot_calibration_results.ipynb
Collects MCMC samples from all chains and plots various results

Results and plots corresponding to `run_1.py` in `run_1` directory are used in the article. `run_2.py` generates posterior samples where the forward model surrogate is not used, i.e., `true posterior` upto numerical and sampling errors.

## plot_multi_results.ipynb
Essentially compares the posterior from two methods: when the surrogate of the forward model is used `run_1.py` and when not used `run_2.py`.

## posterior_update_codes.ipynb
Notebook was used to develop script to run MCMC. 
