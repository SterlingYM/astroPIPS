# Period-determination and Identification Pipeline Suite (PIPS)

![GitHub tag (latest SemVer pre-release)](https://img.shields.io/github/v/tag/SterlingYM/PIPS?include_prereleases)
![GitHub](https://img.shields.io/github/license/SterlingYM/PIPS)
[![Run tests](https://github.com/SterlingYM/astroPIPS/actions/workflows/run_tests.yml/badge.svg?branch=master)](https://github.com/SterlingYM/astroPIPS/actions/workflows/run_tests.yml)
[![codecov](https://codecov.io/gh/SterlingYM/PIPS/branch/master/graph/badge.svg?token=R1W2S30XV2)](https://codecov.io/gh/SterlingYM/PIPS)
[![Documentation Status](https://readthedocs.org/projects/pips/badge/?version=latest)](https://pips.readthedocs.io/en/latest/?badge=latest)

PIPS is a Python pipeline designed to analyze the lightcurves of astronomical objects whose brightness changes periodically. Our pipeline can be imported quickly and is designed to be user friendly. PIPS was originally developed to determine the periods of RR Lyrae variable stars and offers many features designed for variable star analysis. We have expanded PIPS into a suite that can obtain period values for almost any type of lightcurve with both speed and accuracy. PIPS can determine periods through several different methods, analyze the morphology of lightcurves via fourier analysis, estimate the statistical significance of the detected signal, and determine stellar properties based on preexisting stellar models. Currently our team is also exploring the possibility of using this pipeline to detect periods of exoplanets as well.

A detailed description of PIPS and its algorithms is provided in [our paper](https://arxiv.org/abs/2107.14223).

__NOTE: We are currently updating the documentation to match the content with the paper__

(last edit: February 24th, 2022) 

![](sample_lightcurve.png)

--------------------------
## Developers

* Head developer: [Yukei S. Murakami](https://www.fromthecalmsea.com) (sterling.astro@berkeley.edu)
* Developers: [Arjun Savel](https://www.arjunsavel.com) (asavel@umd.edu), [Andrew Hoffman]() (andrew@hoffman.aero), [James Sunseri](https://sites.google.com/view/jamessunseri/home) (jamessunseri@berkeley.edu)

--------------------------
## Publications
Please cite the following if PIPS is utilized for a scientific project:
* Hoffman et al. 2021 [![DOI:10.1093/mnras/stab010](https://zenodo.org/badge/DOI/10.1093/mnras/stab010.svg)](https://doi.org/10.1093/mnras/stab010) (arxiv: [2008.09778](https://arxiv.org/abs/2008.09778))
* Murakami et al. (Submitted to MNRAS. arxiv: [2107.14223](https://arxiv.org/abs/2107.14223))


--------------------------
## Dependencies
* python (>=3.7)
* numpy
* scipy
* matplotlib
* time
* astropy
* ~~seaborn~~


--------------------------
## Usage (ver 0.3.0)

1. ```$ git clone https://github.com/SterlingYM/PIPS```
2. ```$ cd PIPS```
4. ```$ jupyter notebook```
5. Start a new Python notebook. In a jupyter cell, run the following:
```python
import PIPS

# data preparation -- create [time,mag,mag_err] list
data = PIPS.data_readin_LPP('sample_data/005.dat',filter='V')
phot_obj = PIPS.photdata(data)

# period detection
star.get_period(multiprocessing=False)

# generate best-fit light curve
x_th,y_th = star.get_bestfit_curve()

# plot light curve
star.plot_lc()
plt.plot(x_th/star.period,y_th,c='yellowgreen',lw=3,alpha=0.7) # x-axis normalized to unitless phase
plt.plot(x_th/star.period+1,y_th,c='yellowgreen',lw=3,alpha=0.7)
```

Sample data credit: UCB SNe Search Team (Filippenko Group)

