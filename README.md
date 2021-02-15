# Period-determination and Identification Pipeline Suite (PIPS)
[![Build Status](https://dev.azure.com/PIPS-developers/PIPS/_apis/build/status/SterlingYM.PIPS?branchName=master)](https://dev.azure.com/PIPS-developers/PIPS/_build/latest?definitionId=1&branchName=master)

PIPS is a pipeline suite designed for analysis of variable lightcurves of astronomical objects. PIPS was originally developed solely to determine the periods of RR Lyrae variable stars but has grown into a suite that can detect periods of variable stars accurately, precisely, and rapidly while offering several other features for variable star anaylsis. With this suite one can determine periods through several different methods, analyze the morphology of lightcurves via fourier analysis, and determine stellar properties based off of preexisting stellar models. Currently our team is also exploring the possibility of using this pipeline to detect periods of exoplanets as well. 

(last edit: February 15th 2021) 

![](sample_output.png)
--------------------------
## Developers

* Head developer: [Yukei S. Murakami](https://www.fromthecalmsea.com) (sterling.astro@berkeley.edu)
* Developers: [Arjun Savel](https://www.arjunsavel.com)(asavel@umd.edu), [Andrew Hoffman]()(andrewmh@berkeley.edu), [James Sunseri](https://sites.google.com/view/jamessunseri/home)(jamessunseri@berkeley.edu)

--------------------------
## Publications
Please cite the following if PIPS is utilized for a scientific project:
* Hoffman et al 2020
* Murakami et al (Paper in prep, Arxiv link pending)


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

1. ```$ mkdir pips_workspace```
2. ```$ cd pips_workspace```
3. ```$ git clone https://github.com/SterlingYM/PIPS```
4. ```$ jupyter notebook```
5. Start a new Python notebook. In a jupyter cell, run the following:
```python
import PIPS
data_V   = PIPS.data_readin_LPP('getPeriod2/sample_data/000.dat',filter='V')
phot_obj = PIPS.photdata(data_V)

filtered_data = phot_obj.phot_err_cut()
filtered_data.detect_period()
# or, as a much quicker method when uncertainty of period is not needed,
# filtered_data.detect_period_quick() 

print(filtered_data.period)
```

Sample data credit: UCB SNe Search Team (Filippenko Group)

