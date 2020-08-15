# Period-determination and Identification Pipeline Suite (PIPS)
A more advanced version of getPeriod, which determines the period of short-period variable stars from photometry data. This code is under development and this documentation may be outdated.

Yukei S. Murakami<br>
sterling.astro@berkeley.edu

-------------------------
![](sample_output.png)


---------------------
## Dependency

* numpy
* scipy
* matplotlib
* time
* astropy
* seaborn


---------------------
## Usage (ver 0.2.0)

1. ```$mkdir gp2_workspace```
2. ```$cd gp2_workspace```
3. ```$git clone https://github.com/SterlingYM/getPeriod2```
4. ```$jupyter notebook```
5. In a jupyter cell, run following:
```python
import getPeriod2 as gp2
data_V,_ = gp2.data_readin('getPeriod2/sample_data/000.dat')
phot_obj = gp2.photdata(data_V)

filtered_data = phot_obj.phot_err_cut()
filtered_data.detect_period()
print(filtered_data.period)
```

Sample data credit: UCB SNe Search Team (Filippenko Group)

---------------------
## object ```photdata```

PhotData object (```getPeriod2.photdata```) is the main component of this pipeline and contains photometry data. This PhotData is consisted of the data itself (time, magnitude/flux, and mag/flux error), parameters for various operations, and method functions to analyze the data. Having correctly set parameters is essential to obtain reliable results from this pipeline, and understanding the correct usage of method functions will allow an easier and faster analysis.

### photdata: data
|name |description |unit |default|
|:----|:-----------|:----|:------|
|```photdata.x```   | An array of time values |days|```data[0]```|
|```photdata.y```   | An array of magnitude or flux values |mag (or flux)|```data[1]```|
|```photdata.yerr```| An array of error for ```photdata.y``` values. This error is in the absolute value (e.g. ```photdata.yerr[0]=0.3``` when ```photdata.y[0]=16``` means the magnitude for the first datapoint is 16+-0.3.)| mag (or flux) | ```data[3]```|
|```photdata.t```   | Identical to ```photdata.x``` by default. The values are only copied to each other when the object is initialized, and the arrays are not identical to each other anymore when one of them is manually edited.|days|```photdata.x```|
|```photdata.mag``` | Identical to ```photdata.y```. Similar to ```photdata.t```. |mag (or flux)|```photdata.y```|
|```photdata.mag_err```   | Identical to ```photdata.yerr```. Similar to ```photdata.t```. |mag(flux)|```photdata.yerr```|
|```photdata.period```    | Period value determined by the pipeline. Initially ```None```, and the value appears once ```photdata.detect_period()``` is executed. | days | ```None```|
|```photdata.period_err```| Uncertainty for the period value | days | ```None```|
|```photdata.amplitude``` | Amplitude of the best-fit lightcurve (not the data itself).|mag (or flux) |```None```|

### photdata: parameters
|name |description |unit |default|
|:----|:-----------|:----|:------|
|```photdata.A0```| Initial guess value for the mean magnitude. ```photdata.A0 = np.mean(photdata.y)``` is recommended when manually setting.| mag (or flux) | 15|
|```photdata.K``` | Number of Fourier terms used to obtain best-fit lightcurve for phase-folded data. ||5|
|```photdata.err_cut_threshold```| The tolerance for error values. Data with an error larger than this value will be removed when ```phot_err_cut(Nsigma)``` is executed (see below for more detail). The value is null by default, and is set automaticall at Nsigma\*std value from the mean of the error when the cut function is called.|mag (or flux)|```None```|
|```photdata.quiet```| A switch to enable/disable text outputs during the compututation. ```True``` will suppress text messages. Not all text messages are controlled by this variable. **Under development**|boolean|```True```|
|```photdata.LS_min_f```| The minimum frequency to search with Lomb-Scargle. The value of frequency is 1/period, and this variable corresponds to maximum period searched by LS function. |1/days|0.5|
|```photdata.LS_max_f```| The maximum frequency to search with Lomb-Scargle. |1/days|10|

### photdata: functions 
##### (i) Initialization
|name |description |required arguments |optional arguments|return|
|:----|:-----------|:------------------|:-----------------|:-----|
|```photdata(data)``` (```photdata.__init__()```)|Initialization function. the argument ```data``` is a list of three arrays: \[x,y,yerr\]. |```data```||PhotData|

##### (ii) Fourier composition
|name |description |required arguments |optional arguments|return|
|:----|:-----------|:------------------|:-----------------|:-----|
|```photdata.fourier_composition(x,omega,A0,*ab_list)```|A function to generate y-values at given x based on Fourier parameters and angular frequency given.|```x```: array, ```omega```: 2\*pi/period, ```A0```: see ```photdata.A0```, ```ab_list```: Fourier parameters||array(y value)|
|```photdata.fourier_composition_folded(x,period,A0,*ab_list)```|A function to generate y-values at the phase, which is generated by folding x-values at given period.|```x```: array, ```period```: period value, ```A0```: see ```photdata.A0```, ```ab_list```: Fourier parameters||array(y value)|

##### (iii) Data manipulation
|name |description |required arguments |optional arguments|return|
|:----|:-----------|:------------------|:-----------------|:-----|
|```photdata.phot_err_cut(Nsigma=1)```|Performs a cut based on the error value. The threshold is determined by either manually set ```photdata.err_cut_threshold``` or the argument ```Nsigma``` (only when threshold is ```None```). Returns another photdata. **This function does not overwrite the existing data in the object. Returned new object needs to be stored somewhere else.**||```Nsigma```: |PhotData|

##### (iv) Data analysis
|name |description |required arguments |optional arguments|return|
|:----|:-----------|:------------------|:-----------------|:-----|
|```photdata.calc_chisq(FF_popt)```|Calculates the chi square value based on given parameters. An internally called function for other functions below.|```FF_popt```||chi2,Ndata| 
|```photdata.get_best_fit_at_p(period)```|Determines the best-fit set of Fourier parameters for the phase-folded data at given period.|```period```: float||popt|
|```photdata.test_global_potential_engine(p_test)```|A helper function for the function below. Evaluates the chi2 value at single test period.|```p_test```: float||\[period,chi2/Ndata\]|
|```photdata.test_global_potential(test_p_min,test_p_max,test_num)```|Tests the liear regression algorithm on the global chi2 potential and returns the best-fit period list for each 'test' (=initial guess) period value.|```test_p_min```: minimum test period (float), ```test_p_max```: maximum test period (float), ```test_num```: number of test periods (int)||test_p_list, estimated_p_list, chi2_list|
|```photdata.get_global_potential_engine(p_test)```|A helper function for the function below. Evaluates the chi2 value at a single fixed period.|```p_test```: float||chi2/Ndata|
|```photdata.get_global_potential(test_p_list)```|Obtains the list of chi2 values at each fixed period. Period is not a free parameter unlike ```test_global_potential()```.|```test_p_list```: array||chi2_list|
|```photdata.get_LS()```|Calls astropy's Lomb-Scargle function to obtain the initial guess. |||freq_list,power_list,\[f0,f1,f2,f3\]|

##### (v) Plotting
|name |description |required arguments |optional arguments|return|
|:----|:-----------|:------------------|:-----------------|:-----|
|```photdata.potential_plot(test_p_list,estimated_p_list,chi2_list,chi2_potential)```|||||
|```photdata.phot_lc_folded(period,title='',fitting=True,invert_y=True)```|||||

##### (vi) Period detection
|name |description |required arguments |optional arguments|return|
|:----|:-----------|:------------------|:-----------------|:-----|
|```photdata.detect_period(max_iteration=100,test_num=100,threshold=0.1,initial_search_width=1e-5,convergence_size_ratio=0.03,title='',show_plot=True,show_results=True)```||||period,period_err,amplitude|
