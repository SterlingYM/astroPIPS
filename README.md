# getPeriod2
A more advanced version of getPeriod. Determines the period of short-period variable stars from sky-survey-type data (i.e. widely separated photometry). This code is under development and this documentation may be outdated.

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
---------------------
## A note on Uncertainty

I think we have three uncertainties

* __instrumental (telescope's time precision)__: This is about 1e-6 for us. This becomes smaller as data spans longer period. For example, if the data spans 100 days, the error is 1e-6/100 = 1e-8. Typically this is smaller or within the order of magnitude for our data.
* __systematic (chi2 quadratic fitting)__: this occurs when the potential is in fact not a quadratic function but a combination of two or more of those. This is typically much smaller than the size of statistical uncertainty mentioned below.
* __statistical (scatter of best-fit period)__: This is due to the 'mechanics' of fitting -- the random initial velocity (=random walk width), smoothness of the surface (=microscopic local potentials), and the steepness of the potential well will all affect on the scatter of the resulting period. This is what we are mainly analyzing here.

__Will they add together?__: No. We are talking about period, which is a single quantity. Whichever the largest will be our error.

__Will the size of mag error affect on any of these?__: Probably not. We are taking chi2 as a potential, meaning that it is only realtive to the neighbor values of chi2. When the mag error changes, chi2 changes too, but it does so uniformly, and the location of the potential does not change.

TODO: clean this up 

----------------------------------------------------
## ```search_period()``` function

__This function works with the assumption that the best-fit period has the lowest chi2.__

### Phase zero: potential lock
First move around until the bottom of the potential is found. this phase is added in case Lomb-Scargle's estimation is way off from the true value.

### Phase I: best-fit-chi2-based
This method selects only one potential when there are multiple local minima.
Same result can be achieved by gradually narrowing the search window, but this method is faster especially when the width of the peak is much narrower than the initial search width. 

This phase uses $\sigma_\mathrm{statistical}$ (see below) as a benchmark value for the convergence. It typically 'converges' when a single peak is selected. This is because any fitting with initial guess within that peak (=main peak) can easily fall to the lowest chi2, and selection based on this chi2 value does not exclude those initial guesses, while initial guesses (=search range) outside of this peak will fall to the other direction, yielding much higher chi2 values that will be cut off (based on the threshold).

This phase does not keep the search window centered at the bottom of the potential (peak), but since it includes all initial guess values that yield chi2 value smaller than the threshold, it never rejects the true peak (as long as the overall shape of the best-fit-chi2 distribution is in quadratic).

### Phase II: global_potential_based
This method determines the peak as well as the standard deviation of the period estimation.

$$\sigma = \max(\sigma_\mathrm{statistical},\sigma_\mathrm{systematic})\ ,$$
where 
$$\sigma_\mathrm{statistical}  = \min\left[ \mathrm{std}\big|\mathrm{guess}-\mathrm{estimated}\big| , 
\mathrm{std}(\mathrm{estimated}) \right]\ ,$$
and 
$$\sigma_\mathrm{systematic} = \sqrt{\mathrm{diag}(\mathrm{cov\ matrix})}[\mathrm{idx\ for\ period}] 
\quad(\mathrm{from\ the\ quadratic\ fit})\ .$$

systematic error is only calculated in Phase III, so this phase only considers statistic error.
This phase evaluates $\sigma_\mathrm{statistical}$ and repeats it with smaller search window until the value converges within given threshold (typically 5%). This way the uncertainty has its own uncertainty, assuring the reliability of the uncertainty value up to 2 sig-fig (when threshold is 5%).

This phase keeps the peak centered.


### Phase III: summary
fit again. replaces std with systematic if the fitting is not so good. (5/29/2020 update: now this fitting is done at 5-10$\sigma$ level from the statistical uncertainty, and usually the shape of chi2 potential is a beautiful quadratic, yielding a very small systematic uncertainty.


------------------------

# getPeriod2 (tmp) manual

Yukei S. Murakami
sterling.astro@berkeley.edu
last updated: June 27th, 2020

## introduction

getPeriod2 (name TBD: Free Form Periodic Fitting (FFPF)?) is a python package that is developed for the analysis of variable star photometry data. This algorithm re-constructs light curves of pulsating variable stars through detection of periods. While there are many algorithms and implemented codes available, such as Lomb-Scargle periodogram in Astropy \uk{ref astropy} or Template Fourier Fitting (TFF) \uk{ref paper} for known types of variable stars,  those algorithms may introduce biases to the results due to their function-form dependency, and several improvements were needed upon the analysis of datasets with certain characteristics with which these biases become significant. These characteristics include large separation of photometry data points between each observation window, small signal-to-noise ratio (SNR), small number of data points, and large intrinsic scatter of data points.

We approached this problem by removing assumptions in terms of the periodic shape of light curve and instead by constructing the fit function as the code searches for the best-fit period. This free-form fit function is a Kth order Fourier series, which is constructed by an internal fitting to the folded light curve that is generated at each trial period. Unlike previous studies with a two-step analysis which returns the best-fit period determined by a template (e.g. sinusoidal shape for Lomb-Scargle) and then fits the light curve to this possibly-biased data \uk{ref Gaia?}, this method returns the best-fit period and the light curve determined as a combination. Although the output from this algorithm can be interpreted as a ‘true’ set of data without biases, it should be always kept in mind that this algorithm employs one big assumption that, at the ‘true’ or ‘best-fit’ period, the disagreement (i.e. $\chi^2$ value) between  free-form fit function and data is minimized. \uk{this last sentence should be re-written in more positive way}

The main motivation to develop this algorithm was to adopt existing technique and advance them to the modern form which is capable of processing datasets in the Sky-Survey era \uk{ref Gaia, TESS, WFIRST, etc}. Each year, it is getting less and less likely to have observations optimized and dedicated for the variable star analysis, while more numbers of datasets including variable star photometry are becoming available. This current situation requires that we instead provide upgrades to methods which can process less optimized data.

During the implementation of this algorithm, we encountered several challenges. Although this algorithm is able to completely detect the period and reproduce the light curve of any shape in theory, the incompleteness of the phase coverage and poor quality of data (e.g. large error bars and large intrinsic scatter) determine the limitation of this algorithm. Particularly, limitations in number of Fourier terms K and the need for interpolation of folded data may introduce unwanted biases to the result, and the effects of those are discussed in the validation section.

In addition to the limitation in data, there was another challenge to overcome the limitation of linear-regression tools that are utilized in our implementation: there was sometimes a large initial-guess dependency in the resulting period. This is, we believe, due to an incomplete optimization of linear regression tool. There are more free parameters available for the curve shape (i.e. Fourier parameters) than the parameter for the period (i.e. the period value itself), and this creates many local minima at the surface of $\chi^2$ potential. Best-fit period value may get trapped in the nearest local minimum instead of reaching to the 'true' lowest-$\chi^2$ point, and this occurs when the local minimum is too sharp and small compared to the global minimum.
The linear regression detects the convergence at the local minimum before searching for a larger global minimum, which is the 'true' period value, and this effect is more obvious when the data sampling is less optimal for the variable star analysis (e.g. larger separation between observation, uneven coverage of the phase). This also makes it difficult to determine the uncertainty of resulting value, since the value may have a large offset from the 'true' period at the lowest $\chi^2$ value even though the uncertainty is extremely small (i.e. the size of small local minima) when we analyzed the posterior distribution with MCMC. We approached this issue by combining our initially described free-form fitting and more traditional Fourier fitting which does not include period as its free parameter. 


## algorithm

## application & example

## issues & todo
