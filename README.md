# getPeriod2
A more advanced version of getPeriod. Determines the period of short-period variable stars from sky-survey-type data (i.e. widely separated photometry). This code is under development.

Yukei S. Murakami<br>
sterling.astro@berkeley.edu

-------------------------
![](sample_output.png)

---------------------
# A note on Uncertainty

I think we have three uncertainties

* __instrumental (telescope's time precision)__: This is about 1e-6 for us. This becomes smaller as data spans longer period. For example, if the data spans 100 days, the error is 1e-6/100 = 1e-8. Typically this is smaller or within the order of magnitude for our data.
* __systematic (chi2 quadratic fitting)__: this occurs when the potential is in fact not a quadratic function but a combination of two or more of those. This is typically much smaller than the size of statistical uncertainty mentioned below.
* __statistical (scatter of best-fit period)__: This is due to the 'mechanics' of fitting -- the random initial velocity (=random walk width), smoothness of the surface (=microscopic local potentials), and the steepness of the potential well will all affect on the scatter of the resulting period. This is what we are mainly analyzing here.

__Will they add together?__: No. We are talking about period, which is a single quantity. Whichever the largest will be our error.

__Will the size of mag error affect on any of these?__: Probably not. We are taking chi2 as a potential, meaning that it is only realtive to the neighbor values of chi2. When the mag error changes, chi2 changes too, but it does so uniformly, and the location of the potential does not change.

TODO: clean this up 

----------------------------------------------------
# ```search_period()``` function

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


