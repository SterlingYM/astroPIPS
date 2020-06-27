# getPeriod2
A more advanced version of getPeriod. Determines the period of short-period variable stars from sky-survey-type data (i.e. widely separated photometry).
This code is under development. For any requests/comments, please send an email to
sterling.astro (at) berkeley.edu

Thank you!


------------------------

# getPeriod2 (tmp) manual

Yukei S. Murakami
sterling.astro@berkeley.edu
last updated: June 19th, 2020

## introduction
getPeriod2 (name TBD: Free Form Periodic Fitting (FFPF)?) is a python package that is developed for analysis of variable star photometry data. This algorithm re-constructs light curves of a pulsating variable stars through detection of periods. While there are many algorithms and implemented codes available, such as Lomb-Scargle periodogram in Astropy or Template Fourier Fitting (TFF) for known types of variable stars,  those algorithms may introduce biases to the results due to their function-form dependency, and several improvements were needed upon the analysis of datasets with certain characteristics with which these biases become significant. These characteristics include large separation of photometry data points between each observation window, small signal-to-noise ratio (SNR), small number of data points, and large intrinsic scatter of data points. We approached this problem by removing assumptions in terms of the periodic shape of light curve and instead by constructing the fit function as the code searches for the best-fit period. This free-form fit function is a Kth order Fourier series, which is constructed by an internal fitting to the folded light curve that is generated at each trial period. Unlike previous studies with a two-step analysis (Lomb-Scargle —> Fourier Fitting) which returns the best-fit period to a template (e.g. sinusoidal shape for Lomb-Scargle) and then fits the light curve to this possibly-biased data, this method returns the best-fit combination of both period and the light curve. Although the output from this algorithm can be interpreted as a ‘true’ set of data without biases, it should be always kept in mind that this algorithm employs one big assumption that, at the ‘true’ or ‘best-fit’ period, the disagreement (chi2) between  free-form fit function and data is minimized.

Motivation: 
The era of sky survey

We found, however, several obstacles 
overfitting (limited number of K)
‘Mechanics of chi2 potential’ (lots of local minima)
computing speed

## algorithm

## dependency

## installation

## usage

## application & example

## issues & todo
