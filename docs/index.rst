.. PIPS documentation master file, created by
   sphinx-quickstart on Mon Feb 22 11:47:22 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PIPS's documentation!
==================================

PIPS is a Python pipeline primarily designed to analyze the photometric data of astronomical objects whose brightness changes periodically. Our pipeline can be imported quickly and is designed to be user friendly. PIPS was originally developed to determine the periods of RR Lyrae variable stars and offers many features designed for variable star analysis. We have expanded PIPS into a suite that can obtain period values for almost any type of photometry with both speed and accuracy. PIPS can determine periods through several different methods, analyze the morphology of lightcurves via fourier analysis, and determine stellar properties based on preexisting stellar models. Currently our team is also exploring the possibility of using this pipeline to detect periods of exoplanets as well.

To determine the period value from provided data, PIPS runs a periodogram by evaluating the goodness of fit (chi-squared test) at a range of period values fitted to the data. PIPS will continue to refine the period estimation by re-running a periodogram test around a successively smaller range of values until a confidence threshold is reached. The goodness of fit is evaluated against a model and the period-folded data. PIPS offers choices of fitting models, with both Fourier and Gaussian Mixture currently implemented. PIPS can easily accept custom model functions to evaluate custom data while PIPS' preserving structure and ease of use. The choice of model is important for accuracy, and dependent on the type of data being analyzed. For instance, the relatively gradual changes of variable stars are well suited to Fourier models, which the sharp troughs of exoplanet light curves are much better fit by the Gaussian Mixture model. 

Unlike many other period-determination methods, PIPS calculates the uncertainty of both the period and model parameters. Currently, this uncertainty is calculated using the covariant matrix of the linear regression. However, in the future the PIPS team intends to implement an optional MCMC based method which will offer further accuracy.
While period estimation is the primary function of PIPS, other tools are available for specific uses. These include stellar parameter estimation, lightcurve-based classifications, and visualization helpers for data analysis. The combination of a robust data analysis structure, and the flexibility to work with external tools and models, ensures that PIPS is an excellent solution for a wide variety of periodic data analysis.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   tutorial
   data_manip
   stellar_parameters
   
.. toctree::
   :maxdepth: 2
   :caption: Documentation

   sources/modules

