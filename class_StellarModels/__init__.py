class StellarModels:
    '''
    A supplemental class that provides various stellar property relationships.
    
    (each model is a sub-class that contains unique method functions)
    e.g.
    ~~~~
        >>> from PIPS import photdata, StellarModels
        >>> star = photdata([x,y,yerr])
        >>> model = StellarModels.Hoffman20()
        >>> star_updated = model.calc_Teff(star)
        >>> print(f'{star_updated.Teff:.3e}')
        7.580e3
    ~~~~

    subclasses and method functions for each of them:
        Hoffman20 # <-- this is an example: author of the paper + year is the standard
            calc_color() # <-- This is an example: names don't have to be 'calc_xxxx()'
            calc_Teff()
            calc_luminosity()
        Murakami21
            calc_Teff()
            calc_mass()
            calc_xxx()
        Sunseri22
    '''