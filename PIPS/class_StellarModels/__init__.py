# from scipy.optimize import curve_fit
# import numpy as np
# from PIPS.class_photdata import photdata


class StellarModels:
    """ Work in progress.
    TODO: merge James' code
    """
    def __init__(self):
        raise NotImplementedError
#     '''
#     A supplemental class that provides various stellar property relationships.

#     (each model is a sub-class that contains unique method functions)
#     e.g.
#     ~~~~
#         >>> import PIPS
#         >>> star = photdata([x,y,yerr])
#         >>> model = PIPS.Hoffman20(star)
#         >>> star_updated = model.calc_log_Teff()
#         >>> print(f'{star_updated.Teff:.3e}')
#         7.580e3
#     ~~~~

#     subclasses and method functions for each of them:
#         Hoffman20 # <-- this is an example: author of the paper + year is the standard
#             calc_color() # <-- This is an example: names don't have to be 'calc_xxxx()'
#             calc_Teff()
#             calc_luminosity()
#         Murakami21
#             calc_Teff()
#             calc_mass()
#             calc_xxx()
#         Sunseri22


#     An instance of this class should contain the following traits

#     self.C_k
#     self.phik1
#     self.C_k_err
#     self.phik1_err
#     self.period
#     self.period_err

#     From these properties one should have all they need to use the
#     subclasses to find any stellar properties based on the given models.
#     '''

#     def __init__(self, star):
#         """initialization function for StellarModels Super Class"""
#         if isinstance(star, photdata):
#             if star.period:
#                 if star.epoch_offset:
#                     self.period = star.period
#                     self.period_err = star.period_err
#                     K=4
#                     def fit_cphi(x,y,period,K):
#                         #Curve fit function for series of cosines and phases
#                         def f(x_folded,A0,*cp_list):
#                             # model function for scipy curvefit function
#                             c_list = cp_list[0:K]
#                             p_list = cp_list[K:]
#                             y = A0
#                             for i in range(len(c_list)):
#                                 y += c_list[i]*np.cos((i+1) * 2 * np.pi * x_folded/period + p_list[i])
#                             return y
#                         p0 = [1,*np.zeros(K*2)]
#                         popt,pcov = curve_fit(f,x % period,y,p0=p0)
#                         return popt,pcov

#                     popt,pcov = fit_cphi(star.x - star.epoch_offset % star.period, star.y, star.period, K=4) #epoch may cause error depending on implementation

#                     errors = []
#                     for j in range(len(popt)):
#                         err = np.sqrt(pcov[j,j])
#                         errors.append(err)

#                     self.C_1_err, self.C_2_err, self.C_3_err, self.C_4_err = errors[1], errors[2], errors[3], errors[4]
#                     self.phi21_err = np.sqrt( (errors[6])**2 + (-2 * errors[5])**2 )
#                     self.phi31_err = np.sqrt( (errors[7])**2 + (-3 * errors[5])**2 )
#                     self.phi41_err = np.sqrt( (errors[8])**2 + (-4 * errors[5])**2 )

#                     for i in range(K):
#                         if popt[i+1]<0:
#                             popt[i+K+1] += np.pi
#                             popt[i+1] *= -1
#                         popt[i+5] = popt[i+5]%(2*np.pi)

#                     self.C_1, self.C_2, self.C_3, self.C_4 = popt[1], popt[2], popt[3], popt[4]
#                     self.phi21 = (popt[6] - 2*popt[5])%(2*np.pi)
#                     self.phi31 = (popt[7] - 3*popt[5])%(2*np.pi)
#                     self.phi41 = (popt[8] - 4*popt[5])%(2*np.pi)

#                     # self.fit_terms = [c_1, c_2, c_3, c_4, phi21, phi31, phi41]
#                     # self.fit_terms_err = [err_c_1, err_c_2, err_c_3, err_c_4, err_phi21, err_phi31, err_phi41]

#                 else:
#                     print("Run get_epoch_offset to determine the epoch_offset necessary for calculation")
#             else:
#                 print("Run get_period to determine the period of this photdata object first")
#         else:
#             print("Object given is not an instance of the photdata class")



# class Cacciari2005(StellarModels):
#     """Subclass for StellarModels corresponding to Cacciari's paper from 2005,
#     this paper is commonly referenced in the literature for RR Lyrae Stellar
#     Parameter relationships. We denote the original author's relationship's in
#     the form of method doc strings"""

#     ###########################################################################
#     #                               RRab Stars                                 #
#     ###########################################################################

#     #Values

#     def calc_Fe_H_ab(self):
#         """
#         relationship derived by Jurscik (1998)
#         """
#         #Metallicity
#         phi_s = self.phi31 + np.pi #phi31 + pi
#         self.Fe_H = -5.241 - 5.394*self.period + 1.345*phi_s

#     def calc_BV_0_ab(self):
#         """
#         relationship empirically determined by Kovács & Walker (2001)
#         """
#         #Color
#         self.BV_0 = 0.189*np.log10(self.period) -0.313*self.C_1 + 0.293*self.C_3 + 0.460

#     def calc_log_T_eff_type_ab(self):
#         """
#         relationship derived by Kovács & Walker (2001)
#         """
#         #Effective Temperature
#         self.log_T_eff = 3.930 - 0.322*self.BV_0 + 0.007*self.Fe_H

#     def calc_M_v_ab(self):
#         """
#         relationship derived by Kovács (2002)
#         """
#         #Absolute Magnitude
#         self.M_v = -1.876*np.log10(self.period) - 1.158*self.C_1 + 0.821*self.C_3 + 0.43

#     def calc_log_L_ab(self):
#         """
#         Standard luminosity calculation using absolute magnitude.
#         Absolute Magnitude of the Sun reported by Wilmer (2018)
#         """
#         #Luminosity
#         self.log_L = (self.M_v - 4.83)/(-2.5)

#     def calc_log_mass_ab(self):
#         """
#         Originally derived by Jurscik (1998)
#         """
#         #Mass
#         self.log_M = 20.844 - 1.754*np.log10(self.period) + 1.477*(self.log_L) - 6.272*self.log_T_eff + 0.0367*self.Fe_H

#     #Errors

#     def calc_error_Fe_H_ab(self):
#         self.Fe_H_err = np.sqrt((-5.394*self.period_err)**2 + (1.345*self.phi31_err)**2)

#     def calc_error_BV_0_ab(self):
#         self.BV_0_err = np.sqrt(((0.189/np.log(10))*(self.period_err/self.period))**2 + (-0.313*self.C_1_err)**2 + (0.293*self.C_3_err)**2)

#     def calc_error_log_T_eff_type_ab(self):
#         self.log_T_eff_err = np.sqrt((0.322*self.BV_0_err)**2 + (0.007*self.Fe_H_err)**2)

#     def calc_error_M_v_ab(self):
#         self.M_v_err = np.sqrt(((-1.876/np.log(10))*(self.period_err/self.period))**2 + (-1.158*self.C_1_err)**2 + (0.821*self.C_3_err)**2)

#     def calc_error_log_L_ab(self):
#         self.log_L_err = np.sqrt((((1.876/2.5)/np.log(10))*(self.period_err/self.period))**2 + (1.158*self.C_1_err/2.5)**2 + (0.821*self.C_3_err/2.5)**2)

#     def calc_error_log_mass_ab(self):
#         self.log_M_err = np.sqrt(((-1.754/np.log(10))*(self.period_err/self.period))**2 + (1.477*(self.log_L_err))**2
#         + (-6.272*self.log_T_eff_err)**2 + (0.0367*self.Fe_H_err)**2)

#     ###########################################################################
#     #                               RRc Stars                                 #
#     ###########################################################################

#     #Values

#     def calc_Fe_H_c(self):
#         """
#         relationship derived by Carretta & Gratton (1997)
#         """
#         #phi is phi_31
#         phi_s = self.phi31 + np.pi
#         self.Fe_H = 0.0348*phi_s**2 + 0.196*self.phi31 - 8.507*self.period + 0.367

#     def calc_log_T_eff_type_c(self):
#         """
#         relationship from Simon & Clement (1993)
#         """
#         phi_s = self.phi31 + np.pi
#         self.log_T_eff = 3.7746 - 0.1452*np.log10(self.period) + 0.0056*phi_s

#     def calc_M_v_c(self):
#         """
#         relationship derived by Kovács (1998)
#         """
#         phi_s = self.phi21 - np.pi/2
#         self.M_v = 1.061 - 0.961*np.log10(self.period) - 0.044*phi_s - 4.447*self.C_4

#     def calc_log_L_c(self):
#         """
#         Standard luminosity calculation using absolute magnitude.
#         Absolute Magnitude of the Sun reported by Wilmer (2018)
#         """
#         phi_s = self.phi21 - np.pi/2
#         self.log_L = (self.M_v - 4.83)/(-2.5)

#     def calc_log_mass_c(self):
#         """
#         Derived by Simon & Clement (1993)
#         """
#         phi_s = self.phi31 + np.pi
#         self.log_M = 0.52 * np.log10(self.period) - 0.11*phi_s + 0.39

#     #Errors

#     def calc_error_Fe_H_c(self):
#         #phi is phi_31
#         phi_s = self.phi31 + np.pi
#         self.Fe_H_err = (2*0.0348*phi_s * self.phi31_err)**2 + (0.196*self.phi31_err)**2 + (-8.507*self.period_err)**2

#     def calc_error_log_T_eff_type_c(self):
#         #phi_s = phi + np.pi
#         self.log_T_eff_err = np.sqrt(((- 0.1452/np.log(10))*(self.period_err/self.period))**2 + (0.0056*self.phi31_err)**2)

#     def calc_error_M_v_c(self):
#         self.M_v_err = np.sqrt(((- 0.961/np.log(10))*(self.period_err/self.period))**2 + (-0.044*self.phi21_err)**2 +  (-4.447*self.C_4_err)**2)

#     def calc_error_log_L_c(self):
#         self.log_L_err = (((0.961/2.5)/np.log(10))*(self.period_err/self.period))**2 + (- 0.044*self.phi21_err/2.5)**2 + (-4.447*self.C_4_err/2.5)**2

#     def calc_error_log_mass_c(self):
#         self.log_M_err = np.sqrt(((0.52/np.log(10))*(self.period_err/self.period))**2 + (-0.11*self.phi31_err)**2)

#     ###########################################################################
#     #                            All RR Lyrae Stars                           #
#     ###########################################################################

#     def calc_log_surface_gravity(self):
#         """
#         Cited from Cacciari et al. (2005), standard surface gravity equation
#         """
#         self.log_g = -10.607 + self.log_M - self.log_L + 4*self.log_T_eff

#     def calc_error_log_surface_gravity(self):
#         self.log_g_err = np.sqrt(self.log_M_err**2  + self.log_L_err**2 + (4*self.log_T_eff_err)**2)

#     ###########################################################################
#     #                            Computes All Values                          #
#     ###########################################################################

#     def calc_all_vals(self,star_type):
#         """"This function returns none. It should add several traits to your StellarModel Object:

#         TRAITS

#         self.Fe_H
#         self.BV_0 (RRab only)
#         self.log_T_eff
#         self.M_v
#         self.log_L
#         self.log_M
#         self.Fe_H_err
#         self.BV_0_err (RRab only)
#         self.log_T_eff_err
#         self.M_v_err
#         self.log_L_err
#         self.log_M_err
#         """
#         if star_type == 'RRab':
#             #Values
#             self.calc_Fe_H_ab()
#             self.calc_BV_0_ab()
#             self.calc_log_T_eff_type_ab()
#             self.calc_M_v_ab()
#             self.calc_log_L_ab()
#             self.calc_log_mass_ab()
#             self.calc_log_surface_gravity()

#             #Errors
#             self.calc_error_Fe_H_ab()
#             self.calc_error_BV_0_ab()
#             self.calc_error_log_T_eff_type_ab()
#             self.calc_error_M_v_ab()
#             self.calc_error_log_L_ab()
#             self.calc_error_log_mass_ab()
#             self.calc_error_log_surface_gravity()

#         elif star_type == 'RRc':
#             #Values
#             self.calc_Fe_H_c()
#             self.calc_log_T_eff_type_c()
#             self.calc_M_v_c()
#             self.calc_log_L_c()
#             self.calc_log_mass_c()
#             self.calc_log_surface_gravity()

#             #Errors
#             self.calc_error_Fe_H_c()
#             self.calc_error_log_T_eff_type_c()
#             self.calc_error_M_v_c()
#             self.calc_error_log_L_c()
#             self.calc_error_log_mass_c()
#             self.calc_error_log_surface_gravity()

#         else:
#             print("""Not a valid input. star_type must be a string of the form 'RRab' or 'RRc' in order to work""")
