################################################################################
#################### STRT-WOUND -- OUTLIER CELL DETECTION ######################
################################################################################

"""
Scripts which allow the detection of outlier cells based on a negative binominal
model of gene expression
Python3
"""

################################################################################
################################ DEPENDENCIES ##################################
################################################################################

import numpy as np
import pandas as pd
import itertools
from collections import Counter, OrderedDict

from WND_misc_scripts_v2_0 import *

import rpy2.robjects as robj
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import Vector, DataFrame, FloatVector, IntVector, StrVector, ListVector, Matrix, BoolVector
from rpy2.rinterface import RNULLType
stats = importr('stats')


################################################################################
############################## OUTLIER DETECTION_v2 ############################
################################################################################

def OD_detect_outliers_v2P(data, cells, dview, path_input, genes = [], cutoff_mean = 0.05, 
    score_ctrl = False, pval_min = (1 * 10**-16), BH = True, neg_log = True, tail = 'upper'):

    """
    Function that for that for each gene fits a negative binominal to a reference population and returns
    the probability that the expression observed in a non-reference cells is derived from this population.
    ----------
    data: [pd.Dataset] containing expression counts of n genes x m cells.
    cells: [pd.Series] defining reference/training population (0) or test population (1) membership for each cell.
    dview: ipyparallel DirectView Instance for parallel computing.
    score_ctrl: [bool] whether to score reference cells. Default: False
    pval_min: min p-value to avoid generating inf. Default: 1 * 10**-16.
    BH: whether p-values are Benjamin-Hochberg transformed. Default: True.
    neg_log: whether p-values are returned as negativ log10. Default: True.
    tail: whether 'upper', 'lower' or 'both' tail(s) are to be considered in p-value calculation. Default: 'upper'.
    ----------
    returns p-values for n genes over m cells.
    """
        
    #fit each gene in parallel
    
    l = len(genes)
    
    output_tmp = dview.map_sync(OD_detect_outliers_helper_v2,
                                [data] * l,
                                [cells] * l, 
                                genes,
                                [score_ctrl] * l, 
                                [pval_min] * l, 
                                [tail] * l)
    
    output = pd.concat(output_tmp)
    
    #get BH corrected p-values
    
    output_BH = pd.DataFrame(index = output.index, columns = output.columns)
    
    for c in output_BH.columns:
        output_BH[c] = stats.p_adjust(FloatVector(output[c]), method = 'BH')
        
    if BH == True:
        output = output_BH
        
    if neg_log == True:
        output = -np.log10(output.astype(float))
        
    return output
        
################################################################################

def OD_detect_outliers_helper_v2(data, cells, g, score_ctrl=False, pval_min = (1 * 10**-16), tail = 'upper'):

        """
        Helper function for OD_detect_outliers_v2P.
        """
        
        #define input data
        
        inp = list(data.ix[g, cells[cells==0].index])
        
        if np.sum(inp) == 0:
            inp = inp + [1] * int((len(inp) / 200) + 1)
            
        #define output
        
        output = pd.DataFrame(index = [g], columns = cells[cells==1].index)
        
        #fit negative binominal distribution to training data
        
        try:
            fit = fitdist(np.array(inp), distr="nbinom", method="mle")
        except:
            fit = 'F'
        
        #get upper and lower tail pval for each cell in test data
        
        if score_ctrl == True:
            gr_sel = [0,1]
        else:
            gr_sel = [1]
        
        for c in cells[cells.isin(gr_sel)].index:
            
            if fit == 'F':
                output.ix[g,c] = 1.0
                
            else:
                
                q = data.ix[g,c]

                if tail = 'upper':
                    P = 1 - nbinom_cdf_fromfit(int(q - 1), fit)
                elif tail = 'lower':
                    P = nbinom_cdf_fromfit(int(q), fit)
                elif tail = 'both':
                    P = 2 * np.min([nbinom_cdf_fromfit(int(q), fit),             #lower tail
                                    1 - nbinom_cdf_fromfit(int(q - 1), fit)])    #upper tail
                                
                if P > 1.0:
                    P = 1.0
                    
                if P == 0.0:
                    P = pval_min
                
                output.ix[g,c] = P
            
        return output

################################################################################

def OD_loocv_v2P(data, cells, genes, dview, pval_min = (1 * 10**-16), BH = True, neg_log = True, tail = 'upper'):

    """
    Function that for each gene fits a negative binominal to a reference population and approximates the background
    noise of the reference population by leave-one-out-cross-validation (LOOCV).
    ----------
    data: [pd.Dataset] containing expression counts of n genes x m cells.
    cells: [pd.Series] defining reference/training population (0) or test population (1) membership for each cell.
    dview: ipyparallel DirectView Instance for parallel computing.
    pval_min: min p-value to avoid generating inf. Default: 1 * 10**-16.
    BH: whether p-values are Benjamin-Hochberg transformed. Default: True.
    neg_log: whether p-values are returned as negativ log10. Default: True.
    tail: whether 'upper', 'lower' or 'both' tail(s) are to be considered in p-value calculation. Default: 'upper'.
    ----------
    returns LOOCV-based p-values (background distribution) for n genes over m reference cells.
    """
            
    #define output
    
    output = pd.DataFrame(index = genes, columns = cells[cells==0].index)
    
    #run in parallel
    
    l = len(genes)
    
    output_tmp = dview.map_sync(OD_loocv_helper_v2,
                                [data] * l, 
                                [cells] * l, 
                                genes,
                                [pval_min] * l, 
                                [tail] * l)
    
    output = pd.concat(output_tmp)
    
    #get BH corrected p-values
    
    output_BH = pd.DataFrame(index = output.index, columns = output.columns)
    
    for c in output_BH.columns:
        output_BH[c] = stats.p_adjust(FloatVector(output[c]), method = 'BH')
        
    if BH == True:
        output = output_BH
        
    if neg_log == True:
        output = -np.log10(output.astype(float))
        
    return output


################################################################################

def OD_loocv_helper_v2(data, cells, gene, pval_min = (1 * 10**-16), tail = 'upper'):

    """
    Helper function for OD_loocv_v2P.
    """
        
    #define test cells
        
    c_sel = list(cells[cells==0].index)
        
    #define output
        
    output = pd.DataFrame(index = [gene], columns = c_sel)
        
    #iterate over test cells and fit model
        
    for c in c_sel:
            
        #exclude cell
            
        c_sel_excl = c_sel[::]
        c_sel_excl.remove(c)
            
        #define input data
        
        inp = list(data.ix[gene, c_sel_excl])
        
        if np.sum(inp) == 0:
            inp = inp + [1] * int((len(inp) / 400) + 1)
                
        #fit negative binominal distribution to input data
        
        try:
            fit = fitdist(np.array(inp), distr="nbinom", method="mle")
        except:
            fit = 'F'
            
        #get upper and lower tail pval for each cell in test data
        
        if fit == 'F':
            output.ix[gene,c] = 1.0
            
        else:
                    
            q = data.ix[gene,c]
                
            if tail = 'upper':
                P = 1 - nbinom_cdf_fromfit(int(q - 1), fit)
            elif tail = 'lower':
                P = nbinom_cdf_fromfit(int(q), fit)
            elif tail = 'both':
                P = 2 * np.min([nbinom_cdf_fromfit(int(q), fit),             #lower tail
                                1 - nbinom_cdf_fromfit(int(q - 1), fit)])    #upper tail
                
            if P > 1.0:
                P = 1.0
                
            if P == 0.0:
                P = pval_min
                
            output.ix[gene,c] = P
            
    return output


################################################################################
############################## HELPER FUNCTIONS ################################
################################################################################

def recurList(data):
    
    """
    Important utility that transforms a list object vector returned from an R function to a python dictionary
    """
    
    #define the kind of vector 
    
    # it is not using pandas for dataframes right now but it could be implemented if needed
    
    rDictTypes = [DataFrame,ListVector]
    rArrayTypes = [FloatVector,IntVector,Matrix,BoolVector,RNULLType, Vector]
    rListTypes=[StrVector]

    if type(data) in rDictTypes:
        return OrderedDict(zip(data.names, [recurList(elt) for elt in data]))
        
    elif type(data) in rListTypes:
        return [recurList(elt) for elt in data] #Recoursive call
        
    elif type(data) in rArrayTypes:
        return np.array(data)
        
    else:
        
        if hasattr(data, "rclass"): # An unsupported r class
            raise KeyError('Could not proceed, type {} is not defined'.format(type(data)))
            
        else:
            return data # We reached the end of recursion
            
################################################################################

def fitdist(data, distr="nbinom", method="mle"):
    
    """
    Input
    -----
    data - numpy 1d array
        A numeric vector.

    distr - string
        A character string "name" naming a distribution:
        "norm", "lnorm", "pois", "exp", "gamma", "nbinom", "geom", "beta", "unif" and "logis"

    method - sting 
        A character string coding for the fitting method: 
        "mle" for maximum likeli-hood estimation
        "mme" for moment matching estimation
        "qme" for quantile matching estimation
        "mge" for maximum goodness-of-fit estimation

    (start - NOT IMPLEMENTED
        A named list giving the initial values of parameters of the named distribution)

    
    Returns
    -------
    estimate - 1d array
        the parameter estimates.
    method - string
        the character string coding for the fitting method
    sd - 1d array
        the estimated standard errors, NA if numerically not computable or NULL if not available.
    cor - 1d array
        the estimated correlation matrix, NA if numerically not computable or NULL if not available.
    vcov - 1d array
        the estimated variance-covariance matrix, NULL if not available. loglik the log-likelihood.
    aic - 1d array
        the Akaike information criterion.
    bic - 1d array
        the the so-called BIC or SBC (Schwarz Bayesian criterion). n the length of the data set.
    data - 1d array
        the data set.
    distname - str
        the name of the distribution.
    """
    
    data = FloatVector(data)
    
    fitdistrplus = importr('fitdistrplus')
    
    res = fitdistrplus.fitdist(data= data, distr=distr, method=method)
    
    return recurList(res)
    
################################################################################

def nbinom_rvs(nsamples, size, mu):
    
    rnbinom = robj.r('rnbinom')
    
    return np.array( rnbinom(n=nsamples,size=size,mu=mu) )
    
################################################################################

def nbinom_pdf(x, size, mu):
    
    dnbinom = robj.r('dnbinom')
    
    if np.isscalar(x):
        return np.array( dnbinom(x=x,size=size,mu=mu) )
        
    else:
        return np.array( dnbinom(x=FloatVector(x),size=size,mu=mu) )
        
################################################################################

def nbinom_cdf(q, size, mu):
    
    pnbinom = robj.r('pnbinom')
    
    if np.isscalar(q):
        return np.array( pnbinom(q=q,size=size,mu=mu) )
        
    else:
        return np.array( pnbinom(q=FloatVector(q),size=size,mu=mu) )
        
################################################################################

def nbinom_rvs_fromfit(nsamples, fit_dict):
    
    rnbinom = robj.r('rnbinom')
    
    return np.array( rnbinom(n=nsamples,size=fit_dict['estimate'][0],mu=fit_dict['estimate'][1]) )
    
################################################################################

def nbinom_pdf_fromfit(x, fit_dict):
    
    dnbinom = robj.r('dnbinom')
    
    if np.isscalar(x):
        return np.array( dnbinom(x=x,size=fit_dict['estimate'][0],mu=fit_dict['estimate'][1]) )
        
    else:
        return np.array( dnbinom(x=FloatVector(x),size=fit_dict['estimate'][0],mu=fit_dict['estimate'][1]) )
        
################################################################################

def nbinom_cdf_fromfit(q, fit_dict):
    
    pnbinom = robj.r('pnbinom')
    
    if np.isscalar(q):
        return np.array( pnbinom(q=q,size=fit_dict['estimate'][0],mu=fit_dict['estimate'][1]) )
        
    else:
        return np.array( pnbinom(q=FloatVector(q),size=fit_dict['estimate'][0],mu=fit_dict['estimate'][1]) )

################################################################################
################################ VISUALIZATION #################################
################################################################################

def OD_plot_diff_genes_per_cell(data_bin, cutoff, order = True):
    
    plt.figure(figsize=(10,5), facecolor = 'w')
    
    if order == True:
        data = data_bin.sum(axis = 0).sort_values()
        
    else:
        data = data_bin.sum(axis = 0)
        
    ax = plt.subplot(111)
    
    ax.set_xlim(-0.5, len(data) - 0.5)
    ax.set_ylim(0, np.max(data) * 1.1)
    
    if cutoff != None:
        ax.axhline(cutoff, color = 'red', linewidth = 2)
        clist = []
        for c in data.index:
            if data[c] >= cutoff:
                clist += ['dodgerblue']
            else:
                clist += ['grey']         
    else:
        clist = ['grey' for c in data.index]
            
    
    ax.bar(left = range(len(data)),
           height = data, width = 1,
           color = clist, linewidth = 0)