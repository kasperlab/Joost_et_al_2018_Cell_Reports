
################################################################################
################### STRT-WOUND -- MISCELLANEOUS SCRIPTS ####################
################################################################################

"""
A variety of smaller scripts for data input, data wrangling and transformation,
data plotting and data analysis. Scripts were usually tested interactively in Ipython Notebook
before being transferred here
Python 3
"""

################################################################################
################################ DEPENDENCIES ##################################
################################################################################

import os, math, datetime, random, itertools
from collections import Counter
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import fmin
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import scipy.cluster.hierarchy as sch

import rpy2.robjects as robj
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import Vector, DataFrame, FloatVector, IntVector, StrVector, ListVector, Matrix, BoolVector
from rpy2.rinterface import RNULLType
stats = importr('stats')

################################################################################
################################# DATA INPUT ###################################
################################################################################

def create_ID():
    
    "Creates experiment ID (YmdHm) to identify output"
    
    exp_id = datetime.datetime.now().strftime("%Y%m%d%H%M")
    
    print("\nThe experiment ID is %s" % (exp_id))
    
    return exp_id
    
################################################################################

def setWDandMaster(path, master_inp):
    
    """
    Sets the working directory (path [str]) for the following functions and imports the master file [str]. The working directory requires subfolders containing
    'Sequencing' data, 'Metadata', 'Capturing' data and 'Scepter' data identifiable by IFC barcode. The master file is located in the working 
    directory and contains all IFC barcodes to be included in the analysis divided by linebreaks. Returns the contents of the master file.
    """ 
    
    ### select working directory for master file
    
    os.chdir(path)
               
    ### import master file       
           
    master_file = open('Input/%s' % (master_inp), 'r').read().split()
    print("\nThe master file contains the following %s libraries: % s" % (len(master_file), master_file))
    
    return master_file 
    
################################################################################

def import_and_merge_v2(master_file):
      
    """
    Iterates through the master file and imports sequencing and metadata for every specified IFC / library. Transforms sequencing data 
    into DataFrame with Cells as columns and Genes as rows. Merges general metadata and capturing metadata sets and adds quantitative
    data about sequencing results (number of genes, transcripts etc.). Every cell is uniquely defined using the following format:
    '[IFC-barcode]-[#IFC-row][#IFC-column]. Merged sequencing (= 'seq') and metadata DataFrames (= 'meta') are returned.
    NB: Uses the 'XXXXXX-XXX_expression_for_R.tab' as input!
    """   
    
    ### check whether master file is imported
    
    try:
        master_file
    except:
        print("Master file not imported!")
        
    ###i terate through library / IFC barcodes in master file
        
    for barcode in master_file:
        
        ### import and format capturing data
                
        cap_tmp = pd.read_table('Capturing/capturing_%s.txt' % barcode, sep = '\t', header = 0, index_col = 0)
        ix_sel = ['%s-%s' % (barcode, ix) for ix in cap_tmp['Valid'][cap_tmp['Valid']=='Y'].index]
                           
        ###import and format sequencing data per barcode
        
        seq_tmp = pd.read_table('Sequencing/C1-%s-%s_expression_for_R.tab' % (barcode[:7], barcode[7:]), sep = '\t', header = 0, index_col = 0, low_memory = False)
        seq_tmp = seq_tmp.ix['ERCC-00002':].astype(int)
        seq_tmp.columns = ['%s-%s' % (barcode, column[-3:]) for column in seq_tmp.columns]
        seq_tmp = seq_tmp[ix_sel]
        seq_tmp.index.name = 'Gene'; seq_tmp.columns.name = 'Cell'
        
        ###compile metadata per barcode
                
        #import and transform general (IFC-specific) metadata
        
        meta_gen_tmp = pd.read_table('Metadata/C1_metadata_%s.txt' % (barcode), sep = '\t', header = None, index_col = 0)
        meta_gen_tmp = meta_gen_tmp.reindex(columns = range(1, len(seq_tmp.columns) + 1))
        meta_gen_tmp = meta_gen_tmp.fillna(method = 'ffill', axis = 1)
        meta_gen_tmp.columns = seq_tmp.columns
                
        #import and transform capturing metadata
        
        meta_cap_tmp = cap_tmp.T.ix[['Diameter','Area','Red','Green','Blue']].astype(float)
        meta_cap_tmp.columns = ['%s-%s' % (barcode, col) for col in meta_cap_tmp.columns]
        meta_cap_tmp = meta_cap_tmp[seq_tmp.columns]
                
        #join datasets
        
        meta_tmp = meta_gen_tmp.append([meta_cap_tmp])
        
        #calculate volume
        
        meta_tmp_volume = pd.Series((((meta_tmp.ix['Diameter'] / 2) ** 3) * math.pi * 4/3), name = 'Volume')
        meta_tmp = meta_tmp.append(meta_tmp_volume)
                
        #calculate spikes and move markers
        
        spikes = open('/Users/sijo/Documents/Epidermis map/Scripts/spikes.txt', 'r').read().split('\n')[:-1]
        markers = open('/Users/sijo/Documents/Epidermis map/Scripts/markers.txt','r').read().split('\n')[:-1]
        
        meta_tmp_spikes = pd.DataFrame(seq_tmp.ix[spikes].sum(axis = 0), columns = ['sum_spikes']).T
        meta_tmp_markers = seq_tmp.ix[markers]
            
        meta_tmp = meta_tmp.append([meta_tmp_spikes, meta_tmp_markers])
                
        #calculate transcripts and genes
        
        meta_tmp_transcripts = pd.Series(seq_tmp.ix['Xkr4':'Erdr1'].sum(axis = 0), name = 'sum_transcripts')
        meta_tmp_genes = pd.Series(seq_tmp.ix['Xkr4':'Erdr1'].apply(lambda x: x > 0).sum(axis = 0), name = 'sum_genes')
        meta_tmp = meta_tmp.append([meta_tmp_transcripts, meta_tmp_genes])
                
        ###join datasets
        
        print("Adding %s to dataset" % (barcode))
        
        if master_file.index(barcode) == 0:
            seq = seq_tmp
            seq_ix = seq.index
            meta = meta_tmp
            
        else:
            seq = pd.concat([seq, seq_tmp], axis = 1)
            meta = pd.concat([meta, meta_tmp], axis = 1)
            
    seq = seq.ix[seq_ix]  #to prevent lexicographical ordering of big dataset
    
    return seq, meta
    
################################################################################

def merge_isoforms_v1(data):
    
    """
    Merges all isoforms (in the case of STRT data, only isoforms distinguished by alternative start sites) of each gene
    in the dataset.
    ----------
    data: gene expression dataset of n genes (inlcuding isoforms) in m samples. Isoforms must be specified by "genename_vX".
    ----------
    returns gene expression dataset of g genes with mergen isoforms in m samples.
    """
    
    output = pd.DataFrame(columns = data.columns)
    
    for g in data.index:
        
        g_ = g.split("_v")
        
        if len(g_) == 1:
            output.ix[g_[0]] = data.ix[g]
            
        elif len(g_) == 2:
            if g_[0] in output.index:
                output.ix[g_[0]] = output.ix[g_[0]] + data.ix[g]
            else:
                output.ix[g_[0]] = data.ix[g]
                
    return output
    
################################################################################

def saveData_v1(dataset, path, id_, name):
    
    """
    Saves pd.DataFrames or pd.Series to csv.
    ----------
    dataset: [pd.DataFrame] or [pd.Series].
    path: path to saving location.
    id_: experimental ID (e.g. YYMMDDHHMM).
    name: name of saved file. Format: /path/ID_name.
    """
            
    dataset.to_csv('%s/%s_%s.txt' % (path, id_, name), sep = '\t')
    
################################################################################

def saveData_to_pickle_v1(dataset, path, id_, name):
    
    """
    Saves pd.DataFrames or pd.Series to pickle.
    ----------
    dataset: [pd.DataFrame] or [pd.Series].
    path: path to saving location.
    id_: experimental ID (e.g. YYMMDDHHMM).
    name: name of saved file. Format: /path/ID_name.
    """
        
    dataset.to_pickle('%s/%s_%s.txt' % (path, id_, name))

################################################################################

def loadData_v1(path, id_, name, datatype):
    
    """
    loads pd.DataFrames or pd.Series from csv.
    ----------
    path: path to saving location.
    id_: experimental ID (e.g. YYMMDDHHMM).
    name: name of saved file. Format: /path/ID_name.
    datatype: 'DataFrame' or 'Series'.
    ----------
    returns [pd.DataFrame] or [pd.Series]
    """
    
    if datatype == 'DataFrame':
        
        dataset = pd.DataFrame.from_csv('%s/%s_%s.txt' % (path, id_, name), sep = '\t', header = 0, index_col = 0)
    
    elif datatype == 'Series':
        
        dataset = pd.Series.from_csv('%s/%s_%s.txt' % (path, id_, name), sep = '\t', header = None, index_col = 0)
        
    return dataset
    
################################################################################
    
def loadData_from_pickle_v1(path, id_, name):
    
    """
    loads pd.DataFrames or pd.Series from csv.
    ----------
    path: path to saving location.
    id_: experimental ID (e.g. YYMMDDHHMM).
    name: name of saved file. Format: /path/ID_name.
    ----------
    returns [pd.DataFrame] or [pd.Series]
    """
    
    return pd.read_pickle('%s/%s_%s.txt' % (path, id_, name))

################################################################################
################# DATA TRANSFORMATION AND FEATURE SELECTION ####################
################################################################################

def dropNull_v2(dataset, path, drop_spikes = True, drop_markers = True, cutoff_mean = 0):

    """
    Takes the merged sequencing dataset, drops spike values (based on 'spike.txt' containing all current spike index names) unless 
    set False and removes all unexpressed genes (sum == 0 over the whole dataset)
    ----------
    dataset: seq dataset [pd.DataFrame] containing m cells x n genes.
    path: path where txt. files specifying spikes and repeats are stored.
    drop_spikes: [bool] indicating whether ERCC spikes are removed from the dataset.
    drop_markers: [bool] indicating whether markers are removed from the dataset.
    cutoff_mean: Average expression count [float] at which a gene is dropped from the dataset.
    ----------
    returns seq dataset [pd.DataFrame] containing m cells x n genes.
    """
      
    ###drop spike and repeat indices unless specifically disabled
    
    spikes = open('%s/spikes.txt' % (path),'r').read().split('\n')[:-1]
    markers = open('%s/markers.txt' % (path),'r').read().split('\n')[:-1]
    
    if drop_spikes == True:
        print("\nDropping spikes from dataset")
        dataset = dataset.drop(spikes)
        
    if drop_markers == True:
        print("\nDropping markers from dataset")
        dataset = dataset.drop(markers)   
    
        
    ###drop rows of non-expressed genes
    
    print("\nDropping unexpressed genes from dataset")
    
    g_sel = dataset.mean(axis=1)[dataset.mean(axis=1)>cutoff_mean].index
    
    dataset = dataset.ix[g_sel]
    
    return dataset
    
################################################################################

def cellCutoff(dataset, cutoff):
    
    """
    Removes all observations / cells whose total number of transcripts lies below a
    specified cutoff.
    ----------
    dataset: seq dataset [pd.DataFrame] containing m cells x n genes.
    cutoff: number of total transcript / molecule count [int] under which a cell is dropped from the dataset.
    ----------
    returns seq dataset [pd.DataFrame] containing m cells x n genes.
    """
    
    print("\nRemoving cells with less than %s transcripts" % (cutoff))
    
    dataset = dataset[dataset.columns[dataset.sum() >= cutoff]]
    
    return dataset

################################################################################

def log2Transform(dataset, add = 1):
    
    """
    Calculates the binary logarithm (log2(x + y)) for every molecule count / cell x in dataset. 
    Unless specified differently, y = 1.
    ----------
    dataset: seq dataset [pd.DataFrame] containing m cells x n genes.
    add: y [float or int] in (log2(x + y)).
    ----------
    returns seq dataset [pd.DataFrame] containing m cells x n genes.
    """
    
    print("\nCalculating binary logarithm of x + %s" % (add))
    dataset = np.log2(dataset + add)
    
    return dataset


################################################################################

def log2_cv_fit(dataset):
    
    """
    Fits noise model: log2(CV) = log2(mean^alpha + k).
    ----------
    dataset: seq dataset [pd.DataFrame] containing m cells x n genes. NB: Must be untransformed!
    ----------
    returns pd.Series containing difference between measured and expected cv for every gene and fitted values for [alpha, k].
    """
    
    #########################
    
    def min_cvfit(log2_mean, log2_cv, x0):
        
        """
        Helper function for log_cv_fit
        """
    
        nestedfun = lambda x: np.sum(np.abs(np.log2 ((2 ** log2_mean) ** x[0] + x[1]) - log2_cv))  
        xopt0, xopt1  = fmin(nestedfun, x0 = x0)
        
        return xopt0, xopt1
    
    #########################

    data_mean = dataset.mean(axis = 1)
    data_cv = dataset.std(axis = 1) / dataset.mean(axis = 1)

    log2_mean = np.log2(data_mean)
    log2_cv = np.log2(data_cv)

    x0 = [-0.5, 0.5]
    xopt0, xopt1 = min_cvfit(log2_mean, log2_cv, x0)
    xopt = [xopt0, xopt1]
    
    log2_cv_fit = np.log2(( 2 ** log2_mean) ** xopt[0] + xopt[1])
    log2_cv_diff = log2_cv - log2_cv_fit

    return log2_cv_diff, xopt

    
################################################################################

def select_features_v3(dataset, cutoff_mean, nr_features, path_input, return_all=False, drop_spikes = True, drop_markers = True):

    """
    Helper function to compress the feature selection workflow.
    ----------
    dataset: [DataFrame] of m cells x n genes. Must be non-transformed.
    cutoff_mean: gene expression cutoff [float]. Average expression required for a gene to be retained
    in the dataset.
    nr_features: number of features to select [int]. Features are ordered according to log2_cv_diff.
    path_input: name of path leading to input files including spike and repeat lists
    percentile: correlation percentile at which two genes are considered highly correlated. Default: 95.
    return_all: if True, additionally returns corr_filt, log2_cv_diff and x_opt. Default: False
    ----------
    return log2-transformed DataFrame of m cells and n selected features. If return_all is True, additionally 
    returns corr_filt, log2_cv_diff and x_opt
    """

    dataset = dropNull_v2(dataset, path_input, drop_spikes = drop_spikes, drop_markers = drop_markers, cutoff_mean=cutoff_mean)
    
    print("\nAfter mean expression cutoff of %s, %s genes remain" % (cutoff_mean, len(dataset.index)))
    
    log2_cv_diff, xopt = log2_cv_fit(dataset)
    
    genes_sel = log2_cv_diff.sort_values()[-nr_features:].index
    
    draw_log2_cv_diff(dataset, log2_cv_diff, xopt, selected=genes_sel)
    
    print("\nAfter high variance feature selection, %s genes remain" % (len(genes_sel)))
    
    dataset = dataset.ix[genes_sel]
    
    dataset = log2Transform(dataset)
    
    if return_all==True:
    
        return dataset, log2_cv_diff, xopt
    
    else:
        
        return dataset

################################################################################
##################################### tSNE #####################################
################################################################################

def tSNE_get_params_v1P(dist_mat, groups, cmap, dview, perplexity, early_exaggeration, learning_rate, n_iter, sec_var = 'early_exaggeration', **kwargs):
    
    """
    Function to crossscreen for adequate tSNE parameters. Uses tSNE implementation from scikit-learn
    ----------
    dist_mat: [pd.DataFrame] containing cell-cell distances.
    groups: [pd.Series] containing group membership (e.g. cluster ID) of cells
    cmap: [dict] containing color information for groups or matplotlib colormap
    dview: ipyparallel DirectView Instance for parallel computing.
    perplexity: [list] of perplexity parameters
    early_exaggeration: [list] of early_exaggeration parameters or single parameter
    learning_rate: [list] of learning_rate parameters or single value
    n_iter: [list] of n_iter or single value
    sec_var: second variable to be used in the screening. default: 'early_exaggeration'
    ----------
    Plots tSNE plots using the specified combinations of parameters.
    Returns [dict] of tSNE coordinates
    """

    ##########################################################
    
    def get_tSNE(param, dist_mat):
        
        """
        Helper function for parallelization of tSNE_get_params.
        """
        
        tsne = TSNE(n_components=2, perplexity=param['perplexity'], early_exaggeration=param['early_exaggeration'], 
                    learning_rate=param['learning_rate'], n_iter=param['n_iter'])
        
        return pd.DataFrame(tsne.fit_transform(dist_mat), index = dist_mat.index, columns = ['x', 'y'])
    
    ##########################################################
    
    #get tSNE init dict
    
    param_dict = {}
    
    cs = perplexity
    if sec_var == 'early_exaggeration':
            rs = early_exaggeration
    elif sec_var == 'learning_rate':
            rs = learning_rate
    elif sec_var == 'n_iter':
            rs = n_iter   
            
    for c in cs:
        for r in rs:
            param_dict[c,r] = {}
            param_dict[c,r]['perplexity'] = c
            param_dict[c,r]['early_exaggeration'] = early_exaggeration
            param_dict[c,r]['learning_rate'] = learning_rate
            param_dict[c,r]['n_iter'] = n_iter
            param_dict[c,r][sec_var] = r

    #get tSNE coords in parallel
    
    keys = param_dict.keys()
            
    tsne_coords = dview.map_sync(get_tSNE, 
                                 [param_dict[k] for k in keys], 
                                 [dist_mat for k in keys])
    
    tsne_coords_dict = {}
    for ix, k in enumerate(keys):
        tsne_coords_dict[k] = tsne_coords[ix]
    
    #print data
    
    #initialize figure

    height = 5 * len(rs)
    width = 5 * len(cs)

    plt.figure(facecolor = 'w', figsize = (width, height))
    
    #initialize gridspec
    
    gs = plt.GridSpec(nrows = len(rs), ncols = len(cs), wspace=0.00, hspace=0.00)
    
    for pos_c, c in enumerate(cs):
        for pos_r, r in enumerate(rs):
            
            tsne_coords = tsne_coords_dict[c,r]

            #define x- and y-limits

            x_min, x_max = np.min(tsne_coords['x']), np.max(tsne_coords['x'])
            y_min, y_max = np.min(tsne_coords['y']), np.max(tsne_coords['y'])
            x_diff, y_diff = x_max - x_min, y_max - y_min

            pad = 3.0

            if x_diff > y_diff:
                xlim = (x_min - pad, x_max + pad)
                ylim = (y_min * (x_diff / y_diff) - pad, y_max * (x_diff / y_diff) + pad)

            if x_diff < y_diff:
                xlim = (x_min * (y_diff/x_diff) - pad, x_max * (y_diff/x_diff) + pad)
                ylim = (y_min - pad, y_max + pad)

            #define x- and y-axes

            ax1 = plt.subplot(gs[pos_r, pos_c])

            ax1.set_xlim(xlim[0], xlim[1])
            ax1.set_ylim(ylim[0], ylim[1])

            remove_ticks(ax1)
            
            if type(cmap) != dict:
                cm = cmap
                cmap = {}
                for ix, gr in enumerate(return_unique(groups)):
                    cmap[gr] = cm(float(ix) / len(set(groups)))

            clist_tsne = [cmap[groups[ix]] for ix in groups.index]

            ax1.scatter(tsne_coords.ix[groups.index, 'x'],
                        tsne_coords.ix[groups.index, 'y'], 
                        s = 10,
                        linewidth = 0.0,
                        c = clist_tsne)

            #draw params

            ax1.text(xlim[0] + ((xlim[1] - xlim[0]) * 0.5), 
                    ylim[0] + ((ylim[1] - ylim[0]) * 0.98), 
                     'perplexity = %s, %s = %s' % (c,sec_var,r), 
                     family = 'Liberation Sans', fontsize = 10, ha = 'center', va = 'center')
            
            #clean_axis(ax1)
            
    return tsne_coords_dict

################################################################################
########################### DIMENSIONALITY REDUCTION ###########################   
################################################################################

def dist_mat_dim_reduc_v2(dataset, dim=50, method='TruncatedSVD', distance = 'euclidean', **kwargs):

    """
    Function to perform dimensionality reduction prior to clustering or embedding. Reverse transforms the dimension-reduced data
    back into the orginal data format and returns a distance matrix.
    ----------
    dataset: [pd.Dataset] containing expression counts of n x genes in m x cells.
    dim: [int] number of dimensions to consider.
    method: dimensionality reduction method. Possible are "PCA", "TruncatedSVD" and "NMF".
    distance: distance metric. Either "euclidean", "sqeuclidean" or "pearson"
    ---------
    returns [pd.Dataset] containing dimensionality reduced expression counts of n x genes in m x cells
    returns [pd.Dataset] containing distances of m x cells
    """

    from sklearn.decomposition import NMF, PCA, TruncatedSVD
    from scipy.spatial.distance import pdist, squareform

    if method == 'PCA':
        pca = PCA(n_components=dim, **kwargs)
        data_tmp = pd.DataFrame(pca.fit_transform(dataset.T).T, index = range(dim), columns = dataset.columns)

    if method == 'TruncatedSVD':
        tSVD = TruncatedSVD(n_components=dim, **kwargs)
        data_tmp =  pd.DataFrame(tSVD.fit_transform(dataset.T).T, index =  range(dim), columns = dataset.columns)
        
    if method == 'NMF':
        nmf = NMF(n_components=dim, **kwargs)
        data_tmp = pd.DataFrame(nmf.fit_transform(dataset.T).T, index = range(dim), columns = dataset.columns)
        
    if distance == 'pearson':
        return data_tmp, 1 - data_tmp.corr()

    elif distance == 'euclidean':
        return data_tmp, pd.DataFrame(squareform(pdist(np.array(data_tmp.T), 'euclidean')), index = dataset.columns, columns = dataset.columns)

    elif distance == 'sqeuclidean':
        return data_tmp, pd.DataFrame(squareform(pdist(np.array(data_tmp.T), 'sqeuclidean')), index = dataset.columns, columns = dataset.columns)


################################################################################
############################ K-MEANS CLUSTERING ################################
################################################################################

def tsne_kmeans(tsne, cells, n_clusters, **kwargs):

    """
    Performs k-means clustering in two-dimensional t-SNE space.
    ----------
    tsne: [pd.DataFrame] of tSNE coordinates.
    cells: [list] of m cellIDs to consider.
    n_clusters: number of clusters.
    ----------
    return [pd.Series] containing the clusterIDs of m cells.
    """
    
    from sklearn.cluster import KMeans
    
    km = KMeans(n_clusters=n_clusters,**kwargs)
    
    return pd.Series(km.fit_predict(tsne.ix[cells]), index = cells)

################################################################################

def tsne_kmeans_select_n(dataset, tsne, cells, n_range, criterion = 'BIC', **kwargs):

    """
    Function returning information criteria scores for k-mean clusterings on tSNE embeddings.
    ----------
    dataset: [pd.Dataset] containing expression counts of n x genes in m x cells.
    tsne: [pd.Dataset] containing the tSNE coordinates of m cells.
    n_range: [list of ints] containing the cluster numbers to be evaluated.
    criterion: 'AIC' or 'BIC'
    ----------
    plots information criteria scores for specified cluster numbers
    """

    ##########################################################

    """
    Helper function to plot scores.
    """

    def plot_crit_n(scores, criterion):
    
        height = 7.5
        width = 10
        plt.figure(facecolor = 'w', figsize = (width, height))
        
        ax = plt.subplot(111)
        ax.set_xlim(0, len(scores)-1)
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels(scores.index)
        ax.set_ylabel('%s score' % criterion)
        
        ax.plot(range(len(scores)), scores, color = 'dodgerblue', linewidth = 3)

    ##########################################################
    
    scores = pd.Series(index = n_range)
    
    for n in n_range:
        km_tmp = tsne_kmeans(tsne, cells, n, **kwargs)
        scores.ix[n] = calculateIC_v2P(dataset, km_tmp, 0, criterion)
                        
    plot_crit_n(scores, criterion)

################################################################################

def calculateIC_v2P(dataset, groups, axis, criterion):
    
    """
    Calculates the Aikike (AIC) or Bayesian information criterion (BIC) using a formula described in
    http://en.wikipedia.org/wiki/Bayesian_information_criterion
    -----
    dataset: [pd.DataFrame] of m samples x n genes.
    groups: [pd.Series] containing group identity (int) for each sample or gene in dataframe.
    axis: 0 for samples, 1 for genes.
    criterion: 'AIC' or 'BIC'.
    """
    
    #for parallel processing, import modules and helper functions to engine namespace
    
    import numpy as np
    import pandas as pd
    from collections import Counter
    
    # main formula: BIC = N * ln (Vc) + K * ln (N)
    # main formula: AIC = 2 * N * ln(Vc) + 2 * K
    # Vc = error variance
    # n = number of data points
    # k = number of free parameters
    try:
        if axis == 0:
            X = dataset

        elif axis == 1:
            X = dataset.T

        Y = groups
        N = len(X.columns)
        K = len(set(Y))

        #1. Compute pd.Series Kl containing cluster lengths

        Kl = pd.Series(index = set(Y))
        Kl_dict = Counter(Y)

        for cluster in set(Y):
            Kl[cluster] = Kl_dict[cluster]

        #2. Compute pd.DataFrame Vc containing variances by cluster

        Vc = pd.DataFrame(index = X.index, columns = set(Y))

        for cluster in set(Y):

            tmp_ix = Y[Y == cluster].index
            tmp_X_var = X[tmp_ix].var(axis = 1) + 0.05 #to avoid -inf values
            Vc[cluster] = tmp_X_var

        #3. Calculate the mean variance for each cluster

        Vc = Vc.mean(axis = 0)

        #4. Calculate the ln of the mean variance

        Vc = np.log(Vc)

        #5. Multiply Vc by group size Kl

        Vc = Vc * Kl

        #6. Calculate accumulative error variance

        Vc = Vc.sum()

        #7a. Calculate BIC

        BIC = Vc + K * np.log(N)

        #7b. Calculate AIC

        AIC = 2 * Vc + 2 * K

        #8. Return AIC or BIC value


        if criterion == 'BIC':
            return BIC

        if criterion == 'AIC':
            return AIC
        
    except: return None

################################################################################

def AP_reorder_inclusters(dataset, groups, axis, linkage='Ward'):
    
    """
    Reorders cells within K-Means clusters using hierarchical clustering.
    ----------
    dataset: [pd.Dataset] containing expression counts of n genes x m cells.
    groups: [pd.Series] containing cluster IDs for m cells.
    axis: axis along which hierarchical clustering method is performed. 0: cells; 1: genes.
    linkage: scipy.cluster.hierarchy linkage method. Default: 'Ward'.
    ----------
    returns reordered [pd.Series] containing cluster IDs for m cells.
    """
    
    #save cluster order
    
    order = return_unique(groups)
    
    #create new incides
    
    ix_new = []
    
    #iterate over groups
    
    for gr in order:
        
        tmp_ix = groups[groups==gr].index
        
        # a. Cells

        if axis == 0:
            tmp_data = dataset[tmp_ix]
            tmp_dist = 1 - tmp_data.corr() #linkage at the moment only implemented with Pearson distance
            tmp_Z = sch.linkage(tmp_dist, method = linkage)
            tmp_leaves = sch.dendrogram(tmp_Z, no_plot = True)['leaves']
            tmp_sorted = [column for column in tmp_dist.ix[tmp_leaves,tmp_leaves].columns]

        # b. Genes
        elif axis == 1:

            tmp_data = dataset.ix[tmp_ix]
            tmp_dist = 1 - tmp_data.T.corr() #linkage at the moment only implemented with Pearson distance
            tmp_Z = sch.linkage(tmp_dist, method = linkage)
            tmp_leaves = sch.dendrogram(tmp_Z, no_plot = True)['leaves']
            tmp_sorted = [index for index in tmp_dist.ix[tmp_leaves,tmp_leaves].index]
        
    
        ix_new += tmp_sorted
            
    return groups[ix_new]

################################################################################

def AP_groups_reorder_v2(groups, order, link_to = None):
    
    """
    Reorders the groups in an sample or gene group Series either completely or partially
    -----
    groups: pd.Series of either samples (Cell ID) or gene (gene ID) linked to groups (int)
    order: list containing either complete or partial new order of groups
    link_to: defines which group position is retained when groups are reorded partially; default == None, groups are linked to
    first group in 'order'
    -----
    returns reordered group Series
    """
    
    # (1) Define new group order
    
    if set(order) == set(groups):
        order_new = order
        
    else:
        
        order_new = return_unique(groups, drop_zero = False)
        
        if link_to in order:
            link = link_to
        
        elif link_to not in order or link_to == None:
            link = order[0]
            
        order.remove(link)
        
        for group in order:
            
            order_new.remove(group)
            ins_ix = order_new.index(link) + 1
            gr_ix = order.index(group)
            order_new.insert(ins_ix + gr_ix, group)
            
    # (2) Reorder groups
    
    groups_new = pd.Series()
    
    for group in order_new:
        
        groups_new = groups_new.append(groups[groups == group])
        
    groups_new = groups_new
    
    return groups_new

################################################################################
################################## PLOTTING ####################################
################################################################################

def draw_log2_cv_diff(dataset, log2_cv_diff, xopt, selected = None):
    
    """
    Plots the average gene expression versus the coefficient of variation of log2 normalized expression data and overlays
    the expected cv per expression level from the the fitted noise model.
    -----
    dataset: seq dataset [pd.DataFrame] containing m cells x n genes. NB: Must be untransformed!
    log2_cv_diff: pd.Series containing difference between measured and expected cv for every gene.
    xopt: fitted parameters of noise model.
    selected: list or pd Index file containing the indices of selected cells. Default: none.
    """
    
    data_mean = dataset.mean(axis = 1)
    data_cv = dataset.std(axis = 1) / data_mean

    log2_mean = np.log2(data_mean)
    log2_cv = np.log2(data_cv)
    
    line_x = np.arange(log2_mean.min(), log2_mean.max(), 0.01)
    line_y = np.log2(( 2 ** line_x) ** xopt[0] + xopt[1])
    
    clist = pd.Series('blue', index = log2_mean.index)
    clist[log2_cv_diff[log2_cv_diff > 0].index] = 'red'
    
    if np.all(selected != None):
        clist[selected] = 'green'

    fig = plt.figure(figsize = [10,10], facecolor = 'w')
    ax0 = plt.axes()
    
    ax0.set_xlabel('Average number of transcripts [log2]')
    ax0.set_ylabel('Coefficient of variation (CV) [log2]')
    
    ax0.set_xlim(log2_mean.min() - 0.5, log2_mean.max() + 0.5)
    ax0.set_ylim(log2_cv.min() - 0.5, log2_cv.max() + 0.5)
    
    ax0.scatter(log2_mean, log2_cv, c = clist, linewidth = 0,)
    ax0.plot(line_x, line_y, c = 'k', linewidth = 3)
    
################################################################################

def draw_heatmap(dataset, cell_groups, gene_groups, cmap = plt.cm.jet):
    
    """
    Draw heatmap showing gene expression ordered according to cell_groups and
    gene_groups Series (e.g. AP clustering). Cell and gene groups membership is 
    visualized in two additional panels:
    ----------
    dataset: pd.DataFrame of m cells * n genes.
    cell_groups: pd.Series with ordered cluster identity of m cells.
    gene_groups: pd.Series with ordered cluster identity of n genes.
    cmap: matplotlib color map. Default: plt.cm.jet.
    """
    
    dataset = dataset.ix[gene_groups.index, cell_groups.index]
    dataset = dataset.apply(lambda x: x / max(x), axis = 1)

    plt.figure(figsize=(20,20), facecolor = 'w')
    
    #draw heatmap

    axSPIN1 = plt.axes()
    axSPIN1.set_position([0.05, 0.05, 0.9, 0.9])
    
    axSPIN1.imshow(dataset, aspect = 'auto', interpolation = 'nearest')
    
    remove_ticks(axSPIN1)
    
    #draw genes bar

    divider = make_axes_locatable(axSPIN1)

    axGene_gr = divider.append_axes("right", size= 0.5, pad=0.05)

    axGene_gr.set_xlim(-0.5,0.5)
    axGene_gr.imshow(np.matrix(gene_groups).T, aspect = 'auto')
    
    remove_ticks(axGene_gr)
    
    #draw genes bar ticks
    
    gene_groups_ticks = pd.Series(index = set(gene_groups))
    
    for gr in gene_groups_ticks.index:
                
        first_ix = list(gene_groups.values).index(gr)
        last_ix = len(gene_groups) - list(gene_groups.values)[::-1].index(gr)
        gene_groups_ticks[gr] = first_ix + ((last_ix - first_ix) / 2.0)
        
    axGene_gr.set_yticks(gene_groups_ticks.values)
    axGene_gr.set_yticklabels(gene_groups_ticks.index)
    axGene_gr.yaxis.set_ticks_position('right')
    
    #draw cells bar
    
    axCell_gr = divider.append_axes("bottom", size= 0.5, pad=0.05)

    axCell_gr.set_ylim(-0.5, 0.5)
    axCell_gr.imshow(np.matrix(cell_groups), aspect = 'auto')
    
    remove_ticks(axCell_gr)
    
    #draw cells bar ticks
    
    cell_groups_ticks = pd.Series(index = set(cell_groups))
        
    for gr in cell_groups_ticks.index:
                
        first_ix = list(cell_groups.values).index(gr)
        last_ix = len(cell_groups) - list(cell_groups.values)[::-1].index(gr)
        cell_groups_ticks[gr] = first_ix + ((last_ix - first_ix) / 2.0)
        
    axCell_gr.set_xticks(cell_groups_ticks.values)
    axCell_gr.set_xticklabels(cell_groups_ticks.index)
    axCell_gr.xaxis.set_ticks_position('bottom')

################################################################################

def draw_AP_dist_mat(dist_mat, groups, **kwargs):
    
    """
    Draws distance matrices of either m cells or n genes randomly shuffled and
    ordered according to group Series (e.g. AP clustering).
    ----------
    dist_mat: pd.DataFrame with distances of either m x m cells or n x n genes.
    groups: pd.Series with ordered cluster identity of m cells or n genes.
    """
    
    plt.figure(figsize = [20,10], facecolor = 'w')

    ax0 = plt.subplot(121)

    tmp_ix = list(dist_mat.index)
    random.shuffle(tmp_ix)

    ax0.matshow(dist_mat.ix[tmp_ix, tmp_ix], **kwargs)

    ax1 = plt.subplot(122)

    ax1.matshow(dist_mat.ix[groups.index, groups.index], **kwargs)
    
################################################################################
    
def draw_tSNE(tsne_coords, cell_groups, number = None, cmap = plt.cm.jet):
    
    """
    Function to draw tSNE plots.
    ----------
    tsne_coords: DataFrame of tSNE coordinates in two dimensions.
    cell_groups: Series of cell group identity.
    number: int for indentification of plot in tSNE screen.
    """
    
    #initialize figure

    height = 14
    width = 14

    plt.figure(facecolor = 'w', figsize = (width, height))

    #define x- and y-limits

    x_min, x_max = np.min(tsne_coords['x']), np.max(tsne_coords['x'])
    y_min, y_max = np.min(tsne_coords['y']), np.max(tsne_coords['y'])
    x_diff, y_diff = x_max - x_min, y_max - y_min

    pad = 2.0

    if x_diff > y_diff:
        xlim = (x_min - pad, x_max + pad)
        ylim = (y_min * (x_diff / y_diff) - pad, y_max * (x_diff / y_diff) + pad)

    if x_diff < y_diff:
        xlim = (x_min * (y_diff/x_diff) - pad, x_max * (y_diff/x_diff) + pad)
        ylim = (y_min - pad, y_max + pad)

    text_pad = 2
    
    #define x- and y-axes

    ax1 = plt.subplot()

    ax1.set_xlim(xlim[0], xlim[1])
    ax1.set_ylim(ylim[0], ylim[1])

    remove_ticks(ax1)
    
    #define colormap
    
    if type(cmap) != dict:
        cm = cmap
        cmap = {}
        for ix, gr in enumerate(return_unique(cell_groups)):
            cmap[gr] = cm(float(ix) / len(set(cell_groups)))
            
    clist_tsne = [cmap[cell_groups[c]] for c in cell_groups.index]

    ax1.scatter(tsne_coords.ix[cell_groups.index, 'x'],
                tsne_coords.ix[cell_groups.index, 'y'], 
                s = 100,
                linewidth = 0.0,
                c = clist_tsne)
    
    #draw number
    
    ax1.text(xlim[1] * 0.9, ylim[1] * 0.9, number, family = 'Arial', fontsize = 25)
    
################################################################################

def draw_barplots_v2(dataset, cell_groups, genes, cmap = plt.cm.jet):
    
    """
    draws expression of selected genes in order barplot with juxtaposed group identity
    dataset: pd.DataFrame of n samples over m genes
    sample_group_labels: ordered (!) pd.Series showing sample specific group indentity 
    list_of_genes: list of selected genes
    color: matplotlib cmap
    """
    
    # set figure framework
    
    plt.figure(facecolor = 'w', figsize = (21, len(genes) * 3 + 1))
        
    gs0 = plt.GridSpec(1,1, left = 0.05, right = 0.95, top = 1 - 0.05 / len(genes),
                       bottom = 1 - 0.15 / len(genes), hspace = 0.0, wspace = 0.0)
    
    gs1 = plt.GridSpec(len(genes), 1, hspace = 0.05, wspace = 0.0, left = 0.05, right = 0.95, 
                       top = 1 - 0.2 / len(genes) , bottom = 0.05)
    
    #define dataset
    
    dataset = dataset.ix[genes, cell_groups.index]
    
    #define max group ID for color definition
    
    val_max = float(len(return_unique(cell_groups)))
    
    #draw genes
    
    for ix, g in enumerate(genes):
    
        ax = plt.subplot(gs1[ix])
        ax.set_xlim(left = 0, right = (len(dataset.columns)))
                     
        if g != genes[-1]:
            ax.xaxis.set_ticklabels([])
        
        elif g == genes[-1]:
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(15) 
                
        ax.set_ylabel(g, fontsize = 25)
        ax.yaxis.labelpad = 10
        
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
            tick.label.set_label(10)
            
        ax.bar(np.arange(0, len(dataset.columns),1), 
               dataset.ix[g],
               width=1,
               color=[cmap(return_unique(cell_groups).index(val)/val_max) for val in cell_groups.values],
               linewidth=0)
    
    #create groups bar
    
    ax_bar = plt.subplot(gs0[0])
    
    ax_bar.set_xlim(left = 0, right = (len(dataset.columns)))
    ax_bar.set_ylim(bottom = 0, top = 1)
    
    for ix, val in enumerate(cell_groups.values):
        
        ax_bar.axvspan(ix,
                       ix+1,
                       ymin=0,
                       ymax=1, 
                       color = cmap(return_unique(cell_groups).index(val) / val_max))
        
    remove_ticks(ax_bar)
    
################################################################################

def draw_barplots_QC(data,param,plate,cmap,**kwargs):

    """
    Creates barplots showing total number of unique molecules or genes in each single cell.
    ----------
    data: [pd.DataFrame] of m samples x n genes.
    param: 'reads' or 'genes'
    plate: [pd.Series] containing plate ID (or any other group identifier) for m samples.
    cmap: [dict] with color-specifications for plate.
    """
    
    #compile data based on param
    
    if param == 'reads': data = data.sum(axis=0)
    elif param == 'genes': data = (data>0).sum(axis=0)
        
    #plot data
    
    #initialize figure

    height = 10
    width = 20
    plt.figure(facecolor = 'w', figsize = (width, height))
    
    ax = plt.subplot(111)
    ax.set_xlim(0, len(data.index))
    if param == 'reads': ax.set_ylabel('Number of reads per cell')
    elif param == 'genes': ax.set_ylabel('Number of genes per cell')
        
    #define patient colorscheme
    
    clist = [cmap[plate[ix]] for ix in data.index]
    
    #plot bars
    
    ax.bar(range(len(data.index)),
           data, color = clist,
           width = 1.0,
           **kwargs)
    
    #plot median and mean
    
    ax.axhline(np.median(data), color = 'blue')
    ax.axhline(np.mean(data), color = 'red')

################################################################################
######################## RECEPTOR-LIGAND-INTERACTIONS ##########################
################################################################################

def quantify_LigRec(gene_dict_Lig, gene_dict_Rec, pairs):

    """
    Quantifies potential receptor-ligand-pairs based on the database presented by Ramilowki et al.
    ----------
    gene_dict_Lig: [dict] containing the ligands expressed in each population.
    gene_dict_Lig: [dict] containing the receptors expressed in each population.
    pairs: [pd.Series] of ligand-receptor pairs with ligands as indices and receptors as values.
    ----------
    returns receptor-ligand pairs between populations as [dict] or [pd.DataFrame]
    returns receptor-ligand pair counts between populations as [dict] or [pd.DataFrame]
    """
    
    output_df = pd.DataFrame(index = gene_dict_Lig.keys(), columns = gene_dict_Rec.keys())
    output_dict = {}
    
    output_quant_df = pd.DataFrame(index = gene_dict_Lig.keys(), columns = gene_dict_Rec.keys())
    output_quant_dict = {}
    
    for kLig in gene_dict_Lig.keys():
        lig_exp = gene_dict_Lig[kLig]
        
        for kRec in gene_dict_Rec.keys():
            rec_exp = gene_dict_Rec[kRec]
                
            rl = []
            
            for l in lig_exp:
                for r in rec_exp:
                    if np.any(pairs.ix[l] == r):
                        rl.append((l,r))
            
            output_df.ix[kLig,kRec] = rl
            output_dict['%s - %s' % (kLig,kRec)] = rl
            
            output_quant_df.ix[kLig,kRec] = len(rl)
            output_quant_dict['%s - %s' % (kLig,kRec)] = len(rl)
            
    return output_df, output_dict, output_quant_df, output_quant_dict

################################################################################

def sim_lig_rec(dict_rec, dict_lig, pairs, repeats = 10000):

    """
    Simulates background distribution of receptor-ligand pairs between populations based on random sampling.
    ----------
    dict_lig: [dict] containing the ligands expressed in each population.
    dict_rec: [dict] containing the receptors expressed in each population.
    pairs: [pd.Series] of ligand-receptor pairs with ligands as indices and receptors as values.
    repeats: bootstrapping repeats for each population pair. Default: 10000.
    ----------
    returns [dict] containing for each population pair the number of R-L pairs matched randomly during each repeat.
    """
    
    output = {}
    
    for kLig in dict_lig.keys():
        nLig = len(dict_lig[kLig])
        
        for kRec in dict_rec.keys():
            nRec = len(dict_rec[kRec])
            
            output_tmp = []
            
            for r in range(repeats):
                gLig = np.random.choice(list(set(pairs.index)), size = nLig, replace = False)
                gRec = np.random.choice(list(set(pairs.values)), size = nRec, replace = False)
                                
                output_tmp.append(len([g for g in gRec if g in list(pairs.ix[gLig].values)]))
                
            output['%s - %s' % (kLig,kRec)] = output_tmp
            
    return output

################################################################################

def sim_lig_reg_get_p(data, sim):
    
    output = {}
    
    for k in sim.keys():
        output[k] = np.sum(np.array(sim[k])>data[k]) / len(sim[k])
        
    return output
    
################################################################################
############################## HELPER FUNCTIONS ################################
################################################################################

def chunks(l, n):
    """ 
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]

################################################################################

def remove_ticks(axes, linewidth = 0.5):
    """
    Removes ticks from matplotlib Axes instance
    """
    axes.set_xticklabels([]), axes.set_yticklabels([])
    axes.set_xticks([]), axes.set_yticks([])
    for axis in ['top','bottom','left','right']:
        axes.spines[axis].set_linewidth(linewidth)

################################################################################

def return_unique(groups, drop_zero = False):
    """
    Returns unique instances from a list (e.g. an AP cluster Series) in order 
    of appearance.
    """
    unique = []
    
    for element in groups.values:
        if element not in unique:
            unique.append(element)
            
    if drop_zero == True:
        unique.remove(0)
        
    return unique

################################################################################

def clean_axis(ax):
    """Remove ticks, tick labels, and frame from axis"""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

################################################################################

def get_CLR(corr_mat, method = 'normal', normalize = False):
    
    """
    Calculates context likelihood of relatedness (Faith et al. 2007).
    ----------
    corr_mat: DataFrame containing the similarity (e.g. Pearson correlation) of n x n genes.
    method: CLR method. 'normal'.
    ----------
    returns DataFrame of CLR-corrected distance of n x n genes.
    """
    
    #get zscores along both axes
           
    Z0 = corr_mat.apply(lambda x: (x - np.mean(x)) / np.std(x), axis = 0)
    Z1 = corr_mat.apply(lambda x: (x - np.mean(x)) / np.std(x), axis = 1)
    
    #combine zscores uing unweighted Stouffer method
        
    Z = (Z0 + Z1) / np.sqrt(2)
    
    if normalize == True: Z = (Z - Z.min().min()) / (Z.max().max() - Z.min().min())
        
    return Z

################################################################################

def draw_barplots_QC(data,param,plate,cmap,**kwargs):
    
    #compile data based on param
    
    if param == 'reads': data = data.sum(axis=0)
    elif param == 'genes': data = (data>0).sum(axis=0)
        
    #plot data
    
    #initialize figure

    height = 10
    width = 20
    plt.figure(facecolor = 'w', figsize = (width, height))
    
    ax = plt.subplot(111)
    ax.set_xlim(0, len(data.index))
    if param == 'reads': ax.set_ylabel('Number of reads per cell')
    elif param == 'genes': ax.set_ylabel('Number of genes per cell')
        
    #define patient colorscheme
    
    clist = [cmap[plate[ix]] for ix in data.index]
    
    #plot bars
    
    ax.bar(range(len(data.index)),
           data, color = clist,
           width = 1.0,
           **kwargs)
    
    #plot median and mean
    
    ax.axhline(np.median(data), color = 'blue')
    ax.axhline(np.mean(data), color = 'red')
    
################################################################################

def get_size_factors(counts):
    return np.exp2(np.log2(counts+1).apply(lambda x: x - np.mean(x), axis=1).median(axis=0))

################################################################################

def get_pval_binomial_distr(n, p, N):
    
    """
    n: number of trials == number of cells at a time point
    p: baseline success probability == percentage of false positive wound cells in control area
    N: number of successes == number of putative wound cells in population
    """
    
    return scipy.stats.binom.cdf(n-N, n, 1 - p)

################################################################################

def get_enrichment(meta, cells_ctrl, cells_ctrl_area, cells_wnd, cells_wnd_area):
    
    output = pd.DataFrame(index = ['1 d','4 d','7 d','10 d','1 m+'], 
                          columns = ['count - ctrl','perc - ctrl','count - wnd','perc - wnd','pval - wnd'])

    for ix in output.index:
        output.ix[ix, 'count - ctrl'] = Counter(meta.ix['Days', cells_ctrl])[ix]
        output.ix[ix, 'perc - ctrl'] = Counter(meta.ix['Days', cells_ctrl])[ix] / Counter(meta.ix['Days', cells_ctrl_area])[ix]
        output.ix[ix, 'count - wnd'] = Counter(meta.ix['Days', cells_wnd])[ix]
        output.ix[ix, 'perc - wnd'] = Counter(meta.ix['Days', cells_wnd])[ix] / Counter(meta.ix['Days', cells_wnd_area])[ix]
        output.ix[ix, 'pval - wnd'] = get_pval_binomial_distr(Counter(meta.ix['Days', cells_wnd_area])[ix], 
                                                              output.ix[ix, 'perc - ctrl'],
                                                              output.ix[ix, 'count - wnd'])
    
    return output

################################################################################
