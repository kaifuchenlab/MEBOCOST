#!/usr/bin/env python

# ================================
# @auther: Rongbin Zheng
# @email: Rongbin.Zheng@childrens.harvard.edu
# @date: May 2022
# ================================

import os,sys
import time, re
import pickle as pk
from datetime import datetime
import numpy as np
import pandas as pd
from operator import itemgetter
import scipy
from scipy import sparse
import scanpy as sc
import collections
import multiprocessing
import configparser
import tracemalloc
import seaborn as sns
from adjustText import adjust_text
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr

import mebocost.MetEstimator as ME
import mebocost.crosstalk_calculator as CC
import mebocost.crosstalk_plots as CP

plt.rcParams.update(plt.rcParamsDefault)
rc={"axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.titleweight":"bold", #"font.size":14,
    "figure.figsize":(5.5,4.2), "font.weight":"regular", "legend.fontsize":10,
    'axes.labelpad':8, 'figure.dpi':300}
plt.rcParams.update(**rc)

#!/usr/bin/env python

# ================================
# @auther: Rongbin Zheng
# @email: Rongbin.Zheng@childrens.harvard.edu
# @date: May 2024
# ================================

def info(string):
    """
    print information
    """
    today = datetime.today().strftime("%B %d, %Y")
    now = datetime.now().strftime("%H:%M:%S")
    current_time = today + ' ' + now
    print("[{}]: {}".format(current_time, string))


def _correct_colname_meta_(scRNA_meta, cellgroup_col=[]):
    """
    sometime the column names have different
    """
#     print(scRNA_meta)
    if scRNA_meta is None or scRNA_meta is pd.DataFrame:
        raise KeyError('Please provide cell_ann data frame!')
    
    if cellgroup_col:
        ## check columns names
        for x in cellgroup_col:
            if x not in scRNA_meta.columns.tolist():
                info('ERROR: given cell group identifier {} is not in meta table columns'.format(x))
                raise ValueError('given cell group identifier {} is not in meta table columns'.format(x))
        ## get cell group name
        scRNA_meta['cell_group'] = scRNA_meta[cellgroup_col].astype('str').apply(lambda row: '_'.join(row), axis = 1).tolist()
    else:
        info('no cell group given, try to search cluster and cell_type')
        col_names = scRNA_meta.columns.tolist()
        if 'cell_type' in col_names:
            pass
        elif 'cell_type' not in col_names and 'Cell_Type' in col_names:
            scRNA_meta.columns = ['cell_type' if x.upper() == 'CELL_TYPE' else x for x in col_names]
        elif 'cell_type' not in col_names and 'celltype' in col_names:
            scRNA_meta.columns = ['cell_type' if x.upper() == 'CELLTYPE' else x for x in col_names]
        elif 'cell_type' not in col_names and 'CellType' in col_names:
            scRNA_meta.columns = ['cell_type' if x.upper() == 'CELL TYPE' else x for x in col_names]
        else:
            info('ERROR: "cell_type" not in scRNA meta column names, will try cluster')
            if 'cluster' not in col_names and 'Cluster' in col_names:
                scRNA_meta.columns = ['cluster' if x.upper() == 'CLUSTER' else x for x in col_names]
            else:
                raise KeyError('cluster cannot find in the annotation, and cell_group does not specified')
            raise KeyError('cell_type cannot find in the annotation, and cell_group does not specified'.format(x))
        
        if 'cell_type' in scRNA_meta.columns.tolist():
            scRNA_meta['cell_group'] = scRNA_meta['cell_type'].tolist()
        elif 'cluster' in scRNA_meta.columns.tolist():
            scRNA_meta['cell_group'] = scRNA_meta['cluster'].tolist()
        else:
            raise KeyError('Please a group_col to group single cell')
    return(scRNA_meta)

def _read_config(conf_path):
    """
    read config file
    """
    #read config
    cf = configparser.ConfigParser()
    cf.read(conf_path)
    config = cf._sections
    # remove the annotation:
    for firstLevel in config.keys():
        for secondLevel in config[firstLevel]:
            if '#' in config[firstLevel][secondLevel]:
                config[firstLevel][secondLevel] = config[firstLevel][secondLevel][:config[firstLevel][secondLevel].index('#')-1].rstrip()
    return(config)

def load_obj(path):
    """
    read mebocost object
    """
    try:
        f = open(path, 'rb')
        obj_vars = pk.load(f)
        f.close()
    except:
        ## sometime the pandas version affects pickle reading
        obj_vars = pd.read_pickle(path)

    keys = list(obj_vars.keys())
    mebocost_obj = create_obj(exp_mat = obj_vars['exp_mat'],
                        adata = obj_vars['adata'],
                        cell_ann = obj_vars['cell_ann'],
                        group_col = obj_vars['group_col'] if 'group_col' in keys else ['celltype'],
                        config_path = obj_vars['config_path'],
                        met_enzyme = obj_vars['met_enzyme'],
                        met_sensor = obj_vars['met_sensor'],
                        met_ann = obj_vars['met_ann'], 
                        scFEA_ann = obj_vars['scFEA_ann'],
                        compass_met_ann = obj_vars['compass_met_ann'],
                        compass_rxn_ann = obj_vars['compass_rxn_ann']
                       )
    
    mebocost_obj.exp_mat_indexer = obj_vars['exp_mat_indexer'] if 'exp_mat_indexer' in keys else []
    mebocost_obj.exp_mat_columns = obj_vars['exp_mat_columns'] if 'exp_mat_columns' in keys else []
    mebocost_obj.avg_exp = obj_vars['avg_exp'] if 'avg_exp' in keys else pd.DataFrame()
    mebocost_obj.avg_exp_indexer = obj_vars['avg_exp_indexer'] if 'avg_exp_indexer' in keys else []
    mebocost_obj.avg_exp_columns = obj_vars['avg_exp_columns'] if 'avg_exp_columns' in keys else []
    mebocost_obj.avg_exp_columns = obj_vars['avg_exp_columns'] if 'avg_exp_columns' in keys else []
    mebocost_obj.met_mat = obj_vars['met_mat'] if 'met_mat' in keys else pd.DataFrame()
    mebocost_obj.met_mat_indexer = obj_vars['met_mat_indexer'] if 'met_mat_indexer' in keys else pd.DataFrame()
    mebocost_obj.met_mat_columns = obj_vars['met_mat_columns'] if 'met_mat_columns' in keys else pd.DataFrame()
    mebocost_obj.avg_met = obj_vars['avg_met'] if 'avg_met' in keys else pd.DataFrame()
    mebocost_obj.avg_met_indexer = obj_vars['avg_met_indexer'] if 'avg_met_indexer' in keys else pd.DataFrame()
    mebocost_obj.avg_met_columns = obj_vars['avg_met_columns'] if 'avg_met_columns' in keys else pd.DataFrame()

    mebocost_obj.species = obj_vars['species'] if 'species' in keys else 'human'
    mebocost_obj.met_est = obj_vars['met_est'] if 'met_est' in keys else 'mebocost'
    mebocost_obj.met_pred = obj_vars['met_pred'] if 'met_pred' in keys else pd.DataFrame()
    mebocost_obj.cutoff_exp = obj_vars['cutoff_exp'] if 'cutoff_exp' in keys else 0
    mebocost_obj.cutoff_met = obj_vars['cutoff_met'] if 'cutoff_met' in keys else 0
    mebocost_obj.cutoff_prop = obj_vars['cutoff_prop'] if 'cutoff_prop' in keys else 0.1
    mebocost_obj.sensor_type = obj_vars['sensor_type'] if 'sensor_type' in keys else ['Receptor', 'Transporter', 'Nuclear Receptor']
    mebocost_obj.thread = obj_vars['thread'] if 'thread' in keys else 1
    mebocost_obj.commu_res = obj_vars['commu_res'] if 'commu_res' in keys else pd.DataFrame()
    mebocost_obj.original_result = obj_vars['original_result'] if 'original_result' in keys else pd.DataFrame()
    mebocost_obj.commu_bg = obj_vars['commu_bg'] if 'commu_bg' in keys else dict()
    mebocost_obj.exp_prop = obj_vars['exp_prop'] if 'exp_prop' in keys else pd.DataFrame()
    mebocost_obj.met_prop = obj_vars['met_prop'] if 'met_prop' in keys else pd.DataFrame()

    return mebocost_obj


def save_obj(obj, path = 'mebocost_result.pk', filetype = 'pickle'):
    """
    save object to pickle
    """
    if filetype == 'pickle':
        obj_vars = vars(obj)
        out = open(path, 'wb')
        pk.dump(obj_vars, out)
        out.close()
    
def _check_exp_mat_(exp_mat, cell_ann):
    """
    check if the expression matrix are all numerical
    """
    str_cols = exp_mat.apply(lambda col: np.array_equal(col, col.astype(str)))
    str_rows = exp_mat.apply(lambda row: np.array_equal(row, row.astype(str)), axis = 1)
    if np.any(str_cols == True):
        info("%s column is a str, will be removed as only int or float accepted in expression matrix"%(exp_mat.columns[str_cols]))
    if np.any(str_rows == True):
        info("%s row is a str, will be removed as only int or float accepted in expression matrix"%(exp_mat.columns[str_cols]))
    exp_mat = exp_mat.loc[~str_rows, ~str_cols]
    ## check cell names
    if len(np.intersect1d(exp_mat.index, cell_ann.index)) ==  len(cell_ann.index) and len(np.intersect1d(exp_mat.columns, cell_ann.index)) ==  0:
        info("It seems the exp_mat needs to be transposed to match cell_ann, proceed...")
        exp_mat = exp_mat.T
    return(exp_mat)

class create_obj:
    """
    MEBOCOST for predicting metabolite-based cell-cell communication. The modules of the package include communication inference and visualization.

    Params
    -------
    exp_mat
        python pandas data frame, single cell expression matrix, rows are genes, columns are cells
        'exp_mat' is a exclusive parameter to 'adata'
    adata
        scanpy adata object, the expression will be extracted, 'adata' is an exclusive parameter to 'exp_mat'
    cell_ann
        data frame, cell annotation information, cells in row names
    group_col
        a list, specify the column names in 'cell_ann' for grouping cells, by default 'cell_type' or 'cluster' will be detected and used
    species
        human or mouse, this determines which database will be used in our collection

    met_est
        the method for estimating metabolite level in cell, should be one of:
        mebocost: estimated by the enzyme network related to the metabolite
        scFEA-flux: flux result of published software scFEA (https://pubmed.ncbi.nlm.nih.gov/34301623/)
        scFEA-balance: balance result of published software scFEA (https://pubmed.ncbi.nlm.nih.gov/34301623/)
        compass-reaction: reaction result of published software Compass (https://pubmed.ncbi.nlm.nih.gov/34216539/)
        compass-uptake: uptake result of published software Compass (https://pubmed.ncbi.nlm.nih.gov/34216539/)
        compass-secretion: secretion result of published software Compass (https://pubmed.ncbi.nlm.nih.gov/34216539/)
    met_pred
        data frame, if scFEA or Compass is used to impute the metabolite level in cells, please provide the original result from scFEA or Compass, cells in row names, metabolite/reaction/module in column names, 
        Noted that this parameter will be ignored if 'met_est' was set as mebocost.

    config_path
        str, the path for a config file containing the path of files for metabolite annotation, enzyme, sensor, scFEA annotation, compass annotation. These can also be specified separately by paramters as following:

        if config_path not given, please set:
    met_enzyme
        data frame, metabolite and gene (enzyme) relationships, required columns include HMDB_ID, gene, direction, for instance:
        
        HMDB_ID     gene                                                direction
        HMDB0003375 Cyp2c54[Unknown]; Cyp2c38[Unknown]; Cyp2c50[Un...   substrate
        HMDB0003375 Cyp2c54[Unknown]; Cyp2c38[Unknown]; Cyp2c50[Un...   substrate
        HMDB0003375 Cyp2c54[Unknown]; Cyp2c38[Unknown]; Cyp2c50[Un...   substrate
        HMDB0003450 Cyp2c54[Unknown]; Cyp2c38[Unknown]; Cyp2c50[Un...   product
        HMDB0003948 Tuba8[Unknown]; Ehhadh[Unknown]; Echs1[Enzyme]...   product

    met_sensor
        data frame, metabolite sensor information, each row is a pair of metabolite and sensor, must include columns  HMDB_ID, Gene_name, Annotation, for instance:
        
        HMDB_ID Gene_name   Annotation
        HMDB0006247 Abca1   Transporter
        HMDB0000517 Slc7a1  Transporter
        HMDB0000030 Slc5a6  Transporter
        HMDB0000067 Cd36    Transporter
        
    met_ann:
        data frame, the annotation of metabolite collected from HMDB website, these are basic annotation info including HMDB_ID, Kegg_ID, metabolite, etc

    scFEA_ann
        data frame, module annotation of metabolite flux in scFEA, usually is the file at https://github.com/changwn/scFEA/blob/master/data/Human_M168_information.symbols.csv

    compass_met_ann
        data frame, the metabolite annotation used in Compass software, usually is the file at https://github.com/YosefLab/Compass/blob/master/compass/Resources/Recon2_export/met_md.csv

    compass_rxn_ann
        data frame, the reaction annotation used in Compass software, usually is the file at https://github.com/YosefLab/Compass/blob/master/compass/Resources/Recon2_export/rxn_md.csv

    cutoff_exp
        auto or float, used to filter out cells which are lowly expressed for the given gene, by default is auto, meaning that automatically decide cutoffs for sensor expression to exclude the lowly 25% non-zeros across all sensor or metabolites in all cells in addition to zeros 

    cutoff_met
        auto or float, used to filter out cells which are lowly abundant of the given metabolite, by default is auto, meaning that automatically decide cutoffs for metabolite aggregated enzyme to exclude the lowly 25% non-zeros across all sensor or metabolites in all cells in addition to zeros 

    cutoff_prop
        float from 0 to 1, used to filter out metabolite or genes if the proportion of their abundant cells less than the cutoff

    sensor_type
        a list, provide a list of sensor type that will be used in the communication modeling, must be one or more from ['Receptor', 'Transporter', 'Nuclear Receptor'], default is all the three

    thread
        int, number of cores used for running job, default 1
        
    """
    def __init__(self,  
                exp_mat=None, 
                adata=None, 
                cell_ann=None,
                group_col=[],
                species = 'human',

                met_est=None,
                met_pred=pd.DataFrame(), 

                config_path=None,
                met_enzyme=pd.DataFrame(),
                met_sensor=pd.DataFrame(),
                met_ann=pd.DataFrame(), 
                scFEA_ann=pd.DataFrame(),
                compass_met_ann=pd.DataFrame(),
                compass_rxn_ann=pd.DataFrame(),

                cutoff_exp='auto',
                cutoff_met='auto',
                cutoff_prop=0.15,

                sensor_type=['Receptor', 'Transporter', 'Nuclear Receptor'],
                thread = 1
                ):
        tic = time.time()

        self.exp_mat = exp_mat
        self.adata = adata
        ## check cell group information
        ## add a column "cell_group" if successfull
        if (self.exp_mat is None and cell_ann is None) and (self.adata is not None):
            cell_ann = adata.obs.copy()
        self.group_col = ['cell_type', 'cluster'] if not group_col else group_col
        self.cell_ann = _correct_colname_meta_(cell_ann, cellgroup_col = self.group_col)
        self.species = species

        self.met_est = 'mebocost' if not met_est else met_est # one of [scFEA-flux, scFEA-balance, compass-reaction, compass-uptake, compass-secretion]
        self.met_pred = met_pred

        ## the path of config file
        self.config_path = config_path
        ## genes (enzyme) related to met
        self.met_enzyme = met_enzyme
        ## gene name in metaboltie sensor
        self.met_sensor = met_sensor
        ## met basic ann
        self.met_ann = met_ann
        ## software ann
        self.scFEA_ann = scFEA_ann
        self.compass_met_ann = compass_met_ann
        self.compass_rxn_ann = compass_rxn_ann

        if not self.config_path and (self.met_sensor is None or self.met_sensor.shape[0] == 0):
            raise KeyError('Please either provide config_path or a data frame of met_enzyme, met_sensor, met_ann, etc')

        ## cutoff for expression, metabolite, and proportion of cells
        self.cutoff_exp = cutoff_exp
        self.cutoff_met = cutoff_met
        self.cutoff_prop = cutoff_prop
        self.sensor_type = sensor_type
        self.thread = thread
        
        ## ============== initial ===========

        if self.exp_mat is None and self.adata is None:
            raise ValueError('ERROR: please provide expression matrix either from exp_mat or adata (scanpy object)')  
        elif self.exp_mat is None and self.adata is not None:
            ## check the adata object
            ngene = len(self.adata.var_names)
            ncell = len(self.adata.obs_names)
            info('We get expression data with {n1} genes and {n2} cells.'.format(n1 = ngene, n2 = ncell))
            if ngene < 5000:
                info('scanpy object contains less than 5000 genes, please make sure you are using raw.to_adata()')
            self.exp_mat = sparse.csc_matrix(self.adata.X.T)
            self.exp_mat_indexer = self.adata.var_names
            self.exp_mat_columns = self.adata.obs_names
            self.adata = None
        else:
            if 'scipy.sparse' in str(type(self.exp_mat)):
                ## since the scipy version problem leads to the failure of using sparse.issparse
                ## use a simple way to check!!!
                #sparse.issparse(self.exp_mat):
                pass 
            elif type(self.exp_mat) is type(pd.DataFrame()):
                self.exp_mat = _check_exp_mat_(self.exp_mat, self.cell_ann)
                self.exp_mat_indexer = self.exp_mat.index ## genes
                self.exp_mat_columns = self.exp_mat.columns ## columns
                self.exp_mat = sparse.csc_matrix(self.exp_mat)
                ngene, ncell = self.exp_mat.shape
                print(1)
                info('We get expression data with {n1} genes and {n2} cells.'.format(n1 = ngene, n2 = ncell))
            else:
                info('ERROR: cannot read the expression matrix, please provide pandas dataframe or scanpy adata')

        ## end preparation
        toc = time.time()
        info('Data Preparation Done in {:.4f} seconds'.format(toc-tic))
    
    def _load_config_(self):
        """
        load config and read data from the given path based on given species
        """
        ## the path of config file
        info('Load config and read data based on given species [%s].'%(self.species))
        if self.config_path:
            if not os.path.exists(self.config_path):
                raise KeyError('ERROR: the config path is not exist!')
            config = _read_config(conf_path = self.config_path)
            ## common
            self.met_ann = pd.read_csv(config['common']['hmdb_info_path'], sep = '\t')
            if self.met_est.startswith('scFEA'):
                    self.scFEA_ann = pd.read_csv(config['common']['scfea_info_path'], index_col = 0)
            if self.met_est.startswith('compass'):
                self.compass_met_ann = pd.read_csv(config['common']['compass_met_ann_path'])
                self.compass_rxn_ann = pd.read_csv(config['common']['compass_rxt_ann_path'])
            ## depends on species
            if self.species == 'human':
                self.met_enzyme = pd.read_csv(config['human']['met_enzyme_path'], sep = '\t')
                met_sensor = pd.read_csv(config['human']['met_sensor_path'], sep = '\t')
#                 met_sensor['gene'] = met_sensor['Gene_name'].apply(lambda x: x.split('[')[0])
                self.met_sensor = met_sensor
            elif self.species == 'mouse':
                self.met_enzyme = pd.read_csv(config['mouse']['met_enzyme_path'], sep = '\t')
                met_sensor = pd.read_csv(config['mouse']['met_sensor_path'], sep = '\t')
#                 met_sensor['gene'] = met_sensor['Gene_name'].apply(lambda x: x.split('[')[0])
                self.met_sensor = met_sensor
            else:
                raise KeyError('Species should be either human or mouse!')
                            ## check row and columns, we expect rows are genes, columns are cells
                if len(set(self.met_sensor['Gene_name'].tolist()) & set(self.exp_mat_indexer.tolist())) < 10 and len(set(self.met_sensor['Gene_name'].tolist()) & set(self.exp_mat_columns.tolist())) < 10:
                    raise KeyError('it looks like that both the row and columns are not matching to gene name very well, please check the provided matrix or species!')
                if len(set(self.met_sensor['Gene_name'].tolist()) & set(self.exp_mat_indexer.tolist())) < 10 and len(set(self.met_sensor['Gene_name'].tolist()) & set(self.exp_mat_columns.tolist())) > 10:
                    info('it is likely the columns of the exp_mat are genes, will transpose the matrix')
                    self.exp_mat = self.exp_mat.T
                    columns = self.exp_mat_indexer.copy()
                    index = self.exp_mat_columns.copy()
                    self.exp_mat_indexer = index
                    self.exp_mat_columns = columns
        else:
            info('please provide config path')

    def estimator(self):
        """
        estimate of metabolite level in cells using the expression of related enzymes
        """
        info('Aggregate metabolite enzymes using %s'%self.met_est)
        mtd = self.met_est

        if mtd == 'mebocost':
            met_mat, met_indexer, met_columns = ME._met_from_enzyme_est_(exp_mat=self.exp_mat, 
                                                   indexer = self.exp_mat_indexer,
                                                   columns = self.exp_mat_columns,
                                                    met_gene=self.met_enzyme, 
                                                    method = 'mean')
        elif mtd == 'scFEA-flux':
            met_mat = ME._scFEA_flux_est_(scFEA_pred = self.met_pred, 
                                            scFEA_info=self.scFEA_ann, 
                                            hmdb_info=self.met_ann)
        elif mtd == 'scFEA-balance':
            met_mat = ME._scFEA_balance_est_(scFEA_pred = self.met_pred, 
                                                scFEA_info=self.scFEA_ann, 
                                                hmdb_info=self.met_ann)
        elif mtd == 'compass-reaction':
            met_mat = ME._compass_react_est_(compass_pred=self.met_pred, 
                                                compass_react_ann=self.compass_rxn_ann, 
                                                compass_met_ann=self.compass_met_ann, 
                                                hmdb_info=self.met_ann)
        else:
            raise KeyError('Please specify "met_est" to be one of [mebocost, scFEA-flux, scFEA-balance, compass-reaction, compass-uptake, compass-secretion]')
        
        self.met_mat = sparse.csc_matrix(met_mat)
        self.met_mat_indexer = np.array(met_indexer)
        self.met_mat_columns = np.array(met_columns)
#         return met_mat


    def infer(self, met_mat=pd.DataFrame(), n_shuffle = 1000, seed = 12345, thread = None):
        """
        excute communication prediction
        met_mat
            data frame, columns are cells and rows are metabolites
        """
        info('Infer communications')
        if met_mat.shape[0] != 0: ## if given met_mat in addition
            self.met_mat_indexer = np.array(met_mat.index)
            self.met_mat_columns = np.array(met_mat.columns)
            self.met_mat = sparse.csc_matrix(met_mat)
        ## focus on met and gene of those are in the data matrix
        met_sensor = self.met_sensor[self.met_sensor['Gene_name'].isin(self.exp_mat_indexer) & 
                                     self.met_sensor['HMDB_ID'].isin(self.met_mat_indexer)]
        self.met_sensor = met_sensor

        ## init
        cobj = CC.InferComm(exp_mat = self.exp_mat,
                            exp_mat_indexer = self.exp_mat_indexer, 
                            exp_mat_columns = self.exp_mat_columns,
                            avg_exp = self.avg_exp,
                            avg_exp_indexer = self.avg_exp_indexer,
                            avg_exp_columns = self.avg_exp_columns,
                            met_mat = self.met_mat,
                            met_mat_indexer = self.met_mat_indexer,
                            met_mat_columns = self.met_mat_columns,
                            avg_met = self.avg_met,
                            avg_met_indexer = self.avg_met_indexer,
                            avg_met_columns = self.avg_met_columns,
                            cell_ann = self.cell_ann,
                            met_sensor = self.met_sensor,
                            sensor_type = self.sensor_type,
                            thread = thread
                           )

        commu_res_df, commu_res_bg = cobj.pred(n_shuffle = n_shuffle, seed = seed)
    
        ## add metabolite name
        hmdbid_to_met = {}
        for Id, met in self.met_ann[['HMDB_ID', 'metabolite']].values.tolist():
            hmdbid_to_met[Id] = met
        ## add name
        commu_res_df['Metabolite_Name'] = list(map(lambda x: hmdbid_to_met.get(x) if x in hmdbid_to_met else None,
                                                   commu_res_df['Metabolite']))

        ## add annotation
        sensor_to_ann = {}
        for s, a in self.met_sensor[['Gene_name', 'Annotation']].values.tolist():
            sensor_to_ann[s] = a
        commu_res_df['Annotation'] = list(map(lambda x: sensor_to_ann.get(x) if x in sensor_to_ann else None,
                                              commu_res_df['Sensor']))
        
        return commu_res_df, commu_res_bg


    def _filter_lowly_aboundant_(self, 
                                 pvalue_res,
                                 cutoff_prop,
                                 met_prop=None,
                                 exp_prop=None,
                                 min_cell_number=50
                                ):
        """
        change p value to 1 if either metabolite_prop or transporter_prop equal to 0 
        (meaning that no metabolite or transporter level in the cluster)
        """
        res = pvalue_res.copy()
        ## add the metabolite abudance proportion
        if met_prop is not None:
            res['metabolite_prop_in_sender'] = [met_prop.loc[s, m] for s, m in res[['Sender', 'Metabolite']].values.tolist()]
        ## add the metabolite abudance proportion
        if exp_prop is not None:
            res['sensor_prop_in_receiver'] = [exp_prop.loc[r, s] for r, s in res[['Receiver', 'Sensor']].values.tolist()]
        
        if 'original_result' not in list(vars(self)):
            self.original_result = res.copy()
        ## minimum cell number
        cell_count = pd.Series(dict(collections.Counter(self.cell_ann['cell_group'].tolist())))
        bad_cellgroup = cell_count[cell_count<min_cell_number].index.tolist() 
        
        info('Set p value and fdr to 1 if sensor or metaboltie expressed cell proportion less than {}'.format(cutoff_prop))
        bad_index = np.where((res['metabolite_prop_in_sender'] <= cutoff_prop) |
                             (res['sensor_prop_in_receiver'] <= cutoff_prop) |
                             (res['Commu_Score'] < 0) |
                             (res['Sender'].isin(bad_cellgroup)) | 
                             (res['Receiver'].isin(bad_cellgroup))
                            )[0]
        if len(bad_index) > 0:
            pval_index = np.where(res.columns.str.endswith('_pval'))[0]
            res.iloc[bad_index, pval_index] = 1 # change to 1
            fdr_index = np.where(res.columns.str.endswith('_fdr'))[0]
            res.iloc[bad_index, fdr_index] = 1 # change to 1
        
        ## norm communication score
        res['Commu_Score'] = res['Commu_Score']/np.array(res['bg_mean']).clip(min = 0.05)
        
        ## reorder columns
        columns = ['Sender', 'Metabolite', 'Metabolite_Name', 
                   'Receiver', 'Sensor', 'Commu_Score', 
                   'metabolite_prop_in_sender',
                   'sensor_prop_in_receiver', 'Annotation',
#                    'bg_mean', 'bg_std',
                   'ztest_stat', 'ztest_pval', 'ttest_stat',
                   'ttest_pval', 'ranksum_test_stat', 'ranksum_test_pval',
                   'permutation_test_stat', 'permutation_test_pval',
                   'ztest_fdr', 'ttest_fdr', 'ranksum_test_fdr',
                   'permutation_test_fdr']
        get_columns = [x for x in columns if x in res.columns.tolist()]
        res = res.reindex(columns = get_columns).sort_values('permutation_test_fdr')
        return(res)

    def _auto_cutoff_(self, mat, q = 0.25):
        """
        given a matrix, such as gene-by-cell matrix,
        find 25% percentile value as a cutoff
        meaning that, for example, sensor in cell with lowest 25% expression will be discarded, by default.
        """
        v = []
        for x in mat:
            if np.all(x.toarray() <= 0):
                continue
            xx = x.toarray()
            xx = xx[xx>0]
            v.extend(xx.tolist())
        v = np.array(sorted(v))
        c = np.quantile(v, q)
        return(c)


    def _check_aboundance_(self, cutoff_exp=None, cutoff_met=None):
        """
        check the aboundance of metabolite or transporter expression in cell clusters,
        return the percentage of cells that meet the given cutoff
        by default, cutoff for metabolite aboundance is 0, expression of transporter is 0
        """
        info('Calculating metabolite aggregated enzyme and sensor expression in cell groups')
        ## this will re-write the begin values
        j1 = cutoff_exp is None or cutoff_exp is False
        j2 = self.cutoff_exp is None or self.cutoff_exp is False
        j3 = self.cutoff_exp == 'auto'
        j4 = isinstance(self.cutoff_exp, float) or isinstance(self.cutoff_exp, int)

        if cutoff_exp == 'auto':
            # decide cutoff by taking 75% percentile across all sensor in all cells
            sensor_loc = np.where(self.exp_mat_indexer.isin(self.met_sensor['Gene_name']))[0]
            sensor_mat = self.exp_mat[sensor_loc,:]
            cutoff_exp = self._auto_cutoff_(mat = sensor_mat)
            self.cutoff_exp = cutoff_exp
            info('automated cutoff for sensor expression, cutoff=%s'%cutoff_exp)
        elif j1 and (j2 or j3):
            ## decide cutoff by taking 75% percentile across all sensor in all cells
            sensor_loc = np.where(self.exp_mat_indexer.isin(self.met_sensor['Gene_name']))[0]
            sensor_mat = self.exp_mat[sensor_loc,:]
            cutoff_exp = self._auto_cutoff_(mat = sensor_mat)
            self.cutoff_exp = cutoff_exp
            info('automated cutoff for sensor expression, cutoff=%s'%cutoff_exp)
        elif j1 and j4:
            cutoff_exp = self.cutoff_exp 
            info('provided cutoff for sensor expression, cutoff=%s'%cutoff_exp)
        elif j1 and j2:
            cutoff_exp = 0
            info('cutoff for sensor expression, cutoff=%s'%cutoff_exp)
        else:
            cutoff_exp = 0 if not cutoff_exp else cutoff_exp
            info('cutoff for sensor expression, cutoff=%s'%cutoff_exp)
        ## met 
        j1 = cutoff_met is None or cutoff_met is False
        j2 = self.cutoff_met is None or self.cutoff_met is False
        j3 = self.cutoff_met == 'auto'
        j4 = isinstance(self.cutoff_met, float) or isinstance(self.cutoff_met, int)

        if cutoff_met == 'auto':
            ## decide cutoff by taking 75% percentile across all sensor in all cells
            cutoff_met = self._auto_cutoff_(mat = self.met_mat)
            self.cutoff_met = cutoff_met
            info('automated cutoff for metabolite aggregated enzyme, cutoff=%s'%cutoff_met)
        elif j1 and (j2 or j3):
            ## decide cutoff by taking 75% percentile across all sensor in all cells
            cutoff_met = self._auto_cutoff_(mat = self.met_mat)
            self.cutoff_met = cutoff_met
            info('automated cutoff for metabolite aggregated enzyme, cutoff=%s'%cutoff_met)
        elif j1 and j4:
            cutoff_met = self.cutoff_met 
            info('provided cutoff for metabolite aggregated enzyme, cutoff=%s'%cutoff_met)
        elif j1 and j2:
            cutoff_met = 0
            info('cutoff for metabolite aggregated enzyme, cutoff=%s'%cutoff_met)
        else:
            cutoff_met = 0 if not cutoff_met else cutoff_met
            info('cutoff for metabolite aggregated enzyme, cutoff=%s'%cutoff_met)

        ## expression for all transporters
        sensors = self.met_sensor['Gene_name'].unique().tolist()
        info('cutoff_exp: {}'.format(cutoff_exp))
        
        sensor_loc = {g:i for i,g in enumerate(self.exp_mat_indexer) if g in sensors}
        exp_prop = {}
        for x in self.cell_ann['cell_group'].unique().tolist():
            cells = self.cell_ann[self.cell_ann['cell_group'] == x].index.tolist()
            cell_loc = [i for i, c in enumerate(self.exp_mat_columns) if c in cells]
            s = self.exp_mat[list(sensor_loc.values()),:][:,cell_loc]
            exp_prop[x] = pd.Series([v[v>cutoff_exp].shape[1] / v.shape[1] for v in s],
                                   index = list(sensor_loc.keys()))
        exp_prop = pd.DataFrame.from_dict(exp_prop, orient = 'index')
         
        # ====================== #
        info('cutoff_metabolite: {}'.format(cutoff_met))
        ## metabolite aboundance
        metabolites = self.met_sensor['HMDB_ID'].unique().tolist()
        met_prop = {}
        for x in self.cell_ann['cell_group'].unique().tolist():
            cells = self.cell_ann[self.cell_ann['cell_group'] == x].index.tolist()
            cell_loc = [i for i, c in enumerate(self.met_mat_columns) if c in cells]
            m = self.met_mat[:,cell_loc]
            met_prop[x] = pd.Series([v[v>cutoff_met].shape[1] / v.shape[1] for v in m],
                                   index = self.met_mat_indexer.tolist())
        met_prop = pd.DataFrame.from_dict(met_prop, orient = 'index')

        return exp_prop, met_prop ## cell_group x sensor gene, cell_group x metabolite
    
    def _get_gene_exp_(self):
        """
        only sensor and enzyme gene expression are needed for each cells
        """
        sensors = self.met_sensor['Gene_name'].unique().tolist()
        enzymes = []
        for x in self.met_enzyme['gene'].tolist():
            enzymes.extend([i.split('[')[0] for i in x.split('; ')])
        genes = list(set(sensors+enzymes))
        ## gene loc
        gene_loc = np.where(pd.Series(self.exp_mat_indexer).isin(genes))[0]
        
        gene_dat = self.exp_mat[gene_loc].copy()
        ## update the exp_mat and indexer
        self.exp_mat = sparse.csr_matrix(gene_dat)
        self.exp_mat_indexer = self.exp_mat_indexer[gene_loc]
                                   
    def _avg_by_group_(self):
        ## avg exp by cell_group for met sensor
        group_names = self.cell_ann['cell_group'].unique().tolist()
        avg_exp = np.empty(shape = (self.exp_mat.shape[0],0)) ## save exp data

        for x in group_names:
            cells = self.cell_ann[self.cell_ann['cell_group'] == x].index.tolist()
            cell_loc = np.where(pd.Series(self.exp_mat_columns).isin(cells))[0]
            # arithmatic mean
            avg_exp = np.concatenate((avg_exp, self.exp_mat[:,cell_loc].mean(axis = 1)), axis = 1)
        
        self.avg_exp = sparse.csr_matrix(avg_exp)
        self.avg_exp_indexer = np.array(self.exp_mat_indexer)
        self.avg_exp_columns = np.array(group_names)
    
    
    def _avg_met_group_(self):
        """
        take average of sensor expression and metabolite by cell groups
        """
        ## avg met by cell_group for met
        avg_met = np.empty(shape = (self.met_mat.shape[0],0)) ## save exp data
        group_names = self.cell_ann['cell_group'].unique().tolist()

        for x in group_names:
            cells = self.cell_ann[self.cell_ann['cell_group'] == x].index.tolist()
            cell_loc = np.where(pd.Series(self.met_mat_columns).isin(cells))[0]
            ## mean
            avg_met = np.concatenate((avg_met, self.met_mat[:,cell_loc].mean(axis = 1)), axis = 1)

        self.avg_met = sparse.csr_matrix(avg_met)
        self.avg_met_indexer = np.array(self.met_mat_indexer)
        self.avg_met_columns = group_names

        
    def infer_commu(self, 
                      n_shuffle = 1000,
                      seed = 12345, 
                      Return = True, 
                      thread = None,
                      save_permuation = False,
                      min_cell_number = 50,
                      pval_method='permutation_test_fdr', 
                      pval_cutoff = 0.05
                     ):
        """
        execute mebocost to infer communications

        Params
        -----
        n_shuffle
            int, number of cell label shuffling for generating null distribution when calculating p-value
            
        seed
            int, a random seed for shuffling cell labels, set seed to get reproducable shuffling result 
            
        Return
            True or False, set True to return the communication event in a data frame
            
        thread
            int, the number of cores used in the computing, default None, thread set when create the object has the highest priority to be considered, so only set thread here if you want to make a change
            
        save_permuation
            True or False, set True to save the communication score for each permutation, this could occupy a higher amount of space when saving out, so default is False

        min_cell_number
            int, the cell groups will be excluded and p-value will be replaced to 1 if there are not enough number of cells (less than min_cell_number), default is 50
        pval_method
            should be one of ['zztest_pval', 'ttest_pval', 'ranksum_test_pval', 'permutation_test_pval', 'zztest_fdr', 'ttest_fdr', 'ranksum_test_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        
        pval_cutoff
            float, a value in range between 0 and 1, pvalue less than the cutoff considered as significant event
        progress_info
            print out the progress notification
        """
        tic = time.time()
        today = datetime.today().strftime("%B %d, %Y")
        now = datetime.now().strftime("%H:%M:%S")
        current_time = today + ' ' + now
        self.commu_time_stamp = current_time
        
        tracemalloc.start()
        ## load config
        self._load_config_()
        
        ## take average by cell group, this must be done before extract sensor and enzyme gene expression of cells
        self._avg_by_group_()
        
        ## extract exp data for sensor and enzyme genes for all cells
        self._get_gene_exp_()
        
        ## estimate metabolite
        self.estimator()
        
        ## avg met mat
        self._avg_met_group_()
    
        # running communication inference
        commu_res_df, commu_res_bg = self.infer(
                                                n_shuffle = n_shuffle, 
                                                seed = seed,
                                                thread = self.thread if thread is None else thread ## allow to set thread in this function
                                                )
        
        ## update self
        self.commu_res = commu_res_df
        if save_permuation:
            self.commu_bg = commu_res_bg
        
        ## check cell proportion
        exp_prop, met_prop = self._check_aboundance_()
        ## update self
        self.exp_prop = exp_prop
        self.met_prop = met_prop 
        
        ## check low and set p val to 1
        commu_res_df_updated = self._filter_lowly_aboundant_(pvalue_res = commu_res_df,
                                                             cutoff_prop = self.cutoff_prop,
                                                             met_prop=self.met_prop, 
                                                             exp_prop=self.exp_prop,
                                                             min_cell_number = min_cell_number)
        ## update self
        self.commu_res = commu_res_df_updated[(commu_res_df_updated[pval_method]<pval_cutoff)]
        
        current, peak = tracemalloc.get_traced_memory()
        
        # stopping the library
        tracemalloc.stop()
        
        toc = time.time()

        info('Prediction Done in {:.4f} seconds'.format(toc-tic))
        info('Memory Usage in Peak {:.2f} GB'.format(peak / 1024 / 1024 / 1024))
        if Return:
            return(self.commu_res)
        
## ============================== constrain by flux ============================
    def _get_compass_flux_(self, compass_folder):  
        if os.path.exists(compass_folder):
            uptake_path = os.path.join(compass_folder, 'uptake.tsv')
            secret_path = os.path.join(compass_folder, 'secretions.tsv')
            if os.path.exists(uptake_path) and os.path.exists(secret_path):
                uptake = pd.read_csv(uptake_path, index_col = 0, sep = '\t')
                secretion = pd.read_csv(secret_path, index_col = 0, sep = '\t')
            else:
                uptake_path = os.path.join(compass_folder, 'uptake.tsv.gz')
                secret_path = os.path.join(compass_folder, 'secretions.tsv.gz')
                if os.path.exists(uptake_path) and os.path.exists(secret_path):
                    uptake = pd.read_csv(uptake_path, index_col = 0, sep = '\t')
                    secretion = pd.read_csv(secret_path, index_col = 0, sep = '\t')
                else:
                    raise ValueError('Failed to identify COMPASS output files')
        else:
            raise ValueError('compass_folder path does not exist')
        ## load compass annotation
        compass_met_ann = pd.read_csv(_read_config(self.config_path)['common']['compass_met_ann_path'])
        # compass_rxn_ann = pd.read_csv(_read_config(self.config_path)['common']['compass_rxt_ann_path'])
        ## annotate compass result
        efflux_mat = pd.merge(secretion, compass_met_ann[['met', 'hmdbID']],
                                left_index = True, right_on = 'met').dropna()
        efflux_mat = pd.merge(efflux_mat, self.met_ann[['Secondary_HMDB_ID', 'metabolite']],
                                left_on = 'hmdbID', right_on = 'Secondary_HMDB_ID')
        efflux_mat = efflux_mat.drop(['met','hmdbID','Secondary_HMDB_ID'], axis = 1).groupby('metabolite').max()
        influx_mat = pd.merge(uptake, compass_met_ann[['met', 'hmdbID']],
                                left_index = True, right_on = 'met').dropna()
        influx_mat = pd.merge(influx_mat, self.met_ann[['Secondary_HMDB_ID', 'metabolite']],
                                left_on = 'hmdbID', right_on = 'Secondary_HMDB_ID')
        influx_mat = influx_mat.drop(['met','hmdbID','Secondary_HMDB_ID'], axis = 1).groupby('metabolite').max()
        self.efflux_mat = efflux_mat
        self.influx_mat = influx_mat
        
    def _ConstainFlux_(self, compass_folder, efflux_cut = 'auto', influx_cut='auto', inplace=True):
        """
        a function to filter out communications with low efflux and influx rates based on COMPASS output, the commu_res will be replaced by updated table
        Params
        -----
        compass_folder: a path to indicate COMPASS output folder. The folder should include secretions.tsv and uptake.tsv for cell group level.
        efflux_cut: a numeric efflux threshold to indicate active efflux event. Default sets to 'auto', which determines the threshold by taking 25th percentile of COMPASS values after square root transfermation ((x/np.abs(x)) * np.sqrt(np.abs(x)))
        influx_cut: a numeric ifflux threshold to indicate active influx event. Default sets to 'auto', which determines the threshold by taking 25th percentile of COMPASS values after square root transfermation ((x/np.abs(x)) * np.sqrt(np.abs(x)))
        inplace: True for updating the commu_res in the object, False for return the updated communication table without changing the mebo_obj
        """
        comm_res = self.commu_res.sort_values(['Sender', 'Receiver', 'Metabolite', 'Sensor'])
        ## compass
        self._get_compass_flux_(compass_folder = compass_folder)
        x1 = 'sender_transport_flux'
        x2 = 'receiver_transport_flux'
        comm_res[x1] = [self.efflux_mat.loc[m,c] if m in self.efflux_mat.index.tolist() else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
        comm_res[x2] = [self.influx_mat.loc[m,c] if m in self.influx_mat.index.tolist() else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]
        flux_norm = lambda x: (x/np.abs(x)) * np.sqrt(np.abs(x)) if x != 0 else 0
        comm_res[x1] = [flux_norm(x) for x in comm_res[x1].tolist()]
        comm_res[x2] = [flux_norm(x) for x in comm_res[x2].tolist()]
        if efflux_cut == 'auto':
            all_efflux = [flux_norm(self.efflux_mat.loc[m,c]) if m in self.efflux_mat.index.tolist() else np.nan for c, m in self.original_result[['Sender', 'Metabolite_Name']].values.tolist()]
            efflux_cut = np.percentile(all_efflux, 25)
        if influx_cut == 'auto':
            all_influx = [flux_norm(self.influx_mat.loc[m,c]) if m in self.influx_mat.index.tolist() else np.nan for c, m in self.original_result[['Receiver', 'Metabolite_Name']].values.tolist()]
            influx_cut = np.percentile(all_influx, 25)
        print('efflux_cut:', efflux_cut)
        print('influx_cut:', influx_cut)
        ## base_efflux_influx_cut
        tmp_na = comm_res[pd.isna(comm_res[x1]) | pd.isna(comm_res[x2])]
        tmp1 = comm_res.query('Annotation != "Receptor"').copy()
        tmp2 = comm_res.query('Annotation == "Receptor"').copy()
        tmp1 = tmp1[(tmp1[x1]>efflux_cut) & (tmp1[x2]>influx_cut)]
        tmp2 = tmp2[(tmp2[x1]>efflux_cut)]
        update_commu_res = pd.concat([tmp1, tmp2, tmp_na])
        if inplace:
            self.commu_res = update_commu_res.copy()
        else:
            return(update_commu_res)

## ============================== examining bais to abundant metabolites in the blood ============================
    def _blood_correct_test_(self, 
                            met_cont_file, 
                            commu_score_col = 'Commu_Score', 
                            title = '',
                            show_plot = False,
                            pdf = False):
        blood_cont = pd.read_table(met_cont_file, index_col = 0).iloc[:,0]
        commu_res = self.commu_res.copy()
        commu_res['blood_level'] = [np.nan if x not in blood_cont.index.tolist() else blood_cont[x] for x in commu_res['Metabolite_Name'].tolist()]
        commu_res['blood_level'] = np.log(commu_res['blood_level'])
        commu_res['blood_level'] = (commu_res['blood_level'] - commu_res['blood_level'].min()) / (commu_res['blood_level'].max() - commu_res['blood_level'].min())
        commu_res = commu_res[~pd.isna(commu_res['blood_level'])]#[['blood_level', 'Commu_Score']]
        plotm = commu_res.drop_duplicates(['Sender', 'Metabolite_Name'])
        r1, p1 = pearsonr(commu_res['blood_level'], commu_res['Commu_Score'])
        sr1, sp1 = spearmanr(commu_res['blood_level'], commu_res['Commu_Score'])
        model = LinearRegression(fit_intercept = True)
        model.fit(commu_res[['blood_level']], commu_res['Commu_Score'])
        commu_res['pred'] = model.predict(commu_res[['blood_level']])
        commu_res['Corrected_Commu_Score'] = commu_res['Commu_Score'] - commu_res['pred']
        r2, p2 = pearsonr(commu_res['blood_level'], commu_res['Corrected_Commu_Score'])
        sr2, sp2 = spearmanr(commu_res['blood_level'], commu_res['Corrected_Commu_Score'])
        if show_plot:
            fig, ax = plt.subplots(figsize = (10, 4), nrows = 1, ncols = 2)
            sns.regplot(data = commu_res,
                        x = 'blood_level', y = 'Commu_Score', ci = False, ax = ax[0],
                       scatter_kws={'alpha':.5})
            # ax[0].set_title('PCC: %.2f, p-val: %.2e\nSp Rho: %.2f, p-val:%.2e'%(r1, p1, sr1, sp1))
            ax[0].set_title('PCC: %.2f, p-val: %.2e'%(r1, p1))
            ax[0].set_xlabel('Metabolite level in blood')
            ax[0].set_ylabel('mCCC score')
            sns.regplot(data = commu_res,
                        x = 'blood_level', y = 'Corrected_Commu_Score', ci = False, ax = ax[1],
                       scatter_kws={'alpha':.5})
            # ax[1].set_title('PCC: %.2f, p-val: %.2eSp Rho: %.2f, p-val:%.2e'%(r2, p2, sr2, sp2))
            ax[1].set_title('PCC: %.2f, p-val: %.2e'%(r2, p2))
            ax[1].set_xlabel('Metabolite level in blood')
            ax[1].set_ylabel('Corrected mCCC score')
            # ax.set_ylabel('Communication score')
            fig.suptitle(title)
            sns.despine()
            plt.tight_layout()
            pdf.savefig(fig) if pdf else plt.show()
            plt.close()
        return(commu_res)

## ============================== communication plot functions ============================
    def eventnum_bar(self,
                    sender_focus = [],
                    metabolite_focus = [],
                    sensor_focus = [],
                    receiver_focus = [],
                    and_or = 'and',
                    xorder = [],
                    pval_method = 'permutation_test_fdr',
                    pval_cutoff = 0.05,
                    comm_score_col = 'Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = None,
                    figsize = 'auto',
                    save = None,
                    show_plot = True,
                    show_num = True,
                    include = ['sender-receiver', 'sensor', 'metabolite', 'metabolite-sensor'],
                    group_by_cell = True,
                    colorcmap = 'tab20',
                    return_fig = False
                  ):
        """
        this function summarize the number of communication events
        
        Params
        ------
        sender_focus
            a list, set a list of sender cells to be focused, only plot related communications
        metabolite_focus
            a list, set a list of metabolites to be focused, only plot related communications
        sensor_focus
            a list, set a list of sensors to be focused, only plot related communications
        receiver_focus
            a list, set a list of receiver cells to be focused, only plot related communications
        and_or
            eithor 'and' or 'or', 'and' for finding communications that meet to all focus, 'or' for union
        xorder
            a list to order the x axis
        pval_method
            should be one of ['zztest_pval', 'ttest_pval', 'ranksum_test_pval', 'permutation_test_pval', 'zztest_fdr', 'ttest_fdr', 'ranksum_test_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        save
            str, the file name to save the figure
        show_plot
             True or False, whether print the figure on the screen
        show_num
            True or False, whether label y-axis value to the top of each bar
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        cutoff_prop
            a float between 0 and 1, set a cutoff to further filter out lowly abundant cell populations by the fraction of cells expressed sensor genes or metabolite, Note that this parameter will lost the function if cutoff_prop was set lower than the one user set at begaining of running mebocost.infer_commu or preparing mebocost object. This parameter were designed to further strengthen the filtering.
        include
            a list, contains one or more elements from ['sender-receiver', 'sensor', 'metabolite', 'metabolite-sensor'], we try to summarize the number of communications grouping by the given elements, if return_fig set to be True, only provide one for each run.
        group_by_cell
            True or False, only effective for metabolite and sensor summary, True to further label number of communications in cell groups, False to do not do that
        colormap
            only effective when group_by_cell is True, should be a python camp str, default will be 'tab20', or can be a dict where keys are cell group, values are RGB readable color
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself.
            
        """
        
#         if show_plot is None and self.show_plot is not None:
#             show_plot = self.show_plot
        
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None
            
        fig = CP._eventnum_bar_(commu_res = self.commu_res,
                    sender_focus = sender_focus,
                    metabolite_focus = metabolite_focus,
                    sensor_focus = sensor_focus,
                    receiver_focus = receiver_focus,
                    and_or = and_or,
                    xorder = xorder,
                    pval_method = pval_method,
                    pval_cutoff = pval_cutoff,
                    comm_score_col = comm_score_col,
                    comm_score_cutoff = comm_score_cutoff,
                    cutoff_prop = cutoff_prop,
                    figsize = figsize,
                    pdf = Pdf,
                    show_plot = show_plot,
                    show_num = show_num,
                    include = include,
                    group_by_cell = group_by_cell,
                    colorcmap = colorcmap,
                    return_fig = return_fig
                  )
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)
    
    def histogram(self, 
                    met_name=None,
                    sensor=None,
                    title = '', 
                    save = None, 
                    show_plot = True,
                    bins = 100,
                    alpha = .6,
                    bg_color = 'grey',
                    obs_color = 'red',
                    figsize = (5.5, 4.2),
                    comm_score_col = 'Commu_Score',
                   return_fig = False):
        """
        histogram plot to show the communication score distribution in background and given pairs of metabolite and sensor

        Params
        -----
        met
            str, metabolite name in the communication table
        sensor
            str, gene name of metabolite sensor in communication table
        title
            str, the title for the figure
        save
            str, the file name to save the figure
        show_plot
            True or False, whether print the figure on the screen
        bins
            int, how many bins plot in the histogram, 100 for default
        alpha
            float, set for color transparent, defaut is 0.6
        bg_color
            color for bars of backgroud communication scores
        obs_color
            color for bars of observed communication score
        comm_score_col
            column name of communication score, can be Commu_Score
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself

        """
#         if show_plot is None and self.show_plot is not None:
#             show_plot = self.show_plot

        commu_mat = self.original_result[(self.original_result['Metabolite_Name'] == met_name) &
                                    (self.original_result['Sensor'] == sensor)]
        if commu_mat.shape[0] == 0:
            info('ERROR: no data found, please check met_name and sensor') 
            return
        ## find met HMDB ID
        hmdbId = commu_mat['Metabolite'].tolist()[0]
        commu_bg = self.commu_bg[hmdbId+'~'+sensor]
        ## create pdf if true
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None

        fig = CP._histogram_(commu_mat=commu_mat, commu_bg=commu_bg, title = title, pdf = Pdf, 
                 show_plot = show_plot, bins = bins, alpha = alpha, bg_color = bg_color,
                 obs_color = obs_color, figsize = figsize, comm_score_col = comm_score_col,
                            return_fig = return_fig)

        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)

    def commu_dotmap(self,
                sender_focus = [],
                metabolite_focus = [],
                sensor_focus = [],
                receiver_focus = [],
                and_or = 'and',
                pval_method='permutation_test_fdr',
                pval_cutoff=0.05, 
                figsize = 'auto',
                cmap = 'Reds',
                cmap_vmin = None,
                cmap_vmax = None,
                cellpair_order = [],
                met_sensor_order = [],
                dot_size_norm = (10, 150),
                save = None, 
                show_plot = True,
                comm_score_col = 'Commu_Score',
                comm_score_range = None,
                comm_score_cutoff = None,
                cutoff_prop = None,
                swap_axis = False,
                return_fig = False):
        """
        commu_dotmap to show all significant communication events, cell pairs by metabolite sensor pairs
        
        Params
        -----
        sender_focus
            a list, set a list of sender cells to be focused, only plot related communications
        metabolite_focus
            a list, set a list of metabolites to be focused, only plot related communications
        sensor_focus
            a list, set a list of sensors to be focused, only plot related communications
        receiver_focus
            a list, set a list of receiver cells to be focused, only plot related communications
        and_or
            eithor 'and' or 'or', 'and' for finding communications that meet to all focus, 'or' for union
        pval_method
            should be one of ['zztest_pval', 'ttest_pval', 'ranksum_test_pval', 'permutation_test_pval', 'zztest_fdr', 'ttest_fdr', 'ranksum_test_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        cmap
            colormap for dot color for -log10(pvalues), default is Reds
        cmap_vmin
            the lower limit of the color scale. Values smaller than vmin are plotted with the same color as vmin, default None
        cmap_vmax
            the upper limit of the color scale. Values greater than vmin are plotted with the same color as vmax, default None
        cellpair_order
            a list to order the cell group pairs, default None
        met_sensor_order
            a list to order the metabolite sensor pairs, default None
        dot_size_norm
            two values in a tuple, used to normalize the dot size in matplotlib for comm_score_col, such as (10, 150)
        save
            str, the file name to save the figure
        show_plot
             True or False, whether print the figure on the screen
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_range   
            a typle, e.g. (smallest, largest), to set the range of comm_score_col represented by dot size, this is useful when you want to compare multiple plots and want to set the dot representing the same range. 
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        cutoff_prop
            a float between 0 and 1, set a cutoff to filter out lowly abundant cell populations by the fraction of cells expressed sensor genes or metabolite, Note that this parameter will lost the function if cutoff_prop was set lower than the one user set at begaining of running mebocost.infer_commu or preparing mebocost object. This parameter were designed to further strengthen the filtering.
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
        """
        
#         if show_plot is None and self.show_plot is not None:
#             show_plot = self.show_plot

        comm_res = self.commu_res
        ## pdf
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None
        
        fig = CP._commu_dotmap_(comm_res=comm_res, 
                     sender_focus = sender_focus,
                     metabolite_focus = metabolite_focus,
                     sensor_focus = sensor_focus,
                     receiver_focus = receiver_focus,
                     and_or = and_or,
                     pval_method=pval_method, 
                     pval_cutoff=pval_cutoff,
                     cmap_vmin = cmap_vmin,
                     cmap_vmax = cmap_vmax,
                     cellpair_order = cellpair_order,
                     met_sensor_order = met_sensor_order,
                     figsize = figsize, 
                     comm_score_col = comm_score_col,
                     comm_score_range = comm_score_range,
                     comm_score_cutoff = comm_score_cutoff,
                     cutoff_prop = cutoff_prop,
                     cmap = cmap,
                     dot_size_norm = dot_size_norm,
                     pdf = Pdf, 
                     show_plot = show_plot,
                     swap_axis = swap_axis,
                     return_fig = return_fig
                    )
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)
        
    def FlowPlot(self, 
                pval_method='permutation_test_fdr',
                pval_cutoff=0.05,
                sender_focus = [],
                metabolite_focus = [],
                sensor_focus = [],
                receiver_focus = [],
                remove_unrelevant = False,
                and_or = 'and',
                node_label_size = 8,
                node_alpha = .8,
                figsize = 'auto',
                node_cmap = 'Set1',
                line_cmap = 'spring_r',
                line_cmap_vmin = None,
                line_cmap_vmax = None,
                linewidth_norm = (0.1, 1),
                linewidth_value_range = None,
                node_size_norm = (10, 150),
                node_value_range = None,
                save=None, 
                show_plot = False,
                comm_score_col = 'Commu_Score',
                comm_score_cutoff = None,
                cutoff_prop = None,
                text_outline = False,
                return_fig = False):
        """
        Flow plot to show the communication connections from sender to metabolite, to sensor, to receiver

        Params
        ------
        pval_method
            should be one of ['zztest_pval', 'ttest_pval', 'ranksum_test_pval', 'permutation_test_pval', 'zztest_fdr', 'ttest_fdr', 'ranksum_test_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        sender_focus
            a list, set a list of sender cells to be focused, only plot related communications
        metabolite_focus
            a list, set a list of metabolites to be focused, only plot related communications
        sensor_focus
            a list, set a list of sensors to be focused, only plot related communications
        receiver_focus
            a list, set a list of receiver cells to be focused, only plot related communications
        remove_unrelevant
            True or False, set True to hide unrelated nodes 
        and_or
            eithor 'and' or 'or', 'and' for finding communications that meet to all focus, 'or' for union
        node_label_size
            float, font size of text label on node, default will be 8
        node_alpha
            float, set to transparent node color
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        node_cmap
            node color map or a four-element list, used to color sender, metabolite, sensor, receiver, set one from https://matplotlib.org/stable/tutorials/colors/colormaps.html
        line_cmap
            line color map, used to indicate the communication score, set one from https://matplotlib.org/stable/tutorials/colors/colormaps.html
        line_cmap_vmin
            the lower limit of the color scale. Values smaller than vmin are plotted with the same color as vmin, default None
        line_cmap_vmax
            the upper limit of the color scale. Values greater than vmax are plotted with the same color as vmax, default None
        node_size_norm
            two values in a tuple, used to normalize the dot size, such as (10, 150)
        node_value_range
            a typle, e.g. (smallest, largest), to set the range of dot represented values, this is useful when you want to compare multiple plots and want to set the dot representing the same range. 
        linewidth_norm
            two values in a tuple, used to normalize the line width, such as (0.1, 1), in the matplotlib
        linewidth_value_range
            a typle, e.g. (smallest, largest), to set the range of line width represented values, this is useful when you want to compare multiple plots and want to set the line width representing the same range. 
        save
            str, the file name to save the figure
        show_plot
            True or False, whether print the figure on the screen
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        cutoff_prop
            a float between 0 and 1, set a cutoff to filter out lowly abundant cell populations by the fraction of cells expressed sensor genes or metabolite, Note that this parameter will lost the function if cutoff_prop was set lower than the one user set at begaining of running mebocost.infer_commu or preparing mebocost object. This parameter were designed to further strengthen the filtering.
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
        """
        
#         if show_plot is None and self.show_plot is not None:
#             show_plot = self.show_plot

        comm_res = self.commu_res
        ## pdf
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None

        fig = CP._FlowPlot_(comm_res=comm_res, pval_method=pval_method, pval_cutoff=pval_cutoff, 
                      sender_focus = sender_focus, metabolite_focus = metabolite_focus,
                      sensor_focus = sensor_focus, receiver_focus = receiver_focus, 
                      remove_unrelevant = remove_unrelevant, and_or = and_or,
                      node_label_size = node_label_size, node_alpha = node_alpha, figsize = figsize, 
                      node_cmap = node_cmap, line_cmap = line_cmap, line_cmap_vmin = line_cmap_vmin,
                      line_cmap_vmax = line_cmap_vmax, linewidth_norm = linewidth_norm, 
                      linewidth_value_range = linewidth_value_range, node_value_range = node_value_range,
                      node_size_norm = node_size_norm, pdf=Pdf, show_plot = show_plot, 
                      comm_score_col = comm_score_col, comm_score_cutoff = comm_score_cutoff, cutoff_prop = cutoff_prop,
                      text_outline = text_outline, return_fig = return_fig)
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)

    def count_dot_plot(self, 
                    pval_method='permutation_test_pval', 
                    pval_cutoff=0.05, 
                    cmap='RdBu_r', 
                    figsize = 'auto',
                    save = None,
                    dot_size_norm = (5, 100),
                    dot_value_range = None,
                    dot_color_vmin = None,
                    dot_color_vmax = None,
                    show_plot = True,
                    comm_score_col = 'Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = None,
                    dendrogram_cluster = True,
                    sender_order = [],
                    receiver_order = [],
                    return_fig = False):
        """
        dot plot to show the summary of communication numbers between sender and receiver 

        Params
        -----
        pval_method
            should be one of ['zztest_pval', 'ttest_pval', 'ranksum_test_pval', 'permutation_test_pval', 'zztest_fdr', 'ttest_fdr', 'ranksum_test_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        cmap
            color map to set dot color 
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        save
            str, the file name to save the figure
        dot_size_norm
            two values in a tuple, used to normalize the dot size, such as (10, 150)
        dot_value_range
            a typle, e.g. (smallest, largest), to set the range of dot represented values, this is useful when you want to compare multiple plots and want to set the dot representing the same range. 
        dot_color_vmin
            float, the value limits the color map in maximum
        dot_color_vmax
            float, the value limits the color map in minimum
        show_plot
            True or False, whether print the figure on the screen
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        cutoff_prop
            a float between 0 and 1, set a cutoff to filter out lowly abundant cell populations by the fraction of cells expressed sensor genes or metabolite, Note that this parameter will lost the function if cutoff_prop was set lower than the one user set at begaining of running mebocost.infer_commu or preparing mebocost object. This parameter were designed to further strengthen the filtering.
        dendrogram_cluster
            True for clustering rows and columns by hierarchical clustering, False to disable, default True
        sender_order
            a list to set the sender cell group orders
        receiver_order
            a list to set the receiver cell group orders
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
        """
        
        
#         if show_plot is None and self.show_plot is not None:
#             show_plot = self.show_plot

        comm_res = self.commu_res
        ## pdf
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None

        fig = CP._count_dot_plot_(commu_res=comm_res, pval_method = pval_method, pval_cutoff = pval_cutoff, 
                        cmap = cmap, figsize = figsize, pdf = Pdf, dot_size_norm = dot_size_norm, dot_value_range = dot_value_range,
                        dot_color_vmin = dot_color_vmin, dot_color_vmax = dot_color_vmax, show_plot = show_plot,
                        comm_score_col = comm_score_col, comm_score_cutoff = comm_score_cutoff, cutoff_prop = cutoff_prop,
                        dendrogram_cluster = dendrogram_cluster,
                        sender_order = sender_order, receiver_order = receiver_order,
                        return_fig = return_fig)
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)

    def commu_network_plot(self,
                        sender_focus = [],
                        metabolite_focus = [],
                        sensor_focus = [],
                        receiver_focus = [],
                        remove_unrelevant = False,
                        and_or = 'and',
                        pval_method = 'permutation_test_fdr',
                        pval_cutoff = 0.05,
                        node_cmap = 'tab20',
                        figsize = 'auto',
                        line_cmap = 'RdBu_r',
                        line_color_vmin = None,
                        line_color_vmax = None,
                        linewidth_value_range = None,
                        linewidth_norm = (0.1, 1),
                        node_size_norm = (50, 300),
                        node_value_range = None,
                        adjust_text_pos_node = True,
                        node_text_hidden = False,
                        node_text_font = 10,
                        save = None,
                        show_plot = True,
                        comm_score_col = 'Commu_Score',
                        comm_score_cutoff = None,
                        cutoff_prop = None,
                        text_outline = False,
                        return_fig = False):

        """
        Network plot to show the communications between cell groups

        Params
        ------
        sender_focus
            a list, set a list of sender cells to be focused, only plot related communications
        metabolite_focus
            a list, set a list of metabolites to be focused, only plot related communications
        sensor_focus
            a list, set a list of sensors to be focused, only plot related communications
        receiver_focus
            a list, set a list of receiver cells to be focused, only plot related communications
        remove_unrelevant
            True or False, set True to hide unrelated nodes
        and_or
            eithor 'and' or 'or', 'and' for finding communications that meet to all focus, 'or' for union
        pval_method
            should be one of ['zztest_pval', 'ttest_pval', 'ranksum_test_pval', 'permutation_test_pval', 'zztest_fdr', 'ttest_fdr', 'ranksum_test_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        node_cmap
            node color map, used to indicate different cell groups, set one from https://matplotlib.org/stable/tutorials/colors/colormaps.html
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        line_cmap
            line color map, used to indicate number of communication events, set one from https://matplotlib.org/stable/tutorials/colors/colormaps.html
        line_color_vmin
            float, the value limits the line color map in minimum
        line_color_vmax
            float, the value limits the line color map in maximum
        linewidth_norm
            two values in a tuple, used to normalize the dot size, such as (0.1, 1)
        linewidth_value_range
            a typle, e.g. (smallest, largest), to set the range of line width represented values, this is useful when you want to compare multiple plots and want to set the line width representing the same range. 
        node_size_norm
            two values in a tuple, used to normalize the node size, such as (50, 300)
        node_value_range
            a typle, e.g. (smallest, largest), to set the range of node size represented values, this is useful when you want to compare multiple plots and want to set the node size representing the same range. 
        adjust_text_pos_node 
            True or Flase, whether adjust the text position to avoid overlapping automatically
        node_text_font
            float, font size for node text annotaion
        save
            str, the file name to save the figure
        show_plot
            True or False, whether print the figure on the screen
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        cutoff_prop
            a float between 0 and 1, set a cutoff to filter out lowly abundant cell populations by the fraction of cells expressed sensor genes or metabolite, Note that this parameter will lost the function if cutoff_prop was set lower than the one user set at begaining of running mebocost.infer_commu or preparing mebocost object. This parameter were designed to further strengthen the filtering.
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
        """
        
        comm_res = self.commu_res
        ## pdf
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None

        fig = CP._commu_network_plot_(commu_res=comm_res, sender_focus = sender_focus, metabolite_focus = metabolite_focus, 
                            sensor_focus = sensor_focus, receiver_focus = receiver_focus, and_or = and_or, 
                            pval_method = pval_method, remove_unrelevant = remove_unrelevant,
                            pval_cutoff = pval_cutoff, node_cmap = node_cmap, figsize = figsize, line_cmap = line_cmap, 
                            line_color_vmin = line_color_vmin, line_color_vmax = line_color_vmax,
                            linewidth_norm = linewidth_norm, linewidth_value_range = linewidth_value_range, node_text_hidden = node_text_hidden,
                            node_size_norm = node_size_norm, node_value_range = node_value_range, adjust_text_pos_node = adjust_text_pos_node, 
                            comm_score_col = comm_score_col, comm_score_cutoff = comm_score_cutoff, cutoff_prop = cutoff_prop,
                            node_text_font = node_text_font, pdf = Pdf, show_plot = show_plot, text_outline = text_outline,
                            return_fig = return_fig)
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        
        if return_fig:
            return(fig)
            
    def violin_plot(self,
                    sensor_or_met,
                    cell_focus = [],
                    cell_order = [],
                    row_zscore = False,
                    cmap = None,
                    vmin = None,
                    vmax = None,
                    figsize = 'auto',
                    cbar_title = '',
                    save = None,
                    show_plot = True,
                    return_fig = False):
        """
        Violin plot to show the distribution of sensor expression or metabolite aggregated enzyme expression across cell groups

        Params
        -----
        sensor_or_met
            a list, provide a list of sensor gene name or metabolite name
        cell_focus
            a list, provide a list of cell group that you want to focus, otherwise keep empty
        cell_order
            a list to set the order of cell group in the x axis
        row_zscore
            True for z score normalization applied for each row, default False
        cmap
            the color map used to draw the violin
        vmin
            float, maximum value for the color map
        vmin
            float, minimum value for the color map
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        title
            str, figure title on the top
        save
            str, the file name to save the figure
        show_plot
            True or False, whether print the figure on the screen
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
        """
        ## cell group
        cell_ann = self.cell_ann.copy()
        if 'cell_group' not in cell_ann.columns.tolist():
            raise ValueError('ERROR: "cell_group" not in cell_ann column names!')
        ### extract expression for sensor
        sensors = []
        if self.exp_mat is not None and self.exp_mat_indexer is not None:
            sensor_loc = np.where(pd.Series(self.exp_mat_indexer).isin(sensor_or_met))
            #[i for i,j in enumerate(self.exp_mat_indexer.tolist()) if j in sensor_or_met]
            sensors = self.exp_mat_indexer[sensor_loc]
            #[j for i,j in enumerate(self.exp_mat_indexer.tolist()) if j in sensor_or_met]
            exp_dat = pd.DataFrame(self.exp_mat[sensor_loc].toarray(),
                                   index = sensors,
                                   columns = self.exp_mat_columns)
            if len(sensors) > 0:
                info('Find genes %s to plot violin'%(sensors))
                ## expression
                if save is not None and save is not False and isinstance(save, str):
                    save = save.replace('.pdf', '_sensor_exp.pdf')
                    Pdf = PdfPages(save)
                else:
                    Pdf = None
                if cmap is None:
                    ccmap = 'Reds'
                else:
                    ccmap = cmap

                if cbar_title == '':
                    if row_zscore:
                        sensor_cbar_title = 'Mean Z score of sensor expression'
                    else:
                        sensor_cbar_title = 'Mean sensor expression'
                else:
                    sensor_cbar_title = cbar_title
                ## data mat for plot
                dat_mat = pd.merge(exp_dat.T, cell_ann[['cell_group']], left_index = True, right_index = True)
                fig = CP._violin_plot_(dat_mat=dat_mat, sensor_or_met=list(sensors),
                                       cell_focus = cell_focus, cell_order = cell_order, 
                                       cmap = ccmap, row_zscore = row_zscore,
                                       vmin = vmin, vmax = vmax, figsize = figsize, 
                                       cbar_title = sensor_cbar_title, pdf = Pdf,
                                       show_plot = show_plot, return_fig = return_fig)

                if save is not None and save is not False and isinstance(save, str):
                    Pdf.close()
                if return_fig:
                    return(fig)
            else:
                info('Warnings: no sensors to plot')
        else:
            info('Warnings: failed to load metabolite data matrix')
            
        ### extract metabolite level
        metabolites = list(set(sensor_or_met) - set(sensors))
        metabolites = list(set(metabolites) & set(self.met_ann['metabolite'].unique().tolist()))
        if metabolites:
            # to HMDBID
            met_name_to_id = {}
            for m, iD in self.met_ann[['metabolite', 'HMDB_ID']].values.tolist():
                met_name_to_id[m] = iD
            metaboliteIds = {x: met_name_to_id.get(x) for x in metabolites}
            ## metabolite matrix
            if self.met_mat is not None and self.met_mat_indexer is not None:
                met_loc = np.where(pd.Series(self.met_mat_indexer).isin(list(metaboliteIds.values())))[0]
                met_Ids = self.met_mat_indexer[met_loc]
                met_names = [list(metaboliteIds.keys())[list(metaboliteIds.values()).index(x)] for x in met_Ids]
                met_dat = pd.DataFrame(self.met_mat[met_loc].toarray(),
                                   index = met_names,
                                   columns = self.met_mat_columns)
                dat_mat = pd.merge(met_dat.T, cell_ann[['cell_group']], left_index = True, right_index = True)
                if len(met_names) > 0:
                    info("Find metabolites %s to plot violin"%metabolites)
                    ## expression
                    if save is not None and save is not False and isinstance(save, str):
                        save = save.replace('.pdf', '_metabolite.pdf')
                        Pdf = PdfPages(save)
                    else:
                        Pdf = None
                    if cmap is None:
                        ccmap = 'Purples'
                    else:
                        ccmap = cmap
                    if cbar_title == '':
                        if row_zscore:
                            met_cbar_title = 'Mean Z score of\n aggregated enzyme expression'
                        else:
                            met_cbar_title = 'Mean aggregated enzyme expression'
                    else:
                        met_cbar_title = cbar_title
                        
                    fig = CP._violin_plot_(dat_mat=dat_mat, sensor_or_met=list(metaboliteIds.keys()),
                                     cell_focus = cell_focus, cmap = ccmap,
                                     cell_order = cell_order, row_zscore = row_zscore, 
                                    vmin = vmin, vmax = vmax, figsize = figsize,
                                    cbar_title = met_cbar_title, pdf = Pdf,
                                    show_plot = show_plot, return_fig = return_fig)

                    if save is not None and save is not False and isinstance(save, str):
                        Pdf.close()
                    if return_fig:
                        return(fig)
                else:
                    info('Warnings: no metabolites to plot')
            else:
                info('Warnings: failed to load metabolite data matrix')
        else:
            info('Warnings: no metabolites to plot')
            
    def communication_in_notebook(self,
                                  pval_method = 'permutation_test_fdr',
                                  pval_cutoff = 0.05,
                                  comm_score_col = 'Commu_Score',
                                  comm_score_cutoff = None, 
                                  cutoff_prop = None
                                 ):

        # some handy functions to use along widgets
        from IPython.display import display, Markdown, clear_output, HTML
        import ipywidgets as widgets
        import functools

        outt = widgets.Output()

        df = self.commu_res.copy()
        
        if not comm_score_cutoff:
            comm_score_cutoff = 0
        if not cutoff_prop:
            cutoff_prop = 0
        ## basic filter
        df = df[(df[pval_method] <= pval_cutoff) & 
                (df[comm_score_col] >= comm_score_cutoff) &
                (df['metabolite_prop_in_sender'] >= cutoff_prop) &
                (df['sensor_prop_in_receiver'] >= cutoff_prop)
                ]
        
        senders = ['All']+sorted(list(df['Sender'].unique()))
        receivers = ['All']+sorted(list(df['Receiver'].unique()))
        metabolites = ['All']+sorted(list(df['Metabolite_Name'].unique()))
        transporters = ['All']+sorted(list(df['Sensor'].unique()))
        
        logic_butt = widgets.RadioButtons(
                            options=['and', 'or'],
                            description='Logic',
                            disabled=False
                        )

        sender_sel = widgets.SelectMultiple(description='Sender:',
                                            options=senders,
                                            layout=widgets.Layout(width='30%'))
        receiver_sel = widgets.SelectMultiple(description='Receiver:',
                                              options=receivers,
                                              layout=widgets.Layout(width='30%'))
        metabolite_sel = widgets.SelectMultiple(description='Metabolite:',
                                                options=metabolites,
                                                layout=widgets.Layout(width='30%'))
        sensor_sel = widgets.SelectMultiple(description='Sensor:',
                                                 options=transporters,
                                                layout=widgets.Layout(width='30%'))
        
        flux_butt = widgets.Button(description='Communication Flow (FlowPlot)',
                              layout=widgets.Layout(width='100%'))
        net_butt = widgets.Button(description='Communication Network (CirclePlot)',
                              layout=widgets.Layout(width='100%'))
        dotHeatmap_butt = widgets.Button(description='Communication Details (Dot-shaped Heatmap)',
                              layout=widgets.Layout(width='100%'))
        violin_butt = widgets.Button(description='ViolinPlot to show metabolite or sensor level in cell groups',
                              layout=widgets.Layout(width='100%'))

        def _flowplot_filter_(b):
            with outt:
                clear_output()
                print('+++++++++++++++++++++++++++ Running, Please Wait +++++++++++++++++++++++++++ ')
                print('[Selection]: Sender{}; Metabolite{}; Transporter{}; Receiver{}'.format(sender_sel.value,
                                                                                                  metabolite_sel.value,
                                                                                                  sensor_sel.value,
                                                                                                  receiver_sel.value))
                and_or = logic_butt.value
        
                self.FlowPlot(pval_method=pval_method,
                            pval_cutoff=pval_cutoff,
                            sender_focus = [x for x in sender_sel.value if x != 'All'],
                            metabolite_focus = [x for x in metabolite_sel.value if x != 'All'],
                            sensor_focus = [x for x in sensor_sel.value if x != 'All'],
                            receiver_focus = [x for x in receiver_sel.value if x != 'All'],
                            remove_unrelevant = True,
                            and_or = and_or,
                            node_label_size = 8,
                            node_alpha = .8,
                            figsize = 'auto',
                            node_cmap = 'Set1',
                            line_cmap = 'bwr',
                            line_cmap_vmin = None,
                            line_cmap_vmax = None,
                            node_size_norm = (10, 150),
                            linewidth_norm = (0.5, 5),
                            save=None, 
                            show_plot = True,
                            comm_score_col = comm_score_col,
                            comm_score_cutoff = comm_score_cutoff,
                            cutoff_prop = cutoff_prop,
                            text_outline = False,
                            return_fig = False)
                
                
        def _networkplot_filter_(b):
            with outt:
                clear_output()
                print('+++++++++++++++++++++++++++ Running, Please Wait +++++++++++++++++++++++++++ ')
                print('[Selection]: Sender{}; Metabolite{}; Transporter{}; Receiver{}'.format(sender_sel.value,
                                                                                                  metabolite_sel.value,
                                                                                                  sensor_sel.value,
                                                                                                  receiver_sel.value))
                and_or = logic_butt.value
                self.commu_network_plot(
                                sender_focus = [x for x in sender_sel.value if x != 'All'],
                                metabolite_focus = [x for x in metabolite_sel.value if x != 'All'],
                                sensor_focus = [x for x in sensor_sel.value if x != 'All'],
                                receiver_focus = [x for x in receiver_sel.value if x != 'All'],
                                remove_unrelevant = False,
                                and_or = and_or,
                                pval_method = pval_method,
                                pval_cutoff = pval_cutoff,
                                node_cmap = 'tab20',
                                figsize = 'auto',
                                line_cmap = 'bwr',
                                line_color_vmin = None,
                                line_color_vmax = None,
                                linewidth_norm = (0.1, 1),
                                node_size_norm = (50, 300),
                                adjust_text_pos_node = False,
                                node_text_font = 10,
                                save = None,
                                show_plot = True,
                                comm_score_col = comm_score_col,
                                comm_score_cutoff = comm_score_cutoff,
                                cutoff_prop = cutoff_prop,
                                text_outline = False
                                )
        def _dotHeatmapPlot_(b):
            with outt:
                clear_output()
                print('+++++++++++++++++++++++++++ Running, Please Wait +++++++++++++++++++++++++++ ')
                print('[Selection]: Sender{}; Metabolite{}; Transporter{}; Receiver{}'.format(sender_sel.value,
                                                                                                  metabolite_sel.value,
                                                                                                  sensor_sel.value,
                                                                                                  receiver_sel.value))
                and_or = logic_butt.value
                self.commu_dotmap(
                            sender_focus = [x for x in sender_sel.value if x != 'All'],
                            metabolite_focus = [x for x in metabolite_sel.value if x != 'All'],
                            sensor_focus = [x for x in sensor_sel.value if x != 'All'],
                            receiver_focus = [x for x in receiver_sel.value if x != 'All'],
                            and_or = and_or,
                            pval_method=pval_method,
                            pval_cutoff=pval_cutoff, 
                            figsize = 'auto',
                            cmap = 'bwr',
                            dot_size_norm = (10, 150),
                            save = None, 
                            show_plot = True,
                            comm_score_col = comm_score_col,
                            comm_score_cutoff = comm_score_cutoff,
                            cutoff_prop = cutoff_prop
                )

        def _violinPlot_(b):
            with outt:
                clear_output()
                print('+++++++++++++++++++++++++++ Running, Please Wait +++++++++++++++++++++++++++ ')
                print('[Selection]: Sender{}; Metabolite{}; Transporter{}; Receiver{}'.format(sender_sel.value,
                                                                                                  metabolite_sel.value,
                                                                                                  sensor_sel.value,
                                                                                                  receiver_sel.value))
                
                self.violin_plot(
                                sensor_or_met = [x for x in metabolite_sel.value + sensor_sel.value if x != 'All'],
                                cell_focus = [x for x in sender_sel.value + receiver_sel.value if x != 'All'],
                                cmap = None,
                                vmin = None,
                                vmax = None,
                                figsize = 'auto',
                                cbar_title = '',
                                save = None,
                                show_plot = True)
                
                
        flux_butt.on_click(_flowplot_filter_)
        net_butt.on_click(_networkplot_filter_)
        dotHeatmap_butt.on_click(_dotHeatmapPlot_)
        violin_butt.on_click(_violinPlot_)


        h1 = widgets.HBox([sender_sel, metabolite_sel, sensor_sel, receiver_sel])
        h2 = widgets.VBox([flux_butt, net_butt, dotHeatmap_butt, violin_butt])

        mk = Markdown("""<b>Select and Click button to visulize</b>""")
        display(mk, widgets.VBox([logic_butt, h1, h2, outt]))

            
            
            
            
            
            
            
            
