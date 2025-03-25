import pip
# Packages
import igraph
import scanpy as sc
import pandas as pd
import numpy as np

# Loading the data in

print("starting reading")
data = sc.read_10x_mtx("E:\\pycharm-python\\pre_processing_scRNA_data\\filtered_feature_bc_matrix",  # the directory with the `.mtx` file
    var_names='gene_symbols',       # use gene symbols for the variable names (variables-axis index)
    cache=True)                     # write a cache file for faster subsequent reading
print("finish reading")

print("data")
print(data)

# Preprocessing
print("filtering genes and cells")
sc.pp.filter_cells(data, min_genes=100) #removes cells with genes >100
sc.pp.filter_genes(data, min_cells=3) #removes cells with genes detected >3
print("finish filtering")

print("labelling mitochondrial genes") #Always double check you actually labeled MT

data.var['mt'] = data.var_names.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'

#print (data.var[data.var.mt == True]) # checking mt gene annotation

print(" quality control metrics")
sc.pp.calculate_qc_metrics(data, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
print(data.obs) # acts like a pandas dataframe.


#sc.pl.violin(data, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True) # check distribution
#sc.pl.scatter(data, x='total_counts', y='pct_counts_mt', color= "red") #Scatter plots of gene distribution
#sc.pl.scatter(data, x='total_counts', y='n_genes_by_counts', color= "blue")
#sc.pl.scatter(data, "total_counts", "n_genes_by_counts", color="pct_counts_mt") # plotted together

#instead of picking subjectively, you can use quanitle function in numpy.
upper_lim = np.quantile(data.obs.n_genes_by_counts.values, .98)
lower_lim = np.quantile(data.obs.n_genes_by_counts.values, .02)
print('lower and upper limits of gene counts')
print(f'{lower_lim} to {upper_lim}')

#print(data.X[1,:].sum())

# Normalisation of counts
print("normalisation")
sc.pp.normalize_total(data, target_sum=1e4) #normalize every cell to 10,000 UMI

#print(data.X[1,:].sum())
print("Log counts")
sc.pp.log1p(data) #change to log counts
print("saving data")
data.raw = data # Saving the data as is

# Clustering 
print("top genes")
sc.pp.highly_variable_genes(data, n_top_genes = 2000)
print(data.var)

#sc.pl.highly_variable_genes(data) #plotting of high variance genes

data = data[:, data.var.highly_variable] # filtering highly variable genes

data = data.copy() #making a copy to avoid warnings or issues
print("regression")
sc.pp.regress_out(data, ['total_counts', 'pct_counts_mt']) #cleans up data variation

print("normalise variance")
sc.pp.scale(data, max_value=10) # normalise each gene to unit variance of each gene

sc.tl.pca(data, svd_solver='arpack')# reducing the data to principle components

#sc.pl.pca_variance_ratio(data, log=True, n_pcs = 50) #plotting pca variances

print("nearest neighbours calculation")
sc.pp.neighbors(data, n_pcs = 20) #calculating nearest neighbours of cells using top 20 where the elbow flattens
print("creating umap")
sc.tl.umap(data) # creating umap & plotting
#sc.pl.umap(data)

sc.tl.leiden(data, resolution = 0.5, flavor= 'leidenalg', n_iterations = -1) # Clustering cells using the Leiden algorithm

sc.pl.umap(data, color=['leiden']) # Clusters from algorithm


