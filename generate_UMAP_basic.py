#!/usr/bin/env python
# coding: utf-8

# # Step 2, alternative a: Generate UMAP representations from spectrograms  -  Basic pipeline

# ## Introduction

# This script creates UMAP representations from spectrograms using the basic pipeline.
# 
# #### The following  structure and files are required in the project directory:
# 
#     ├── data
#     │   ├── df.pkl            <- pickled pandas dataframe with metadata and spectrograms (generated in
#     |                            01_generate_spectrograms.ipynb)
#     ├── parameters         
#     ├── functions             <- the folder with the function files provided in the repo                
#     ├── notebooks             <- the folder with the notebook files provided in the repo    
#     ├── ...  
#      
# 
# #### The following columns must exist (somewhere) in the pickled dataframe df.pkl:
# 
#     | spectrograms    |    ....
#     ------------------------------------------
#     |  2D np.array    |    ....
#     |  ...            |    ....
#     |  ...            |    .... 
#     
# 
# #### The following files are generated in this script:
# 
#     ├── data
#     │   ├── df_umap.pkl         <- pickled pandas dataframe with metadata, spectrograms AND UMAP coordinates

# ## Import statements, constants and functions

# In[1]:


import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import umap
import sys 
sys.path.insert(0, '..')

from functions.preprocessing_functions import calc_zscore, pad_spectro
from functions.custom_dist_functions_umap import unpack_specs


# In[3]:


P_DIR = str(Path(os.getcwd()).parents[0])       # project directory
DATA = os.path.join(os.path.sep, P_DIR, 'data') # path to data subfolder in project directory
DF_NAME = 'df.pkl'                              # name of pickled dataframe with metadata and spectrograms


# Specify UMAP parameters. If desired, other inputs can be used for UMAP, such as denoised spectrograms, bandpass filtered spectrograms or other (MFCC, specs on frequency scale...) by changining the INPUT_COL parameter.

# In[4]:


INPUT_COL = 'spectrograms'  # column that is used for UMAP
                            #  could also choose 'denoised_spectrograms' or 'stretched_spectrograms' etc etc...
    
METRIC_TYPE = 'euclidean'     # distance metric used in UMAP. Check UMAP documentation for other options
                              # e.g. 'euclidean', correlation', 'cosine','manhattan' ...
    
N_COMP = 2                    # number of dimensions desired in latent space  


# ## 1. Load data

# In[7]:


df = pd.read_pickle(os.path.join(os.path.sep, DATA, DF_NAME))
df.shape


# ## 2. UMAP
# ### 2.1. Prepare UMAP input

# In this step, the spectrograms are z-transformed, zero-padded and concatenated to obtain numeric vectors.

# In[8]:


# Basic pipeline
# No time-shift allowed, spectrograms should be aligned at the start. All spectrograms are zero-padded 
# to equal length
    
specs = df[INPUT_COL] # choose spectrogram column
specs = [calc_zscore(s) for s in specs] # z-transform each spectrogram

maxlen= np.max([spec.shape[1] for spec in specs]) # find maximal length in dataset
flattened_specs = [pad_spectro(spec, maxlen).flatten() for spec in specs] # pad all specs to maxlen, then row-wise concatenate (flatten)
data = np.asarray(flattened_specs) # data is the final input data for UMAP


# ### 2.2. Specify UMAP parameters

# In[9]:


reducer = umap.UMAP(n_components=N_COMP, metric = METRIC_TYPE,  # specify parameters of UMAP reducer
                    min_dist = 0, random_state=2204) 


# ### 2.2. Fit UMAP

# In[10]:


embedding = reducer.fit_transform(data)  # embedding contains the new coordinates of datapoints in 3D space


# ## 3. Save dataframe

# In[11]:


# Add UMAP coordinates to dataframe
for i in range(N_COMP):
    df['UMAP'+str(i+1)] = embedding[:,i]

# Save dataframe
df.to_pickle(os.path.join(os.path.sep, DATA, 'df_umap.pkl'))


# In[12]:


df


# In[ ]:





# In[ ]:




