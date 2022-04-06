#!/usr/bin/env python
# coding: utf-8

# In[12]:


import geopandas as gpd
import os
import pandas as pd
import numpy as np
import glob
from osgeo import osr
import time
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import sys
import qgrid
from scipy.spatial import cKDTree

homeFolder = r'/Users/mrd/emarde/osga'
extHDloc = r'/Volumes/Transcend/Data'
vecFolder = os.path.join(homeFolder,'Data/Vector')
rastFolder = os.path.join(homeFolder,'Data/Raster')
aspatialFolder = os.path.join(homeFolder,'Data/Aspatial')
moduleFolder =  os.path.join('/Users/mrd/emarde/osga/GitHub/geospytial')
    
# adding modules folder to the system path
sys.path.append(moduleFolder)

#Import personal module
import emdeMods as md 


# In[16]:


#coordinate reference system, in epsg.
dst_crs = 'epsg:2193'

#converting epsg:#### to a wkt format (well-known text projection)
srs = osr.SpatialReference()
srs.ImportFromEPSG(int(dst_crs.split(':')[1]))
srs_wkt = srs.ExportToPrettyWkt()


# In[17]:


years = [1999,2001,2007,2012,2017]
Islands = ['North','South']
farmTypes = ['BEF','DAI','DRY','SHP','SNB','DEE','NAT','GRA']
subCols = ['farm_id','farm_type','REG','region','RegKey','bef_nos','dai_nos','shp_nos','dee_nos','geometry']

cols2017 = ['LandCover','UniqueID','farm_id','farm_type','REG','Region','AreaMtr2','AreaHa','geometry']
#all other years don't have landcover and uID because they weren't joined with the WFC layer.
colsOther = ['farm_id','farm_type','REG','Region','AreaMtr2','AreaHa','geometry'] 


# In[18]:


farmKey = pd.read_csv(os.path.join(aspatialFolder,'Grazing','AgriBase_FarmCodeKey.csv'))


# ## Stocking grazed pastoral lands based on hexagon stock populations

# In[ ]:


for Island in Islands:
    for year in [2017, 1999]:
        pastoralLoc = os.path.join(vecFolder,'Grazing','pastoralGdb'+str(year)+Island+'NZ.shp')
        farmStockClean = gpd.read_file(pastoralLoc)

        #setting default for final farm populations to 0, these will be overrode later, where relevant.
        farmStockClean['dai'+str(year)] = 0; farmStockClean['daiDen'+str(year)] = 0
        farmStockClean['bef'+str(year)] = 0; farmStockClean['befDen'+str(year)] = 0
        farmStockClean['shp'+str(year)] = 0; farmStockClean['shpDen'+str(year)] = 0
        farmStockClean['dee'+str(year)] = 0; farmStockClean['deeDen'+str(year)] = 0    
        
        begTime = time.time()
        for grID in farmStockClean.grid_id.unique(): 
        #statsNZ total population of each stock type for the current grid cell

            daiGridPop = int(farmStockClean[farmStockClean.grid_id == grID]['dairy'+str(year)].unique()[0])
            shpGridPop = int(farmStockClean[farmStockClean.grid_id == grID]['sheep'+str(year)].unique()[0])
            befGridPop = int(farmStockClean[farmStockClean.grid_id == grID]['beef'+str(year)].unique()[0])
            deeGridPop = int(farmStockClean[farmStockClean.grid_id == grID]['deer'+str(year)].unique()[0])
            #printing data check
            #print('dsb grid pop totals: {0},{1},{2}'.format(daiGridPop,shpGridPop,befGridPop))

        #total area of each farm type within the grid:
            daiGridArea = np.sum(farmStockClean[(farmStockClean.grid_id == grID)&(farmStockClean.farm_type=='DAI')].AreaHa)
            befGridArea = np.sum(farmStockClean[(farmStockClean.grid_id == grID)&(farmStockClean.farm_type=='BEF')].AreaHa)
            shpGridArea = np.sum(farmStockClean[(farmStockClean.grid_id == grID)&(farmStockClean.farm_type=='SHP')].AreaHa)
            snbGridArea = np.sum(farmStockClean[(farmStockClean.grid_id == grID)&(farmStockClean.farm_type=='SNB')].AreaHa)
            dryGridArea = np.sum(farmStockClean[(farmStockClean.grid_id == grID)&(farmStockClean.farm_type=='DRY')].AreaHa)
            natGridArea = np.sum(farmStockClean[(farmStockClean.grid_id == grID)&(farmStockClean.farm_type=='NAT')].AreaHa)
            deeGridArea = np.sum(farmStockClean[(farmStockClean.grid_id == grID)&(farmStockClean.farm_type=='DEE')].AreaHa)
            #printing data check
            #print('dsb farm areas: {0},{1},{2}'.format((daiGridArea+dryGridArea),(shpGridArea+snbGridArea+natGridArea),(befGridArea+snbGridArea+natGridArea)))

        #actual density of stock types for the grid, based on total farm area within the current grid
            if (daiGridPop == 0) | ((daiGridArea+dryGridArea) == 0): dai_PopDens = 0
            elif (daiGridPop > 0) & ((daiGridArea+dryGridArea) > 0): dai_PopDens = daiGridPop / (daiGridArea+dryGridArea)

            if (shpGridPop == 0) | ((shpGridArea+snbGridArea+natGridArea) == 0): shp_PopDens = 0
            elif (shpGridPop > 0) & ((shpGridArea+snbGridArea+natGridArea) > 0): shp_PopDens = shpGridPop / (shpGridArea+snbGridArea+natGridArea)

            if (befGridPop == 0) | ((befGridArea+snbGridArea+natGridArea) == 0): bef_PopDens = 0
            elif (befGridPop > 0) & ((befGridArea+snbGridArea+natGridArea) > 0): bef_PopDens = befGridPop / (befGridArea+snbGridArea+natGridArea)

            if (deeGridPop == 0) | ((deeGridArea) == 0): dee_PopDens = 0
            elif (deeGridPop > 0) & ((deeGridArea) > 0): dee_PopDens = deeGridPop / (deeGridArea)

            #printing data check
            #print('dsb grid popDens totals: {0},{1},{2}'.format(dai_PopDens,shp_PopDens,bef_PopDens))

            #Calculating stock populations at farm scales (multiply average grid density x farm area (km2))
            farmStockClean.loc[(farmStockClean.grid_id==grID)&(farmStockClean.farm_type.isin(['DAI','DRY'])), 'dai'+str(year)] = dai_PopDens * farmStockClean.loc[(farmStockClean.grid_id==grID)&(farmStockClean.farm_type.isin(['DAI','DRY'])),'AreaHa']
            farmStockClean.loc[(farmStockClean.grid_id==grID)&(farmStockClean.farm_type.isin(['SHP','SNB','NAT'])),'shp'+str(year)] = shp_PopDens * farmStockClean.loc[(farmStockClean.grid_id==grID)&(farmStockClean.farm_type.isin(['SHP','SNB','NAT'])),'AreaHa']
            farmStockClean.loc[(farmStockClean.grid_id==grID)&(farmStockClean.farm_type.isin(['BEF','SNB','NAT'])),'bef'+str(year)] = bef_PopDens * farmStockClean.loc[(farmStockClean.grid_id==grID)&(farmStockClean.farm_type.isin(['BEF','SNB','NAT'])),'AreaHa']  
            farmStockClean.loc[(farmStockClean.grid_id==grID)&(farmStockClean.farm_type.isin(['DEE'])),'dee'+str(year)] = dee_PopDens * farmStockClean.loc[(farmStockClean.grid_id==grID)&(farmStockClean.farm_type.isin(['DEE'])),'AreaHa']  

            farmStockClean.loc[(farmStockClean.grid_id==grID)&(farmStockClean.farm_type.isin(['DAI','DRY'])), 'daiDen'+str(year)] = dai_PopDens
            farmStockClean.loc[(farmStockClean.grid_id==grID)&(farmStockClean.farm_type.isin(['SHP','SNB','NAT'])),'shpDen'+str(year)] = shp_PopDens
            farmStockClean.loc[(farmStockClean.grid_id==grID)&(farmStockClean.farm_type.isin(['BEF','SNB','NAT'])),'befDen'+str(year)] = bef_PopDens
            farmStockClean.loc[(farmStockClean.grid_id==grID)&(farmStockClean.farm_type.isin(['DEE'])),'deeDen'+str(year)] = dee_PopDens
        
        #After all calcs have finished:
        #Convert final populations to integers, rounding up to the nearest stock
        farmStockClean['dai'+str(year)]=np.round(farmStockClean['dai'+str(year)],0).astype(int)
        farmStockClean['shp'+str(year)]=np.round(farmStockClean['shp'+str(year)],0).astype(int)
        farmStockClean['bef'+str(year)]=np.round(farmStockClean['bef'+str(year)],0).astype(int)
        farmStockClean['dee'+str(year)]=np.round(farmStockClean['dee'+str(year)],0).astype(int)
        
        if 'Region' in farmStockClean.columns:
            farmStockClean.rename(columns={'Region':'region'},inplace=True)

        if year == 2017:
            farmStockClean=farmStockClean[['farm_id','farm_type','region','REG','grid_id',
                               'dai'+str(year),'shp'+str(year),'bef'+str(year),'dee'+str(year),
                               'daiDen'+str(year),'shpDen'+str(year),'befDen'+str(year),'deeDen'+str(year),
                               'AreaMtr2','AreaHa','LandCover','UniqueID','geometry']]

        elif year!=2017:
            farmStockClean=farmStockClean[['farm_id','farm_type','region','REG','grid_id',
                                           'dai'+str(year),'shp'+str(year),'bef'+str(year),'dee'+str(year),
                                           'daiDen'+str(year),'shpDen'+str(year),'befDen'+str(year),'deeDen'+str(year),
                                           'AreaMtr2','AreaHa','geometry']]

        #Removing all rows that don't have any livestock (ie., ALL stock values are 0.)
        #These polygons are too small for livestock counts, most likely.
        farmStockClean = farmStockClean.loc[~(farmStockClean[['dai'+str(year),'shp'+str(year),'bef'+str(year),'dee'+str(year)]]==0).all(axis=1)]


#4. Saving output
        pastoralOutLoc = os.path.join(vecFolder,'Grazing','pastoralGdb'+str(year)+Island+'NZ.shp')
        farmStockClean.to_file(pastoralOutLoc)

        endTime = time.time()
        print(r'Full runtime: {0} seconds'.format(np.round(endTime-begTime,1)))
                                 
#5. Clearing and printing current variables and their memory storage usage
        for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                                 key= lambda x: -x[1])[:10]:
            #print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
            thresh = 500000 #equivalent of ~50Mb
            if (size>thresh and name.__contains__('_')):
                print('Deleting large excess file > {0}!'.format(sizeof_fmt(thresh)))
                locals()[name]=None


# # Checking for correct output and columns

# In[ ]:


viewDf(farmStockClean)

