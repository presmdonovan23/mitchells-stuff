#!/usr/bin/env python
# coding: utf-8

# ### Calculating surficial erosion after adjusting for impacts from grazing.
# 
# - 1. Import subfactors: $K_{tr}, LS, Rf_{season}, C_{gr}$
# - 2. Calculate seasonal erosion $ErSe_{Gr}$
# - 2. Calculate annual grazing-adjusted erosion: $ErYr_{Gr}$
# - 3. Calculated % change in erosion: $/delta Er$
# 

# In[1]:


# Import system modules
import os
import glob
from osgeo import gdal, osr
from osgeo.gdal import gdalconst, GA_ReadOnly
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import numpy as np
import rasterio as rio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
from shapely.geometry import mapping
#import skimage.transform as st
import matplotlib.pyplot as plt
import sys
import time
from tqdm import tqdm

import sys
#defining function to print out variables and their current memory usage
#see original: https://stackoverflow.com/questions/24455615/python-how-to-display-size-of-all-variables
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

# setting home folder and sub-directories where input and output files lie
homeFolder = r'/Users/mrd/emarde/osga/Data'
extHDloc = r'/Volumes/Transcend/Data'
vecFolder = os.path.join(homeFolder,'Vector')
rastFolder = os.path.join(homeFolder,'Raster')
aspatialFolder = os.path.join(homeFolder,'Aspatial')


# In[2]:


def arr2rast(inArr,refRastLoc,outLoc,noDataValue,outDataType,**kwargs):
    
    #writing output data to new masked raster file using gdal
    reference = gdal.Open(refRastLoc, gdalconst.GA_ReadOnly)

    referenceProj = reference.GetProjection()
    referenceTrans = reference.GetGeoTransform()
    bandreference = reference.GetRasterBand(1)
    refDtype = reference.GetRasterBand(1).DataType
    refNoDatVal = reference.GetRasterBand(1).GetNoDataValue()
    x = reference.RasterXSize 
    y = reference.RasterYSize

    reference=None
        
    outputLoc = outLoc #Path to output file
    outDriver = gdal.GetDriverByName('GTiff')
    outRaster = outDriver.Create(outputLoc,x,y,1,outDataType,options=['COMPRESS=LZW', 'TILED=YES'])
    outRaster.SetGeoTransform(referenceTrans)
    outRaster.SetProjection(referenceProj)
    outRaster.GetRasterBand(1).SetNoDataValue(noDataValue)  
    
    outRaster.GetRasterBand(1).WriteArray(inArr) #calling the band that we want to write the array into

    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(2193)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outRaster.FlushCache()

#function for clipping raster to polygon layer.
def clipRast2Poly(rastLoc,shpLoc,outLoc):
    # clipping the raster by polygon
    input_shape = shpLoc
    shapefile = gpd.read_file(input_shape)
    shapefilePrj = shapefile.to_crs(epsg=dst_crs[-4:])#4326)
    input_raster = rastLoc # input raster
    geoms = shapefilePrj.geometry.values # list of shapely geometries
    output_raster = outLoc # output raster
    geometry = geoms[0] # shapely geometry
    geoms = [mapping(geoms[0])]
    
    with rio.open(input_raster) as src:
        out_image, out_transform = mask(src, geoms, crop=True) 
        out_meta = src.meta.copy()
    
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    
    with rio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image)


# In[3]:


#Conversion factor for adjusting from t/ha/yr --> t/yr (the same as: t/pixel/yr); where pixel = 225 m^2
#This allows summing of the output per farm and/or catchment to estimate total potential losses.
ha2sqm = 0.0001 #1 square meter = 0.0001 ha
pxArea = 15**2 #225 m2 pixels
conv = ha2sqm*pxArea #(conversion factor of 0.0225)


# In[4]:


Seasons = ['Spring','Summer','Autumn','Winter'] #list of seasons
Islands=['North','South']


#%% 1. Seasonal ungrazed erosion calculations

for Island in Islands:
    #if Island == 'South':
    for Season in Seasons:
        print('Currently processing: ', Season,' ', Island, ' Island')
        KfLoc = os.path.join(rastFolder,'Soil',Island+'NZ','KfSt'+Island+'NZ.tif')
        CfLoc = os.path.join(rastFolder,'Landuse',Island+'NZ','Cf'+Season[0:2]+Island+'NZ.tif')
        LSfLoc = os.path.join(extHDloc,'Raster/Terrain',Island+'NZ','LSfThresh'+Island+'NZ.tif')
        RfLoc = os.path.join(extHDloc,'Raster/Climate',Island+'NZ','Rf'+Season[0:2]+Island+'NZ.tif')

        refRastLoc = LSfLoc

        in_files =[LSfLoc, KfLoc, CfLoc, RfLoc]
        in_files

        in_rasters = []
        for rast in in_files:
            dat = rio.open(rast)
            in_rasters.append(dat)
        n = int(len(in_rasters)) #total number of rasters to be included in calculations

        #print('rasters:', in_rasters.name) #printing list of rasters to be included
        #print([in_rasters[i].name.split('/')[-1] for i in range(n)])

        #output folder path location for the erosion calculation for the base scenario
        outLoc = os.path.join(rastFolder,'Erosion',Island+'NZ','Er'+Season[0:2]+Island+'NZ.tif')

        #generating the blocks from default block/window sizes within the first raster. assumes same grid size/dimensions
        #Must rerun this to refresh value of 'blocks' as it doesn't update after iterating within the loop below
        blocks = in_rasters[0].block_windows()
        blocks

        # Get the number of rows and columns and block sizes
        x_size = in_rasters[0].height
        y_size = in_rasters[0].width
        x_block_size = in_rasters[0].block_shapes[0][0]
        y_block_size = in_rasters[0].block_shapes[0][1]
        x=0; y=0; index = 0

        rastDtype=in_rasters[0].dtypes[0]
        noData = in_rasters[0].nodata

        outArr = np.empty((x_size, y_size), dtype=np.byte) #empty input array to be converted to masked array)
        outMskArr = ma.asarray(outArr,dtype=np.float32) #creating masked array
        outMskArr.set_fill_value(noData)
        outArr=None #deleting unmasked array

        #print('Raster dims:',x_size, y_size)
        #print('Block size:', x_block_size, y_block_size)
        #print('Raster data type: ', rastDtype)
        #print('NoData Value:',noData)
        #print('Rasters (# of):',n)

        #loading windowed masked arrays using rasterio
        for (_, window) in tqdm(blocks,desc='Processing raster blocks...', position=0, ncols=80, units = 'blocks',total=len(blocks)):
            # Bring in a block's worth of data from all in_rasters
            # Stacking masked arrays of data from each raster
            inStack = ma.stack(
                [in_rasters[i].read(1, window=window, masked=True) \
                    for i in range(n)]).astype(np.float32)
            
            #appropriately masking the output calculations
            outMask = ma.any([[inStack[i].mask==True] for i in range(n)],axis=0) #creating a mask where *any* layer is masked
            for i in range(n):
                inStack[i].mask = outMask.data
            
            #calculating erosion from all factors per window
            outWindow = (np.ma.prod(inStack,axis=0)/4)*conv #the product of the input datasets (KfTr,CfGr,LSf,Rf)
            #outWindow.data[outMask.data[0]==True]=noData 
            #outWindow = np.ma.masked_where(outMask.data==True, outWindow)
            
            #Because the product spits out 1.0 as the no data/fill value, have to update
            outWindow.data[outWindow.data==1.0]=noData
            #outWindow = np.ma.masked_where((outWindow.data == 1.0), outWindow)
            #outWindow.data[outMask.data[0]==True]=noData    
            #outWindow.set_fill_value(noData)
            
            # Find dimensions of this block
            x_start = window.col_off
            y_start = window.row_off
            x_wsize = window.width + x_start 
            y_wsize = window.height + y_start

            #storing window into final grid
            outMskArr[y_start : y_wsize, x_start : x_wsize] = outWindow 

        #Outputs are tonnes/px/season
        arr2rast(inArr=outMskArr,refRastLoc=refRastLoc,noDataValue=noData,outLoc=outLoc,outDataType=gdal.GDT_Float32)

        print('Raster output {0}'.format(outLoc))
        del(inStack);del(outMskArr);del(outWindow)


#%% 2. Seasonal grazed erosion calculations

Islands = ['North','South']

for Island in Islands:
    #if Island == 'South':
    for Season in Seasons:
        print(Season,' has started for the ', Island, ' Island')
        KfTrLoc = os.path.join(rastFolder,'Soil',Island+'NZ','KfStTr'+Season[0:2]+Island+'NZ.tif')
        CfGrLoc = os.path.join(rastFolder,'Landuse',Island+'NZ','Cf'+Season[0:2]+'Gr'+Island+'NZ.tif')
        LSfLoc = os.path.join(extHDloc,'Raster/Terrain',Island+'NZ','LSfThresh'+Island+'NZ.tif')
        RfLoc = os.path.join(extHDloc,'Raster/Climate',Island+'NZ','Rf'+Season[0:2]+Island+'NZ.tif')

        refRastLoc = LSfLoc

        in_files =[KfTrLoc, CfGrLoc, LSfLoc, RfLoc]
        in_files

        in_rasters = []

        for rast in in_files:
            dat = rio.open(rast)
            in_rasters.append(dat)
        n = int(len(in_rasters)) #total number of rasters to be included in calculations

        #print('rasters:', in_rasters.name) #printing list of rasters to be included
        #print([in_rasters[i].name.split('/')[-1] for i in range(n)])

        #output folder path location for the erosion calculation for the base scenario
        outLoc = os.path.join(rastFolder,'Erosion',Island+'NZ','ErGr'+Season[0:2]+Island+'NZ.tif')

        #generating the blocks from default block/window sizes within the first raster. assumes same grid size/dimensions
        #Must rerun this to refresh value of 'blocks' as it doesn't update after iterating within the loop below
        blocks = in_rasters[0].block_windows()
        blocks

        # Get the number of rows and columns and block sizes
        x_size = in_rasters[0].height
        y_size = in_rasters[0].width
        x_block_size = in_rasters[0].block_shapes[0][0]
        y_block_size = in_rasters[0].block_shapes[0][1]
        x=0; y=0; index = 0

        rastDtype=in_rasters[0].dtypes[0]
        noData = in_rasters[0].nodata

        #in_arr = np.empty((x_block_size, y_block_size, n), dtype=np.uint16)
        outArr = np.empty((x_size, y_size), dtype=np.byte) #empty input array to be converted to masked array)
        outMskArr = ma.asarray(outArr,dtype=np.float32) #creating masked array
        outArr=None #deleting unmasked array

        #print('Raster dims:',x_size, y_size)
        #print('Block size:', x_block_size, y_block_size)
        #print('Raster data type: ', rastDtype)
        #print('NoData Value:',noData)
        #print('Rasters (# of):',n)

        #loading windowed masked arrays using rasterio
        for (_, window) in tqdm(blocks,desc='Processing raster blocks...', position=0, ncols=80):
            # Bring in a block's worth of data from all in_rasters
            # Stacking masked arrays of data from each raster
            inStack = ma.stack(
                [in_rasters[i].read(1, window=window, masked=True) \
                    for i in range(n)]).astype(np.float32)

            outMask = ma.any([[inStack[i].mask==True] for i in range(n)],axis=0) #creating a mask where *any* layer is masked
            for i in range(n):
                inStack[i].mask = outMask.data

            #calculating erosion from all factors per window
            outWindow = (np.ma.prod(inStack,axis=0)/4)*conv #the product of the input datasets (KfTr,CfGr,LSf,Rf)
            #Because the product spits out 1.0 as the no data/fill value, have to update
            outWindow.data[outWindow.data==1.0]=noData
                        
            # Find dimensions of this block
            x_start = window.col_off
            y_start = window.row_off
            x_wsize = window.width + x_start 
            y_wsize = window.height + y_start
            #storing window into final grid
            outMskArr[y_start : y_wsize, x_start : x_wsize] = outWindow 

    #Outputs are tonnes/px/season
        arr2rast(inArr=outMskArr,refRastLoc=refRastLoc,noDataValue=noData,outLoc=outLoc,outDataType=gdal.GDT_Float32)

        print('Raster written to {0}'.format(outLoc))
        del(inStack);del(outMskArr);del(outWindow)


#%% Annual grazed erosion calculations

Islands = ['North','South']

for Island in Islands:
    #if Island == 'South':

    ErSpLoc = os.path.join(rastFolder, 'Erosion',Island+'NZ','ErGrSp'+Island+'NZ.tif')
    ErSuLoc = os.path.join(rastFolder, 'Erosion',Island+'NZ','ErGrSu'+Island+'NZ.tif')
    ErAuLoc = os.path.join(rastFolder, 'Erosion',Island+'NZ','ErGrAu'+Island+'NZ.tif')
    ErWiLoc = os.path.join(rastFolder, 'Erosion',Island+'NZ','ErGrWi'+Island+'NZ.tif')
    LSfLoc = os.path.join(extHDloc,'Raster/Terrain',Island+'NZ','LSf'+Island+'NZ.tif')
    
    refRastLoc = LSfLoc

    in_files =[ErSpLoc, ErSuLoc, ErAuLoc, ErWiLoc]
    in_files

    in_rasters = []

    for rast in in_files:
        dat = rio.open(rast)
        in_rasters.append(dat)
    n = int(len(in_rasters)) #total number of rasters to be included in calculations

    #output folder path location for the erosion calculation for the base scenario
    outLoc = os.path.join(rastFolder,'Erosion',Island+'NZ','ErGrYr'+Island+'NZ.tif')

    #generating the blocks from default block/window sizes within the first raster. assumes same grid size/dimensions
    #Must rerun this to refresh value of 'blocks' as it doesn't update after iterating within the loop below
    blocks = in_rasters[0].block_windows()
    blocks

    # Get the number of rows and columns and block sizes
    x_size = in_rasters[0].height
    y_size = in_rasters[0].width
    x_block_size = in_rasters[0].block_shapes[0][0]
    y_block_size = in_rasters[0].block_shapes[0][1]
    x=0; y=0; index = 0

    rastDtype=in_rasters[0].dtypes[0]
    noData = in_rasters[0].nodata

    #in_arr = np.empty((x_block_size, y_block_size, n), dtype=np.uint16)
    outArr = np.empty((x_size, y_size), dtype=np.byte) #empty input array to be converted to masked array)
    outMskArr = ma.asarray(outArr,dtype=np.float32) #creating masked array
    outArr=None #deleting unmasked array

    #print('Raster dims:',x_size, y_size)
    #print('Block size:', x_block_size, y_block_size)
    #print('Raster data type: ', rastDtype)
    #print('NoData Value:',noData)
    #print('Rasters (# of):',n)

    #loading windowed masked arrays using rasterio
    for (_, window) in tqdm(blocks,desc='Processing raster blocks...'):
        # Bring in a block's worth of data from all in_rasters
        # Stacking masked arrays of data from each raster
        inStack = ma.stack(
            [in_rasters[i].read(1, window=window, masked=True) \
                for i in range(n)]).astype(np.float32)
        
        outMask = ma.any([[inStack[i].mask==True] for i in range(n)],axis=0) #creating a mask where *any* layer is masked
        for i in range(n):
            inStack[i].mask = outMask.data

        #calculating erosion from all factors per window
        outWindow = inStack.sum(axis=0) #the sum of the seasonal erosion calculations
        #Because the product spits out 1.0 as the no data/fill value, have to update
        outWindow.data[outWindow.data==1.0]=noData
        
        # Find dimensions of this block
        x_start = window.col_off
        y_start = window.row_off
        x_wsize = window.width + x_start 
        y_wsize = window.height + y_start

        #storing window into final grid
        outMskArr[y_start : y_wsize, x_start : x_wsize] = outWindow 
    
    #Outputs are tonnes/px/year
    arr2rast(inArr=outMskArr,refRastLoc=refRastLoc,noDataValue=noData,outLoc=outLoc,outDataType=gdal.GDT_Float32)

    print('Raster written to {0}'.format(outLoc))
    del(inStack);del(outMskArr);del(outWindow)

#%% Annual UNgrazed erosion calculations

Islands = ['North','South']

for Island in Islands:
    #if Island == 'South':

    ErSpLoc = os.path.join(rastFolder, 'Erosion',Island+'NZ','ErSp'+Island+'NZ.tif')
    ErSuLoc = os.path.join(rastFolder, 'Erosion',Island+'NZ','ErSu'+Island+'NZ.tif')
    ErAuLoc = os.path.join(rastFolder, 'Erosion',Island+'NZ','ErAu'+Island+'NZ.tif')
    ErWiLoc = os.path.join(rastFolder, 'Erosion',Island+'NZ','ErWi'+Island+'NZ.tif')
    LSfLoc = os.path.join(extHDloc,'Raster/Terrain',Island+'NZ','LSf'+Island+'NZ.tif')
    #output folder path location for the erosion calculation for the base scenario
    outLoc = os.path.join(rastFolder,'Erosion',Island+'NZ','ErYr'+Island+'NZ.tif')
    
    refRastLoc = LSfLoc

    in_files =[ErSpLoc, ErSuLoc, ErAuLoc, ErWiLoc]
    in_files

    in_rasters = []

    for rast in in_files:
        dat = rio.open(rast)
        in_rasters.append(dat)
    n = int(len(in_rasters)) #total number of rasters to be included in calculations

    #generating the blocks from default block/window sizes within the first raster. assumes same grid size/dimensions
    #Must rerun this to refresh value of 'blocks' as it doesn't update after iterating within the loop below
    blocks = in_rasters[0].block_windows()
    blocks

    # Get the number of rows and columns and block sizes
    x_size = in_rasters[0].height
    y_size = in_rasters[0].width
    x_block_size = in_rasters[0].block_shapes[0][0]
    y_block_size = in_rasters[0].block_shapes[0][1]
    x=0; y=0; index = 0

    rastDtype=in_rasters[0].dtypes[0]
    noData = in_rasters[0].nodata

    #in_arr = np.empty((x_block_size, y_block_size, n), dtype=np.uint16)
    outArr = np.empty((x_size, y_size), dtype=np.byte) #empty input array to be converted to masked array)
    outMskArr = ma.asarray(outArr,dtype=np.float32) #creating masked array
    outArr=None #deleting unmasked array

    #print('Raster dims:',x_size, y_size)
    #print('Block size:', x_block_size, y_block_size)
    #print('Raster data type: ', rastDtype)
    #print('NoData Value:',noData)
    #print('Rasters (# of):',n)

    #loading windowed masked arrays using rasterio
    for (_, window) in tqdm(blocks,desc='Processing raster blocks...'):
        # Bring in a block's worth of data from all in_rasters
        # Stacking masked arrays of data from each raster
        inStack = ma.stack(
            [in_rasters[i].read(1, window=window, masked=True) \
                for i in range(n)]).astype(np.float32)
        
        outMask = ma.any([[inStack[i].mask==True] for i in range(n)],axis=0) #creating a mask where *any* layer is masked
        for i in range(n):
            inStack[i].mask = outMask.data

        #calculating erosion from all factors per window
        outWindow = inStack.sum(axis=0) #the sum of the seasonal erosion calculations
        #Because the product spits out 1.0 as the no data/fill value, have to update
        outWindow.data[outWindow.data==1.0]=noData
        
        # Find dimensions of this block
        x_start = window.col_off
        y_start = window.row_off
        x_wsize = window.width + x_start 
        y_wsize = window.height + y_start

        #storing window into final grid
        outMskArr[y_start : y_wsize, x_start : x_wsize] = outWindow 

#Outputs are tonnes/px/year
    arr2rast(inArr=outMskArr,refRastLoc=refRastLoc,noDataValue=noData,outLoc=outLoc,outDataType=gdal.GDT_Float32)

    print('Raster written to {0}'.format(outLoc))
    del(inStack);del(outMskArr);del(outWindow)


# In[ ]:




