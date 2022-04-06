#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 08:43:57 2022

@author: mrd
"""

#defining function to print out variables and their current memory usage
#see original: https://stackoverflow.com/questions/24455615/python-how-to-display-size-of-all-variables
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Kb','Mb','Gb','Tb','Pb','Eb','Zb']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)
 
def sliverFinder(inShp,thresh):
    """

    Parameters
    ----------
    inShp : Variabe name
        Name of polygon shapefile variable..
    thresh : Value, integer or float.
        Value should be in units of the dataframe CRS (most likely square meters).

    Returns
    -------
    No values returned. Prints a statement of the number of slivers

    """
    import numpy as np
    
    shpNum = len(inShp)
    slivNum = len(inShp[inShp.geometry.area<thresh])
    slivPct = np.round((slivNum/shpNum)*100,2)
    print('{0}/{1} polygons exist ({2}%) that are less than {3} m^2.'.format(slivNum,shpNum,slivPct,thresh))
    

def delSlivers(inShp,thresh):
    
    """
    This function deletes all slivers above a specified area threshold (m2).
    
    Parameters
    ----------
    inShp : Variabe name
        Name of polygon shapefile variable..
    thresh : Value, integer or float.
        Threshold value, below which polygons will be deleted. Should be in units of the dataframe CRS (most likely square meters).

    Returns
    -------
    Returns the Gdf less the slivers.

    """
    import numpy as np
    
    slivNum = len(inShp[inShp.geometry.area<thresh])
    if slivNum == 0:
        print ('No slivers found.')
        return(inShp)
    elif slivNum >0:
        shpNum = len(inShp)
        slivPct = np.round((slivNum/shpNum)*100,2)
        inShp = inShp[inShp.geometry.area>thresh]
        print('Removed {0}/{1} polygons ({2}%) that were <{3} m^2.'.format(slivNum,shpNum,slivPct,thresh))
        return(inShp)

def eqAreaCalc(inShp,units):
    """
    This converts the Gdf into a equal area projection and calculates the area of each polygon.
    Parameters
    ----------
    inShp : Gdf name
        Name of shapefile to calculate area columns for.
    units : ['Ha','Mtr2']
        Units for which area should be calculated. Options are only the those listed.

    Returns
    -------
    Returns a gdf with area calculated in the specified units.

    """
    import geopandas as gpd
    
    gdfPrj = inShp.to_crs({'proj':'cea'}) #'cea'is a cylindrical equal area projection
    gdfPrj['AreaMtr2'] = gdfPrj['geometry'].area #calculating new column of area
    for unit in units:
        if unit == 'Mtr2':
            inShp['AreaMtr2'] = gdfPrj['AreaMtr2']
        elif unit == 'Ha':
            inShp['AreaHa'] = (gdfPrj['AreaMtr2']/1000000)*100
        elif unit == 'Km2':
            inShp['AreaKm2'] = gdfPrj['AreaMtr2']/1000000
    del(gdfPrj)
    return(inShp)
 
#a function to check for the nearest polygon adjacent to another using their centroids.
#this can be used on points as well, just drop the 'centroid' call between geometry and apply.
def ckdnearest(gdA, gdB):
    import np
    import cKDTree
    
    nA = np.array(list(gdA.geometry.centroid.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.centroid.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ],
        axis=1)
 
    return gdf
 
#Functions to dissolve over multiple (first) and a single (second) field(s)
#edited from: https://gis.stackexchange.com/questions/149959/dissolving-polygons-based-on-attributes-with-python-shapely-fiona
def dissolveMultField(inLoc, outLoc, fields):
    """
    
    Parameters
    ----------
    inLoc : C:/....../.shp'  
        str: Path from the shape file
    outLoc : 'C:/....../.shp' 
        location/path to output dissolved shapefile.
    fields : list of strings
        A list of strings that are fields on which the gdf should be dissolved.

    Returns
    -------
    None.

    """
    with fiona.open(inLoc) as inDat:
        with fiona.open(outLoc, 'w', **inDat.meta) as output:
            grouper = itemgetter(*fields)
            key = lambda k: grouper(k['properties'])
            for k, group in itertools.groupby(sorted(input, key=key), key):
                properties, geom = zip(*[(feature['properties'], shape(feature['geometry'])) for feature in group])
                output.write({'geometry': mapping(unary_union(geom)), 'properties': properties[0]})

#Modified from https://gis.stackexchange.com/a/150001/2856
def dissolve(inLoc, outLoc, field):              
    with fiona.open(inLoc) as inDat:
        # preserve the schema of the original shapefile, including the crs
        meta = inDat.meta
        with fiona.open(outLoc, 'w', **meta) as output:
            # groupby clusters consecutive elements of an iterable which have the same key so you must first sort the features by the 'STATEFP' field
            e = sorted(inDat, key=lambda k: k['properties'][field])
            # group by the 'STATEFP' field
            for key, group in itertools.groupby(e, key=lambda x:x['properties'][field]):
                properties, geom = zip(*[(feature['properties'],shape(feature['geometry'])) for feature in group])
                # write the feature, computing the unary_union of the elements in the group with the properties of the first element in the group
                output.write({'geometry': mapping(unary_union(geom)), 'properties': properties[0]})

# defining a function that rounds the coordinates of every geometry in the array
#geomRound = np.vectorize(lambda geom: pg.apply(geom, lambda g: g.round(3)))

#widget view of pandas dataframe
def viewDf(df):
    """
    This function returns a qgrid view of a geodataframe without the geometry column.
    
    Parameters
    ----------
    df : Name of gdf/df
        Variable name

    Returns
    -------
    Table view of dataframe.

    """
    import qgrid
    from osgeo import gdal, gdalconst
    
    if True in df.columns.str.contains('geometry'):
        widget = qgrid.show_grid(df[df.columns.difference(['geometry'])])
    else:
        widget = qgrid.show_grid(df)
        
    return(widget)

def resampleRast(inRastLoc,refRastLoc,outLoc):
    inputfile = inRastLoc #Path to input file
    inRast = gdal.Open(inputfile, gdalconst.GA_ReadOnly)
    inputProj = inRast.GetProjection()
    #inputTrans = inRast.GetGeoTransform()

    referencefile = refRastLoc #Path to reference file
    reference = gdal.Open(referencefile, gdalconst.GA_ReadOnly)
    referenceProj = reference.GetProjection()
    referenceTrans = reference.GetGeoTransform()
    bandreference = reference.GetRasterBand(1)    
    x = reference.RasterXSize 
    y = reference.RasterYSize

    outputfile = outLoc #Path to output file
    driver= gdal.GetDriverByName('GTiff')
    output = driver.Create(outputfile,x,y,1,bandreference.DataType)
    output.SetGeoTransform(referenceTrans)
    output.SetProjection(referenceProj)

    gdal.ReprojectImage(inRast,output,inputProj,referenceProj,gdalconst.GRA_Bilinear)

    del output

def vec2rast(shpLoc, rastRefLoc, outLoc, fieldName):
    
    """
    This function creates a raster of a shp file based on an existing raster reference layer

    Keyword arguments:
    shpLoc -- 'C:/....../.shp'  
        str: Path from the shape file
    rastRef -- 'C:/....../.tif'     
        str: Path to an example tiff file (all arrays will be reprojected to this example)
    outLoc -- 
        str: path to the output folder
    fieldName -- 
        str: path to the name of the field to be rasterized
         
    """
    
    from osgeo import gdal, ogr, osr
    import numpy as np
    
    #Opening a tiff info, for example size of array, projection and transform matrix
    refRast = gdal.Open(rastRefLoc)
    
    if refRast is None:
        print('%s does not exists' %rastRefLoc)
    else:
        geo = refRast.GetGeoTransform()
        wkt_proj = refRast.GetProjection()
        size_X = refRast.RasterXSize
        size_Y = refRast.RasterYSize
        
        #extent of raster
        x_min = geo[0]
        x_max = geo[0] + size_X * geo[1]
        y_min = geo[3] + size_Y * geo[5]
        y_max = geo[3]
        
        #resolution/pixel size of raster
        pixel_size = geo[1]
        
        refRast = None
    
    # Create the destination data source
    x_res = int(round((x_max - x_min) / pixel_size))
    y_res = int(round((y_max - y_min) / pixel_size))
    
    #making the shapefile as an object.
    input_shp = ogr.Open(shpLoc)
    
    #getting layer information of shapefile.
    shp_layer = input_shp.GetLayer()

    #get GeoTiff driver by 
    image_type = 'GTiff'
    driver = gdal.GetDriverByName(image_type)
    
    #passing the filename, x and y direction resolution, no. of bands, new raster.
    new_raster = driver.Create(outLoc, x_res, y_res, 1, gdal.GDT_Float32, ['COMPRESS=LZW'])
    
    #transforms between pixel raster space to projection coordinate space.
    new_raster.SetGeoTransform(geo)
    
    #get required raster band.
    band = new_raster.GetRasterBand(1)
    
    #assign no data value to empty cells.
    no_data_value = np.NaN
    band.SetNoDataValue(no_data_value)
    band.FlushCache()
    
    # Getting info on which field to rasteriz.
    options = ['ALL_TOUCHED=TRUE']
    if fieldName and len(fieldName) > 0:
        options.append('ATTRIBUTE=' + fieldName)
    
    #main conversion method
    gdal.RasterizeLayer(new_raster, [1], shp_layer, None, None, options=options)
    
    #adding a spatial reference
    out_SRS = osr.SpatialReference()
    #out_SRS.ImportFromEPSG(2975)
    new_raster.SetProjection(wkt_proj)
    
    new_raster.FlushCache()  # Write to disk.
       
def shp2newRast(shpLoc, outLoc, pixelSize, epsg):
    """
    This function creates a raster of a shp file WITHOUT an existing raster reference layer
    
    Parameters
    ----------
    shpLoc : 'C:/....../.shp' 
    location/path to shapefile.
    outLoc : 'C:/....../.tif' 
    location/path to output raster.
    pixelSize : eg. 10
        number reflecting pixel size (m).
    epsg : eg. 2193
        Four number EPSG code of the out projection.

    Returns
    -------
    TYPE: GTiff
        Outputs geotiff raster with coordinate system, no data set to -9999.
    
    """
    from osgeo import gdal, ogr, osr
    #import numpy as np

    #making the shapefile as an object.
    input_shp = ogr.Open(shpLoc)
    
    #getting layer information of shapefile.
    shp_layer = input_shp.GetLayer()
    
    #pixel_size determines the size of the new raster.
    #pixel_size is proportional to size of shapefile.
    pixel_size = pixelSize
    
    #get extent values to set size of output raster.
    x_min, x_max, y_min, y_max = shp_layer.GetExtent()
    
    #calculate size/resolution of the raster.
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)
    
    #get GeoTiff driver by 
    image_type = 'GTiff'
    driver = gdal.GetDriverByName(image_type)
    
    #passing the filename, x and y direction resolution, no. of bands, new raster.
    new_raster = driver.Create(outLoc, x_res, y_res, 1, gdal.GDT_Byte)
    
    #transforms between pixel raster space to projection coordinate space.
    new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))
    
    #get required raster band.
    band = new_raster.GetRasterBand(1)
    
    #assign no data value to empty cells.
    no_data_value = -9999
    band.SetNoDataValue(no_data_value)
    band.FlushCache()
    
    #main conversion method
    gdal.RasterizeLayer(new_raster, [1], shp_layer, burn_values=[255])
    
    #adding a spatial reference
    new_rasterSRS = osr.SpatialReference()
    new_rasterSRS.ImportFromEPSG(2975)
    new_raster.SetProjection(new_rasterSRS.ExportToWkt())
    return gdal.Open(output_raster)

def shpDissolve(inLoc, outLoc, field):  
    """
    Parameters
    ----------
    inLoc : 'C:/....../.shp' 
    location/path to input shapefile to be dissolved
    outLoc : 'C:/....../.shp' 
    location/path to output dissolved shapefile.
    field : eg. 'Region'
        name of field to be dissolved by, in quotations.
    
    Returns
    -------
    TYPE: .shp
        Outputs shapefile that has been dissolved/aggregated on the field chosen.
    
    """
    
    import fiona
    import itertools
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
                
    with fiona.open(inLoc) as inDat:
        # preserve the schema of the original shapefile, including the crs
        meta = inDat.meta
        with fiona.open(outLoc, 'w', **meta) as output:
            # groupby clusters consecutive elements of an iterable which have the same key so you must first sort the features by the 'STATEFP' field
            e = sorted(inDat, key=lambda k: k['properties'][field])
            # group by the 'STATEFP' field
            for key, group in itertools.groupby(e, key=lambda x:x['properties'][field]):
                properties, geom = zip(*[(feature['properties'],shape(feature['geometry'])) for feature in group])
                # write the feature, computing the unary_union of the elements in the group with the properties of the first element in the group
                output.write({'geometry': mapping(unary_union(geom)), 'properties': properties[0]})


def clipRast2shp(rastLoc,shpLoc,outLoc,dst_crs):
    """
    Parameters
    ----------
    rastLoc : C:/....../.tif'     
        str: Path to an tiff file that will be clipped
    shpLoc : 'C:/....../.shp'  
        str: Path to the shape file that will define the clipping bounds
    outLoc : C:/....../.tif' 
        str: path to the output folder
    dst_crs : Te.g., 2193
        integer: 4 digit integer that represent the EPSG that the raster should be projected to.

    Returns
    -------
    None.

    """
    
    import geopandas as gpd
    import rasterio as rio
    from rasterio.mask import mask
    from shapely.geometry import mapping
    #import skimage.transform as st
    
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

        
def arr2rast(inArr,refRastLoc,outLoc,noDataValue,outDataType,**kwargs):
    from osgeo import gdal, gdalconst,osr
    
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


def zipReadDf(datLoc,zipSearch):
    """
    
    Parameters
    ----------
    datLoc : 'C:/..../Folder'
        Folder location where zip files can be found
    zipSearch : "...*.zip"
        String indicating what glob should search for within zip folder.

    Returns: pandas df from zip file
    -------
    None.

    """
    from zipfile import ZipFile
    from glob import glob
    import pandas as pd
    import os
    from tqdm import tqdm
    
    datLoc = datLoc
    zipSearch = "*.zip" 
    zipFiles = glob(os.path.join(datLoc,'*.zip')) #creates list of files within path/dir
     
    df = pd.DataFrame([])
    searchStr = ''
    
    for zFile in zipFiles: # write a loop to access
            
        with ZipFile(zFile, 'r') as zipObject:
           listOfFileNames = zipObject.namelist()
           for fileName in tqdm(listOfFileNames):
               #if fileName.__contains__('30min'):
               if searchStr in fileName:
                   # Extract a single file from zip
                   basename = fileName.split('/')[-1]
                   #siteName = basename.split('.')[2]
                  
                   tempDf = pd.read_csv(zipObject.open(fileName))
                   
                   return(tempDf)
               
                   tempDf['Column Name'] = basename
                  
                   df=pd.concat([df,tempDf])


#IN PROGRESS
def crsMatch(shp1,shp2,dst_epsg,outLoc):
    if shp1.crs == shp2.crs:
        print('Coordinate references match.')
    elif shp1.crs != shp2.crs:
        print('Reprojecting')
        shp1 = shp1.to_crs(epsg=dst_epsg)
        shp1.to_file(os.path.join(vecFolder,'Grazing','livestockGrid'+Island+'NZ.shp'))


def checkVals(inShp,inCol,how,val):
    
    import numpy as np
    
    if how == '==':
       abvThresh = len(inShp[inShp[str(inCol)]==val])
       shpNum = len(inShp)
       abvThreshPct = np.round((abvThresh/shpNum)*100,2)
       print('{0}/{1} polygons ({2}%) had {3} that were {4} your threshold of: {5}'.format(abvThresh,shpNum,abvThreshPct,inCol,how,val))
       #display(inShp[inShp[str(inCol)]==val])
      
    if how == '>':
       abvThresh = len(inShp[inShp[str(inCol)]>val])
       shpNum = len(inShp)
       abvThreshPct = np.round((abvThresh/shpNum)*100,2)
       print('{0}/{1} polygons ({2}%) had {3} that were {4} your threshold of: {5}'.format(abvThresh,shpNum,abvThreshPct,inCol,how,val))
       #display(inShp[inShp[str(inCol)]>val])
    
    if how == '<':
       blwThresh = len(inShp[inShp[str(inCol)]<val])
       shpNum = len(inShp)
       blwThreshPct = np.round((blwThresh/shpNum)*100,2)
       print('{0}/{1} polygons ({2}%) had {3} that were {4} your threshold of: {5}'.format(blwThresh,shpNum,blwThreshPct,inCol,how,val))
       #display(inShp[inShp[str(inCol)]<val])   
