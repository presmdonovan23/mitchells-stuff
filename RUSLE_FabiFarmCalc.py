#!/usr/bin/env python
# coding: utf-8

# In[13]:

import pandas as pd
import numpy as np
import numpy.ma as ma
import os
import matplotlib.pyplot as plt
import glob
import sys

from osgeo import osr, gdal

import rasterio as rio
import geopandas as gpd
from rasterstats import zonal_stats

from tqdm import tqdm

# Prettier plotting with seaborn
import seaborn as sns; 
sns.set(font_scale=1.5)
sns.set_style("white")
sns.set_theme(style="ticks")

# Set standard plot parameters for uniform plotting
plt.rcParams['figure.figsize'] = (12, 12)

moduleFolder = os.path.join('/Users/mrd/emarde/osga/GitHub/geospytial')
    
# adding modules folder to the system path
sys.path.append(moduleFolder)

#Import personal module
import emdeMods as md


# In[2]:


#coordinate reference system, in epsg.
dst_crs = 'epsg:2193'

#converting epsg:#### to a wkt format (well-known text projection)
srs = osr.SpatialReference()
srs.ImportFromEPSG(int(dst_crs.split(':')[1]))
srs_wkt = srs.ExportToPrettyWkt()

print('Output projection information:' + '\n' + srs_wkt)
    
#%% Setting inputs/outputs

# Which NZ Island?
Island = 'South'
# Farm name?
farmName = 'MtGrand'

# setting home folder and sub-directories where input and output files lie
homeFolder = r'/Users/mrd/emarde/osga'
vecFolder = os.path.join(homeFolder,'Data/Vector')
rastFolder = os.path.join(homeFolder,'Data/Raster')
aspatialFolder = os.path.join(homeFolder,'Data/Aspatial')
farmFolder = os.path.join(homeFolder,'Data/Farms')
os.chdir(homeFolder)

#folder where the vector data is located
farmBoundLoc = os.path.join(farmFolder,farmName,'aInputs',farmName+'FarmBound.shp')
farmPdkLoc = os.path.join(farmFolder,farmName,'aInputs',farmName+'Pdks.shp')
farmDatLoc = os.path.join(farmFolder,farmName,'aInputs',farmName+'StockDat.csv')

KfShpLoc = os.path.join(vecFolder,'Soil','KfTr'+Island+'NZ.shp')
CfShpLoc = os.path.join(vecFolder,'Landuse',Island+'NZ','Cf'+Island+'NZ.shp')
LSfLoc = os.path.join(rastFolder,'Terrain',Island+'NZ','LSfThresh'+Island+'NZ.tif')
RfLocs = glob.glob(os.path.join(rastFolder,'Climate',Island+'NZ','Rf*'+'[!Reg]'+Island+'NZ.tif'))

#output folder path location
outFolder = os.path.join(vecFolder,'Farms',farmName)

Seasons = ['Spring','Summer','Autumn','Winter'] #list of seasons

# ## Model of treading Impact on Soil Properties (Donovan & Monaghan, 2021)
# - $\Delta_{tr}$ = 1 + $p - (ph)e^{-0.5i\omega}$
# - $\Delta_{tr}$ = Treading damage to soil
# - $\Delta$ = Damage to soil; $p$ = hoof pressure (kPa/kPa), normalized
# - $h$ = 1 - ${0.05y}$; h=history of grazing; y = years
# - $\omega_{c}$ = $-0.003\phi(75\sigma-20\phi )^{2} + 8\phi + 2)$ compression factor
# - $\omega_{p}$ = $10*(1-e^{-500\phi(max(0,\sigma^{3} - (0.25\phi)))^{2}})$ pugging factor
# - $\phi$ = clay fraction (0-1); $\sigma$ = soil moisture (%v/%v)
# - $i$ = Grazing Intensity*    $\frac{RSU}{m^{2} daily break}$
# 
# ## Stock hoof pressures normalized by actual pressures:
# - Dairy cattle (Friesian) hoof pressure: 220 kPa (Di et al., 2001 [p 113]; Scholefield & Hall, 1986)
# - Sheep hoof pressure: 83 kPa (Drewry 2006, p 160)
# - Beef cattle hoof pressure: 205 kPa (beef cattle tend to weigh 85-95% of dairy cattle)

# ## Step 1. Calculating seasonal pugging and compaction scalars.

#%% Defining constants and equations to be applied

#assumed grazing years/history
y=1 #years
hPasture = 1 - 0.05*y
hWfc = 1 - 0.05*y #assume 1 year per wfc paddock
hNon = 0

#SI unit converter:
siConv= 0.1317 #final units = (metric ton * ha * hr)/ (ha* MJ *mm)

#p = hoof pressure
#d = duration of grazing per season (0-1)
#h = history of grazing (calc'd earlier)
#I = grazing intensity
#sm = soil moisture
#c = clay fraction
#wC/wP=omega C/P, compaction/pugging factor

#m = soil moisture, c = clay fraction (0-1)
maxFinder = lambda m,c: np.maximum(0,(m**3)-(c*0.25))

#omega C and omega P calculations (compaction and pugging scalars, respectively)
wCompScalarCalc = lambda c, sm: -0.003* c *((75*sm)- (20*(c+1)))**2+8*c+2
wPugScalarCalc = lambda c,phi: 10*(1-np.exp(-500*c*(phi**2)))

dTcompCalc = lambda p,h,d,I,wC: 1 + (p - (p*(1-(h*(d/90)**0.4)*0.05) * np.exp(np.prod([-0.5,I,wC],dtype=object))))
dTpugCalc = lambda p,h,d,I,wP: 1 + (p - (p*(1-(h*(d/90)**0.4)*0.05) * np.exp(np.prod([-0.5,I,wP],dtype=object))))

#final K-factor calculation:
KfCalc = lambda Mf, OM, SV, P, conv: ((0.00021*(Mf**1.14)*(12-OM*100)+3.25*(SV-2)+2.5*(P-3))/100)*conv
KfTrCalc = lambda Mf, OM, SVtr, Ptr, conv: ((0.00021*(Mf**1.14)*(12-OM*100)+3.25*(SVtr-2)+2.5*(Ptr-3))/100)*conv

#normalized stock hoof pressure equivalents:
hoofPresDict = {'pNon':0,'pShp':0.38,'pBef':0.65,'pDai':0.7,'pDer':0.6,'pSnb':0.515}

pShp=0.38
pBef=0.65
pDai=0.7
pSnb=0.515 #assumes a 50:50 mix between sheep and beef
pDry=0.515
pNat=0.515
pMix = 0.515
pDer = 0.6

#seasons soil water content (%/%) assumptions
swcDict={'swcSp' : 0.4 , 'swcSu' : 0.15, 'swcAu' : 0.2, 'swcWi' : 0.55}

#setting constants (stock units) for sheep bef and dairy as defined in Donovan & Monaghan (2021)
cNon = 0
cShp=1.35
cBef=6.9
cDai=8.0
cDer=2.3
cSnb=np.mean([cShp,cBef])

suDict = {'Non':cNon,'Shp':cShp,'Bef':cBef,'Dai':cDai,'Der':cDer,'Snb':cSnb}

#%% Loading farm paddock data and adjusting headers

farmStockDat = pd.read_csv(farmDatLoc) #farm stock data (population, type, season, etc.)
farmStockDat.Paddock = farmStockDat.Paddock.astype(str)
farmBound = gpd.read_file(farmBoundLoc) #farm boundary shapefile


#%% Initial data cleaning/updates

#Adjusting/updating any duplicate scenarios.
dupVals = farmStockDat[farmStockDat.duplicated(subset=['Scenario','Paddock'])][['Scenario','Paddock']]

for dupScen,dupPaddock in zip(dupVals.Scenario,dupVals.Paddock):
    print(dupScen,'is duplicated')
    farmStockDat.loc[(farmStockDat.Scenario == dupScen)&(farmStockDat.Paddock == dupPaddock),'Scenario'] = [str(dupScen) + '_' + str(i) if i != 0 else dupScen for i in range(len(farmStockDat.loc[(farmStockDat.Scenario == dupScen)&(farmStockDat.Paddock == dupPaddock),'Scenario']))]

#adding stock type  to df
farmStockDat.loc[farmStockDat.stock=='shp','stock']='Shp'
farmStockDat.loc[farmStockDat.stock=='bef','stock']='Bef'
farmStockDat.loc[farmStockDat.stock=='dai','stock']='Dai'
farmStockDat.loc[(farmStockDat.stock=='der')|(farmStockDat.stock=='deer')|(farmStockDat.stock=='dee'),'stock']='Der'
farmStockDat.loc[(farmStockDat.stock=='snb')|(farmStockDat.stock=='bns'),'stock']='Snb'


#%% Loading farm paddock data and running initial area calcs and filtering unncessary columns

farmPdks = gpd.read_file(farmPdkLoc) #farm paddocks shapefile
farmPdks['pdkNa']=farmPdks['PdkID'].astype(str) #paddock id names
farmPdks.drop(columns=['PdkID'],inplace=True)

farmPdks.drop(farmPdks.filter(regex='Area').columns, axis=1, inplace=True)
farmPdks.drop(farmPdks.filter(regex='SHAPE').columns, axis=1, inplace=True)
farmPdks.drop(farmPdks.filter(regex='Shape').columns, axis=1, inplace=True)

#using cylindrical equal area projection as in: https://proj.org/operations/projections/cea.html
farmPdks = md.eqAreaCalc(farmPdks,['Ha','Mtr2'])

#adding status of grazed/ungrazed depending on land use ID
#if 'Status' not in farmPdks.columns:
#    print('No Status column provided, adding defaults.')
#    farmPdks['Status'] = 'Ungrazed'
#    farmPdks.loc[farmPdks.LUID.isin([40,41,42,83]),'Status'] = 'Grazed'
#elif 'Status' in farmPdks.columns:
#    print('Status column provided.')

#Ensuring no duplicate paddock names exist.
pdkNames=pd.Series(farmPdks.pdkNa)

for dup in pdkNames[pdkNames.duplicated()].unique(): 
    pdkNames[pdkNames[pdkNames == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(pdkNames == dup))]
    
# rename the columns with the cols list.
farmPdks.pdkNa=pdkNames.astype(str)

#print(np.sort(farmPdks.pdkNa.unique()))


#%% Clipping factors to farm boundary

LSfFarmLoc = os.path.join(farmFolder,farmName,'Rast','LSf'+farmName+'.tif')
if not os.path.exists(LSfFarmLoc):
    print('Clipping LSf data..')
    md.clipRast2shp(LSfLoc,farmBoundLoc,LSfFarmLoc)

KfFarmLoc = os.path.join(farmFolder,farmName,'Kf'+farmName+'_Fabi.shp')
if not os.path.exists(KfFarmLoc):
    print('Clipping Kf data..')
    KfIsl = gpd.read_file(KfShpLoc)
    KfFarmDat = gpd.clip(KfIsl,farmPdks)
    KfFarmDat.to_file(KfFarmLoc)
    del(KfIsl)
else:
    print('KfTr already exists for {0}, loading data.'.format(farmName))
    KfFarmDat = gpd.read_file(KfFarmLoc)

#%%
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


#%% Kf Tr Calculations

for scen in tqdm(farmStockDat.Scenario.unique(),total=len(farmStockDat.Scenario.unique())):
    if scen == 31:
    
        #farm stock data for ONLY this scenario
        farmStockScen=farmStockDat[farmStockDat.Scenario==scen].drop(columns='Scenario')
        #list of unique stock types in this scenario
        stockList = farmStockScen.stock.unique().tolist()
    
        #merging geospatial farm paddocks with scenario stock info
        farmPdkScen = farmPdks.merge(farmStockScen,how='left', left_on='pdkNa', right_on='Paddock',suffixes=[None,'_drop'])
        farmPdkScen['Status']='Grazed' #adding grazed status default
        #dropping unnecessary columns
        farmPdkScen.drop(columns='Paddock', axis=1, inplace=True)
        farmPdkScen.drop(farmPdkScen.filter(regex='_drop').columns, axis=1, inplace=True)
        #setting default to 0 when data doesn't exist
        farmPdkScen.loc[farmPdkScen.stockDen.isnull()==True,'stockDen'] = 0
        farmPdkScen.loc[farmPdkScen.stock.isnull()==True,'stock'] = 'Non'
        farmPdkScen.loc[farmPdkScen.stock=='Non',['Status','OP']] = ['Ungrazed',0]
        farmPdkScen.loc[farmPdkScen.OP.isnull()==True,'OP'] = 0
        #adding columns for stock populations
    
    #2.
        KfPdkJoin = gpd.overlay(farmPdkScen, KfFarmDat,how='union',keep_geom_type=True)
        KfPdkJoin.drop(KfPdkJoin.filter(regex='Area').columns, axis=1, inplace=True) #resetting the area calculations for new areas 
        KfPdkJoin.drop(KfPdkJoin.filter(regex='Perm').columns, axis=1, inplace=True)
        KfPdkJoin.drop(KfPdkJoin.filter(regex='farm_type').columns, axis=1, inplace=True)
    
        KfPdkJoin = md.eqAreaCalc(KfPdkJoin,['Ha','Mtr2'])#recalculating area of new polygons:
        sliverThresh = 10 #threshold (m2) for slivers to be detected
        KfPdkJoin = md.delSlivers(KfPdkJoin,sliverThresh)#removing small polygon slivers
    
        #testing for unique geometries within a shapefile
        uniqueGeoms=[]
        for geom in KfPdkJoin.geometry:
            if any(g.equals(geom) for g in uniqueGeoms):
                uniqueGeoms.append(geom);
        # Removing any rows that have 'nontype' as their PS value.
        errNo=0
        for index, row in KfPdkJoin.iterrows():
            if KfPdkJoin.loc[index,'PS'] is None:
                #print('NAY!')
                errNo=errNo+1
                KfPdkJoin.loc[index,'PS'] = 'undef'
        if scen==0:
            #print('{0}% of polygons with areas < {1} m^2 were removed.'.format(np.round(sliverPct,1),sliverThresh))
            print('{0}/{1} polygons are unique.'.format(len(uniqueGeoms),len(KfPdkJoin)))
            print(errNo, 'rows with nonetype were adjusted.')
    
        #deleting unnecessary columns that will be calculated later.
        delCols = ['KfTrStYr','KfTrStSp', 'KfTrStSu', 'KfTrStAu','KfTrStWi','dKSp','dKSu','dKAu','dKWi','dKfStYr',
                   'PtrSp','PtrSu','PtrAu','PtrWi','SVtrSp','SVtrSu','SVtrAu','SVtrWi']
    
        for col in delCols:
            if col in KfPdkJoin.columns:
                KfPdkJoin.drop(columns=col,inplace=True)
    
                stockList = KfPdkJoin.stock.unique().tolist()
    #3.
        #creating a column of hoof pressures to simplify the calculation length in the next loops (will delete in final output)
        KfPdkJoin['p']=0
        for stock in stockList:
             KfPdkJoin.loc[KfPdkJoin.stock==stock,'p']=hoofPresDict['p'+stock]
        #!!!!!! ADD ALT CALC FOR MIXED GRZING SYSTEM!!!!!
        #Need to bring back frStock to get the proportions
            #KfPdkJoin.loc[KfPdkJoin.farm_type.isin(['SNB','DRY','NAT']),'p']=pMix
    
        #assuming constant histories (will delete in final output) h = 1-(0.05*y)
        KfPdkJoin.loc[KfPdkJoin.stock=='Non','h']=y
        KfPdkJoin.loc[KfPdkJoin.stock!='Non','h']=y
    
        seasonDays = 90
        for season in Seasons:
            se = str(season[0:2])
    
            swc = swcDict['swc'+se]
    
            KfTrStS = str('KfTrSt'+se); dltaKs = str('dKf'+se); SVtrS = str('SV'+se);
            PtrS = str('Ptr'+se); SVtrS = str('SVtr'+se); dCompS = str('dComp'+se); dPugS = str('dPug'+se);
            GrIntS = str('GrInt'+se)
    
            wC = wCompScalarCalc(KfPdkJoin['Clay'],swc) #compression factor
            phiMax = maxFinder(swc,KfPdkJoin['Clay'])
            wP = wPugScalarCalc(KfPdkJoin['Clay'],phiMax) #pugging factor
    
            #default duration set to 0
            KfPdkJoin.loc[:,'dur'+se] = 0
    
            #updating duration seasonally
            if season=='Spring':
                KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Sp','dur'+se]=KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Sp','OP']
                KfPdkJoin.loc[KfPdkJoin.GrSeasonInit.isin(['Su','Au','Wi','Non']),'dur'+se]=0
            elif season=='Summer':
                KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Sp','dur'+se]=KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Sp','OP'].sub(seasonDays)
                KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Su','dur'+se]=KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Su','OP']
                KfPdkJoin.loc[KfPdkJoin.GrSeasonInit.isin(['Au','Wi','Non']),'dur'+se]=0 
            elif season=='Autumn':
                KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Sp','dur'+se]=KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Sp','OP'].sub(seasonDays*2)
                KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Su','dur'+se]=KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Su','OP'].sub(seasonDays)
                KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Au','dur'+se]=KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Au','OP']
                KfPdkJoin.loc[KfPdkJoin.GrSeasonInit.isin(['Wi','Non']),'dur'+se]=0  
            elif season=='Winter':
                KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Sp','dur'+se]=KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Sp','OP'].sub(seasonDays*3)
                KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Su','dur'+se]=KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Su','OP'].sub(seasonDays*2)
                KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Au','dur'+se]=KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Au','OP'].sub(seasonDays)
                KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Wi','dur'+se]=KfPdkJoin.loc[KfPdkJoin.GrSeasonInit=='Wi','OP']
                KfPdkJoin.loc[KfPdkJoin.GrSeasonInit.isin(['Non']),'dur'+se]=0
    
            KfPdkJoin.loc[KfPdkJoin['dur'+se]<0,'dur'+se] = 0 #where duration is < 0, set to 0
            KfPdkJoin.loc[KfPdkJoin['dur'+se]>90,'dur'+se] = 90 #where duration is >90, set to 90
    
            #calculate p & grInt based on stock type
            KfPdkJoin.loc[:,'p']=0
            KfPdkJoin.loc[:,'GrInt'+se] = 0
            
            for stock in stockList:
                KfPdkJoin.loc[(KfPdkJoin['dur'+se]>0)&(KfPdkJoin.stock==stock),'p'] =  hoofPresDict['p'+stock]  
                KfPdkJoin.loc[(KfPdkJoin['dur'+se]>0)&(KfPdkJoin.stock==stock),'GrInt'+se] = np.prod([KfPdkJoin.loc[(KfPdkJoin['dur'+se]>0)&(KfPdkJoin.stock==stock),'stockDen'],hoofPresDict['p'+stock],suDict[stock]]).div(1000)
    
    # % change to permeability/porosity (p) after treading/compaction
            KfPdkJoin[dCompS] = dTcompCalc(KfPdkJoin.p,KfPdkJoin.h,KfPdkJoin['dur'+se],KfPdkJoin[GrIntS],wC)# note difference in omega C term
    # % change to soil structure (sv) after treading/pugging        
            KfPdkJoin[dPugS]  = dTpugCalc(KfPdkJoin.p,KfPdkJoin.h,KfPdkJoin['dur'+se],KfPdkJoin[GrIntS],wP)# note difference in omega P term
    
            #Calculating permeability (p) and structural (s) subclasses after being damaged by dmgComp and dmgPug
            #Ptr = treaded permeability, SVtr = treaded soil structure
            KfPdkJoin[PtrS] = KfPdkJoin['Pclass']*KfPdkJoin[dCompS] #permeability/porosity (p) after treading/compaction
            KfPdkJoin[SVtrS] = KfPdkJoin['SVclass']*KfPdkJoin[dPugS] #soil structure (sv) after treading/pugging
    
            #Ensuring that "NON" farm locations don't have any change to permeability or structural vulnerability
            #KfPdkJoin.loc[KfPdkJoin.Status=='Ungrazed',PtrS] = KfPdkJoin.loc[KfPdkJoin.Status=='Ungrazed','Pclass'] #permeability/porosity (p) after treading/compaction
            #KfPdkJoin.loc[KfPdkJoin.Status=='Ungrazed',SVtrS] = KfPdkJoin.loc[KfPdkJoin.Status=='Ungrazed','SVclass'] #soil structure (sv) after treading/pugging
    
            #Setting default KTr values to the original Kst. 
            #This will change for locations with grazing, but not those without.
            #Textural factor:
            KfPdkJoin.loc[:,'Mf'] = (KfPdkJoin.loc[:,'Silt']*100+KfPdkJoin.loc[:,'vfSand']*100) * (100 - KfPdkJoin.loc[:,'Clay']*100)
    
            #Simple k-factor calculation
            KfPdkJoin.loc[:,'K'] = KfCalc(KfPdkJoin.Mf, KfPdkJoin.OMadj, KfPdkJoin.SVclass, KfPdkJoin.Pclass, siConv)
            #KfPdkJoin.loc[:,'K'] = ((0.00021*(KfPdkJoin.loc[:,'Mf']**1.14)*(12-KfPdkJoin.loc[:,'OMadj']*100)+3.25*(KfPdkJoin.loc[:,'SVclass']-2)+2.5*(KfPdkJoin.loc[:,'Pclass']-3))/100)*siConv
            KfPdkJoin.loc[:,'St'] = np.exp(-0.04*(((100*(KfPdkJoin.loc[:,'Gravel']+KfPdkJoin.loc[:,'Rock'])))-10))
            KfPdkJoin.loc[(KfPdkJoin.Gravel+KfPdkJoin.Rock)<0.1,'St'] = 1 #the equation only applies to areas with >10% stone cover, so this statement eliminates any locations not meeting that criteria
            KfPdkJoin.loc[:,'KfSt']=KfPdkJoin.loc[:,'K']*KfPdkJoin.loc[:,'St']
    
            #Calculating KfTr and KfTrSt for each season.
            KfTrS = KfTrCalc(KfPdkJoin.Mf, KfPdkJoin.OMadj, KfPdkJoin[SVtrS], KfPdkJoin[PtrS], siConv)
            KfPdkJoin[KfTrStS] = KfTrS * KfPdkJoin.St #(St is stoniness adjustment from Poesen (1994))
    
            #removing areas with no p or s-class information and/or negative values
            KfPdkJoin.loc[KfPdkJoin.PS.isin(['town','rive','lake','estu','ice','BRock','MSoil','quar','undef']),[KfTrStS,'KfSt']] = [0,0]
    
            #Calcuating change in k-value due to treading
            KfPdkJoin[dltaKs] = np.round(((KfPdkJoin[KfTrStS]-KfPdkJoin.KfSt)/KfPdkJoin.KfSt)*100,3).abs()
           #KfPdkJoin.loc[KfPdkJoin[dltaKs].abs()<0.0001,dltaKs] = 0 #if the % change is < 0.0001%, then it is 0
    
        KfPdkJoin['KfTrStYr']= KfPdkJoin.filter(like='KfTrSt').mean(axis=1)
        KfPdkJoin['dKfYr']= np.round((KfPdkJoin['KfTrStYr']-KfPdkJoin['KfSt'])/KfPdkJoin['KfSt']*100,3).abs()
    
        #optional
        #del KfPdkJoin['h']; #del KfPdkJoin['p']
    
        #subset of important columns
        KfPdkData=KfPdkJoin[['Landcover', 'Class_2018', 'LUID', 'pdkNa','Status','stock','stockDen','OP',
                     'SOILTYPE', 'PS', 'Sand', 'vfSand', 'Silt', 'Clay', 'Gravel', 'Rock',
                     'OM', 'OMadj', 'pReten', 'Pclass', 'SVclass','Mf', 'K', 'St', 'KfSt',
                     'durSp','durSu','durAu','durWi',
                     'GrIntSp','GrIntSu','GrIntAu','GrIntWi',
                     'KfTrStSp','KfTrStSu','KfTrStAu','KfTrStWi', 'KfTrStYr',
                     'dKfSp', 'dKfSu', 'dKfAu', 'dKfWi', 'dKfYr',
                     'AreaHa','AreaMtr2',
                     'geometry']]
    
        #Saving output
        KfPdkData.to_file(os.path.join(farmFolder,farmName,'aOutputs/KfTr','KfTrPdk_Scen'+str(scen)+'.shp'))

#%% Intermediate plotting
fig, ax = plt.subplots(figsize = (15,15))
#leg_kwds = {'title':'Paddock landuse', 'loc':'upper left','bbox_to_anchor':(1,1.05), 'ncol':1}

KfPdkData.plot(column='GrIntWi',ax=ax,label='GrIntWi',legend=True)#,legend_kwds = leg_kwds)
ax.set_title(farmName+ ' Paddock Map',fontdict = {'fontsize': 30})
ax.set_ylabel('Latitude',fontdict = {'fontsize': 20});
ax.set_xlabel('Longitude',fontdict = {'fontsize': 20});
#plt.legend(prop = {'size':10} )

fig, ax = plt.subplots(figsize = (15,15))
#leg_kwds = {'title':'Farm Soils', 'loc':'upper left','bbox_to_anchor':(1,1.05), 'ncol':1}

KfPdkData.plot(column='GrIntSp',ax=ax,label='GrIntSp',legend=True,cmap='plasma')#,legend_kwds = leg_kwds)
ax.set_title(farmName+ ' Paddock Map',fontdict = {'fontsize': 30})
ax.set_ylabel('Latitude',fontdict = {'fontsize': 20});
ax.set_xlabel('Longitude',fontdict = {'fontsize': 20});
#plt.legend(prop = {'size':10} )
# In[28]:

#checking for rows where the status is 'grazed', but it doesn't have GrInt values calculated for it.
#In such cases, the column will be set to the average of any/all other rows with the same paddock name.
#The list of column values to be changed is below. THe code will loop through these columns and change the values
naColList = KfPdkJoin.columns[KfPdkJoin.isna().any()].tolist()

for feature in KfPdkJoin.iterfeatures():
    for col in naColList:
        if KfPdkJoin[col].dtype != 'O':
            fId = int(feature['id']) #converting index to int, as it defaults to string.
            pdkId = feature['properties']['pdkNa']
            #print('Replacing values for paddock {0}, column:{1}'.format(pdkId, col))

            KfPdkJoin.loc[KfPdkJoin.index == fId,col] = np.mean(KfPdkJoin.loc[(KfPdkJoin.index != fId)&(KfPdkJoin.pdkNa == pdkId) ,col])
            
            #if np.isnan(KfPdkJoin.loc[KfPdkJoin.index == fId,col][0]):
            #    print('still na!')


# In[29]:

#checking for rows where the status is 'grazed', but it doesn't have GrInt values calculated for it.
#In such cases, the column will be set to the average of any/all other rows with the same paddock name.
#The list of column values to be changed is below. THe code will loop through these columns and change the values
naColList = ['GrInt','Class_2018']

for feature in KfPdkJoin.iterfeatures():
    for season in Seasons:
        se = season[0:2]
        if (feature['properties']['Status']=='Grazed') & (feature['properties']['GrInt'+se]==0) & (feature['properties']['stockDen']>0):
            fId = int(feature['id']) #converting index to int, as it defaults to string.
            pdkId = feature['properties']['pdkNa']
            print('Replacing values for paddock {0}, index:{1}'.format(pdkId, fId))
            
            for col in naColList:
                KfPdkJoin.loc[KfPdkJoin.index == fId,col] = np.mean(KfPdkJoin.loc[(KfPdkJoin.index != fId)&(KfPdkJoin.pdkNa == pdkId) ,col])


#%% Save K-treading factor Data to Shapefile

KfPdkJoin=KfPdkJoin[['Landcover', 'Class_2018', 'LUID', 'pdkNa','Status','OP','stockDen',
                     'SOILTYPE', 'PS', 'Sand', 'vfSand', 'Silt', 'Clay', 'Gravel', 'Rock',
                     'OM', 'OMadj', 'Perm', 'pReten', 'Pclass', 'SVclass','GrInt', 'Mf', 'K', 'St', 'Kst',
                     'KfTrStSp','KfTrStSu','KfTrStAu','KfTrStWi', 'KfTrStYr',
                     'dKSp', 'dKSu', 'dKAu', 'dKWi', 'dKfStYr',
                     'AreaHa','AreaMtr2',
                     'geometry']]
KfTrPdkOutLoc = os.path.join(farmFolder,farmName,'KfTrPdk'+farmName+'.shp')
KfPdkJoin.to_file(KfTrPdkOutLoc)

#%% Getting Cf Lookup table

CfDefs = pd.read_csv(os.path.join(aspatialFolder,'Landuse/LCDB Cf Lookup Table.csv'))
CfDefs= CfDefs[['Name_2018','CfAu','CfAuGr','CfSp','CfSpGr','CfSu','CfSuGr','CfWi','CfWiGr','Class_2018']]
CfDefs

#%% Calculating Cf for each paddock

for scen in tqdm(farmStockDat.Scenario.unique(),total=len(farmStockDat.Scenario.unique())):
    #if scen == 0:
    KfPdkJoinLoc = os.path.join(farmFolder,farmName,'aOutputs/KfTr/KfTrPdk_Scen'+str(scen)+'.shp')
    CfKfPdks = gpd.read_file(KfPdkJoinLoc)
    CfKfPdks = CfKfPdks[['K','KfSt','KfTrStSp','KfTrStSu','KfTrStAu','KfTrStWi','KfTrStYr','dKfYr','AreaHa',
                      'AreaMtr2','Class_2018','Landcover', 'LUID', 'pdkNa','Status','stock','stockDen','geometry']]

    CfKfPdks = CfKfPdks.merge(CfDefs, how='left', left_on='LUID',right_on='Class_2018',suffixes=[None,'_drop'])
    CfKfPdks.drop(CfKfPdks.filter(regex='_drop').columns, axis=1, inplace=True) #dropping second unnecessary columns
    CfKfPdks[CfKfPdks.columns.difference(['geometry'])]

    naLUNAs = CfKfPdks.loc[CfKfPdks.CfSp.isna()==True,'Landcover'].unique().tolist()
    naLUIDs = CfKfPdks.loc[CfKfPdks.CfSp.isna()==True,'LUID'].unique()

    if len(naLUIDs) >0:
        replaceNAs = True
    else:
        replaceNAs = False

    if replaceNAs == True:
        #replacing Cf values as weighted combination of grass (41) and forest (54) for mixed-LU paddocks
        wgtIDs = [41,54]
        wgts = [0.85,0.15] #weighted sum of 41 (85%) and 54 (15%) for mixed grassland+forest
        for col in CfKfPdks.filter(regex='Cf').columns.tolist():
            #print(col)
            CfWgt = np.sum(CfDefs[CfDefs.Class_2018.isin(wgtIDs)][col]*wgts)
            CfKfPdks.loc[CfKfPdks.LUID==naLUIDs[0],col] = CfWgt
            CfKfPdks.loc[CfKfPdks.LUID==naLUIDs[0],'Name_2018'] = naLUNAs[0]

    CfKfPdkOutLoc = os.path.join(farmFolder,farmName,'aOutputs/CfGr/CfKf'+farmName+'_Scen'+str(scen)+'.shp')
    CfKfPdks.to_file(CfKfPdkOutLoc)


#%% Convert Kf and Cf to raster

for scen in tqdm(farmStockDat.Scenario.unique(),total=len(farmStockDat.Scenario.unique())):
    pdkShpLoc = os.path.join(farmFolder,farmName,'aOutputs/CfGr','CfKf'+farmName+'_Scen'+str(scen)+'.shp')
    fieldNames = ['CfSpGr','CfSuGr','CfAuGr','CfWiGr'] #'KfTrStSp','KfTrStSu','KfTrStAu','KfTrStWi',
    for fieldName in fieldNames:
        rastOutLoc = os.path.join(farmFolder,farmName,'aOutputs/Rast',fieldName+farmName+'_Scen'+str(scen)+'.tif')
        md.vec2rast(shpLoc=pdkShpLoc,rastRefLoc=LSfFarmLoc,outLoc=rastOutLoc,fieldName=fieldName)


#%% Cropping and resampling Rf data to align with other factors

#loading LSf data as reference grid
LSfDat = gdal.Open(LSfFarmLoc)
w,h = LSfDat.RasterXSize, LSfDat.RasterYSize
#print(w,h)

#clipping data
for RfSeLoc in RfLocs:
    baseName = os.path.basename(RfSeLoc)[0:4]
    RfSeOutLoc = os.path.join(farmFolder,farmName,'Rast',baseName+farmName+'.tif')
    if not os.path.exists(RfSeOutLoc):
        print('Clipping {0} data.'.format(baseName))
        md.clipRast2shp(RfSeLoc,farmBoundLoc,RfSeOutLoc)
    else:
        print('{0} already exists, no need to clip.'.format(baseName))

#testing for resample
RfFarmLocs = glob.glob(os.path.join(farmFolder,farmName,'Rf*'+'*.tif'))
for RfSeLoc in RfFarmLocs:
    baseName = os.path.basename(RfSeLoc)[0:-4]
    resampOutLoc = os.path.join(farmFolder,farmName,baseName+'_r.tif')
    
    #obtaining pixel size of Rf
    RfDat = gdal.Open(RfSeLoc)
    wRf,hRf = RfDat.RasterXSize, RfDat.RasterYSize
    pxSizeRf = RfDat.GetGeoTransform()[1]
    RfDat=None
    #obtaining pixel size of LSf
    LSfDat = gdal.Open(LSfFarmLoc)
    w,h = LSfDat.RasterXSize, LSfDat.RasterYSize
    pxSizeLSf = LSfDat.GetGeoTransform()[1]
    LSfDat=None
        
    if (pxSizeRf != pxSizeLSf) or ([w,h]!=[wRf,hRf]):
        md.resampleRast(RfSeLoc,LSfFarmLoc,resampOutLoc)
    #gdal.Translate(destName= resampOutLoc, srcDS = RfSeLoc, xRes=pxSizeLSf, yRes=pxSizeLSf, width=w, height=h,
    #                   resampleAlg="nearest", format='GTiff')
        print('Resampled Rf raster to {0} m^2 pixels with w:h of {1}:{2}.'.format(pxSizeLSf,w,h))
        
        #replacing original Rf filename with resampled and deleting the resampled one.
        os.remove(RfSeLoc)
        os.rename(resampOutLoc, RfSeLoc)
    else:
        print('No need to resample, Rf rastesr are same size.')

#%% Calculating seasonal erosion

#Conversion factor for adjusting from t/ha/yr --> t/yr (the same as: t/pixel/yr); where pixel = 225 m^2
#This allows summing of the output per farm and/or catchment to estimate total potential losses.
ha2sqm = 0.0001 #1 square meter = 0.0001 ha

for scen in tqdm(farmStockDat.Scenario.unique()):
    for Season in Seasons:
        #print('Currently processing: ', Season,' for ', farmName, ' Farm')
        seAbv = Season[0:2]

        #KfLoc = os.path.join(farmFolder,farmName,'KfSt'+seAbv+farmName+'.tif')
        KfTrLoc = os.path.join(farmFolder,farmName,'aOutputs/Rast','KfTrSt'+seAbv+farmName+'_Scen'+str(scen)+'.tif')
        #CfLoc = os.path.join(farmFolder,farmName,'Cf'+seAbv+farmName+'_copy.tif')
        CfGrLoc = os.path.join(farmFolder,farmName,'aOutputs/Rast','Cf'+seAbv+'Gr'+farmName+'_Scen'+str(scen)+'.tif')
        LSfLoc = os.path.join(farmFolder,farmName,'Rast/LSf'+farmName+'.tif')
        RfLoc = os.path.join(farmFolder,farmName,'Rast/Rf'+seAbv+farmName+'.tif')

        #output folder path location for the erosion calculation for the base scenario
        outLoc = os.path.join(farmFolder,farmName,'aOutputs/ErGrRast/ErGr'+seAbv+'Px'+farmName+'_Scen'+str(scen)+'.tif')
        
        refRastLoc = LSfLoc

        in_files =[LSfLoc, KfTrLoc, CfGrLoc, RfLoc]
        in_files

        in_rasters = []
        for rast in in_files:
            dat = rio.open(rast)
            in_rasters.append(dat)
        n = int(len(in_rasters)) #total number of rasters to be included in calculations

        
        # Get the number of rows and columns and block sizes
        x_size = in_rasters[0].width
        y_size = in_rasters[0].height
        x=0; y=0; index = 0

        rastDtype=in_rasters[0].dtypes[0]
        noData = in_rasters[0].nodata

        # Stacking masked arrays of data from each raster
        inStack = ma.stack(
            [in_rasters[i].read(1, masked=True) \
                for i in range(n)]).astype(np.float32)

        pxArea = np.prod(in_rasters[0].res) #m2 pixels
        conv = ha2sqm*pxArea #conversion factor from t/ha TO t/pixel

        #appropriately masking the output calculations
        outMask = ma.any([[inStack[i].data==-99999] for i in range(n)],axis=0) #creating a mask where *any* layer is masked
        for i in range(n):
            inStack[i].mask = outMask.data

        #calculating erosion from all factors
        outData = (np.ma.prod(inStack,axis=0)/4)*conv #the product of the input datasets (KfTr,CfGr,LSf,Rf)

        #Because the product spits out 1.0 as the no data/fill value, have to update
        outData.data[outData.data==1.0]=noData

        md.arr2rast(inArr=outData,refRastLoc=refRastLoc,noDataValue=noData,outLoc=outLoc,outDataType=gdal.GDT_Float32)

        #print('Raster output {0}'.format(outLoc))


# In[147]:

for scen in tqdm(farmStockDat.Scenario.unique()):
    ErSpLoc = os.path.join(farmFolder,farmName,'aOutputs/ErGrRast/ErGrSpPx'+farmName+'_Scen'+str(scen)+'.tif')
    ErSuLoc = os.path.join(farmFolder,farmName,'aOutputs/ErGrRast/ErGrSuPx'+farmName+'_Scen'+str(scen)+'.tif')
    ErAuLoc = os.path.join(farmFolder,farmName,'aOutputs/ErGrRast/ErGrAuPx'+farmName+'_Scen'+str(scen)+'.tif')
    ErWiLoc = os.path.join(farmFolder,farmName,'aOutputs/ErGrRast/ErGrWiPx'+farmName+'_Scen'+str(scen)+'.tif')

    refRastLoc = LSfLoc

    in_files = [ErSpLoc, ErSuLoc, ErAuLoc, ErWiLoc]
    in_files

    in_rasters = []
    for rast in in_files:
        dat = rio.open(rast)
        in_rasters.append(dat)
    n = int(len(in_rasters)) #total number of rasters to be included in calculations

    #output folder path location for the erosion calculation for the base scenario
    outLoc = os.path.join(farmFolder,farmName,'aOutputs/ErGrRast/ErGrYrPx'+farmName+'_Scen'+str(scen)+'.tif')

    # Get the noData value
    noData = in_rasters[0].nodata

    # Stacking masked arrays of data from each raster
    inStack = ma.stack(
        [in_rasters[i].read(1, masked=True) \
            for i in range(n)]).astype(np.float32)

    #appropriately masking the output calculations
    outMask = ma.any([[inStack[i].data==noData] for i in range(n)],axis=0) #creating a mask where *any* layer is masked
    for i in range(n):
        inStack[i].mask = outMask.data

    #calculating erosion from all factors
    outData = inStack.sum(axis=0) #the sum of the seasonal datasets (KfTr,CfGr,LSf,Rf)

    #Because the product spits out 1.0 as the no data/fill value, have to update
    outData.data[outData.data==1.0]=noData

    md.arr2rast(inArr=outData,refRastLoc=refRastLoc,noDataValue=noData,outLoc=outLoc,outDataType=gdal.GDT_Float32)

    #additional clipping and renaming, because arr2rast isn't doing it properly.
    clipOutLoc = outLoc[:-4]+'_clip.tif'
    md.clipRast2shp(outLoc,farmBoundLoc,clipOutLoc,2193)
    os.remove(outLoc)
    os.rename(clipOutLoc,outLoc)

    #print('Raster output: {0}'.format(outLoc))

    #second output in t/ha/yr
    outDataHa = outData/conv
    outLocHa = os.path.join(farmFolder,farmName,'aOutputs/ErGrRast/ErGrYrHa'+farmName+'_Scen'+str(scen)+'.tif')
    md.arr2rast(inArr=outDataHa,refRastLoc=refRastLoc,noDataValue=noData,outLoc=outLocHa,outDataType=gdal.GDT_Float32)
    
    clipOutLocHa = outLocHa[:-4]+'_clip.tif'
    md.clipRast2shp(outLocHa,farmBoundLoc,clipOutLocHa,2193)
    os.remove(outLocHa)
    os.rename(clipOutLocHa,outLocHa)
    #print('Raster output: {0}'.format(outLocHa))


# In[82]:

outLoc = os.path.join(farmFolder,farmName,'ErGrYrPx'+farmName+'.tif')
with rio.open(outLoc) as rastDat:
    pxArea = np.prod(rastDat.res) #m2 pixels
    print(pxArea)


#%% Aggregating paddocks back to full extent for final rasterstats to be calculated

for scen in tqdm(farmStockDat.Scenario.unique()):
    pdkScenDat = gpd.read_file(os.path.join(farmFolder,farmName,'aOutputs/KfTr','KfTrPdk_Scen'+str(scen)+'.shp'))
    
    pdkDat= pdkScenDat.dissolve(by='pdkNa',aggfunc={'Landcover':pd.Series.mode, 'LUID': pd.Series.mode, 
                                                    'Status':pd.Series.mode,'stock':pd.Series.mode,'stockDen':'median',
                                                    'OP':pd.Series.mode,#'PS':pd.Series.mode,
                                                    'Sand':'mean','vfSand':'mean','Silt':'mean','Clay':'mean',
                                                    'OM':'mean','OMadj':'mean','Pclass':'mean','SVclass':'mean',
                                                    'Mf':'mean','KfSt':'mean','GrIntSp':'mean','GrIntSu':'mean',
                                                    'GrIntAu':'mean','GrIntWi':'mean','KfTrStSp':'mean','KfTrStSu':'mean',
                                                    'KfTrStAu':'mean','KfTrStWi':'mean','KfTrStYr':'mean','dKfSp':'mean',
                                                    'dKfSu':'mean','dKfAu':'mean','dKfWi':'mean','dKfYr':'mean',
                                                    'AreaHa':'sum','AreaMtr2':'sum'
                                                    })
    
    pdkDat.to_file(os.path.join(farmFolder,farmName,'Final','PdkSummary_Scen'+str(scen)+'.shp'))
       
#%% Zonal statistics of soil losses per paddock

seAbvs = ['Sp','Su','Au','Wi','Yr']

for scen in tqdm(farmStockDat.Scenario.unique(),total=len(farmStockDat.Scenario.unique())):
#scen= 1

    pdkScenDat = gpd.read_file(os.path.join(farmFolder,farmName,'Final','PdkSummary_Scen'+str(scen)+'.shp'))
    PdksDf = pdkScenDat.loc[:,['Landcover', 'LUID', 'pdkNa', 'Status','stock', 'OP', 'stockDen','AreaHa','geometry']]
    
    slpLoc = os.path.join(farmFolder,farmName,'Rast/slpDeg'+farmName+'.tif')
    LSfLoc = os.path.join(farmFolder,farmName,'Rast/LSf'+farmName+'.tif')
    
    zStatSlp = zonal_stats(pdkScenDat,
             slpLoc,
             stats=['max','mean','median'])#'min'
    
    #zSlpMin = [item['min'] for item in zStatSlp if item is not None]
    zSlpMax = [item['max'] for item in zStatSlp if item is not None]
    zSlpMean = [item['mean'] for item in zStatSlp if item is not None]
    zSlpMed = [item['median'] for item in zStatSlp if item is not None]
    
    #PdksDf['slpMin']=zSlpMin
    PdksDf['slpMax']=zSlpMax
    PdksDf['slpMean']=zSlpMean
    PdksDf['slpMed']=zSlpMed
    
    zStatLSf = zonal_stats(pdkScenDat,
                           LSfLoc,
                           stats=['max','mean','median'])#'min',
    
    #zLSfMin = [item['min'] for item in zStatLSf if item is not None]
    zLSfMax = [item['max'] for item in zStatLSf if item is not None]
    zLSfMean = [item['mean'] for item in zStatLSf if item is not None]
    zLSfMed = [item['median'] for item in zStatLSf if item is not None]
    
    #PdksDf['LSfMin']=zLSfMin
    PdksDf['LSfMax']=zLSfMax
    PdksDf['LSfMean']=zLSfMean
    PdksDf['LSfMed']=zLSfMed
    
    for se in seAbvs:
        ErLoc = os.path.join(farmFolder,farmName,'aOutputs/ErGrRast/ErGr'+se+'Px'+farmName+'_Scen'+str(scen)+'.tif')
        
        zStatEr = zonal_stats(pdkScenDat,
                         ErLoc,
                         stats=['sum'])
    
        zEr = [item['sum'] for item in zStatEr if item is not None]
    
        PdksDf.loc[:,'ErGr'+se+'Sum']=zEr
        PdksDf.loc[:,'ErGr'+se+'YldHa']=PdksDf.loc[:,'ErGr'+se+'Sum']/PdksDf.loc[:,'AreaHa']
    
    PdksDf.replace([np.inf, -np.inf], 0, inplace=True)
    
    outCsvLoc = os.path.join(farmFolder,farmName,'Final','PdkSummary_Scen'+str(scen)+'.csv')
    PdksDf[PdksDf.columns.difference(['geometry'])].to_csv(outCsvLoc,sep=',')
    
    outGpdLoc = os.path.join(farmFolder,farmName,'Final','PdkSummary_Scen'+str(scen)+'.shp')
    PdksDf.to_file(outGpdLoc)


# In[186]:


seAbvs = ['Sp','Su','Au','Wi','Yr']

scen = 0
pdkScenDat = gpd.read_file(os.path.join(farmFolder,farmName,'Final','PdkSummary_Scen'+str(scen)+'.shp'))
PdksDf = pdkScenDat.loc[:,['Landcover', 'LUID', 'pdkNa', 'Status','stock', 'stockDen','OP','AreaHa','geometry']]

slpLoc = os.path.join(farmFolder,farmName,'Rast/slpDeg'+farmName+'.tif')
LSfLoc = os.path.join(farmFolder,farmName,'Rast/LSf'+farmName+'.tif')

zStatSlp = zonal_stats(pdkScenDat,
             slpLoc,
             stats=['min','max','mean','median'])

zSlpMin = [item['min'] for item in zStatSlp if item is not None]
zSlpMax = [item['max'] for item in zStatSlp if item is not None]
zSlpMean = [item['mean'] for item in zStatSlp if item is not None]
zSlpMed = [item['median'] for item in zStatSlp if item is not None]

PdksDf['slpMin']=zSlpMin
PdksDf['slpMax']=zSlpMax
PdksDf['slpMean']=zSlpMean
PdksDf['slpMed']=zSlpMed

zStatLSf = zonal_stats(pdkScenDat,
                       LSfLoc,
                       stats=['min','max','mean','median'])

zLSfMin = [item['min'] for item in zStatLSf if item is not None]
zLSfMax = [item['max'] for item in zStatLSf if item is not None]
zLSfMean = [item['mean'] for item in zStatLSf if item is not None]
zLSfMed = [item['median'] for item in zStatLSf if item is not None]

PdksDf['LSfMin']=zLSfMin
PdksDf['LSfMax']=zLSfMax
PdksDf['LSfMean']=zLSfMean
PdksDf['LSfMed']=zLSfMed

for se in seAbvs:
    ErLoc = os.path.join(farmFolder,farmName,'aOutputs/ErGrRast/ErGr'+se+'Px'+farmName+'_Scen'+str(scen)+'.tif')

    zStatEr = zonal_stats(pdkScenDat,
                     ErLoc,
                     stats=['sum'])

    zEr = [item['sum'] for item in zStatEr if item is not None]

    PdksDf.loc[:,'ErGr'+se+'Sum']=zEr
    PdksDf.loc[:,'ErGr'+se+'YldHa']=PdksDf.loc[:,'ErGr'+se+'Sum']/PdksDf.loc[:,'AreaHa']

PdksDf.replace([np.inf, -np.inf], 0, inplace=True)

outCsvLoc = os.path.join(farmFolder,farmName,'Final','PdkSummaryBase.csv')
PdksDf[PdksDf.columns.difference(['geometry'])].to_csv(outCsvLoc,sep=',')

outGpdLoc = os.path.join(farmFolder,farmName,'Final','PdkSummaryBase.shp')
PdksDf.to_file(outGpdLoc)


#%%
    
fullMelt = pd.DataFrame()

for scen in tqdm(farmStockDat.Scenario.unique()):
#scen= 1   
    pdkScenDat = pd.read_csv(os.path.join(farmFolder,farmName,'Final','PdkSummary_Scen'+str(scen)+'.csv'))
    pdkMelt = pdkScenDat.melt(
        id_vars=['Landcover', 'LUID', 'pdkNa', 'Status', 'stock', 'stockDen', 'AreaHa','OP',
                 'slpMax', 'slpMed', 'LSfMax', 'LSfMed'],
        value_vars=['ErGrSpYldHa', 'ErGrSuYldHa','ErGrAuYldHa', 'ErGrWiYldHa', 'ErGrYrSum','ErGrYrYldHa'])
    pdkMelt['scenario']=scen
    
    fullMelt = pd.concat([fullMelt,pdkMelt])

fullMelt.to_csv(os.path.join(farmFolder,farmName,farmName+'Summary Long Format.csv'))

#%% Plots

farmLongDf = pd.read_csv(os.path.join(farmFolder,farmName,farmName+'Summary Long Format.csv'))
farmPdkFull = gpd.read_file(os.path.join(farmFolder,farmName,'Final',farmName+'PdkSummaryBase.shp'))
#farmPdkFull[farmPdkFull.columns.difference(['geometry','ErGrAuSum','ErGrAuYldH','ErGrSpSum','ErGrSpYldH','ErGrSuSum','ErGrSuYldH','ErGrWiSum','ErGrWiYldH'])]
farmPdkFull = md.eqAreaCalc(farmPdkFull, ['Ha','Mtr2'])

colors = ['yellowgreen','mediumseagreen','cadetblue','bisque','sandybrown','lightcoral','khaki','mediumpurple']
colorsdark = ['olivedrab','darkolivegreen','darkslategrey','tan','peru','indianred','goldenrod','blueviolet']

#luAllList = farmPdkFull.Landcover.unique() #use if LU types are unknown

#Int he case of fabi's data, I know how to order
luAllList = ['Mixed Exotic Shrubland', 'Exotic Forest','Broadleaved Indigenous Hardwoods',
       'Low Producing Grassland', 'Low Producing Grassland with Forest','High Producing Grassland',
       'Sub Alpine Shrubland', 'Built-up Area (settlement)'] #use if LU types are unknown
luErList = [lu for lu in luAllList if lu != 'Built-up Area (settlement)']

def sorter(column):
    reorder = luAllList
    # This also works:
    # mapper = {name: order for order, name in enumerate(reorder)}
    # return column.map(mapper)
    cat = pd.Categorical(column, categories=reorder, ordered=True)
    return pd.Series(cat)

farmPdkFull = farmPdkFull.sort_values(by="Landcover", key=sorter)
luGroup = farmPdkFull.sort_values(by="Landcover", key=sorter)


# In[200]:

fig, ax = plt.subplots(figsize = (15,15))
#leg_kwds = {'title':'Farm Soils', 'loc':'upper left','bbox_to_anchor':(1,1.05), 'ncol':1}

farmPdkFull.plot('ErGrYrYldH',ax=ax,label='Soil loss yield (t/ha/yr)',legend=True,cmap='rocket_r',edgecolor='black')#,legend_kwds = leg_kwds)
farmPdkFull[farmPdkFull.ErGrYrYldH==0].plot('ErGrYrYldH',ax=ax,color='white',edgecolor='black')#,legend_kwds = leg_kwds)

ax.set_title(farmName+ ' Paddock soil loss ',fontdict = {'fontsize': 24})
ax.set_ylabel('Latitude',fontdict = {'fontsize': 20});
ax.set_xlabel('Longitude',fontdict = {'fontsize': 20});

#for additional paddock labels:
labelThresh = 50000 #Area (m2) below which labels won't be shown.
farmPdkFull[(farmPdkFull.Landcover.isin(luErList))&(farmPdkFull.AreaMtr2>labelThresh)].apply(lambda x: ax.annotate(text=np.round(x.ErGrYrYldH,1), 
                                                                              xy=x.geometry.centroid.coords[0],
                                                                              weight='bold',
                                                                              color='k',
                                                                              size=14,
                                                                              ha='center'), axis=1);

#plt.legend(prop = {'size':10} )
plt.savefig(os.path.join(farmFolder,farmName,'aOutputs/Figures',farmName+' Erosion Yield Map.pdf'))


# In[201]:

fig, ax = plt.subplots(figsize = (15,15))
#leg_kwds = {'title':'Farm Soils', 'loc':'upper left','bbox_to_anchor':(1,1.05), 'ncol':1}

farmPdkFull[farmPdkFull.Landcover.isin(luErList)].plot('slpMean',ax=ax,label='Slope (degrees)',legend=True,cmap='coolwarm',edgecolor='black')#,legend_kwds = leg_kwds)
farmPdkFull[~farmPdkFull.Landcover.isin(luErList)].plot('slpMean',ax=ax,color='white',edgecolor='black')#,legend_kwds = leg_kwds)
ax.set_title(farmName+ ' Mean Paddock Slopes (degrees)',fontdict = {'fontsize': 24})
ax.set_ylabel('Latitude',fontdict = {'fontsize': 20});
ax.set_xlabel('Longitude',fontdict = {'fontsize': 20});

#for additional paddock labels:
farmPdkFull[(farmPdkFull.Landcover.isin(luErList))&(farmPdkFull.AreaMtr2>labelThresh)].apply(lambda x: ax.annotate(text=np.round(x.slpMean,1),
                                                                                                                   xy=x.geometry.centroid.coords[0], 
                                                                                                                   ha='center',weight='bold',size=14),axis=1);
plt.savefig(os.path.join(farmFolder,farmName,'aOutputs/Figures/Paddock Slope Mean Map.pdf'))

# %% ErGr Yield/ha boxplots

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(7, 6))
#ax.set_xscale("log")

# Plot the horizontal boxes
sns.boxplot(x="ErGrYrYldH", y="Landcover", data=farmPdkFull[farmPdkFull.ErGrYrYldH<9],
            whis=[0, 100], width=.6, palette=colors)

# Add in points to show each observation
sns.stripplot(x="ErGrYrYldH", y="Landcover", data=farmPdkFull[farmPdkFull.ErGrYrYldH<9],
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(xlabel="Soil loss yield, per paddock (t/ha/y)")
ax.set(ylabel="")
#ax.set_x_lim

sns.despine(trim=True, left=True)
plt.savefig(os.path.join(farmFolder,farmName,'aOutputs/Figures/LULC ErGrYldHa Boxplots Horiz.pdf'))

#%% LSf boxplots

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(7, 6))
#ax.set_xscale("log")

# Plot the horizontal boxes
sns.boxplot(x="LSfMean", y="Landcover", data=farmPdkFull,
            whis=[0, 100], width=.6, palette=colors)

# Add in points to show each observation
sns.stripplot(x="LSfMean", y="Landcover", data=farmPdkFull,
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(xlabel="Slope-length factor (LSf, dimensionless)",ylabel="")
sns.despine(trim=True, left=True)

plt.savefig(os.path.join(farmFolder,farmName+'/aOutputs/Figures/LULC LSf Boxplots Horiz.pdf'))

#%% Median slope boxplots

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(7, 6))
#ax.set_xscale("log")

# Plot the horizontal boxes
sns.boxplot(x="slpMed", y="Landcover", data=farmPdkFull,
            whis=[0, 100], width=.6, palette=colors)

# Add in points to show each observation
sns.stripplot(x="slpMed", y="Landcover", data=farmPdkFull,
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(xlabel="Median paddock slopes (degrees)",ylabel="")
sns.despine(trim=True, left=True)

plt.savefig(os.path.join(farmFolder,farmName+'/aOutputs/Figures/LULC SlpMed Boxplots Horiz.pdf'))


# In[211]:

f, ax = plt.subplots(figsize=(17, 6))
sns.boxenplot(x="Landcover", y="LSfMean",
              palette=colors,
              #color='b', #hue="LULC",  #alternative color calls
              #order=clarity_ranking, #adjusting order
              scale="linear", data=farmPdkFull)
ax.set(ylabel='Slope-length factor (LSf, dimensionless)',xlabel='');
plt.savefig(os.path.join(farmFolder,farmName+'/aOutputs/Figures/LULC LSf Boxplots.pdf'))


# In[169]:


f, ax = plt.subplots(figsize=(17, 6))
sns.boxenplot(x="Landcover", y="slpMean",
              palette=colors,
              #color='b', #hue="LULC",  #alternative color calls
              #order=clarity_ranking, #adjusting order
              scale="linear", data=farmPdkFull)
ax.set(ylabel='Mean slope (degrees)',xlabel='');
plt.savefig(os.path.join(farmFolder,farmName+'/aOutputs/Figures/LULC SlopeMean Boxplots.pdf'))


# In[212]:
#f, ax = plt.subplots(figsize=(30, 20))

g=sns.jointplot(x="slpMean", y="ErGrYrYldH", hue="Landcover", data=farmPdkFull, 
                  palette=colors[0:len(farmPdkFull.Landcover.unique())],
                  #kind="reg",#truncate=False,#
                  xlim=(0, 40), ylim=(0, 6),
                  color="m", height=7)
g.set_axis_labels('Mean slope (degrees)','Soil loss yield, per paddock (t/ha/y)')
plt.savefig(os.path.join(farmFolder,farmName+'/aOutputs/Figures/Slope-Yield Relationship By Landuse.pdf'))


# In[212]:
#f, ax = plt.subplots(figsize=(30, 20))

var = 'ErGrYrYldHa' #variable to plot
condition = (farmLongDf.variable==var)# & (farmLongDf.pdkNa==12)# & (farmLongDf.Landcover.isin(['Low Producing Grassland','Low Producing Grassland with Forest','High Producing Grassland']))
g=sns.jointplot(x="slpMed", y="value", hue="Landcover", 
                data=farmLongDf[condition], 
                #palette=colors[3:6],
                #kind="reg",#truncate=False,#
                #xlim=(0, 380), #ylim=(0, 6),
                color="m", height=7)
g.set_axis_labels('Occupation period (days)','Soil loss, per paddock (t/y)')
#plt.savefig(os.path.join(farmFolder,farmName+'/aOutputs/Figures/Slope-Yield Relationship By Landuse.pdf'))


#%% 

ErLu = farmPdkFull[['Landcover','LUID','AreaHa','ErGrYrSum']].groupby('Landcover').agg('sum')
#luGroup.reindex(luAllList)
ErLu.reset_index(inplace=True)

areaTot = ErLu.AreaHa.sum()
areaTotEr = ErLu[ErLu.Landcover.isin(luErList)].AreaHa.sum()
ErYrTot = np.sum(ErLu.ErGrYrSum)

ErLu['LuAreaPct'] = ErLu.AreaHa.div(areaTot)
ErLu['ErGrYrPct'] = ErLu.ErGrYrSum.div(ErYrTot)

ErLu= ErLu.astype({'Landcover': pd.CategoricalDtype(luAllList, ordered=True)})
ErLu = ErLu.sort_values(by='Landcover')


# In[214]:

fs=14

#Plotting
colors = ['yellowgreen','mediumseagreen','cadetblue','bisque','sandybrown','lightcoral','khaki','mediumpurple']
colorsdark = ['olivedrab','darkolivegreen','darkslategrey','tan','peru','indianred','goldenrod','blueviolet']
#explode = tuple([0.05] * len(colors))

fig, ax = plt.subplots(figsize=(12,12));

#pie chart for LULC land area (outer)
wedges, texts, autotexts = ax.pie(ErLu.LuAreaPct,
                                  #labels=ErLu.LULC[(ErLu.LULC.isin(luAllList))],
                                  autopct='%1.1f%%', pctdistance=0.85,  radius=4, #explode = 0.05,
                                  colors=colors, labeldistance=1.1,textprops={'fontsize': fs})

#pie chart for surface erosion losses (inner)
wedges2, texts2, autotexts2 = ax.pie(ErLu.ErGrYrPct,autopct='%1.1f%%',
       pctdistance=0.85,radius=3,colors=colorsdark,textprops={'fontsize': fs})

#legend for land area pie chart (outer)
fig.legend(wedges, luAllList,loc='lower right')

#legend for surface erosion pie chart (inner)
fig.legend(wedges2,luAllList,
           loc='lower left')

#labelling inner and outer circles:
ax.text(0.0,2,'Surface erosion (%)',weight='bold',horizontalalignment='center',fontsize=16)
ax.text(0.2,3.6,'Landuse Area (%)',weight='bold',horizontalalignment='center',fontsize=16)

#labelling text in center
ax.text(0.0,0.4, 'Farm: {}'.format(farmName),weight='bold',
    horizontalalignment='center',fontsize=16)
ax.text(0.0,-0.3,
         'Total area: {:,} $km^2$'.format(np.round(areaTot,0)),
        horizontalalignment='center',fontsize=14)
ax.text(0.0,0.0,
         'Surface erosion: {:,} $t/yr$'.format(np.round(ErYrTot,0)),
        horizontalalignment='center',fontsize=14)

#draw center blank/white circle
centre_circle = plt.Circle((0,0),1.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

#plt.tight_layout()
plt.axis('equal');
#plt.show()

plt.savefig(os.path.join(farmFolder,farmName+'/aOutputs/Figures',farmName+' ErGr Nested Pie Chart.pdf'))





