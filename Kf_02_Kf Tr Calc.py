#!/usr/bin/env python
# coding: utf-8

# In[1]:
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from osgeo import osr, gdal

# Prettier plotting with seaborn
import seaborn as sns; 
sns.set(font_scale=1.5)
sns.set_style("white")

# Set standard plot parameters for uniform plotting
plt.rcParams['figure.figsize'] = (12, 12)

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# In[2]:


#coordinate reference system, in epsg.
dst_crs = 'epsg:2193'

#converting epsg:#### to a wkt format (well-known text projection)
srs = osr.SpatialReference()
srs.ImportFromEPSG(int(dst_crs.split(':')[1]))
srs_wkt = srs.ExportToPrettyWkt()

print('Output projection information:' + '\n' + srs_wkt)


# ### Setting inputs/outputs

# In[3]:

Island = 'North'

# setting home folder and sub-directories where input and output files lie
homeFolder = r'/Users/mrd/emarde/osga'
extHDloc = r'/Volumes/Transcend/Data'
vecFolder = os.path.join(homeFolder,'Data/Vector')
rastFolder = os.path.join(homeFolder,'Data/Raster')
aspatialFolder = os.path.join(homeFolder,'Data/Aspatial')

#input folders path location
KfGrILoc = os.path.join(vecFolder,'Soil/KfGrI'+Island+'NZ.shp')


# In[4]:


#Loading KTr-factor soil data generated above
KfTrDat = gpd.read_file(KfGrILoc) #inherent soil erodibility
kfCRS = KfTrDat.crs
KfTrDat.loc[KfTrDat.farm_type=='NON','LandCover'] = 99
KfTrDat.loc[KfTrDat.LandCover.isin([5,6,7,8,9,10]),'Class_2018'] = 83
KfTrDat.rename(columns={'Kst':'KfSt'},inplace=True)

# #### Previous setting 'default' GrInt values for locations WITH a farm type, but with no Intensity value
# KfTrDat.loc[(KfTrDat.GrInt==0) & (KfTrDat.farm_type=='BEF')& (KfTrDat.Class_2018!=83),'GrInt']= 0.02
# KfTrDat.loc[(KfTrDat.GrInt==0) & (KfTrDat.farm_type=='SHP')& (KfTrDat.Class_2018!=83),'GrInt']= 0.01
# KfTrDat.loc[(KfTrDat.GrInt==0) & (KfTrDat.farm_type=='SNB')& (KfTrDat.Class_2018!=83),'GrInt']= 0.015
# KfTrDat.loc[(KfTrDat.GrInt==0) & (KfTrDat.farm_type=='DRY')& (KfTrDat.Class_2018!=83),'GrInt']= 0.015
# KfTrDat.loc[(KfTrDat.GrInt==0) & (KfTrDat.farm_type=='DAI')& (KfTrDat.Class_2018!=83),'GrInt']= 0.03
# KfTrDat.loc[(KfTrDat.GrInt==0) & (KfTrDat.farm_type=='NAT')& (KfTrDat.Class_2018!=83),'GrInt']= 0.005
# KfTrDat.loc[(KfTrDat.GrInt==0) & (KfTrDat.Class_2018==83),'GrInt']= 1.5 #typical expected value

# #### Removing rows that have 'nontype' as their PS value.

# In[5]:

errNo=0
for index, row in KfTrDat.iterrows():
    if KfTrDat.loc[index,'PS'] is None:
        #print('NAY!')
        errNo=errNo+1
        KfTrDat.loc[index,'PS'] = 'undef'
print(errNo, 'rows with nonetype adjusted')


# ## Model of treading Impact on Soil Properties (Donovan & Monaghan, 2021)
# - $\Delta_{tr}$ = 1 + $p - (ph)e^{-0.5i\omega}$
# - $\Delta_{tr}$ = Treading damage to soil
# - $\Delta$ = Damage to soil; $p$ = hoof pressure (kPa/kPa), normalized
# - $h$ = 1 - ${0.05y}$; h=history of grazing; y = years
# - $\omega_{c}$ = $-0.003\phi(75\sigma-20\phi )^{2} + 8\phi + 2)$ compression factor
# - $\omega_{p}$ = $10*(1-e^{-500\phi(max(0,\sigma^{3} - (0.25\phi)))^{2}})$ pugging factor
# - $\phi$ = clay fraction (0-1); $\sigma$ = soil moisture (%v/%v)
# - $i$ = Grazing Intensity*    $\frac{RSU}{m^{2} daily break}$     *See reference table for values

# ## Step 1. Calculating seasonal pugging and compaction scalars that vary with moisture contents.
#     Assuming seasonal moisture contents of (%v/%v):
#     Spring: 0.4 (40%) 
#     Summer: 0.15 (15%)
#     Autumn: 0.2 (20%)
#     Winter: 0.55 (55%)

# In[13]:

#SI unit converter:
siConv= 0.1317 #end units = (metric ton * ha * hr)/ (ha* MJ *mm)

#normalized stock hoof pressure equivalents:
pShp=0.38
pBef=0.65
pDai=0.7
pDee = 0.6

#seasons soil water content (%/%) assumptions
swcSp = 0.4
swcSu = 0.15
swcAu = 0.2
swcWi = 0.55

#assumed grazing years/history
y=0; yPast = 1; yWfc=1 #years
hPasture = 1 - 0.05*yPast
hWfc = 1 - 0.05*yWfc #assume 1 year per wfc paddock

# In[25]:


#Defining equations to be applied to soil rasters.

#p = hoof pressure
#h = history of grazing (calc'd earlier)
#I = grazing intensity
#m = soil moisture
#c = clay fraction
#wC/wP=omega C/P, compaction/pugging factor

#m = soil moisture, c = clay fraction (0-1)
maxFinder = lambda m,c: np.maximum(0,(m**3)-(c*0.25))

#omega C and omega P calculations (compaction and pugging scalars, respectively)
wCompScalarCalc = lambda c, m: -0.003* c *((75*m)- (20*(c+1)))**2+8*c+2
wPugScalarCalc = lambda c,phi: 10*(1-np.exp(-500*c*(phi**2)))

#p- (p*(1-(np.sqrt(dur[3])+h)*0.05))*np.exp(-0.5*omega*gI)

dTcompCalc = lambda p,h,d,I,wC: 1 + (p - (p*(1-(y*(d/90))*0.05) * np.exp(np.prod([-0.5,I,wC],dtype=object))))
dTpugCalc = lambda p,h,d,I,wP: 1 + (p - (p*(1-(y*(d/90))*0.05) * np.exp(np.prod([-0.5,I,wP],dtype=object))))

#final K-factor calculation:
KfCalc = lambda Mf, OM, SV, P, conv: ((0.00021*(Mf**1.14)*(12-OM*100)+3.25*(SV-2)+2.5*(P-3))/100)*conv
KfTrCalc = lambda Mf, OM, SVtr, Ptr, conv: ((0.00021*(Mf**1.14)*(12-OM*100)+3.25*(SVtr-2)+2.5*(Ptr-3))/100)*conv


# In[26]:

#Calculating KfTr and KfStTr for each season.
KfTrDat.loc[:,'Kf'] = KfCalc(KfTrDat.Mf, KfTrDat.OMadj, KfTrDat['SVclass'], KfTrDat['Pclass'], siConv)
KfTrDat.loc[:,'KfSt'] = KfTrDat.loc[:,'Kf'] * KfTrDat.loc[:,'St'] #(St is stoniness adjustment from Poesen (1994))    

#Setting default KTr values to the original Kst. This will change for locations with grazing, but not those without.
KfTrDat.loc[KfTrDat.farm_type=='NON',['KfTrSp','KfStTrSp']]=KfTrDat.loc[KfTrDat.farm_type=='NON',['Kf','KfSt']]
KfTrDat.loc[KfTrDat.farm_type=='NON',['KfTrSu','KfStTrSu']]=KfTrDat.loc[KfTrDat.farm_type=='NON',['Kf','KfSt']]
KfTrDat.loc[KfTrDat.farm_type=='NON',['KfTrAu','KfStTrAu']]=KfTrDat.loc[KfTrDat.farm_type=='NON',['Kf','KfSt']]
KfTrDat.loc[KfTrDat.farm_type=='NON',['KfTrWi','KfStTrWi']]=KfTrDat.loc[KfTrDat.farm_type=='NON',['Kf','KfSt']]

#creating a column of hoof pressures for non-mixed farm types (will delete in final output)
KfTrDat.loc[:,'pHoof']=0
KfTrDat.loc[KfTrDat.farm_type=='SHP','pHoof']=pShp
KfTrDat.loc[KfTrDat.farm_type=='BEF','pHoof']=pBef
KfTrDat.loc[KfTrDat.farm_type=='DAI','pHoof']=pDai
KfTrDat.loc[KfTrDat.farm_type=='DEE','pHoof']=pDee

#setting null values to 0
KfTrDat.loc[KfTrDat.shp2017.isna()==True,'shp2017'] = 0
KfTrDat.loc[KfTrDat.bef2017.isna()==True,'bef2017'] = 0
KfTrDat.loc[KfTrDat.dai2017.isna()==True,'dai2017'] = 0
KfTrDat.loc[KfTrDat.dee2017.isna()==True,'dee2017'] = 0

#same for histories (will delete in final output) h = 1-(0.05*y)
KfTrDat.loc[:,'h']=1
KfTrDat.loc[KfTrDat.LandCover.isin([5,6,7,8,9,10]),'h']=hWfc
KfTrDat.loc[KfTrDat.LandCover.isin([0]),'h']=hPasture

#assumed seasonal grazing durations (days) for pastoral lands
durPastDict = {'Sp':30,'Su':30,'Au':30,'Wi':10}
durWfcDict = {'Sp':60,'Su':20,'Au':0,'Wi':90} #seasonal grazing durations for WFCs (not set to 0 to account for residual damage)

#seasonal soil water content (%/%) assumptions
swcDict={'swcSp' : 0.4 , 'swcSu' : 0.15, 'swcAu' : 0.2, 'swcWi' : 0.55}

# In[27]:

Seasons = ['Spring','Summer','Autumn','Winter']

for i in np.arange(0,4):
    Season = Seasons[i]; S = str(Seasons[i][0:2])
    
    #setting seasonal variable names
    KfTrS = str('KfTr'+S); KfStTrS = str('KfStTr'+S); dltaKs = str('dK'+S); SVtrS = str('SV'+S);
    PtrS = str('Ptr'+S); SVtrS = str('SVtr'+S);
    
    #setting seasonally variable parameters (soil moisture, duration)
    swc = swcDict['swc'+S] #setting seasonal soil moisture

    KfTrDat.loc[:,'dur'+S] = 0 #setting seasonal default duration as 0
    KfTrDat.loc[KfTrDat.LandCover.isin([5,6,7,8,9,10])&KfTrDat.farm_type!='NON','dur'+S] = durWfcDict[S] #setting seasonal grazing duration for WFCs
    KfTrDat.loc[KfTrDat.LandCover.isin([0]) & KfTrDat.farm_type!='NON','dur'+S] = durPastDict[S] #setting seasonal grazing duration for pasture
    
    #fractional grazing hoof pressure calculation for mixed farm types. updating each season
    KfTrDat.loc[KfTrDat.farm_type.isin(['SNB','DRY','NAT']),'pHoof']=(KfTrDat['shpFr'+S]*pShp)+(KfTrDat['befFr'+S]*pBef)+(KfTrDat['daiFr'+S]*pDai)
    
    #calculating seasonally variable comparession and pugging coefficients
    KfTrDat['wC'+S] = wCompScalarCalc(KfTrDat.Clay,swc) #compression factor
    phiMax = maxFinder(swc,KfTrDat.Clay)
    KfTrDat['wP'+S] = wPugScalarCalc(KfTrDat.Clay,phiMax) #pugging factor
    
    #calculating seasonally variable damage (compaction and pugging)
    KfTrDat['dmgComp'+S] = dTcompCalc(KfTrDat.pHoof,KfTrDat.h,KfTrDat['dur'+S],KfTrDat['GrIntAdj'+S],KfTrDat['wC'+S])# note difference in omega C term
    KfTrDat['dmgPug'+S] = dTpugCalc(KfTrDat.pHoof,KfTrDat.h,KfTrDat['dur'+S],KfTrDat['GrIntAdj'+S],KfTrDat['wP'+S])# note difference in omega P term

    #Calculating permeability (p) and structural (s) subclasses after being damaged by dmgComp and dmgPug
    #Ptr = treaded permeability, SVtr = treaded soil structure
    KfTrDat[PtrS] = KfTrDat['Pclass']*KfTrDat['dmgComp'+S] #permeability/porosity (p) after treading/compaction
    KfTrDat[SVtrS] = KfTrDat['SVclass']*KfTrDat['dmgComp'+S] #soil structure (sv) after treading/pugging

    #Ensuring that "NON" farm locations have same permeability or structural vulnerability as before
    KfTrDat.loc[KfTrDat.farm_type=='NON',PtrS] = KfTrDat.loc[KfTrDat.farm_type=='NON','Pclass'] #permeability/porosity (p) after treading/compaction
    KfTrDat.loc[KfTrDat.farm_type=='NON',SVtrS] = KfTrDat.loc[KfTrDat.farm_type=='NON','SVclass'] #soil structure (sv) after treading/pugging
    
    #subsetting for farms
    #svTr = KfTrDat.loc[KfTrDat.farm_type!='NON',SVtrS] #subset of treaded soil vulnerability
    #pTr = KfTrDat.loc[KfTrDat.farm_type!='NON',PtrS] #subset of treaded soil permeability
    
    #Calculating KfTr and KfStTr for each season.
    KfTrDat.loc[:,KfTrS] = KfTrCalc(KfTrDat.Mf, KfTrDat.OMadj, KfTrDat[SVtrS], KfTrDat[PtrS], siConv)
    KfTrDat.loc[:,KfStTrS] = KfTrDat.loc[:,KfTrS] * KfTrDat.loc[:,'St'] #(St is stoniness adjustment from Poesen (1994))
    
    #removing areas with no p or s-class information and/or negative values
    KfTrDat.loc[KfTrDat.PS.isin(['town','rive','lake','estu','ice','BRock','MSoil','quar','undef']),[KfTrS,KfStTrS,'KfSt']] = [0,0,0]

    #Calcuating change in k-value due to treading
    KfTrDat.loc[:,dltaKs] = (KfTrDat[KfStTrS]-KfTrDat.KfSt)/KfTrDat.KfSt
    
    #KfTrDat['Rp'+S] = np.exp(np.prod([-0.5,KfTrDat['GrIntAdj'+S],KfTrDat['wP'+S]],dtype=object))
    #KfTrDat['Rc'+S] = np.exp(np.prod([-0.5,KfTrDat['GrIntAdj'+S],KfTrDat['wC'+S]],dtype=object))
    
    
KfTrDat.loc[:,'KfStTrYr']= KfTrDat[['KfStTrSp','KfStTrSu','KfStTrAu','KfStTrWi']].mean(axis=1)
KfTrDat.loc[:,'dKfYr']= (KfTrDat['KfStTrYr']-KfTrDat['KfSt'])/KfTrDat['KfSt']

#optional
#del KfTrDat['h']; del KfTrDat['p']

# In[29]:

KfTrDat=KfTrDat[['SOILTYPE', 'PS', 'Sand', 'vfSand', 'Silt', 'Clay', 'Gravel', 'Rock',
                 'OM', 'OMadj', 'Perm', 'pReten', 'Pclass', 'SVclass', 'PtrSp', 'SVtrSp', 'PtrSu', 'SVtrSu', 'PtrAu', 'SVtrAu', 'PtrWi', 'SVtrWi',
                 'Mf', 'K', 'St', 'KfSt','KfStTrSp','KfStTrSu','KfStTrAu','KfStTrWi', 'KfStTrYr',
                 'dKSp', 'dKSu', 'dKAu', 'dKWi', 'dKfYr',
                 'GrIntSp','GrIntAdjSp', 'GrIntSu','GrIntAdjSu', 'GrIntAu','GrIntAdjAu', 'GrIntWi','GrIntAdjWi',
                 'shpDen2017','befDen2017','daiDen2017','deeDen2017',
                 'farm_type', 'Feed', 'LandCover', 'AreaHa','geometry']]
KfTrOutLoc = os.path.join(vecFolder,'Soil/KfTr'+Island+'NZ.shp')
KfTrDat.to_file(KfTrOutLoc)

# In[34]:

KfGroups = KfTrDat.groupby(['farm_type','Feed']).agg({'dmgCompWi': ['mean', 'max'],'dKWi': ['mean', 'max'],'KfStTrWi': ['mean', 'max'],'dKfYr': ['mean', 'max'], 'Clay': 'size'}) #,'dmgPug': ['mean', 'max']

#%%
S='Sp'

farmOrder = ['NON', 'NAT', 'SHP', 'SNB', 'BEF', 'DRY', 'DAI', 'DEE']
feedOrder = ['Pasture', 'Winter grass', 'Fodder beet', 'Cereal', 'Brassica', 'Unknown']

f = plt.figure(figsize=(18,10))
ax = f.add_subplot(1,2,1)
sns.boxplot(data=KfTrDat.loc[KfTrDat.Feed=='Pasture'],y='dK'+S,x='farm_type',order=farmOrder,
            #hue='Feed',hue_order = feedOrder, 
            ax=ax)
#ax.set_xticklabels(['non-Wfc','Winter pasture','Winter cereal','Brassica','Fodder beet','Other/unknown'], 
#                    rotation=25, horizontalalignment='right')
ax.set_ylim(0,.2);
#f.savefig(os.path.join(homeFolder,'Outputs/Figures/Kf Change by Class and Forage '+Island+' NZ.pdf'))


ax1 = f.add_subplot(1,2,2)
sns.boxplot(data=KfTrDat[KfTrDat.Feed!='Pasture'],y='dK'+S,x='Feed',order=feedOrder[1:],
            hue='farm_type',hue_order = farmOrder, ax=ax1)
ax1.set_xticklabels(['Winter grass','Winter cereal','Brassica','Fodder beet','Other/unknown'], 
                    rotation=25, horizontalalignment='right')
ax1.set_ylim(0,1.5);
#f.savefig(os.path.join(homeFolder,'Outputs/Figures/Kf Change by Class and Forage '+Island+' NZ.pdf'))



# In[ ]:

hueOrder = ['NON', 'NAT', 'SHP', 'BEF', 'SNB', 'DAI', 'DRY', 'DEE']

f = plt.figure(figsize=(12,9))
ax = f.add_subplot(1,1,1)
sns.boxplot(data=KfTrDat,y='dKfStYr',x='farm_type',order=farmOrder,
            hue='Feed',hue_order = feedOrder[1:], ax=ax)
#ax.set_xticklabels(['non-Wfc','Winter pasture','Winter cereal','Brassica','Fodder beet','Other/unknown'], 
#                    rotation=25, horizontalalignment='right')
ax.set_ylim(0,2);
#f.savefig(os.path.join(homeFolder,'Outputs/Figures/Kf Change by Class and Forage '+Island+' NZ.pdf'))

# In[ ]:


hueOrder = ['NON', 'NAT', 'SHP', 'BEF', 'SNB', 'DAI', 'DRY', 'DEE']

f = plt.figure(figsize=(12,9))
ax = f.add_subplot(1,1,1)
sns.boxplot(data=KfTrDat[KfTrDat.LandCover==0],y='KfStTrYr',x='farm_type',order=farmOrder,
            ax=ax)
#ax.set_xticklabels(['Non-Wfc'], 
                    #rotation=25, horizontalalignment='right')
ax.set_ylim(0,5);
#f.savefig(os.path.join(homeFolder,'Outputs/Figures/Kf by Class and Forage '+Island+' NZ.pdf'))


# In[ ]:

#%%


# In[ ]:



# In[ ]:




