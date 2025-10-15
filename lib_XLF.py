import numpy as np
import matplotlib.pyplot as plt

import os
import glob
import time
from astropy.table import Table,vstack
from astropy.io import fits
import scipy
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib as mpl

def flux2luminosity(flux):
    Flux = flux*u.erg/u.s/(u.cm)**2
    R = 761*u.kpc
    Luminosity = Flux*4*np.pi*R**2
    return Luminosity.to(u.erg/u.s).value
def luminosity2flux(luminosity):
    Luminosity = luminosity*u.erg/u.s
    R = 761*u.kpc
    Flux = Luminosity/(4*np.pi*R**2)
    return Flux.to(u.erg/u.s/(u.cm)**2).value

from astropy.wcs import WCS
from scipy import interpolate


def filter_source_on_sensmap(Source,band,sensmap):
    
    
    source_RA,source_Dec = Source['RA'],Source['DEC']
    sky_coordinate = SkyCoord(ra=source_RA,dec=source_Dec,unit=u.deg,frame='fk5' )

#     sensmap_file = 'data/snsmap_EP_02000_04500_ML1.fits'
    # detmsk_file = '/Users/rhuang/Data/XMM/M31_data/edetect_stack/edetect_stack_v8_South_for_simulation/detmskM1.fits.gz'
    sensmap_image = fits.getdata(sensmap)
    try:
        sensmap_header = fits.getheader(Sensmap)
        sensmap_header['RA_OBJ']
    except KeyError:
        print('No key RA_OBJ found')
        sensmap_header = fits.getheader(f'data/snsmap_EP_02000_04500_ML6.fits')
    detmsk_image = sensmap_image==-2

    w = WCS(sensmap_header)
    w
    pixel_x,pixel_y = w.world_to_pixel(sky_coordinate)
    pixel_x_floor = np.int_(np.floor(pixel_x))
    pixel_y_floor = np.int_(np.floor(pixel_y))


    mask_index = Source[f'EP_{band}_FLUX']>sensmap_image[[pixel_y_floor,pixel_x_floor]]
    
    return Source[mask_index]

    
def skyarea_function_from_sensmap(sensmap,bins=np.logspace(-20,-10,301),cutoff=0.1,plot=False ):
    # cutoff is the cutoff fraction of the skyarea as the sensitivity limit
    sensmap_image = fits.getdata(sensmap)
    # sensmap_header = fits.getheader(Sensmap)
    try:
        sensmap_header = fits.getheader(Sensmap)
        sensmap_header['RA_OBJ']
    except KeyError:
        print('No key RA_OBJ found')
        sensmap_header = fits.getheader(f'data/snsmap_EP_02000_04500_ML6.fits')

    detmsk_image = sensmap_image==-2
    w = WCS(sensmap_header)
    sky_area = w.proj_plane_pixel_area()*np.sum(sensmap_image>0)
    counts,bins=np.histogram(sensmap_image[sensmap_image>0],bins=bins, )
    sky_area_function = np.cumsum(counts)*w.proj_plane_pixel_area()
    log_bins = np.log10(bins)
    log_bins_mean = (log_bins[:-1]+log_bins[1:])/2.
    FluxBins = 10**(log_bins_mean)
    cutoff_sensitivity = FluxBins[sky_area_function<np.max(sky_area_function)*cutoff][-1]
    print('cutoff_sensitivity',cutoff_sensitivity) 
    if plot==True:
        plt.axvline(cutoff_sensitivity,linestyle='--')
        plt.plot(FluxBins,sky_area_function)
        plt.xscale('log')
        plt.xlabel(r'$S~\mathrm{(ergs~s^{-1}~cm^{-2})} $')
        plt.ylabel('$area~\mathrm{(deg^{2})}$')
    return FluxBins,sky_area_function,cutoff_sensitivity

def skyarea_function_from_sensmap_with_massmap(Sensmap,Mask=None,MassMap=None,bins=np.logspace(-20,-10,301),cutoff=0.1,plot=False ):
    # cutoff is the cutoff fraction of the skyarea as the sensitivity limit
    # the origin skyarea_function_from_sensmap is used to calculate how much area is above the flux
    # now the skyarea_function_from_sensmap_with_massmap is used to calculate how much stellar mass is above the flux.
    
    sensmap_image = fits.getdata(Sensmap)
    sensmap_header = fits.getheader(Sensmap)
    massmap_image = fits.getdata(MassMap)
    if Mask!=None:
        region_mask = fits.getdata(Mask)
        sensmap_image[region_mask==0]=-2
    detmsk_image = sensmap_image==-2
    w = WCS(sensmap_header)
    sky_area = w.proj_plane_pixel_area()*np.sum(sensmap_image>0)
    counts,bins=np.histogram(sensmap_image[sensmap_image>0],weights=massmap_image[sensmap_image>0],bins=bins, )
    mass_function = np.cumsum(counts)#*w.proj_plane_pixel_area()
#     log_bins = np.log10(bins)
#     log_bins_mean = (log_bins[:-1]+log_bins[1:])/2.
#     FluxBins = 10**(log_bins_mean)
#     FluxBins = np.sqrt(bins[:-1] * bins[1:])
    FluxBins = bins[:-1]
    # print(np.max(mass_function))
    # if np.max(mass_function)==0:
    #     cutoff_sensitivity = np.nan
    # else:
    #     cutoff_sensitivity = FluxBins[mass_function<np.max(mass_function)*cutoff][-1]
#     print('cutoff_sensitivity',cutoff_sensitivity) 
    if plot==True:
#         plt.axvline(cutoff_sensitivity,linestyle='--')
        plt.plot(FluxBins,mass_function)
        plt.xscale('log')
        plt.xlabel('$S~\mathrm{(ergs~s^{-1}~cm^{-2})}$')
        plt.ylabel('$stellar mass~\mathrm{10^{10}~(M_{\odot})}$')
    return FluxBins,mass_function

def mass_function(Sensmap,Mask=None,MassMap=None,bins=np.logspace(-20,-10,301),cutoff=0.1,plot=False ):
    # cutoff is the cutoff fraction of the skyarea as the sensitivity limit
    # the origin skyarea_function_from_sensmap is used to calculate how much area is above the flux
    # now the skyarea_function_from_sensmap_with_massmap is used to calculate how much stellar mass is above the flux.
    
    sensmap_image = fits.getdata(Sensmap)
    sensmap_header = fits.getheader(Sensmap)
    massmap_image = fits.getdata(MassMap)
    if Mask!=None:
        region_mask = fits.getdata(Mask)
        sensmap_image[region_mask==0]=-2
    detmsk_image = sensmap_image==-2
    w = WCS(sensmap_header)
    sky_area = w.proj_plane_pixel_area()*np.sum(sensmap_image>0)
    counts,bins=np.histogram(sensmap_image[sensmap_image>0],weights=massmap_image[sensmap_image>0],bins=bins, )
    mass_function = np.cumsum(counts)#*w.proj_plane_pixel_area()
#     log_bins = np.log10(bins)
#     log_bins_mean = (log_bins[:-1]+log_bins[1:])/2.
#     FluxBins = 10**(log_bins_mean)
#     FluxBins = np.sqrt(bins[:-1] * bins[1:])
    FluxBins = bins[:-1]
    print(np.max(mass_function))
    # if np.max(mass_function)==0:
    #     cutoff_sensitivity = np.nan
    # else:
    #     cutoff_sensitivity = FluxBins[mass_function<np.max(mass_function)*cutoff][-1]
#     print('cutoff_sensitivity',cutoff_sensitivity) 
    if plot==True:
#         plt.axvline(cutoff_sensitivity,linestyle='--')
        plt.plot(FluxBins,mass_function)
        plt.xscale('log')
        plt.xlabel('$S~\mathrm{(ergs~s^{-1}~cm^{-2})}$')
        plt.ylabel('$stellar mass~\mathrm{10^{10}~(M_{\odot})}$')
    return FluxBins,mass_function

def SFR_function(Sensmap,Mask=None,SFRMap=None,bins=np.logspace(-20,-10,301),cutoff=0.1,plot=False ):
    # cutoff is the cutoff fraction of the skyarea as the sensitivity limit
    # the origin skyarea_function_from_sensmap is used to calculate how much area is above the flux
    # now the skyarea_function_from_sensmap_with_massmap is used to calculate how much stellar mass is above the flux.
    
    sensmap_image = fits.getdata(Sensmap)
    sensmap_header = fits.getheader(Sensmap)
    SFRmap_image = fits.getdata(SFRMap)
    if Mask!=None:
        region_mask = fits.getdata(Mask)
        sensmap_image[region_mask==0]=-2
    detmsk_image = sensmap_image==-2
    w = WCS(sensmap_header)
    sky_area = w.proj_plane_pixel_area()*np.sum(sensmap_image>0)
    counts,bins=np.histogram(sensmap_image[sensmap_image>0],weights=SFRmap_image[sensmap_image>0],bins=bins, )
    SFR_function = np.cumsum(counts)#*w.proj_plane_pixel_area()
#     log_bins = np.log10(bins)
#     log_bins_mean = (log_bins[:-1]+log_bins[1:])/2.
#     FluxBins = 10**(log_bins_mean)
#     FluxBins = np.sqrt(bins[:-1] * bins[1:])
    FluxBins = bins[:-1]
    print(np.max(SFR_function))
    # if np.max(mass_function)==0:
    #     cutoff_sensitivity = np.nan
    # else:
    #     cutoff_sensitivity = FluxBins[mass_function<np.max(mass_function)*cutoff][-1]
#     print('cutoff_sensitivity',cutoff_sensitivity) 
    if plot==True:
#         plt.axvline(cutoff_sensitivity,linestyle='--')
        plt.plot(FluxBins,SFR_function)
        plt.xscale('log')
        plt.xlabel('$S~\mathrm{(ergs~s^{-1}~cm^{-2})}$')
        plt.ylabel('$SFR$')
    return FluxBins,SFR_function



def get_weight_from_sky_area_function(Source,band,sensmap,bins):
    FluxBins,sky_area_function,cutoff_sensitivity = skyarea_function_from_sensmap(sensmap=sensmap,bins=bins)
    f=interpolate.interp1d(FluxBins,sky_area_function,kind='linear')
    weight = 1./f(Source[f'EP_{band}_FLUX'].value.data)
    return weight
    
def sub_get_luminosity_function(Source,band,weight,bins,plot=False):
    # 需要选择好的 source flux
    # 需要 skyarea_function
    # 然后可以比较两种不同的计算方法？ 
    counts,bins = np.histogram(Source[f'EP_{band}_FLUX'], bins=bins,weights=weight)
    if plot==True:
        counts,bins,_  =plt.hist(Source[f'EP_{band}_FLUX'], bins=bins, weights=weight,cumulative=-1,histtype='step')
        plt.xscale('log')
        plt.yscale('log')
        # plt.xlabel('EP_4_FLUX')
        if band==1:
            plt.xlabel(r'$S\mathrm{_{0.2-0.5 keV}~(ergs~s^{-1}~cm^{-2})} $')
        if band==2:
            plt.xlabel(r'$S\mathrm{_{0.5-1.0 keV}~(ergs~s^{-1}~cm^{-2})} $')
        if band==3:
            plt.xlabel(r'$S\mathrm{_{1.0-2.0 keV}~(ergs~s^{-1}~cm^{-2})} $')
        if band==4:
            plt.xlabel(r'$S\mathrm{_{2.0-4.5 keV}~(ergs~s^{-1}~cm^{-2})} $')
        if band==5:
            plt.xlabel(r'$S\mathrm{_{4.5-12.0 keV}~(ergs~s^{-1}~cm^{-2})} $')
        plt.ylabel('$\mathrm{N(>S)/area~(deg^{-2})}$')    
    return counts,bins


from astropy.wcs import WCS
from scipy import interpolate

def filter_source_on_Sensmap(SourceFlux,SkyCoord,Sensmap,Mask=None):
    
#     sensmap_file = 'data/snsmap_EP_02000_04500_ML1.fits'
    # detmsk_file = '/Users/rhuang/Data/XMM/M31_data/edetect_stack/edetect_stack_v8_South_for_simulation/detmskM1.fits.gz'
    sensmap_image = fits.getdata(Sensmap)
    try:
        sensmap_header = fits.getheader(Sensmap)
        sensmap_header['RA_OBJ']
    except KeyError:
        print('No key RA_OBJ found')
        sensmap_header = fits.getheader(f'data/snsmap_EP_02000_04500_ML6.fits')
    if Mask!=None:
        region_mask = fits.getdata(Mask)
        sensmap_image[region_mask==0]=-2
    detmsk_image = sensmap_image==-2
    sensmap_image[sensmap_image==-2]=1e6
    w = WCS(sensmap_header)
    pixel_x,pixel_y = w.world_to_pixel(SkyCoord)
    pixel_x_floor = np.int_(np.floor(pixel_x))
    pixel_y_floor = np.int_(np.floor(pixel_y))

    mask_index = SourceFlux>sensmap_image[pixel_y_floor,pixel_x_floor]
    print(np.sum(mask_index))
    return SourceFlux[mask_index]

    
def skyarea_function_from_sensmap(Sensmap,Mask=None,bins=np.logspace(-20,-10,301),cutoff=0.1,plot=False ):
    # cutoff is the cutoff fraction of the skyarea as the sensitivity limit
    sensmap_image = fits.getdata(Sensmap)
    try:
        sensmap_header = fits.getheader(Sensmap)
        sensmap_header['RA_OBJ']
    except KeyError:
        print('No key RA_OBJ found')
        sensmap_header = fits.getheader(f'data/snsmap_EP_02000_04500_ML6.fits')
    if Mask!=None:
        region_mask = fits.getdata(Mask)
        sensmap_image[region_mask==0]=-2
    detmsk_image = sensmap_image==-2
    w = WCS(sensmap_header)
    sky_area = w.proj_plane_pixel_area()*np.sum(sensmap_image>0)
    counts,bins=np.histogram(sensmap_image[sensmap_image>0],bins=bins, )
    sky_area_function = np.cumsum(counts)*w.proj_plane_pixel_area()
#     log_bins = np.log10(bins)
#     log_bins_mean = (log_bins[:-1]+log_bins[1:])/2.
#     FluxBins = 10**(log_bins_mean)
    FluxBins = np.sqrt(bins[:-1] * bins[1:])
    cutoff_sensitivity = FluxBins[sky_area_function<np.max(sky_area_function)*cutoff][-1]
#     print('cutoff_sensitivity',cutoff_sensitivity) 
    if plot==True:
#         plt.axvline(cutoff_sensitivity,linestyle='--')
        plt.plot(FluxBins,sky_area_function)
        plt.xscale('log')
        plt.xlabel('$S~\mathrm{(ergs~s^{-1}~cm^{-2})}$')
        plt.ylabel('$area~\mathrm{(deg^{2})}$')
    return FluxBins,sky_area_function,cutoff_sensitivity

def get_weight_from_sky_area_function(SourceFlux,Sensmap,Mask=None,bins=np.logspace(-20,-10,301)):
    FluxBins,sky_area_function,cutoff_sensitivity = skyarea_function_from_sensmap(Sensmap=Sensmap,Mask=Mask,bins=bins)
    f=interpolate.interp1d(FluxBins,sky_area_function,kind='linear')
    try:
        weight = 1./f(SourceFlux)
    except:
        weight = 1./f(SourceFlux.value.data)
    return weight
    
def sub_get_luminosity_function(SourceFlux,weight,bins,plot=False):
    # 需要选择好的 source flux
    # 需要 skyarea_function
    # 然后可以比较两种不同的计算方法？ 
    N,bins = np.histogram(SourceFlux, bins=bins,weights=weight)
    if plot==True:
        counts,bins,_  =plt.hist(SourceFlux, bins=bins, weights=weight,cumulative=-1,histtype='step')
        plt.xscale('log')
        plt.yscale('log')
        # plt.xlabel('EP_4_FLUX')
        plt.xlabel(r'$S~\mathrm{(ergs~s^{-1}~cm^{-2})} $')
        plt.ylabel('$\mathrm{N(>S)/area~(deg^{-2})}$')    
    return N,bins


def luminosity_function(SourceFlux,Skycoord,Sensmap,Mask=None,color='k',bins=np.logspace(-20,-10,301),cutoff=0.1,plot=True):
    print('Sensmap:',Sensmap)
    print('Mask:',Mask)
    print('Source Num:',len(SourceFlux))
    Filtered_SourceFlux = filter_source_on_Sensmap(SourceFlux,Skycoord,Sensmap,Mask=Mask)
    print('Remained Source Num:',len(Filtered_SourceFlux))
    FluxBins,sky_area_function,cutoff_sensitivity = skyarea_function_from_sensmap(Sensmap=Sensmap,Mask=Mask,bins=bins,cutoff=cutoff,plot=False)
    print(f'Sensitivity limit: {cutoff_sensitivity:.3e}')
    # plt.show()
    weight = get_weight_from_sky_area_function(Filtered_SourceFlux,Sensmap,Mask=Mask,bins=np.logspace(-20,-10,2001))
#     N,bins = sub_get_luminosity_function(SourceFlux,weight,bins,plot=False)
    dN,bins = np.histogram(Filtered_SourceFlux, bins=bins,weights=weight)
    N = np.cumsum(dN[::-1])[::-1]
    square_dN_err,bins = np.histogram(Filtered_SourceFlux, bins=bins,weights=weight**2)
    square_N_err = np.cumsum(square_dN_err[::-1])[::-1]
    N_err = np.sqrt(square_N_err)
    index = (FluxBins>cutoff_sensitivity) & (N>0)
    if plot==True:
        # N,bins,_  =plt.hist(Filtered_SourceFlux, bins=bins, weights=weight,cumulative=-1,histtype='step')
#         plt.stairs(N[bins[:-1]>cutoff_sensitivity],bins[bins>cutoff_sensitivity])
#         plt.step(FluxBins[FluxBins>cutoff_sensitivity],N[FluxBins>cutoff_sensitivity],where='pre')
        
#         plt.stairs(N,bins)
#         plt.step(FluxBins[index],N[index],where='pre')
        plt.step(FluxBins[index],N[index],where='mid',color=color)
        plt.errorbar(x=FluxBins[index],y=N[index],yerr=N_err[index],color=color,fmt='none')
        plt.xscale('log')
        plt.yscale('log')
        # plt.xlabel('EP_4_FLUX')
        plt.xlabel(r'$S~\mathrm{(ergs~s^{-1}~cm^{-2})} $')
        plt.ylabel('$N(>S)/area~\mathrm{(deg^{-2})}$')    
        plt.gca().set_ylim(ymin=np.min(N[index])*0.5)
    return FluxBins[index],N[index],N_err[index]


def luminosity_function_dict(SourceFlux,Skycoord,Sensmap,Mask=None,color='k',bins=np.logspace(-20,-10,301),cutoff=0.1,plot=True):
    print('Sensmap:',Sensmap)
    print('Mask:',Mask)
    print('Source Num:',len(SourceFlux))
    Filtered_SourceFlux = filter_source_on_Sensmap(SourceFlux,Skycoord,Sensmap,Mask=Mask)
    print('Remained Source Num:',len(Filtered_SourceFlux))
    FluxBins,sky_area_function,cutoff_sensitivity = skyarea_function_from_sensmap(Sensmap=Sensmap,Mask=Mask,bins=bins,cutoff=cutoff,plot=False)
    print(f'Sensitivity limit: {cutoff_sensitivity:.3e}')
    # plt.show()
    weight = get_weight_from_sky_area_function(Filtered_SourceFlux,Sensmap,Mask=Mask,bins=np.logspace(-20,-10,2001))
#     N,bins = sub_get_luminosity_function(SourceFlux,weight,bins,plot=False)
    dN,bins = np.histogram(Filtered_SourceFlux, bins=bins,weights=weight)
    dS = bins[1:]-bins[:-1]
    N = np.cumsum(dN[::-1])[::-1]
    square_dN_err,bins = np.histogram(Filtered_SourceFlux, bins=bins,weights=weight**2)
    square_N_err = np.cumsum(square_dN_err[::-1])[::-1]
    N_err = np.sqrt(square_N_err)
    index = (FluxBins>cutoff_sensitivity) & (N>0)
    if plot==True:
        # N,bins,_  =plt.hist(Filtered_SourceFlux, bins=bins, weights=weight,cumulative=-1,histtype='step')
#         plt.stairs(N[bins[:-1]>cutoff_sensitivity],bins[bins>cutoff_sensitivity])
#         plt.step(FluxBins[FluxBins>cutoff_sensitivity],N[FluxBins>cutoff_sensitivity],where='pre')
        
#         plt.stairs(N,bins)
#         plt.step(FluxBins[index],N[index],where='pre')
        plt.step(FluxBins[index],N[index],where='mid',color=color)
        plt.errorbar(x=FluxBins[index],y=N[index],yerr=N_err[index],color=color,fmt='none')
        plt.xscale('log')
        plt.yscale('log')
        # plt.xlabel('EP_4_FLUX')
        plt.xlabel('$S~\mathrm{(ergs~s^{-1}~cm^{-2})}$')
        plt.ylabel('$N(>S)/area~\mathrm{(deg^{-2})}$')    
        plt.gca().set_ylim(ymin=np.min(N[index])*0.5)
    return {'S':FluxBins,'N':N,'Nerr':N_err,'dN':dN,'dNerr':np.sqrt(square_dN_err),'bins':bins,'dS':dS,'cutoff':cutoff_sensitivity,'sky_area':sky_area_function[-1]}

def luminosity_function_SNR_dict(SourceFlux,Skycoord,Sensmap,Mask=None,color='k',bins=np.logspace(-20,-10,301),cutoff=0.1,minN=10,plot=True):
    print('Sensmap:',Sensmap)
    print('Mask:',Mask)
    print('Source Num:',len(SourceFlux))
    Filtered_SourceFlux = filter_source_on_Sensmap(SourceFlux,Skycoord,Sensmap,Mask=Mask)
    print('Remained Source Num:',len(Filtered_SourceFlux))
    if len(Filtered_SourceFlux)<minN:
        bins=np.logspace(-20,-10,41)
        return luminosity_function_dict(SourceFlux,Skycoord,Sensmap,Mask=Mask,color=color,bins=bins,cutoff=cutoff,plot=plot)
    FluxBins,sky_area_function,cutoff_sensitivity = skyarea_function_from_sensmap(Sensmap=Sensmap,Mask=Mask,bins=bins,cutoff=cutoff,plot=False)
    print(f'Sensitivity limit: {cutoff_sensitivity:.3e}')
    print(f'minimum counts per bin: {minN}')
    # plt.show()
    weight = get_weight_from_sky_area_function(Filtered_SourceFlux,Sensmap,Mask=Mask,bins=np.logspace(-20,-10,2001))
#     N,bins = sub_get_luminosity_function(SourceFlux,weight,bins,plot=False)
    dN,bins = np.histogram(Filtered_SourceFlux, bins=bins)
    ## rebin the data to get a least minN counts
    new_bins = []
    new_counts = []
    count = 0
    bin_sum = 0
    ########## Loop through original bins and counts ##########
    # Find the last non-zero bin
    i = len(dN) - 1
    while i >= 0 and dN[i] == 0:
        i -= 1
    # Initialize variables
    new_bins = [bins[i+1]]
    new_counts = []
    count = dN[i]
    # Loop through remaining bins and counts, in reverse order
    for j in range(i-1, -1, -1):
        count += dN[j]
        # If count exceeds 10, add a new bin
        if count >= minN:
            new_bins.append(bins[j])
            count = 0

    # Add the first non-zero bin
    new_bins.append(bins[0])
    # Reverse the order of the new bins and counts
    new_bins = new_bins[::-1]
    # Convert to numpy arrays
    new_bins = np.array(new_bins)
    ############################################################
    dN,bins = np.histogram(Filtered_SourceFlux, bins=new_bins,weights=weight)
    dS = bins[1:]-bins[:-1]
    N = np.cumsum(dN[::-1])[::-1]
    square_dN_err,bins = np.histogram(Filtered_SourceFlux, bins=bins,weights=weight**2)
    square_N_err = np.cumsum(square_dN_err[::-1])[::-1]
    N_err = np.sqrt(square_N_err)
    ### FluxBins
#     log_bins = np.log10(bins)
#     log_bins_mean = (log_bins[:-1]+log_bins[1:])/2.
#     FluxBins = 10**(log_bins_mean)
    FluxBins = np.sqrt(bins[:-1] * bins[1:])
    ### 
    index = (FluxBins>cutoff_sensitivity) & (N>0)
    if plot==True:
        # N,bins,_  =plt.hist(Filtered_SourceFlux, bins=bins, weights=weight,cumulative=-1,histtype='step')
#         plt.stairs(N[bins[:-1]>cutoff_sensitivity],bins[bins>cutoff_sensitivity])
#         plt.step(FluxBins[FluxBins>cutoff_sensitivity],N[FluxBins>cutoff_sensitivity],where='pre')
        
#         plt.stairs(N,bins)
#         plt.step(FluxBins[index],N[index],where='pre')
        plt.step(FluxBins[index],N[index],where='mid',color=color)
        plt.errorbar(x=FluxBins[index],y=N[index],yerr=N_err[index],color=color,fmt='none')
        plt.xscale('log')
        plt.yscale('log')
        # plt.xlabel('EP_4_FLUX')
        plt.xlabel('$S~\mathrm{(ergs~s^{-1}~cm^{-2})}$')
        plt.ylabel('$N(>S)/area~\mathrm{(deg^{-2})}$')    
        plt.gca().set_ylim(ymin=np.min(N[index])*0.5)
    return {'S':FluxBins,'N':N,'Nerr':N_err,'dN':dN,'dNerr':np.sqrt(square_dN_err),'bins':bins,'dS':dS,'cutoff':cutoff_sensitivity,'sky_area':sky_area_function[-1]}




def skyarea_function_from_sensmap(Sensmap,Mask=None,bins=np.logspace(-20,-10,301),cutoff=0.1,plot=False,header=None, ):
    # cutoff is the cutoff fraction of the skyarea as the sensitivity limit
    sensmap_image = fits.getdata(Sensmap)
    if header == None:
        sensmap_header = fits.getheader(Sensmap)
    else: 
        sensmap_header = header
    try:
        sensmap_header = fits.getheader(Sensmap)
        sensmap_header['RA_OBJ']
    except KeyError:
        print('No key RA_OBJ found')
        sensmap_header = fits.getheader(f'data/snsmap_EP_02000_04500_ML6.fits')

    if Mask!=None:
        region_mask = fits.getdata(Mask)
        sensmap_image[region_mask==0]=-2
    detmsk_image = sensmap_image==-2
    from astropy.wcs import WCS
    w = WCS(sensmap_header)
    # print(w)
    sky_area = w.proj_plane_pixel_area()*np.sum(sensmap_image>0)
    counts,bins=np.histogram(sensmap_image[sensmap_image>0],bins=bins, )
    sky_area_function = np.cumsum(counts)*w.proj_plane_pixel_area()
#     log_bins = np.log10(bins)
#     log_bins_mean = (log_bins[:-1]+log_bins[1:])/2.
#     FluxBins = 10**(log_bins_mean)
    FluxBins = np.sqrt(bins[:-1] * bins[1:])
    FluxBins = bins[1:]
    cutoff_sensitivity = FluxBins[sky_area_function<np.max(sky_area_function)*cutoff][-1]
#     print('cutoff_sensitivity',cutoff_sensitivity) 
    if plot==True:
#         plt.axvline(cutoff_sensitivity,linestyle='--')
        plt.plot(FluxBins,sky_area_function,'.-')
        plt.xscale('log')
        plt.xlabel('$S~\mathrm{(erg~s^{-1}~cm^{-2})}$')
        plt.ylabel('$area~\mathrm{(deg^{2})}$')
    return FluxBins,sky_area_function,cutoff_sensitivity


def luminosity_function_dict(SourceFlux,Skycoord,Sensmap,Mask=None,color='k',bins=np.logspace(-20,-10,301),cutoff=0.1,plot=True):
    print('Sensmap:',Sensmap)
    print('Mask:',Mask)
    print('Source Num:',len(SourceFlux))
    Filtered_SourceFlux = filter_source_on_Sensmap(SourceFlux,Skycoord,Sensmap,Mask=Mask)
    print('Remained Source Num:',len(Filtered_SourceFlux))
    FluxBins,sky_area_function,cutoff_sensitivity = skyarea_function_from_sensmap(Sensmap=Sensmap,Mask=Mask,bins=bins,cutoff=cutoff,plot=False)
    print(f'Sensitivity limit: {cutoff_sensitivity:.3e}')
    # plt.show()
    weight = get_weight_from_sky_area_function(Filtered_SourceFlux,Sensmap,Mask=Mask,bins=np.logspace(-20,-10,2001))
#     N,bins = sub_get_luminosity_function(SourceFlux,weight,bins,plot=False)
    dN,bins = np.histogram(Filtered_SourceFlux, bins=bins,weights=weight)
    dS = bins[1:]-bins[:-1]
    N = np.cumsum(dN[::-1])[::-1]
    square_dN_err,bins = np.histogram(Filtered_SourceFlux, bins=bins,weights=weight**2)
    square_N_err = np.cumsum(square_dN_err[::-1])[::-1]
    N_err = np.sqrt(square_N_err)
    index = (FluxBins>cutoff_sensitivity) & (N>0)
    if plot==True:
        # N,bins,_  =plt.hist(Filtered_SourceFlux, bins=bins, weights=weight,cumulative=-1,histtype='step')
#         plt.stairs(N[bins[:-1]>cutoff_sensitivity],bins[bins>cutoff_sensitivity])
#         plt.step(FluxBins[FluxBins>cutoff_sensitivity],N[FluxBins>cutoff_sensitivity],where='pre')
        
#         plt.stairs(N,bins)
#         plt.step(FluxBins[index],N[index],where='pre')
        plt.step(FluxBins[index],N[index],where='mid',color=color)
        plt.errorbar(x=FluxBins[index],y=N[index],yerr=N_err[index],color=color,fmt='none')
        plt.xscale('log')
        plt.yscale('log')
        # plt.xlabel('EP_4_FLUX')
        plt.xlabel('$S~\mathrm{(erg~s^{-1}~cm^{-2})}$')
        plt.ylabel('$N(>S)/area~\mathrm{(deg^{-2})}$')    
        plt.gca().set_ylim(ymin=np.min(N[index])*0.5)
    return {'S':FluxBins,'N':N,'Nerr':N_err,'dN':dN,'dNerr':np.sqrt(square_dN_err),'bins':bins,'dS':dS,'cutoff':cutoff_sensitivity,'sky_area':sky_area_function[-1]}


import sherpa
from sherpa.data import Data1D
from sherpa.models.basic import PowLaw1D

from sherpa.models.basic import RegriddableModel1D
from sherpa.models.parameter import Parameter
from sherpa.models.basic import PowLaw1D
class MyModel(RegriddableModel1D):
    """A simpler form of the Exp model.

    The model is f(x) = exp(a + b * x).
    """

    def __init__(self, name='mymodel',S_model=[1],dS_model=[1],sky_area_function=[1],model=PowLaw1D):

        self.A = Parameter(name, 'A', 100)
        self.alpha = Parameter(name, 'alpha', 2)
        self.dS_model = dS_model
        self.sky_area_function = sky_area_function
        self.S_model = S_model
        self.model = model
        # The _exp instance is used to perform the model calculation,
        # as shown in the calc method.
#         self._exp = Exp('hidden')
        return RegriddableModel1D.__init__(self, name, (self.A, self.alpha))

#     def aux_data(self, S_model,dS_model,sky_area_function):
#         self.dS_model = dS_model
#         self.sky_area_function = sky_area_function
#         self.S_model = S_model
    
    def calc(self, pars, *args, **kwargs):
        """Calculate the model"""

        # Tell the exp model to evaluate the model, after converting
        # the parameter values to the required form, and order, of:
        # offset, coeff, ampl.
        #
        A = pars[0]
        alpha  = pars[1]
#         print(pars)
        model_dN = self.model(self.S_model,(A,alpha)) * self.dS_model * self.sky_area_function
        rebin_indices = np.floor(np.arange(len(model_dN)) / expanded_times).astype(int)
        new_model_dN = np.bincount(rebin_indices, weights=model_dN)

        return new_model_dN
    
    import sherpa.astro.ui as ui


class LFmodel:
    def __init__(self,S_model,dS_model,sky_area_function,model):
        self.dS_model = dS_model
        self.sky_area_function = sky_area_function
        self.S_model = S_model
        self.model = model
     
    def __call__(self, pars,x):
#         print(pars)
        model_dN = self.model(self.S_model,pars) * self.dS_model * self.sky_area_function
        rebin_indices = np.floor(np.arange(len(model_dN)) / expanded_times).astype(int)
        new_model_dN = np.bincount(rebin_indices, weights=model_dN)
        return new_model_dN
    
def filter_source_on_Sensmap(SourceFlux,SkyCoord,Sensmap,Mask=None):
    
#     sensmap_file = 'data/snsmap_EP_02000_04500_ML1.fits'
    # detmsk_file = '/Users/rhuang/Data/XMM/M31_data/edetect_stack/edetect_stack_v8_South_for_simulation/detmskM1.fits.gz'
    sensmap_image = fits.getdata(Sensmap)
    try:
        sensmap_header = fits.getheader(Sensmap)
        print('RA_OBJ:',sensmap_header['RA_OBJ'])
    except KeyError:
        print('No key RA_OBJ found')
        sensmap_header = fits.getheader(f'data/snsmap_EP_02000_04500_ML6.fits')
    if Mask!=None:
        region_mask = fits.getdata(Mask)
        sensmap_image[region_mask==0]=-2
    detmsk_image = sensmap_image==-2
    sensmap_image[sensmap_image==-2]=1e6
    from astropy.wcs import WCS
    w = WCS(sensmap_header)
    pixel_x,pixel_y = w.world_to_pixel(SkyCoord)
    pixel_x_floor = np.int_(np.floor(pixel_x))
    pixel_y_floor = np.int_(np.floor(pixel_y))
    mask_index = SourceFlux>sensmap_image[pixel_y_floor,pixel_x_floor]
    return SourceFlux[mask_index]
def filter_source_on_Sensmap_index(SourceFlux,SkyCoord,Sensmap,Mask=None):
    
#     sensmap_file = 'data/snsmap_EP_02000_04500_ML1.fits'
    # detmsk_file = '/Users/rhuang/Data/XMM/M31_data/edetect_stack/edetect_stack_v8_South_for_simulation/detmskM1.fits.gz'
    sensmap_image = fits.getdata(Sensmap)
    sensmap_header = fits.getheader(Sensmap)
    if Mask!=None:
        region_mask = fits.getdata(Mask)
        sensmap_image[region_mask==0]=-2
    detmsk_image = sensmap_image==-2
    sensmap_image[sensmap_image==-2]=1e6
    w = WCS(sensmap_header)
    pixel_x,pixel_y = w.world_to_pixel(SkyCoord)
    pixel_x_floor = np.int_(np.floor(pixel_x))
    pixel_y_floor = np.int_(np.floor(pixel_y))
    mask_index = SourceFlux>sensmap_image[pixel_y_floor,pixel_x_floor]
    return mask_index