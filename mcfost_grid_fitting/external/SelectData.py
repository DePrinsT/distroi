"""
File: SelectData.py
Author: Dr. Akke Corporaal and Dr. Jacques Kluska, modified by Toon De Prins
Description: Modification of the original file stored at '/STER/akke/Python/Image2OIFITS' in the local system of
KU Leuven's Institute of Astronomy. Used to read in data from OIFITS files within certain limits.
"""
from mcfost_grid_fitting.external import ReadOIFITS as oifits
import numpy as np


# This is used to select OIFITS data within certain limits and returns it as a ReadOIFITS 'data' object.
# Mostly used to calculate chi2 from observation OIFITS VS model image OIFITS files
def SelectData(data_dir, data_file, wave_1=None, wave_2=None, lim_V2_err=None, lim_V2=None, base_1=None,
               base_2=None, lim_T3_err=None):
    ''' Returns the data that is selected based on:
    wave_1: the lower bound of the wavelength range
    that should be taken into account (in micron)
    wave_2: the upper bound of the wavelength range (in micron)
    lim_V2_err: limit up to which the errors on the visibility should be taken into account
    lim_V2: limit up to which the visibility should be taken into account. Can be used if some
    visibilities are negative.
    base_1 and base_2: lower and upper limit, respectively, of bases taken into account
    (baseline should be in B/lambda)
    lim_T3_err: limit up to which the closure phases should be taken into account
    All defaults are set to False. Set to a number if applicable
    '''
    data = oifits.read(data_dir, data_file, removeFlagged=True)
    if lim_V2_err != None:
        Select_viserr(data, lim_V2_err)

    if lim_V2 != None:
        Select_vis2_lim(data, lim_V2)
    if wave_1 != None and wave_2 != None or wave_1 != None and wave_2 == None or wave_1 == None and wave_2 != None:
        Select_vis_t3_wavelength(data, wave_1, wave_2)
    if lim_T3_err != None:
        Select_T3err(data, lim_T3_err)
    if base_1 != None and base_2 != None or base_1 != None and base_2 == None or base_1 == None and base_2 != None:
        Select_vis_t3_base(data, base_1, base_2)
    return data


def SelectData_data_and_image(data_dir, data_file, img_dir, img_file, wave_1=None, wave_2=None, lim_V2_err=None,
                              lim_V2=None, base_1=None, base_2=None, lim_T3_err=None):
    ''' Returns the data that is selected based on:
    wave_1: the lower bound of the wavelength range
    that should be taken into account (in micron)
    wave_2: the upper bound of the wavelength range (in micron)
    lim_V2_err: limit up to which the errors on the visibility should be taken into account
    lim_V2: limit up to which the visibility should be taken into account. Can be used if some
    visibilities are negative.
    base_1 and base_2: lower and upper limit, respectively, of bases taken into account
    (baseline should be in B/lambda)
    lim_T3_err: limit up to which the closure phases should be taken into account
    All defaults are set to False. Set to a number if applicable
    '''
    data = oifits.read(data_dir, data_file, removeFlagged=True)
    img_data = oifits.read(img_dir, img_file, removeFlagged=True)
    if lim_V2_err != None:
        Select_viserr(data, lim_V2_err, img_data)

    if lim_V2 != None:
        Select_vis2_lim(data, lim_V2, img_data)
    if wave_1 != None and wave_2 != None or wave_1 != None and wave_2 == None or wave_1 == None and wave_2 != None:
        Select_vis_t3_wavelength(data, wave_1, wave_2, img_data)
    if lim_T3_err != None:
        Select_T3err(data, lim_T3_err, img_data)
    if base_1 != None and base_2 != None or base_1 != None and base_2 == None or base_1 == None and base_2 != None:
        Select_vis_t3_base(data, base_1, base_2, img_data)
    return data, img_data


def Select_viserr(data, lim_V2_err, img_data=None):
    try:
        if data.vis2:
            print('Selecting data up to V2err of {}'.format(lim_V2_err))
            for i in np.arange(len(data.vis2)):
                print(i)
                maskv2 = data.vis2[i].vis2err < lim_V2_err
                data.vis2[i].vis2data = data.vis2[i].vis2data[maskv2]

                data.vis2[i].vis2err = data.vis2[i].vis2err[maskv2]
                data.vis2[i].effwave = data.vis2[i].effwave[maskv2]
                data.vis2[i].uf = data.vis2[i].uf[maskv2]
                data.vis2[i].vf = data.vis2[i].vf[maskv2]
                if img_data != None:
                    img_data.vis2[i].vis2data = img_data.vis2[i].vis2data[maskv2]
                    img_data.vis2[i].vis2err = img_data.vis2[i].vis2err[maskv2]
                    img_data.vis2[i].effwave = img_data.vis2[i].effwave[maskv2]
                    img_data.vis2[i].uf = img_data.vis2[i].uf[maskv2]
                    img_data.vis2[i].vf = img_data.vis2[i].vf[maskv2]
        else:
            raise IndexError
    except IndexError:
        print('No OIVIS2 table detected... not setting limits on V2err')

    try:
        if data.vis:
            print('Selecting data up to visibility amplitudes errors of {}'.format(lim_V2_err))
            for i in np.arange(len(data.vis)):
                maskv = data.vis[i].visamp < lim_V2_err
                data.vis[i].visamp = data.vis[i].visamp[maskv]
                data.vis[i].visamperr = data.vis[i].visamperr[maskv]
                data.vis[i].effwave = data.vis[i].effwave[maskv]
                data.vis[i].uf = data.vis[i].uf[maskv]
                data.vis[i].vf = data.vis[i].vf[maskv]
        else:
            raise IndexError
    except IndexError:
        print('No OIVIS table detected ... - not setting limits on visamperr')
    return data, img_data


def Select_vis2_lim(data, lim_V2, img_data=None):
    try:
        if data.vis2:
            print('Selecting data up to V2 of {}'.format(lim_V2))
            for i in np.arange(len(data.vis2)):
                maskv2 = data.vis2[i].vis2data > lim_V2
                data.vis2[i].vis2data = data.vis2[i].vis2data[maskv2]
                data.vis2[i].vis2err = data.vis2[i].vis2err[maskv2]
                data.vis2[i].effwave = data.vis2[i].effwave[maskv2]
                data.vis2[i].uf = data.vis2[i].uf[maskv2]
                data.vis2[i].vf = data.vis2[i].vf[maskv2]
                if img_data != None:
                    img_data.vis2[i].vis2data = img_data.vis2[i].vis2data[maskv2]
                    img_data.vis2[i].vis2err = img_data.vis2[i].vis2err[maskv2]
                    img_data.vis2[i].effwave = img_data.vis2[i].effwave[maskv2]
                    img_data.vis2[i].uf = img_data.vis2[i].uf[maskv2]
                    img_data.vis2[i].vf = img_data.vis2[i].vf[maskv2]
        else:
            raise IndexError
    except IndexError:
        print('No OIVIS2 table detected... not setting limits on V2')

    try:
        if data.vis:
            print('Selecting data up to visibility amplitudes of {}'.format(lim_V2))
            for i in np.arange(len(data.vis)):
                maskv = data.vis[i].visamp > lim_V2
                data.vis[i].visamp = data.vis[i].visamp[maskv]
                data.vis[i].visamperr = data.vis[i].visamperr[maskv]
                data.vis[i].effwave = data.vis[i].effwave[maskv]
                data.vis[i].uf = data.vis[i].uf[maskv]
                data.vis[i].vf = data.vis[i].vf[maskv]
                if img_data != None:
                    img_data.vis[i].visamp = img_data.vis[i].visamp[maskv]
                    img_data.vis[i].visamperr = img_data.vis[i].visamperr[maskv]
                    img_data.vis[i].effwave = img_data.vis[i].effwave[maskv]
                    img_data.vis[i].uf = img_data.vis[i].uf[maskv]
                    img_data.vis[i].vf = img_data.vis[i].vf[maskv]
        else:
            raise IndexError
    except IndexError:
        print('No OIVIS table detected... not setting limits on visamp')
    return data, img_data


def Select_vis_t3_wavelength(data, wave_1, wave_2, img_data=None):
    try:
        if data.vis2:

            for i in np.arange(len(data.vis2)):
                if wave_1 != None and wave_2 == None:
                    C = np.where(data.vis2[i].effwave > wave_1)[0]
                    print('Selecting data from wave {} to max data wavelength {} m'.format(wave_1, wave_2))
                elif wave_1 != None and wave_2 != None:
                    C = np.where((data.vis2[i].effwave < wave_2) & (data.vis2[i].effwave > wave_1))[0]
                    print('Selecting data from wave {} to wave {} m'.format(wave_1, wave_2))
                elif wave_1 == None and wave_2 != None:
                    C = np.where(data.vis2[i].effwave < wave_2)[0]
                    print('Selecting data from min data wavelength to {} m'.format(wave_2))
                data.vis2[i].vis2data = data.vis2[i].vis2data[C]
                data.vis2[i].vis2err = data.vis2[i].vis2err[C]
                data.vis2[i].effwave = data.vis2[i].effwave[C]
                data.vis2[i].uf = data.vis2[i].uf[C]
                data.vis2[i].vf = data.vis2[i].vf[C]
                if img_data != None:
                    img_data.vis2[i].vis2data = img_data.vis2[i].vis2data[C]
                    img_data.vis2[i].vis2err = img_data.vis2[i].vis2err[C]
                    img_data.vis2[i].effwave = img_data.vis2[i].effwave[C]
                    img_data.vis2[i].uf = img_data.vis2[i].uf[C]
                    img_data.vis2[i].vf = img_data.vis2[i].vf[C]
        else:
            raise IndexError
    except IndexError:
        print('No OIVIS2 table detected...')

    try:
        if data.vis:
            for i in np.arange(len(data.vis)):
                if wave_1 != None and wave_2 == None:
                    C = np.where(data.vis[i].effwave > wave_1)[0]
                    print('Selecting data from wave {} to max data wavelength {} m'.format(wave_1, wave_2))
                elif wave_1 != None and wave_2 != None:
                    C = np.where((data.vis[i].effwave < wave_2) & (data.vis[i].effwave > wave_1))[0]
                    print('Selecting data from wave {} to wave {} m'.format(wave_1, wave_2))
                elif wave_1 == None and wave_2 != None:
                    C = np.where(data.vis[i].effwave < wave_2)[0]
                    print('Selecting data from min data wavelength to {} m'.format(wave_2))
                data.vis[i].visamp = data.vis[i].visamp[C]
                data.vis[i].visamperr = data.vis[i].visamperr[C]
                data.vis[i].effwave = data.vis[i].effwave[C]
                data.vis[i].uf = data.vis[i].uf[C]
                data.vis[i].vf = data.vis[i].vf[C]
                if img_data != None:
                    img_data.vis[i].visamp = img_data.vis[i].visamp[C]
                    img_data.vis[i].visamperr = img_data.vis[i].visamperr[C]
                    img_data.vis[i].effwave = img_data.vis[i].effwave[C]
                    img_data.vis[i].uf = img_data.vis[i].uf[C]
                    img_data.vis[i].vf = img_data.vis[i].vf[C]
        else:
            raise IndexError
    except IndexError:
        print('No OIVIS table detected...')

    try:
        if data.t3:
            for i in np.arange(len(data.t3)):
                if wave_1 != None and wave_2 == None:
                    C = np.where(data.t3[i].effwave > wave_1)[0]
                    print('Selecting data from wave {} to max data wavelength {} m'.format(wave_1, wave_2))
                elif wave_1 != None and wave_2 != None:
                    C = np.where((data.t3[i].effwave < wave_2) & (data.t3[i].effwave > wave_1))[0]
                    print('Selecting data from wave {} to wave {} m'.format(wave_1, wave_2))
                elif wave_1 == None and wave_2 != None:
                    C = np.where(data.t3[i].effwave < wave_2)[0]
                    print('Selecting data from min data wavelength to {} m'.format(wave_2))
                data.t3[i].t3amp = data.t3[i].t3amp[C]
                data.t3[i].t3amperr = data.t3[i].t3amperr[C]
                data.t3[i].t3phi = data.t3[i].t3phi[C]
                data.t3[i].t3phierr = data.t3[i].t3phierr[C]
                data.t3[i].effwave = data.t3[i].effwave[C]
                data.t3[i].uf1 = data.t3[i].uf1[C]
                data.t3[i].vf1 = data.t3[i].vf1[C]
                data.t3[i].uf2 = data.t3[i].uf2[C]
                data.t3[i].vf2 = data.t3[i].vf2[C]
                if img_data != None:
                    img_data.t3[i].t3amp = img_data.t3[i].t3amp[C]
                    img_data.t3[i].t3amperr = img_data.t3[i].t3amperr[C]
                    img_data.t3[i].t3phi = img_data.t3[i].t3phi[C]
                    img_data.t3[i].t3phierr = img_data.t3[i].t3phierr[C]
                    img_data.t3[i].effwave = img_data.t3[i].effwave[C]
                    img_data.t3[i].uf1 = img_data.t3[i].uf1[C]
                    img_data.t3[i].vf1 = img_data.t3[i].vf1[C]
                    img_data.t3[i].uf2 = img_data.t3[i].uf2[C]
                    img_data.t3[i].vf2 = img_data.t3[i].vf2[C]
        else:
            raise IndexError
    except IndexError:
        print('No T3 table detected...')
    return data, img_data


def Select_T3err(data, lim_T3_err, img_data=None):
    try:
        if data.t3:
            print('Selecting data up to T3 of {}'.format(lim_T3_err))
            for i in np.arange(len(data.vis2)):
                maskT3 = data.t3[i].t3amperr < lim_T3_err
                data.t3[i].t3amp = data.t3[i].t3amp[maskT3]
                data.t3[i].t3amperr = data.t3[i].t3amperr[maskT3]
                data.t3[i].t3phi = data.t3[i].t3phi[maskT3]
                data.t3[i].t3phierr = data.t3[i].t3phierr[maskT3]
                data.t3[i].effwave = data.t3[i].effwave[maskT3]
                data.t3[i].uf1 = data.t3[i].uf1[maskT3]
                data.t3[i].vf1 = data.t3[i].vf1[maskT3]
                data.t3[i].uf2 = data.t3[i].uf2[maskT3]
                data.t3[i].vf2 = data.t3[i].vf2[maskT3]
                if img_data != None:
                    img_data.t3[i].t3amp = img_data.t3[i].t3amp[maskT3]
                    img_data.t3[i].t3amperr = img_data.t3[i].t3amperr[maskT3]
                    img_data.t3[i].t3phi = img_data.t3[i].t3phi[maskT3]
                    img_data.t3[i].t3phierr = img_data.t3[i].t3phierr[maskT3]
                    img_data.t3[i].effwave = img_data.t3[i].effwave[maskT3]
                    img_data.t3[i].uf1 = img_data.t3[i].uf1[maskT3]
                    img_data.t3[i].vf1 = img_data.t3[i].vf1[maskT3]
                    img_data.t3[i].uf2 = img_data.t3[i].uf2[maskT3]
                    img_data.t3[i].vf2 = img_data.t3[i].vf2[maskT3]
        else:
            raise IndexError
    except IndexError:
        print('No T3table detected...')

    return data, img_data


def Select_vis_t3_base(data, base_1, base_2, img_data=None):
    try:
        if data.vis2:
            for i in np.arange(len(data.vis2)):
                base = np.sqrt(data.vis2[i].uf ** 2 + data.vis2[i].vf ** 2)
                if base_1 != None and base_2 == None:
                    C = np.where(base > base_1)[0]
                    print('Selecting data from data baseline {} to the maximum baseline'.format(base_1))
                elif base_1 != None and base_2 != None:
                    C = np.where((base < base_2) & (base > base_1))[0]
                    print('Selecting data from base {} to base {} m'.format(base_1, base_2))
                elif base_1 == None and base_2 != None:
                    C = np.where(base < base_2)[0]
                    print('Selecting data from min data baseline to {} m'.format(base_2))
                data.vis2[i].vis2data = data.vis2[i].vis2data[C]
                data.vis2[i].vis2err = data.vis2[i].vis2err[C]
                data.vis2[i].effwave = data.vis2[i].effwave[C]
                data.vis2[i].uf = data.vis2[i].uf[C]
                data.vis2[i].vf = data.vis2[i].vf[C]
        else:
            raise IndexError
    except IndexError:
        print('No OIVIS2 table detected... - no limit on vis2 baselines')

    try:
        if data.vis:
            for i in np.arange(len(data.vis)):
                if base_1 != None and base_2 == None:
                    C = np.where(data.vis[i].base > base_1)[0]
                    print('Selecting data from wave {} to max data wavelength {} m'.format(base_1, base_2))
                elif base_1 != None and base_2 != None:
                    C = np.where((data.vis[i].base < base_2) & (data.t3[i].baSe > base_1))[0]
                    print('Selecting data from wave {} to wave {} m'.format(base_1, base_2))
                elif base_1 == None and base_2 != None:
                    C = np.where(data.vis[i].base < base_2)[0]
                    print('Selecting data from min data wavelength to {} m'.format(base_2))
                data.vis[i].visamp = data.vis[i].visamp[C]
                data.vis[i].visamperr = data.vis[i].visamperr[C]
                data.vis[i].effwave = data.vis[i].effwave[C]
                data.vis[i].uf = data.vis[i].uf[C]
                data.vis[i].vf = data.vis[i].vf[C]

        else:
            raise IndexError
    except IndexError:
        print('No OIVIS table detected... - no limit on vis baselines')

    # try:
    #     if data.t3:
    #         for i in np.arange(len(data.t3)):
    #             if wave_1!=None and wave_2==None:
    #                 C=np.where(data.t3[i].effwave>wave_1)[0]
    #                 print('Selecting data from wave {} to max data wavelength {} m'.format(wave_1))
    #             elif wave_1!=None and wave_2!=None:
    #                 C=np.where((data.t3[i].effwave<wave_2) & (data.t3[i].effwave>wave_1))[0]
    #                 print('Selecting data from wave {} to wave {} m'.format(wave_1, wave_2))
    #             elif wave_1==None and wave_2!=None:
    #                 C=np.where(data.t3[i].effwave<wave_2)[0]
    #                 print('Selecting data from min data wavelength to {} m'.format(wave_2))
    #             data.t3[i].t3amp=data.t3[i].t3amp[C]
    #             data.t3[i].t3amperr=data.t3[i].t3amperr[C]
    #             data.t3[i].t3phi=data.t3[i].t3phi[C]
    #             data.t3[i].t3phierr=data.t3[i].t3phierr[C]
    #             data.t3[i].effwave=data.t3[i].effwave[C]
    #             data.t3[i].uf1=data.t3[i].uf1[C]
    #             data.t3[i].vf1=data.t3[i].vf1[C]
    #             data.t3[i].uf2=data.t3[i].uf2[C]
    #             data.t3[i].vf2=data.t3[i].vf2[C]
    #     else:
    #         raise IndexError
    # except IndexError:
    #     print('No T3 table detected...')
    return data
