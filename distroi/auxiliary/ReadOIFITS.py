"""
Provides a class to read in observation data from OIFITS files. This is a modification of the original module stored at
https://github.com/kluskaj/ReadOIFITS on GitHub.
"""

import astropy.io.fits as fits
import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt


# OIFITS READING MODULE

# colour class used by Blender (svn.blender.org)
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Define some functions to print out messages in different colours
def header(msg):
    print(bcolors.HEADER + msg + bcolors.ENDC)


def bold(msg):
    print(bcolors.BOLD + msg + bcolors.ENDC)


def underline(msg):
    print(bcolors.UNDERLINE + msg + bcolors.ENDC)


def inform(msg):
    print(bcolors.OKBLUE + msg + bcolors.ENDC)


def inform2(msg):
    print(bcolors.OKGREEN + msg + bcolors.ENDC)


def warn(msg):
    print(bcolors.WARNING + msg + bcolors.ENDC)


def fail(msg):
    print(bcolors.FAIL + msg + bcolors.ENDC)


# Function to write out stuff to a log text file.
def log(msg, dir):
    f = open(dir + "log.txt", "a")
    f.write(msg + "\n")
    f.close()


# Function that calls the constructor of the 'data' class, which will itself properly read in the data in the argument
# files,and return the resulting 'data' object
def read(dir, files, removeFlagged=True):
    dataset = data(dir, files, removeFlagged=removeFlagged)
    return dataset


# function to flatten out a long (nested) list of values into a single list. note by itself this returns a generator
# i.e. only when the generator is called/forced to give a value (as with the 'extend' commands below) are any actual
# values returned
def flatten(L):
    for l in L:
        if isinstance(l, list):
            yield from flatten(l)
        else:
            yield l


# returns baseline lengths for your V/V2 data and maximum baseline length for the closure phase measurements
def Bases(data):
    # gives you back the bases lengths for V2 and cpres
    u, u1, u2, u3 = data['u']
    v, v1, v2, v3 = data['v']
    base = np.sqrt(u ** 2 + v ** 2)
    B1 = np.sqrt(u1 ** 2 + v1 ** 2)
    B2 = np.sqrt(u2 ** 2 + v2 ** 2)
    B3 = np.sqrt(u3 ** 2 + v3 ** 2)
    Bmax = np.maximum(B1, B2, B3)

    return base, Bmax


# the major class of ReadOIFITS, containing attributes such as the wavelengths, visibilities, ...
# note these are lists, every element of that list will be its own object, i.e. the self.vis2 list's 1st element
# will be an OIVIS2 object (defined below), which will contain all the data corresponding to 1 OI_VIS2 HDU extension
# in an OIFITS file
class data:
    def __init__(self, dir='./', files='*fits', removeFlagged=True):
        self.files = files
        self.dir = dir
        self.target = []  # OITARGET()
        self.wave = []  # OIWAVE() ; You always have the same 6 wavelengths for one V2 + CP observing sequence
        self.vis2 = []  # OIVIS2()
        self.t3 = []  # OIT3()
        self.vis = []  # OIVIS()
        self.array = []  # OIARRAY()
        self.flux = []  # OIFLUX()
        self.read()  # perform class functions defined below on the object itself (to set all the attributes above)
        self.associateWave()
        self.associateFreq()
        self.extendMJD()
        if removeFlagged:
            self.filterFlagged()
        header('Success! \o/')

    # Function to write the data in the 'data' object to a new OIFITS file. Note it does this for all the .fits
    # extensions included (i.e. if there are multiple OI_WAVELENGTH extensions in the original read-in file) by
    # iterating.
    def writeOIFITS(self, dir, file, overwrite=False):

        hdus = []
        # creat primary hdu
        hdr = fits.Header()
        hdr['COMMENT'] = 'Generated with ReadOIFITS'
        primary = fits.PrimaryHDU([], header=hdr)
        hdus.append(primary)
        # write OI_TARGET
        for i in np.arange(len(self.target)):
            hdr = fits.Header()
            hdr['EXTNAME'] = 'OI_TARGET'
            nt = len(self.target[i].target_id)
            targetid = fits.Column(name='TARGET_ID', format='I1', array=self.target[i].target_id)
            target = fits.Column(name='TARGET', format='A32', array=self.target[i].target)
            cols = fits.ColDefs([targetid, target])
            oitarget = fits.BinTableHDU.from_columns(cols, header=hdr)
            hdus.append(oitarget)

        # write OI_ARRAY
        for i in np.arange(len(self.array)):
            hdr = fits.Header()
            hdr['EXTNAME'] = 'OI_ARRAY'
            hdr['ARRNAME'] = self.array[i].arrname
            telname = fits.Column(name='TEL_NAME', format='A16', array=self.array[i].tel_name)
            staname = fits.Column(name='STA_NAME', format='A16', array=self.array[i].sta_name)
            staid = fits.Column(name='STA_INDEX', format='I1', array=self.array[i].sta_index)
            diam = fits.Column(name='DIAMETER', format='E1', array=self.array[i].diameter)
            cols = fits.ColDefs([telname, staname, staid, diam])
            oiarray = fits.BinTableHDU.from_columns(cols, header=hdr)
            hdus.append(oiarray)

        # write OI_WAVELENGTH
        for i in np.arange(len(self.wave)):
            hdr = fits.Header()
            hdr['EXTNAME'] = 'OI_WAVELENGTH'
            hdr['INSNAME'] = self.wave[i].insname
            effwave = fits.Column(name='EFF_WAVE', format='E1', array=self.wave[i].effwave)
            effband = fits.Column(name='EFF_BAND', format='E1', array=self.wave[i].effband)
            cols = fits.ColDefs([effwave, effband])
            oiwave = fits.BinTableHDU.from_columns(cols, header=hdr)
            hdus.append(oiwave)

        # write OI_VIS, note here we must set the dimensionality based on the amount of wavelengths observed (nw)
        for i in np.arange(len(self.vis)):
            hdr = fits.Header()
            hdr['EXTNAME'] = 'OI_VIS'
            hdr['INSNAME'] = self.vis[i].insname
            hdr['ARRNAME'] = self.vis[i].arrname
            hdr['AMPTYP'] = self.vis[i].amptype
            hdr['PHITYP'] = self.vis[i].phitype
            hdr['DATE-OBS'] = self.vis[i].dateobs
            # print(self.vis[i].targetid)
            targetid = fits.Column(name='TARGET_ID', format='1I', array=self.vis[i].targetid)
            mjd = fits.Column(name='MJD', format='1D', array=self.vis[i].mjd[:, 0])
            nw = self.vis[i].visamp.shape[1]
            visamp = fits.Column(name='VISAMP', format=str(nw) + 'D', array=self.vis[i].visamp)
            visamperr = fits.Column(name='VISAMPERR', format=str(nw) + 'D', array=self.vis[i].visamperr)
            visphi = fits.Column(name='VISPHI', format=str(nw) + 'D', array=self.vis[i].visphi)
            visphierr = fits.Column(name='VISPHIERR', format=str(nw) + 'D', array=self.vis[i].visphierr)
            ucoord = fits.Column(name='UCOORD', format='1D', array=self.vis[i].ucoord)
            vcoord = fits.Column(name='VCOORD', format='1D', array=self.vis[i].vcoord)
            staindex = fits.Column(name='STA_INDEX', format='2I', array=self.vis[i].staid)
            flag = fits.Column(name='FLAG', format=str(nw) + 'L', array=self.vis[i].flag)
            cols = fits.ColDefs([targetid, mjd, visamp, visamperr, visphi, visphierr, ucoord, vcoord, staindex, flag])
            oivis = fits.BinTableHDU.from_columns(cols, header=hdr)
            hdus.append(oivis)

        # write OI_VIS2
        for i in np.arange(len(self.vis2)):
            hdr = fits.Header()
            hdr['EXTNAME'] = 'OI_VIS2'
            hdr['INSNAME'] = self.vis2[i].insname
            hdr['ARRNAME'] = self.vis2[i].arrname
            hdr['DATE-OBS'] = self.vis2[i].dateobs
            targetid = fits.Column(name='TARGET_ID', format='1I', array=self.vis2[i].targetid)
            mjd = fits.Column(name='MJD', format='1D', array=self.vis2[i].mjd[:, 0])
            nw = self.vis2[i].vis2data.shape[1]
            vis2 = fits.Column(name='VIS2DATA', format=str(nw) + 'D', array=self.vis2[i].vis2data)
            vis2err = fits.Column(name='VIS2ERR', format=str(nw) + 'D', array=self.vis2[i].vis2err)
            ucoord = fits.Column(name='UCOORD', format='1D', array=self.vis2[i].ucoord)
            vcoord = fits.Column(name='VCOORD', format='1D', array=self.vis2[i].vcoord)
            staindex = fits.Column(name='STA_INDEX', format='2I', array=self.vis2[i].staid)
            flag = fits.Column(name='FLAG', format=str(nw) + 'L', array=self.vis2[i].flag)
            cols = fits.ColDefs([targetid, mjd, vis2, vis2err, ucoord, vcoord, staindex, flag])
            oivis2 = fits.BinTableHDU.from_columns(cols, header=hdr)
            hdus.append(oivis2)

        # write OI_T3, note how STA_INDEX now holds 3 values (we're now dealing with 3 telescopes)
        for i in np.arange(len(self.t3)):
            hdr = fits.Header()
            hdr['EXTNAME'] = 'OI_T3'
            hdr['INSNAME'] = self.t3[i].insname
            hdr['ARRNAME'] = self.t3[i].arrname
            hdr['DATE-OBS'] = self.t3[i].dateobs
            targetid = fits.Column(name='TARGET_ID', format='1I', array=self.t3[i].targetid)
            mjd = fits.Column(name='MJD', format='1D', array=self.t3[i].mjd[:, 0])
            nw = self.t3[i].t3phi.shape[1]
            t3amp = fits.Column(name='T3AMP', format=str(nw) + 'D', array=self.t3[i].t3amp)
            t3amperr = fits.Column(name='T3AMPERR', format=str(nw) + 'D', array=self.t3[i].t3amperr)
            t3phi = fits.Column(name='T3PHI', format=str(nw) + 'D', array=self.t3[i].t3phi)
            t3phierr = fits.Column(name='T3PHIERR', format=str(nw) + 'D', array=self.t3[i].t3phierr)
            u1coord = fits.Column(name='U1COORD', format='1D', array=self.t3[i].u1coord)
            v1coord = fits.Column(name='V1COORD', format='1D', array=self.t3[i].v1coord)
            u2coord = fits.Column(name='U2COORD', format='1D', array=self.t3[i].u2coord)
            v2coord = fits.Column(name='V2COORD', format='1D', array=self.t3[i].v2coord)
            staindex = fits.Column(name='STA_INDEX', format='3I', array=self.t3[i].staid)
            flag = fits.Column(name='FLAG', format=str(nw) + 'L', array=self.t3[i].flag)
            cols = fits.ColDefs(
                [targetid, mjd, t3amp, t3amperr, t3phi, t3phierr, u1coord, v1coord, u2coord, v2coord, staindex, flag])
            oit3 = fits.BinTableHDU.from_columns(cols, header=hdr)
            hdus.append(oit3)

        # Write OI_FLUX
        for i in np.arange(len(self.flux)):
            hdr = fits.Header()
            hdr['EXTNAME'] = 'OI_FLUX'
            hdr['INSNAME'] = self.flux[i].insname
            hdr['ARRNAME'] = self.flux[i].arrname
            hdr['DATE-OBS'] = self.flux[i].dateobs
            hdr['CALSTAT'] = self.flux[i].calstat
            targetid = fits.Column(name='TARGET_ID', format='1I', array=self.flux[i].targetid)
            mjd = fits.Column(name='MJD', format='1D', array=self.flux[i].mjd)
            nw = self.flux[i].fluxdata.shape[1]
            flux = fits.Column(name='FLUXDATA', format=str(nw) + 'D', array=self.flux[i].fluxdata)
            fluxerr = fits.Column(name='FLUXERR', format=str(nw) + 'D', array=self.flux[i].fluxerr)
            staindex = fits.Column(name='STA_INDEX', format='2I', array=self.flux[i].staid)
            flag = fits.Column(name='FLAG', format=str(nw) + 'L', array=self.flux[i].flag)
            cols = fits.ColDefs([targetid, mjd, flux, fluxerr, staindex, flag])
            oiflux = fits.BinTableHDU.from_columns(cols, header=hdr)
            hdus.append(oiflux)

        # Create a new oifits file
        hdu = fits.HDUList(hdus)
        hdu.writeto(dir + file, overwrite=overwrite)

    # Function to plot the squared visibilities, closure phases and UV coverage. Some clipping limits and axis limits
    # defined as well. Note there's a lot of confusing 1e+/-6 factors in here which I haven't figured out, but the
    # plots look alright.
    def plotV2CP(self, save=False, name='Data.pdf', V2sigclip=1, CPsigclip=180, Blim=0, CPext=200, V2min=0.0, V2max=1.0,
                 xlog=False, ylog=False, lines=True):
        # Plot the vis2 and cp from the data and the model
        data = self.givedataJK()

        # Actual plot
        fig, (ax1, ax2) = plt.subplots(1, 2)

        fig.subplots_adjust(right=0.8)
        cax = fig.add_axes([0.85, 0.15, 0.02, 0.7])

        # Don't know what the lines=True stands for, seems like code executed is very similarly between the 2 options
        # except lines=False seems to include errorbars on the data and doesn't do a scatterplot. I suspect it plots
        # every baseline separately, while lines=False produces scatterplots (doesn't care which baseline points
        # belong to).
        if lines:
            V2data, V2err, u, v, waveV2 = ListV2(data)
            CPdata, CPerr, u1, v1, u2, v2, u3, v3, waveCP = ListCP(data)

            base = []
            for ui, vi, wavei in zip(u, v, waveV2):
                basei = np.sqrt(np.power(ui, 2) + np.power(vi, 2)) / wavei
                base.append(basei * 1e-6)  # express V2 baseline lengths in MegaLambda

            b1, b2, b3, Bmax = [], [], [], []
            for u1i, v1i, u2i, v2i, u3i, v3i, wavei in zip(u1, v1, u2, v2, u3, v3, waveCP):
                b1i = np.sqrt(np.power(u1i, 2) + np.power(v1i, 2)) / wavei
                b2i = np.sqrt(np.power(u2i, 2) + np.power(v2i, 2)) / wavei
                b3i = np.sqrt(np.power(u3i, 2) + np.power(v3i, 2)) / wavei
                Bmaxi = np.maximum(b1i, b2i, b3i)
                Bmax.append(Bmaxi * 1e-6)  # express max baseline length in triangle in MegaLambda

            for basei, V2i in zip(base, V2data):
                ax1.plot(basei, V2i)  # plot V2 curves

        else:
            # after Load(data) seems we don't need to convert to MegaLambda anymore
            waveV2, waveCP, base, Bmax, V2data, V2err, CPdata, CPerr = Load(data)
            maskv2 = V2err < V2sigclip  # V2err < 0.2 and
            maskcp = CPerr < CPsigclip

            sc = ax1.scatter(base[maskv2], V2data[maskv2], s=0.1, c=waveV2[maskv2], cmap='gist_rainbow_r')
            clb = fig.colorbar(sc, cax=cax)
            clb.set_label(r'Wavelength ($\mu$m)', rotation=270, labelpad=15)
            a, b, c = ax1.errorbar(base[maskv2], V2data[maskv2], yerr=V2err[maskv2], marker='', elinewidth=0.05, ls='',
                                   zorder=0)

        if Blim == 0:
            if lines:
                maxi = []
                for b in base:
                    maxi.append(np.max(b))
                Blim = 1.05 * max(maxi)
            else:
                Blim = 1.05 * np.max(base)

        ax1.axhline(0)
        ax1.set_ylim(V2min, V2max)
        ax1.set_xlim(0, Blim)

        if xlog:
            ax1.set_xlim(min(base) - 1, Blim)
            ax1.set_xscale('log')
        if ylog:
            ax1.set_ylim(0.9 * np.min(np.abs(V2data)), 1)
            ax1.set_yscale('log')

        ax1.set_xlabel(r'B (M$\lambda$)', fontsize=8)
        ax1.set_title('Squared visibilities')

        ax2.axhline(y=0, ls='--', c='grey', lw=0.3)
        if lines:
            for bmaxi, CPi in zip(Bmax, CPdata):
                ax2.plot(bmaxi, CPi)
        else:
            sc = ax2.scatter(Bmax[maskcp], CPdata[maskcp], s=0.1, c=waveCP[maskcp], cmap='gist_rainbow_r')
            a2, b2, c2 = ax2.errorbar(Bmax[maskcp], CPdata[maskcp], yerr=CPerr[maskcp], elinewidth=0.05, marker='',
                                      ls='', zorder=0)

        ax2.set_xlabel(r'B$_\mathrm{max}$ (M$\lambda$)', fontsize=8)
        ax2.set_title('Closure phases')
        ax2.set_xlim(0, Blim)
        ax2.set_ylim(-CPext, CPext)
        if xlog:
            ax2.set_xlim(min(Bmax) - 1, Blim)
            ax2.set_xscale('log')

        if save:
            plt.savefig(dir + name + '_Data.pdf')
        else:
            plt.show()

        u, u1, u2, u3 = data['u']
        v, v1, v2, v3 = data['v']

        # Plot of the UV coverage
        fig, ax = plt.subplots(1, 1)
        cax = fig.add_axes(ax)
        if lines:
            waveV2, waveCP, base, Bmax, V2data, V2err, CPdata, CPerr = Load(data)  # Don't know why Load(data) is called
            # when it wasn't previously for the V2 plot if lines=True (there lines=False used Load(data) instead).
            base /= waveV2
            u /= waveV2
            v /= waveV2
        else:
            u *= 1e-6
            v *= 1e-6
        sc = ax.scatter([u, -u], [v, -v], c=[waveV2, waveV2], s=0.1, cmap='gist_rainbow_r')
        clb = fig.colorbar(sc)
        clb.set_label(r'Wavelength ($\mu$m)', rotation=270, labelpad=15)
        ax.set_xlabel('u (M$\lambda$)', fontsize=8)
        ax.set_ylabel('v (M$\lambda$)', fontsize=8)
        ax.set_title('uv plane')
        ax.set_ylim(-Blim, Blim)
        ax.set_xlim(Blim, -Blim)
        fig.tight_layout()

        if save:
            plt.savefig(dir + name + '_Datauv.pdf')
        else:
            plt.show()

        plt.close()

    # Function to filter out flagged data points, work on OIVIS, OIVIS2 and OIT3 tables in the .fits file
    def filterFlagged(self):
        inform('Removing flagged data...')
        # OIVIS
        for i in np.arange(len(self.vis)):
            vis = self.vis[i]
            flag = np.logical_not(vis.flag)
            vis.mjd = vis.mjd[flag]
            vis.visamp = vis.visamp[flag]
            vis.visphi = vis.visphi[flag]
            vis.visamperr = vis.visamperr[flag]
            vis.visphierr = vis.visphierr[flag]
            vis.effwave = vis.effwave[flag]
            vis.uf = vis.uf[flag]
            vis.vf = vis.vf[flag]
        # OIVIS2
        for i in np.arange(len(self.vis2)):
            vis2 = self.vis2[i]
            flag = np.logical_not(vis2.flag)
            vis2.mjd = vis2.mjd[flag]
            vis2.vis2data = vis2.vis2data[flag]
            vis2.vis2err = vis2.vis2err[flag]
            vis2.effwave = vis2.effwave[flag]
            vis2.uf = vis2.uf[flag]
            vis2.vf = vis2.vf[flag]
        # OIT3
        for i in np.arange(len(self.t3)):
            t3 = self.t3[i]
            flag = np.logical_not(t3.flag)
            t3.mjd = t3.mjd[flag]
            t3.t3amp = t3.t3amp[flag]
            t3.t3amperr = t3.t3amperr[flag]
            t3.t3phi = t3.t3phi[flag]
            t3.t3phierr = t3.t3phierr[flag]
            t3.effwave = t3.effwave[flag]
            t3.uf1 = t3.uf1[flag]
            t3.vf1 = t3.vf1[flag]
            t3.uf2 = t3.uf2[flag]
            t3.vf2 = t3.vf2[flag]

    # This iterates over tables, e.g. OIVIS tables, and extends the MJD arrays to 2D so an MJD is associated to every
    # data point, including in the spectral direction.
    def extendMJD(self):
        inform('Assigning mjd...')
        # OIVIS
        for i in np.arange(len(self.vis)):
            mjd = []
            mjd0 = self.vis[i].mjd
            effwave = self.vis[i].effwave
            for j in np.arange(len(mjd0)):
                mjd.append(np.full(len(effwave[j]), mjd0[j]))
            self.vis[i].mjd = np.array(mjd)
        # OIVIS2
        for i in np.arange(len(self.vis2)):
            mjd = []
            mjd0 = self.vis2[i].mjd
            effwave = self.vis2[i].effwave
            for j in np.arange(len(mjd0)):
                mjd.append(np.full(len(effwave[j]), mjd0[j]))
            self.vis2[i].mjd = np.array(mjd)
        # OIT3
        for i in np.arange(len(self.t3)):
            mjd = []
            mjd0 = self.t3[i].mjd
            effwave = self.t3[i].effwave
            for j in np.arange(len(mjd0)):
                mjd.append(np.full(len(effwave[j]), mjd0[j]))
            self.t3[i].mjd = np.array(mjd)

    # Function not used by any other methods
    def giveV2(self, removeflagged=True):
        if self.vis2 == []:
            fail('There is no V2 data in the files')
        else:
            V2, V2e, u, v, lam, mjd = [], [], [], [], [], []
            for data in self.vis2:
                flag = np.logical_not(data.flag)
                V2i = data.vis2data
                V2erri = data.vis2err
                ui = data.uf
                vi = data.vf
                mjdi = data.mjd
                lami = data.effwave

                if removeflagged:
                    V2.append(V2i[flag])
                    V2e.append(V2erri[flag])
                    u.append(ui[flag])
                    v.append(vi[flag])
                    mjd.append(mjdi[flag])
                    lam.append(lami[flag])
                else:
                    V2.append(V2i)
                    V2e.append(V2erri)
                    u.append(ui)
                    v.append(vi)
                    mjd.append(mjdi)
                    lam.append(lami)
        return V2, V2e, u, v, lam, mjd

    # Returns a dictionary
    def givedataJK(self):
        dataJK = {}
        # OIVIS
        u, v, wave, visamp, visamperr, visphi, visphierr = [], [], [], [], [], [], []
        if self.vis != []:
            for i in np.arange(len(self.vis)):
                # fetching
                ui = self.vis[i].uf
                vi = self.vis[i].vf
                wavei = self.vis[i].effwave
                visampi = self.vis[i].visamp
                visamperri = self.vis[i].visamperr
                visphii = self.vis[i].visphi
                visphierri = self.vis[i].visphierr
                # formatting (fully flatten array?)
                ui = flatten(ui)
                vi = flatten(vi)
                wavei = flatten(wavei)
                visampi = flatten(visampi)
                visamperri = flatten(visamperri)
                visphii = flatten(visphii)
                visphierri = flatten(visphierri)
                # loading, i.e. by extending lists containing all values
                u.extend(ui)
                v.extend(vi)
                wave.extend(wavei)
                visamp.extend(visampi)
                visamperr.extend(visamperri)
                visphi.extend(visphii)
                visphierr.extend(visphierri)
            # flattening and np.arraying (why flatten again? I'm not sure)
            u = np.array(list(flatten(u)))
            v = np.array(list(flatten(v)))
            wave = np.array(list(flatten(wave)))
            visamp = np.array(list(flatten(visamp)))
            visphi = np.array(list(flatten(visphi)))
            visamperr = np.array(list(flatten(visamperr)))
            visphierr = np.array(list(flatten(visphierr)))
            try:
                u = np.concatenate(u)
                v = np.concatenate(v)
                wave = np.concatenate(wave)
                visamp = np.concatenate(visamp)
                visphi = np.concatenate(visphi)
                visamperr = np.concatenate(visamperr)
                visphierr = np.concatenate(visphierr)
            except:
                pass
            # writing in the dictionnary
            dataJK['uvV'] = (u.flatten(), v.flatten())
            dataJK['waveV'] = wave.flatten()
            # vis = {}
            # vis['visamp'], vis['visamperr'], vis['visphi'], vis['visphierr'] = visamp, visamperr, visphi, visphierr
            dataJK['vis'] = (visamp.flatten(), visamperr.flatten(), visphi.flatten(), visphierr.flatten())
        else:
            # writing in the dictionnary
            dataJK['uvV'] = (np.array([]), np.array([]))
            dataJK['waveV'] = np.array([])
            # vis = {}
            # vis['visamp'], vis['visamperr'], vis['visphi'], vis['visphierr'] = visamp, visamperr, visphi, visphierr
            dataJK['vis'] = (np.array([]), np.array([]), np.array([]), np.array([]))

        # OIVIS2
        u, v, wave, vis2, vis2err = [], [], [], [], []
        u1, v1, u2, v2, u3, v3, wavecp, t3phi, t3phierr = [], [], [], [], [], [], [], [], []
        if (self.vis2 != [] and self.t3 != []):
            for i in np.arange(len(self.vis2)):
                # fetching
                ui = self.vis2[i].uf
                vi = self.vis2[i].vf
                wavei = self.vis2[i].effwave
                vis2i = self.vis2[i].vis2data
                vis2erri = self.vis2[i].vis2err
                # formatting
                ui = flatten(ui)
                vi = flatten(vi)
                wavei = flatten(wavei)
                vis2i = flatten(vis2i)
                vis2erri = flatten(vis2erri)
                # loading
                u.extend(ui)
                v.extend(vi)
                wave.extend(wavei)
                vis2.extend(vis2i)
                vis2err.extend(vis2erri)
            # flattening and np.arraying
            u = np.array(list(flatten(u)))
            v = np.array(list(flatten(v)))
            wave = np.array(list(flatten(wave)))
            vis2 = np.array(list(flatten(vis2)))
            vis2err = np.array(list(flatten(vis2err)))
            try:
                u = np.concatenate(u)
                v = np.concatenate(v)
                wave = np.concatenate(wave)
                vis2 = np.concatenate(vis2)
                vis2err = np.concatenate(vis2err)
            except:
                pass

            # OIT3
            for i in np.arange(len(self.t3)):
                # fetching
                u1i = self.t3[i].uf1
                v1i = self.t3[i].vf1
                u2i = self.t3[i].uf2
                v2i = self.t3[i].vf2
                u3i, v3i = [], []
                for x1, x2, y1, y2 in zip(u1i, u2i, v1i, v2i):
                    u3i.extend([x1 + x2])  # calculating the 3d closure baseline from the other 2. IS THIS WRONG?
                    v3i.extend([y1 + y2])
                u3i = np.reshape(u3i, np.array(u1i).shape)
                v3i = np.reshape(v3i, np.array(v1i).shape)
                wavecpi = self.t3[i].effwave
                t3phii = self.t3[i].t3phi
                t3phierri = self.t3[i].t3phierr
                # formatting
                u1i = flatten(u1i)
                v1i = flatten(v1i)
                u2i = flatten(u2i)
                v2i = flatten(v2i)
                u3i = flatten(u3i)
                v3i = flatten(v3i)
                wavecpi = flatten(wavecpi)
                t3phii = flatten(t3phii)
                t3phierri = flatten(t3phierri)
                # loading
                u1.extend(u1i)
                v1.extend(v1i)
                u2.extend(u2i)
                v2.extend(v2i)
                u3.extend(u3i)
                v3.extend(v3i)
                wavecp.extend(wavecpi)
                t3phi.extend(t3phii)
                t3phierr.extend(t3phierri)

            # flattening and np.arraying
            u1 = np.array(list(flatten(u1)))
            v1 = np.array(list(flatten(v1)))
            u2 = np.array(list(flatten(u2)))
            v2 = np.array(list(flatten(v2)))
            u3 = np.array(list(flatten(u3)))
            v3 = np.array(list(flatten(v3)))
            wavecp = np.array(list(flatten(wavecp)))
            t3phi = np.array(list(flatten(t3phi)))
            t3phierr = np.array(list(flatten(t3phierr)))
            try:
                u1 = np.concatenate(u1)
                v1 = np.concatenate(v1)
                u2 = np.concatenate(u2)
                v2 = np.concatenate(v2)
                u3 = np.concatenate(u3)
                v3 = np.concatenate(v3)
                wavecp = np.concatenate(wavecp)
                t3phi = np.concatenate(t3phi)
                t3phierr = np.concatenate(t3phierr)
            except:
                pass

            # writing in the dictionnary
            dataJK['u'] = (u.flatten(), u1.flatten(), u2.flatten(), u3.flatten())
            dataJK['v'] = (v.flatten(), v1.flatten(), v2.flatten(), v3.flatten())
            dataJK['wave'] = (wave.flatten(), wavecp.flatten())
            dataJK['v2'] = (vis2.flatten(), vis2err.flatten())
            dataJK['cp'] = (t3phi.flatten(), t3phierr.flatten())
        else:
            # writing in the dictionnary
            dataJK['u'] = (np.array([]), np.array([]), np.array([]), np.array([]))
            dataJK['v'] = (np.array([]), np.array([]), np.array([]), np.array([]))
            dataJK['wave'] = (np.array([]), np.array([]))
            dataJK['v2'] = (np.array([]), np.array([]))
            dataJK['cp'] = (np.array([]), np.array([]))

        return dataJK

    # Function to associate spatial frequencies in radian to data points (instead of baseline lengths). These are stored
    # in .uf attributes of the table classes. I.e. spatial frequency associated to 3d baseline in OIT3 table is uf3, vf3
    # Note that these are not properties associated to e.g. the OIVIS class at initilaization (this function adds them)
    def associateFreq(self):
        inform('Assigning spatial frequencies...')
        # OIVIS
        for i in np.arange(len(self.vis)):
            uf, vf = [], []
            u = self.vis[i].ucoord
            v = self.vis[i].vcoord
            effwave = self.vis[i].effwave
            for j in np.arange(len(u)):
                uf.append(u[j] / effwave[j])
                vf.append(v[j] / effwave[j])
            self.vis[i].uf = np.array(uf)
            self.vis[i].vf = np.array(vf)
            self.vis[i].base = np.sqrt(np.array(uf) ** 2 + np.array(vf) ** 2)
        # OIVIS2
        for i in np.arange(len(self.vis2)):
            uf, vf = [], []
            u = self.vis2[i].ucoord
            v = self.vis2[i].vcoord
            effwave = self.vis2[i].effwave
            for j in np.arange(len(u)):
                uf.append(u[j] / effwave[j])
                vf.append(v[j] / effwave[j])
            self.vis2[i].uf = np.array(uf)
            self.vis2[i].vf = np.array(vf)
            self.vis2[i].base = np.sqrt(np.array(uf) ** 2 + np.array(vf) ** 2)
        # OIT3
        for i in np.arange(len(self.t3)):
            uf1, vf1, uf2, vf2 = [], [], [], []
            u1 = self.t3[i].u1coord
            v1 = self.t3[i].v1coord
            u2 = self.t3[i].u2coord
            v2 = self.t3[i].v2coord
            effwave = self.t3[i].effwave
            for j in np.arange(len(u1)):
                uf1.append(u1[j] / effwave[j])
                vf1.append(v1[j] / effwave[j])
                uf2.append(u2[j] / effwave[j])
                vf2.append(v2[j] / effwave[j])
            self.t3[i].uf1 = np.array(uf1)
            self.t3[i].vf1 = np.array(vf1)
            self.t3[i].uf2 = np.array(uf2)
            self.t3[i].vf2 = np.array(vf2)

    def associateWave(self):
        inform('Assigning wavelengths...')
        # fetch the wavelengths from OIWAVE
        waveid = {}
        for i in np.arange(len(self.wave)):
            name = self.wave[i].insname
            waveid[name] = i

        # Associate the right waves to OIVIS
        for i in np.arange(len(self.vis)):
            name = self.vis[i].insname
            try:
                id = waveid[name]
                effwave = self.wave[id].effwave
                wave = []
                for j in np.arange(self.vis[i].visamp.shape[0]):
                    wave.append(effwave)
                self.vis[i].effwave = np.array(wave)
            except:
                fail('No wavetable corresponding to {} found...'.format(name))

        # Associate the right waves to OIVIS2
        for i in np.arange(len(self.vis2)):
            name = self.vis2[i].insname
            try:
                id = waveid[name]
                effwave = self.wave[id].effwave
                wave = []
                for j in np.arange(self.vis2[i].vis2data.shape[0]):
                    wave.append(effwave)
                self.vis2[i].effwave = np.array(wave)
            except:
                fail('No wavetable corresponding to {} found...'.format(name))

        # Associate the right waves to OIT3
        for i in np.arange(len(self.t3)):
            name = self.t3[i].insname
            try:
                id = waveid[name]
                effwave = self.wave[id].effwave
                wave = []
                for j in np.arange(self.t3[i].t3phi.shape[0]):
                    wave.append(effwave)
                self.t3[i].effwave = np.array(wave)
            except:
                fail('No wavetable corresponding to {} found...'.format(name))

        # Associate the right waves to OIFLUX
        for i in np.arange(len(self.flux)):
            name = self.flux[i].insname
            try:
                id = waveid[name]
                effwave = self.wave[id].effwave
                wave = []
                for j in np.arange(self.flux[i].fluxdata.shape[0]):
                    wave.append(effwave)
                self.flux[i].effwave = np.array(wave)
            except:
                fail('No wavetable corresponding to {} found...'.format(name))

    def read(self):
        header('Reading from {}{}'.format(self.dir, self.files))
        dir = self.dir
        files = self.files
        listOfFiles = os.listdir(dir)
        i = 0
        for entry in listOfFiles:
            if fnmatch.fnmatch(entry, files):
                i += 1
                inform('Reading ' + entry + '...')
                self.readfile(entry)

    def readfile(self, file):
        hdul = fits.open(self.dir + file)
        err = False
        i = 0
        while err == False:
            i += 1
            try:
                extname = hdul[i].header['EXTNAME']
                print('Reading ' + extname)
                if extname == 'OI_TARGET':
                    self.readTARGET(hdul[i])
                elif extname == 'OI_ARRAY':
                    self.readARRAY(hdul[i])
                elif extname == 'OI_WAVELENGTH':
                    self.readWAVE(hdul[i])
                elif extname == 'OI_VIS':
                    self.readVIS(hdul[i])
                elif extname == 'OI_VIS2':
                    self.readVIS2(hdul[i])
                elif extname == 'OI_T3':
                    self.readT3(hdul[i])
                elif extname == 'OI_FLUX':
                    self.readFLUX(hdul[i])
            except IndexError:
                err = True

    def readTARGET(self, hd):
        target_id = hd.data['TARGET_ID']
        target = hd.data['TARGET']
        tar = OITARGET(target_id=target_id, target=target)
        self.target.append(tar)

    def readARRAY(self, hd):
        arrname = hd.header['ARRNAME']
        tel = hd.data['TEL_NAME']
        sta = hd.data['STA_NAME']
        staid = hd.data['STA_INDEX']
        diam = hd.data['DIAMETER']
        arr = OIARRAY(arrname=arrname, tel_name=tel, sta_name=sta, sta_index=staid, diameter=diam)
        self.array.append(arr)

    def readVIS2(self, hd):
        insname = hd.header['INSNAME']
        arrname = hd.header['ARRNAME']
        dateobs = hd.header['DATE-OBS']
        targetid = hd.data['TARGET_ID']
        mjd = hd.data['MJD']
        vis2data = hd.data['VIS2DATA']
        vis2err = hd.data['VIS2ERR']
        u = hd.data['UCOORD']
        v = hd.data['VCOORD']
        sta = hd.data['STA_INDEX']
        flag = hd.data['FLAG']
        vis2 = OIVIS2(arrname, insname, dateobs=dateobs, mjd=mjd, vis2data=vis2data, vis2err=vis2err, ucoord=u,
                      vcoord=v, flag=flag, targetid=targetid, staid=sta)
        self.vis2.append(vis2)

    def readT3(self, hd):
        insname = hd.header['INSNAME']
        arrname = hd.header['ARRNAME']
        dateobs = hd.header['DATE-OBS']
        t3amp = hd.data['T3AMP']
        t3phi = hd.data['T3PHI']
        t3amperr = hd.data['T3AMPERR']
        t3phierr = hd.data['T3PHIERR']
        mjd = hd.data['MJD']
        targetid = hd.data['TARGET_ID']
        u1 = hd.data['U1COORD']
        u2 = hd.data['U2COORD']
        v1 = hd.data['V1COORD']
        v2 = hd.data['V2COORD']
        staid = hd.data['STA_INDEX']
        flag = hd.data['FLAG']
        T3 = OIT3(arrname, insname, dateobs=dateobs, mjd=mjd, t3amp=t3amp, t3amperr=t3amperr, t3phi=t3phi,
                  t3phierr=t3phierr, u1coord=u1, v1coord=v1, u2coord=u2, v2coord=v2, flag=flag, targetid=targetid,
                  staid=staid)
        self.t3.append(T3)

    def readVIS(self, hd):
        insname = hd.header['INSNAME']
        arrname = hd.header['ARRNAME']
        try:
            amptype = hd.header['AMPTYP']
            phitype = hd.header['PHITYP']
        except KeyError:
            amptype = 'unknown'
            phitype = 'unknown'
        dateobs = hd.header['DATE-OBS']
        mjd = hd.data['MJD']
        targetid = hd.data['TARGET_ID']
        visamp = hd.data['VISAMP']
        visamperr = hd.data['VISAMPERR']
        visphi = hd.data['VISPHI']
        visphierr = hd.data['VISPHIERR']
        u = hd.data['UCOORD']
        v = hd.data['VCOORD']
        staid = hd.data['STA_INDEX']
        flag = hd.data['FLAG']
        VIS = OIVIS(arrname, insname, amptype=amptype, phitype=phitype, dateobs=dateobs, mjd=mjd, visamp=visamp,
                    visamperr=visamperr, visphi=visphi, visphierr=visphierr, ucoord=u, vcoord=v, flag=flag,
                    targetid=targetid, staid=staid)
        self.vis.append(VIS)

    def readWAVE(self, hd):
        insname = hd.header['INSNAME']
        effwave = np.array(hd.data['EFF_WAVE'])
        effband = np.array(hd.data['EFF_BAND'])
        wave0 = OIWAVE(insname, effwave=effwave, effband=effband)
        self.wave.append(wave0)
        # print(self.wave, self.wave[0].effwave)

    def readFLUX(self, hd):
        dateobs = hd.header['DATE-OBS']
        insname = hd.header['INSNAME']
        arrname = hd.header['ARRNAME']
        calstat = hd.header['CALSTAT']
        targetid = hd.data['TARGET_ID']
        mjd = hd.data['MJD']
        try:
            flux = hd.data['FLUXDATA']
        except KeyError:
            flux = hd.data['FLUX']  # for GRAVITY
        fluxerr = hd.data['FLUXERR']
        staid = hd.data['STA_INDEX']
        flag = hd.data['FLAG']
        fl = OIFLUX(insname, arrname, calstat=calstat, dateobs=dateobs, mjd=mjd, fluxdata=flux, fluxerr=fluxerr,
                    flag=flag, staid=staid, targetid=targetid)
        self.flux.append(fl)


class OITARGET:
    def __init__(self, target_id=[], target=[]):
        self.target_id = target_id
        self.target = target

    def addtarget(self, target, target_id):
        self.target.extend(target)
        self.target_id.extend(target_id)

    def printtarget(self):
        for t, i in zip(self.target, self.target_id):
            inform('Target #{} is {}'.format(i, t))

    def givetargetid(self):
        return self.target, self.target_id

    def givetarget(self):
        return self.target

    def giveid(self):
        return self.target_id

    def givetheid(self, target):
        id = []
        for t, i in zip(self.target, self.target_id):
            if t == target:
                id.extend(i)
        return id

    def givethetarget(self, id):
        tar = []
        for t, i in zip(self.target, self.target_id):
            if i == id:
                tar.extend(t)
        return tar


class OIARRAY:
    def __init__(self, arrname='UNKNOWN', tel_name=[], sta_name=[], sta_index=[], diameter=[]):
        self.arrname = arrname
        self.tel_name = tel_name
        self.sta_name = sta_name
        self.sta_index = sta_index
        self.diameter = diameter

    def addarray(self, arrname, tel_name, sta_name, sta_index, diameter):
        self.arrname.extend(arrname)
        self.tel_name.extend(tel_name)
        self.sta_name.extend(sta_name)
        self.sta_index.extend(sta_index)
        self.diameter.extend(diameter)


class OIWAVE:
    def __init__(self, insname, effwave=[], effband=[]):
        self.insname = insname
        self.effwave = effwave
        self.effband = effband

    def addwave(self, insname, effwave, effband):
        self.effwave.extend(effwave)
        self.effband.extend(effband)
        self.insname.extend(insname)


class OIVIS2:
    def __init__(self, arrname, insname, dateobs=0, mjd=[], vis2data=[], vis2err=[], ucoord=[], vcoord=[], flag=[],
                 staid=[], targetid=[]):
        self.arrname = arrname
        self.insname = insname
        self.dateobs = dateobs
        self.mjd = mjd
        self.vis2data = vis2data
        self.vis2err = vis2err
        self.ucoord = ucoord
        self.vcoord = vcoord
        self.flag = flag
        self.staid = staid
        self.targetid = targetid


class OIVIS:
    def __init__(self, arrname, insname, amptype='UNKNOWN', phitype='UNKNOWN', dateobs=0, mjd=[], visamp=[],
                 visamperr=[], visphi=[], visphierr=[], ucoord=[], vcoord=[], flag=[], targetid=[], staid=[]):
        self.arrname = arrname
        self.insname = insname
        self.dateobs = dateobs
        self.amptype = amptype
        self.phitype = phitype
        self.mjd = mjd
        self.visamp = visamp
        self.visphi = visphi
        self.visamperr = visamperr
        self.visphierr = visphierr
        self.ucoord = ucoord
        self.vcoord = vcoord
        self.flag = flag
        self.targetid = targetid
        self.staid = staid


class OIT3:
    def __init__(self, arrname, insname, dateobs=0, mjd=[], t3amp=[], t3amperr=[], t3phi=[], t3phierr=[], u1coord=[],
                 v1coord=[], u2coord=[], v2coord=[], flag=[], targetid=[], staid=[]):
        self.arrname = arrname
        self.insname = insname
        self.dateobs = dateobs
        self.mjd = mjd
        self.t3amp = t3amp
        self.t3amperr = t3amperr
        self.t3phi = t3phi
        self.t3phierr = t3phierr
        self.u1coord = u1coord
        self.v1coord = v1coord
        self.u2coord = u2coord
        self.v2coord = v2coord
        self.flag = flag
        self.targetid = targetid
        self.staid = staid


class OIFLUX:
    def __init__(self, insname, arrname, calstat='unknown', dateobs=0, mjd=[], fluxdata=[], fluxerr=[], flag=[],
                 targetid=[], staid=[]):
        self.insname = insname
        self.arrname = arrname
        self.dateobs = dateobs
        self.mjd = mjd
        self.fluxdata = fluxdata
        self.fluxerr = fluxerr
        self.flag = flag
        self.calstat = calstat
        self.targetid = targetid
        self.staid = staid


def ListV2(data):
    u, v, wavev2, wavecp = DataUnpack(data)
    V2, V2err, CP, CPerr = GiveDataValues(data)

    nV2 = len(V2)

    u, u1, u2, u3 = u
    v, v1, v2, v3 = v

    u *= wavev2
    v *= wavev2

    ubase = u[0]
    vbase = v[0]
    newV2 = []
    newV2err = []
    newwave = []
    newu = []
    newv = []
    tmpV2 = []
    tmpV2err = []
    tmpwave = []
    tmpu = []
    tmpv = []
    for i in np.arange(nV2):
        # print (u[i] - ubase)
        if np.abs(u[i] - ubase) < 1e-5 and np.abs(v[i] - vbase) < 1e-5:
            tmpV2.append(V2[i])
            tmpV2err.append(V2err[i])
            tmpu.append(u[i])
            tmpv.append(v[i])
            tmpwave.append(wavev2[i])
        else:
            newV2.extend([tmpV2])
            newV2err.extend([tmpV2err])
            newu.extend([tmpu])
            newv.extend([tmpv])
            newwave.extend([tmpwave])
            ubase = u[i]
            vbase = v[i]
            tmpV2 = []
            tmpV2err = []
            tmpwave = []
            tmpu = []
            tmpv = []
            tmpV2.append(V2[i])
            tmpV2err.append(V2err[i])
            tmpu.append(u[i])
            tmpv.append(v[i])
            tmpwave.append(wavev2[i])

    newV2.extend([tmpV2])
    newV2err.extend([tmpV2err])
    newu.extend([tmpu])
    newv.extend([tmpv])
    newwave.extend([tmpwave])

    return newV2, newV2err, newu, newv, newwave


def DataUnpack(data):
    # Unpacking the data
    u = data['u']
    v = data['v']
    wave = data['wave']
    # vis2, vis2err = data['v2']
    # cp, cperr = data['cp']
    wavev2, wavecp = wave

    return u, v, wavev2, wavecp


def GiveDataValues(data):
    V2, V2err = data['v2']
    CP, CPerr = data['cp']

    return V2, V2err, CP, CPerr


def ListCP(data):
    u, v, wavev2, wavecp = DataUnpack(data)
    V2, V2err, CP, CPerr = GiveDataValues(data)

    nCP = len(CP)

    u, u1, u2, u3 = u
    v, v1, v2, v3 = v

    u1 *= wavecp
    v1 *= wavecp
    u2 *= wavecp
    v2 *= wavecp
    u3 *= wavecp
    v3 *= wavecp

    u1base = u1[0]
    v1base = v1[0]
    u2base = u2[0]
    v2base = v2[0]
    newCP = []
    newCPerr = []
    newwave = []
    newu1 = []
    newv1 = []
    newu2 = []
    newv2 = []
    newu3 = []
    newv3 = []
    tmpCP = []
    tmpCPerr = []
    tmpwave = []
    tmpu1 = []
    tmpv1 = []
    tmpu2 = []
    tmpv2 = []
    tmpu3 = []
    tmpv3 = []
    for i in np.arange(nCP):
        # print (u[i] - ubase)
        if np.abs(u1[i] - u1base) < 1e-5 and np.abs(v1[i] - v1base) < 1e-5 and np.abs(u2[i] - u2base) < 1e-5 and np.abs(
                v2[i] - v2base) < 1e-5:
            tmpCP.append(CP[i])
            tmpCPerr.append(CPerr[i])
            tmpu1.append(u1[i])
            tmpv1.append(v1[i])
            tmpu2.append(u2[i])
            tmpv2.append(v2[i])
            tmpu3.append(u3[i])
            tmpv3.append(v3[i])
            tmpwave.append(wavecp[i])
        else:
            newCP.extend([tmpCP])
            newCPerr.extend([tmpCPerr])
            newu1.extend([tmpu1])
            newv1.extend([tmpv1])
            newu2.extend([tmpu2])
            newv2.extend([tmpv2])
            newu3.extend([tmpu3])
            newv3.extend([tmpv3])
            newwave.extend([tmpwave])
            u1base = u1[i]
            v1base = v1[i]
            u2base = u2[i]
            v2base = v2[i]
            tmpCP = []
            tmpCPerr = []
            tmpwave = []
            tmpu = []
            tmpv = []
            tmpu1 = []
            tmpv1 = []
            tmpu2 = []
            tmpv2 = []
            tmpu3 = []
            tmpv3 = []
            tmpCP.append(CP[i])
            tmpCPerr.append(CPerr[i])
            tmpu1.append(u1[i])
            tmpv1.append(v1[i])
            tmpu2.append(u2[i])
            tmpv2.append(v2[i])
            tmpu3.append(u3[i])
            tmpv3.append(v3[i])
            tmpwave.append(wavecp[i])

    print(np.array(tmpCP).shape, np.array(tmpu1).shape)
    newCP.extend([tmpCP])
    newCPerr.extend([tmpCPerr])
    newu1.extend([tmpu1])
    newv1.extend([tmpv1])
    newu2.extend([tmpu2])
    newv2.extend([tmpv2])
    newu3.extend([tmpu3])
    newv3.extend([tmpv3])
    newwave.extend([tmpwave])

    return newCP, newCPerr, newu1, newv1, newu2, newv2, newu3, newv3, newwave


def Load(data):
    # Loading the dataset
    wave = data['wave']
    V2data, V2err = data['v2']
    CPdata, CPerr = data['cp']
    base, Bmax = Bases(data)

    # Setting things for the plot
    base *= 1e-6  # set baseline lengths in MegaLambda? (don't ask me why)
    Bmax *= 1e-6

    waveCP = wave[1]
    waveV2 = wave[0]
    waveV2 *= 1e6  # set wavelength in micron (that way the division done u/wave in plotV2CP if lines=False works out)
    waveCP *= 1e6

    return waveV2, waveCP, base, Bmax, V2data, V2err, CPdata, CPerr
