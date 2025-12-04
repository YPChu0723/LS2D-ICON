#
# This file is part of LS2D.
#
# Copyright (c) 2017-2024 Wageningen University & Research
# Author: Bart van Stratum (WUR)
#
# LS2D is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# LS2D is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with LS2D.  If not, see <http://www.gnu.org/licenses/>.
#

# Python modules
import datetime
import sys, os

# Third party modules
import netCDF4 as nc4
import xarray as xr
import numpy as np
from scipy import interpolate

# LS2D modules
import ls2d.src.spatial_tools as spatial
import ls2d.src.finite_difference as fd
from ls2d.src.messages import *

import ls2d.ecmwf.ICON_tools as ICON_tools
# from ls2d.ecmwf.IFS_tools import IFS_tools
from ls2d.ecmwf.utils import utils

from ls2d.ecmwf.patch_cds_ads import patch_netcdf

# Constants
Rd = 287.04
Rv = 461.5
ep = Rd/Rv
g = 9.80665
# ifs_tools = IFS_tools('L137')

utils = utils()

class Slice:
    def __init__(self, istart, iend, jstart, jend):
        self.istart = istart
        self.iend   = iend
        self.jstart = jstart
        self.jend   = jend

    def __call__(self, dj, di):
        return np.s_[:,:,self.jstart+dj:self.jend+dj,\
                         self.istart+di:self.iend+di]

class Read_ICON:
    """
    Read the ERA5 model/pressure/surface level data,
    and optionally calculate the LES/SCM forcings
    """

    def __init__(self, settings):

        self.settings = settings
        self.start = settings['start_date']
        self.end   = settings['end_date']

        # Open all required NetCDF files:
        self.open_netcdf_files()

        # Read all the required variables:
        self.read_data()

        # Calculate derived properties needed for LES:
        self.calc_derived_data()


    def open_netcdf_files(self):
        """
        Open all NetCDF files required for start->end period
        """
        header('Reading ICON from {} to {}'.format(self.start, self.end))

        # Get list of required forecast and analysis times
        an_dates = ICON_tools.get_required_analysis(self.start, self.end)
        # fc_dates = era_tools.get_required_forecast(self.start, self.end)

        # Check if output directory ends with '/'
        if self.settings['ICON_path'][-1] != '/':
            self.settings['ICON_path'] += '/'

        # Create lists with required files
        path = self.settings['ICON_path']
        case = self.settings['case_name']

        an_model_files   = [ICON_tools.file_path(
            d.year, d.month, d.day, d.hour, d.minute, path, case, 'rmp_CLOUDLAB_MIP_input_130_',  False) for d in an_dates]
        # an_model_files = [ICON_tools.era5_file_path(
        #     d.year, d.month, d.day, path, case, 'model_an',    False) for d in an_dates]
        an_pres_files  = [ICON_tools.file_path(
            d.year, d.month, d.day, d.hour, d.minute, path, case + '/pressure_levels_output', 'pl_rmp_CLOUDLAB_MIP_input_130_',  False) for d in an_dates]

        # Check if all files exist, and exit if not..
        def check_files(files):
            file_missing = False
            for f in files:
                if not os.path.exists(f):
                    error('File \"{}\" does not exist...'.format(f), exit=False)
                    file_missing = True
            return file_missing

        files_missing = False
        files_missing += check_files(an_model_files)
        # files_missing += check_files(an_model_files)
        files_missing += check_files(an_pres_files )
        if files_missing:
            error('One or more required ERA5 files are missing..')

        # Check if files are from new CDS, which require patching.
        # This is only a fallback option, in case someone has unpatched NetCDF files.
        # The patching is now automatically done after downloading the files.

        # Open NetCDF files: MFDataset automatically merges the files / time dimensions
        # self.fma = nc4.MFDataset(an_model_files, aggdim='time')
        self.fma = xr.open_mfdataset(an_model_files, combine='by_coords')
        self.fpa = xr.open_mfdataset(an_pres_files, combine='by_coords').isel(plev_3=slice(0,35))  # Reverse pressure levels



    def read_data(self):
        """
        Read all the required variables from the NetCDF files
        """

        def flip(array):
            """
            Flip the height and/or latitude dimensions
            """
            # Note: array is now a numpy array here
            if len(array.shape) == 4:
                return np.flip(array, axis=1)
            elif len(array.shape) == 3:
                return array
            elif len(array.shape) == 1:
                return np.flip(array, axis=0)

        def get_variable(nc, var, dslice, wrap_func=None, dtype=None):
            """
            Read NetCDF variable, convert to numpy, and flip.
            """
            # 1. Access variable from xarray Dataset (nc[var])
            # 2. Slice it ([dslice])
            # 3. Convert to Numpy array immediately (.values)
            raw_data = nc[var][dslice].values
            
            # 4. Apply flip (now operating on numpy array)
            data = flip(raw_data)
            
            # Apply wrapper function (if provided):
            data = wrap_func(data) if wrap_func is not None else data
            # Cast to requested data type (if provided):
            data = data.astype(dtype) if dtype is not None else data

            return data
        

        def decode_time(t):
            # Ensure t is a standard python/numpy scalar, not xarray object
            return datetime.datetime.strptime(str(int(t)), "%Y%m%d") + datetime.timedelta(days=(t % 1))
        
        # EXTRACT COORDINATES AS NUMPY ARRAYS
        # .values converts xarray/dask arrays to in-memory numpy arrays
        raw_times = self.fma['time'].values 

        # This creates the "Master Time Array" used for everything below
        all_datetimes = np.array([decode_time(t) for t in raw_times])

        # 4. Find start and end indices
        idx_start = np.abs(all_datetimes - self.start).argmin()
        idx_end   = np.abs(all_datetimes - self.end).argmin() 

        # 5. Create the slice
        time_slice = slice(idx_start, idx_end + 1)

        self.datetime = all_datetimes[time_slice] 
        self.time     = raw_times[time_slice]
        
        # Extract Lat/Lon as numpy arrays
        self.lats     = self.fma.coords['lat'].values
        self.lons     = self.fma.coords['lon'].values

        self.time_sec = np.array([(t - self.datetime[0]).total_seconds() for t in self.datetime])
        
        if np.any(self.lons > 180):
            self.lons = -360 + self.lons

        # Grid and time dimensions
        self.nfull = self.fma.coords['height'].size # .size returns int
        self.nhalf = self.nfull + 1
        self.nlat  = self.fma.coords['lat'].size
        self.nlon  = self.fma.coords['lon'].size
        self.ntime = self.time.size

        # Slices
        s1d  = np.s_[:]         
        s2d  = np.s_[time_slice,:,:]    
        s3d  = np.s_[time_slice,:,:,:]
        s3d_half = np.s_[time_slice,:-1,:,:]    
        s3ds = np.s_[time_slice,0,:,:]    

        # Model level analysis data:
        # All these will now be numpy arrays because get_variable converts them
        self.u  = get_variable(self.fma, 'u',    s3d)  
        self.v  = get_variable(self.fma, 'v',    s3d)  
        self.w  = get_variable(self.fma, 'w',    s3d_half)  
        self.T  = get_variable(self.fma, 'temp', s3d)  
        self.qc = get_variable(self.fma, 'qc',   s3d) 
        self.qi = get_variable(self.fma, 'qi',   s3d) 
        self.qr = get_variable(self.fma, 'qr',   s3d)  
        self.qs = get_variable(self.fma, 'qs',   s3d)  
        self.q  = get_variable(self.fma, 'qv',   s3d)  
        
        self.p  = get_variable(self.fma, 'pres', s3d)  
        self.zh = get_variable(self.fma, 'z_ifc',s3d) 

        # Surface variables:
        self.qvs  = get_variable(self.fma, 'qv_s',   s2d) 
        self.Ts  = get_variable(self.fma, 't_g',    s2d)  
        self.H   = -get_variable(self.fma, 'shfl_s', s2d) 
        self.LH  = -get_variable(self.fma, 'lhfl_s', s2d) 
        self.z0  = get_variable(self.fma, 'gz0',    s2d)
        self.ps  = get_variable(self.fma, 'pres_sfc', s2d) 

        # Soil variables:
        # Note: np.flip works fine on numpy arrays returned by get_variable
        self.T_soil = np.flip(get_variable(self.fma, 't_so', s3d), axis=1)
        self.T_depth = np.flip(get_variable(self.fma, 'depth', s1d), axis=0)
        
        self.T_soil1 = self.T_soil[:,0,:,:] 
        self.T_soil2 = self.T_soil[:,4,:,:] 
        self.T_soil3 = self.T_soil[:,5,:,:]  
        self.T_soil4 = self.T_soil[:,8,:,:]  

        self.theta_soil = np.flip(get_variable(self.fma, 'smi', s3d), axis=1)
        self.theta_depth = np.flip(get_variable(self.fma, 'depth_2', s1d), axis=0)
        
        self.theta_soil1 = self.theta_soil[:,0,:,:]  
        self.theta_soil2 = self.theta_soil[:,3,:,:]  
        self.theta_soil3 = self.theta_soil[:,4,:,:]  
        self.theta_soil4 = self.theta_soil[:,7,:,:]

        self.z_p = get_variable(self.fpa, 'z_ifc', s3d) # Geopotential height on pressure levels (m)
        self.p_p = get_variable(self.fpa, 'plev_3', s1d)      # Pressure levels (Pa)


    def calc_derived_data(self):
        """
        Calculate derived properties; conversion model levels to pressure/height,
        prognostic variables used by LES, etc.
        """

        self.ql  = self.qc + self.qi + self.qr + self.qs  # Total liquid/solid specific humidity (kg kg-1)
        self.qt  = self.q + self.ql                       # Total specific humidity (kg kg-1)
        self.Tv  = utils.calc_virtual_temp(
                self.T, self.q, self.qc, self.qi, self.qr, self.qs)  # Virtual temp on full levels (K)

        # Calculate half level pressure and heights
        self.ph  = np.zeros((self.ntime, self.nhalf, self.nlat, self.nlon))  # Half level pressure (Pa)
        self.z  = np.zeros((self.ntime, self.nfull, self.nlat, self.nlon))  # Full level geometric height (m)
        
        self.ph[:,1:-1,:,:] = 0.5 * (self.p[:,:-1,:,:] + self.p[:,1:,:,:])
        self.ph[:,0,:,:] =  self.p[:,0,:,:] - 0.5 * (self.p[:,1,:,:] - self.p[:,0,:,:])
        self.ph[:,-1,:,:] = self.p[:,-1,:,:] + 0.5 * (self.p[:,-1,:,:] - self.p[:,-2,:,:])

        self.z = 0.5 * (self.zh[:,1:,:,:] + self.zh[:,:-1:,:]) 

        # Other derived quantities
        self.exn  = utils.calc_exner(self.p)  # Exner on full model levels (-)
        self.th   = (self.T / self.exn)  # Potential temperature (K)
        self.thl  = self.th - utils.Lv / (utils.cpd * self.exn) * self.ql  # Liquid water potential temperature (K)
        self.rho  = self.p / (utils.Rd * self.Tv)  # Density at full levels (kg m-3)
        self.wls  = self.w
        self.U    = (self.u**2. + self.v**2)**0.5  # Absolute horizontal wind (m s-1)


        self.Tvs  = utils.calc_virtual_temp(self.Ts, self.qvs)  # Estimate surface Tv using lowest model q (...)
        self.rhos = self.ph[:,0] / (utils.Rd * self.Tvs)  # Surface density (kg m-3)
        self.exns = utils.calc_exner(self.ps)  # Exner at surface (-)
        self.wths = self.H / (self.rhos * utils.cpd * self.exns)  # Surface kinematic heat flux (K m s-1)
        self.wqs =  self.LH / (self.rhos * utils.Lv)# Surface kinematic moisture flux (kg kg-1 m s-1)
        self.fc = 2 * 7.2921e-5 * np.sin(np.deg2rad(self.settings['central_lat']))  # Coriolis parameter

        # Store soil temperature, and moisture content, in 3D array
        self.z_T_soil = self.T_depth[[0,4,5,8]] / 1000
        self.z_theta_soil = self.theta_depth[[0,3,4,7]] / 1000
        
        self.T_soil = np.zeros((self.ntime, 4, self.nlat, self.nlon))
        self.theta_soil = np.zeros((self.ntime, 4, self.nlat, self.nlon))

        self.T_soil[:,0,:,:] = self.T_soil1[:,:,:]
        self.T_soil[:,1,:,:] = self.T_soil2[:,:,:]
        self.T_soil[:,2,:,:] = self.T_soil3[:,:,:]
        self.T_soil[:,3,:,:] = self.T_soil4[:,:,:]

        self.theta_soil[:,0,:,:] = self.theta_soil1[:,:,:]
        self.theta_soil[:,1,:,:] = self.theta_soil2[:,:,:]
        self.theta_soil[:,2,:,:] = self.theta_soil3[:,:,:]
        self.theta_soil[:,3,:,:] = self.theta_soil4[:,:,:]


    def calculate_forcings(self, n_av_lon=0, n_av_lat=0, method='4th'):
        """
        Calculate the advective tendencies, geostrophic wind, et cetera.
        """
        header('Calculating large-scale forcings')

        # Find nearest location on (regular lat/lon) grid
        self.i = np.abs(self.lons - self.settings['central_lon']).argmin()
        self.j = np.abs(self.lats - self.settings['central_lat']).argmin()

        # Some debugging output
        distance = spatial.haversine(
                self.lons[self.i], self.lats[self.j],
                self.settings['central_lon'], self.settings['central_lat'])

        message('Using nearest lat/lon = {0:.8f}/{1:.8f} (requested = {2:.8f}/{3:.8f}), distance ~= {4:.8f} km'\
                .format(self.lats[self.j], self.lons[self.i],
                        self.settings['central_lat'], self.settings['central_lon'], distance/1000.))

        # Print averaging area.
        dlon = (1+2*n_av_lon) * float(self.lons[1] - self.lons[0])
        dlat = (1+2*n_av_lat) * float(self.lats[1] - self.lats[0])
        self.area = f'{dlon:.2f}°×{dlat:.2f}°'
        message(f'Averaging ERA5 over a {self.area} spatial area.')

        # Start and end indices of averaging domain:
        istart = self.i - n_av_lon
        iend   = self.i + n_av_lon + 1
        jstart = self.j - n_av_lat
        jend   = self.j + n_av_lat + 1

        # Numpy slicing tuples for averaging domain
        center4d = np.s_[:,:,jstart:jend,istart:iend]
        center3d = np.s_[:,  jstart:jend,istart:iend]

        # Variables averaged from (time, height, lon, lat) to (time, height):
        var_4d_mean = [
                'z', 'zh', 'p', 'ph', 'T', 'thl', 'qt', 'qc', 'qi',
                'u', 'v', 'U', 'wls', 'rho',
                'T_soil', 'theta_soil']
        for var in var_4d_mean:
            mean = getattr(self, var)[center4d].mean(axis=(2,3))
            setattr(self, '{}_mean'.format(var), mean)

        # Variables averaged from (time, lon, lat) to (time):
        var_3d_mean = [
                'ps', 'Ts', 'wths', 'wqs', 'ps', 'rhos',
                # 'lai_low', 'lai_high', 
                # 'z0m', 'z0h', 'cveg_low', 'cveg_high'
                'z0'
                ]
        for var in var_3d_mean:
            mean = getattr(self, var)[center3d].mean(axis=(1,2))
            setattr(self, '{}_mean'.format(var), mean)

        # Variables selected as nearest-neighbour
        # var_nn = ['soil_type', 'veg_type_low', 'veg_type_high']
        # for var in var_nn:
        #     data = getattr(self, var)
        #     setattr(self, '{}_nn'.format(var), data[0, self.j, self.i])

        # if self.soil_type_nn == 0:
        #     warning('Selected grid point is water/sea! Setting vegetation/soil indexes to 1e9.')

        #     self.soil_type_nn = int(1e9)
        #     self.veg_type_low_nn = int(1e9)
        #     self.veg_type_high_nn = int(1e9)

        #     gridpoint_is_land = False
        # else:
        #     message('Selected grid point is over land.')
        #     gridpoint_is_land = True

        # Half level values temperature for radiation
        self.Th_mean = np.zeros_like(self.zh_mean)
        self.Th_mean[:,1:-1] = 0.5 * (self.T_mean[:,1:] + self.T_mean[:,:-1])

        dTdz = (self.Th_mean[:,1] - self.T_mean[:,0]) / (self.zh_mean[:,1] - self.z_mean[:,0])
        self.Th_mean[:,0] = self.T_mean[:,0] - dTdz * self.z_mean[:,0]

        dTdz = (self.T_mean[:,-1] - self.Th_mean[:,-2]) / (self.z_mean[:,-1] - self.zh_mean[:,-2])
        self.Th_mean[:,-1] = self.T_mean[:,-1] + dTdz * (self.zh_mean[:,-1] - self.z_mean[:,-1])

        # Estimate horizontal grid spacing (assumed constant in averaging domain)\
        dx = spatial.dlon(self.lons[self.i-1], self.lons[self.i+1], self.lats[self.j]) / 2.
        dy = spatial.dlat(self.lats[self.j-1], self.lats[self.j+1]) / 2.

        print(dx, dy)

        if (method == '2nd'):

            r_earth = 6.37e6

            lat_rad = np.deg2rad(self.lats)
            lon_rad = np.deg2rad(self.lons)
            cos_lat = np.cos(lat_rad)

            dxdi = np.zeros((self.nlat, self.nlon))
            dydj = np.zeros((self.nlat, self.nlon))

            dxdi[:,:] = r_earth * cos_lat[:,None]*np.gradient(lon_rad[None, :], axis=1)
            dydj[:,:] = r_earth * np.gradient(lat_rad[:, None], axis=0)

            def advec(var):
                dvardx = np.gradient(var, axis=3) / dxdi[None, None, :, :]
                dvardy = np.gradient(var, axis=2) / dydj[None, None, :, :]
                dtvar  = -self.u * dvardx - self.v * dvardy
                return dvardx[center4d].mean(axis=(2,3)), dvardy[center4d].mean(axis=(2,3)), dtvar[center4d].mean(axis=(2,3))

            # Calculate advective tendencies:
            self.dqtdx_advec_mean = advec(self.qt)[0]
            self.dqtdy_advec_mean = advec(self.qt)[1]

            self.dtthl_advec_mean = advec(self.thl)[2]
            self.dtqt_advec_mean  = advec(self.qt)[2]
            self.dtu_advec_mean   = advec(self.u)[2]
            self.dtv_advec_mean   = advec(self.v)[2]

            # Geostrophic wind:
            dzdx = np.gradient(self.z_p, axis=3) / dxdi[None, None, :, :]
            dzdy = np.gradient(self.z_p, axis=2) / dydj[None, None, :, :]

            self.ug_p = -utils.grav / self.fc * dzdy
            self.vg_p =  utils.grav / self.fc * dzdx

            ug_p_mean = self.ug_p[center4d].mean(axis=(2,3))
            vg_p_mean = self.vg_p[center4d].mean(axis=(2,3))

            # ug_p_mean = np.nanmean(self.ug_p[center4d], axis=(2,3))
            # vg_p_mean = np.nanmean(self.vg_p[center4d], axis=(2,3))

            # Bonus for large domains; spatial (ug,vg) on model levels.
            # Use Scipy's interpolation, as it can extrapolate (in case ps > 1000 hPa)
            self.ug = np.zeros_like(self.u)
            self.vg = np.zeros_like(self.u)

            for t in range(self.ntime):
                for j in range(self.nlat):
                    for i in range(self.nlon):
                        self.ug[t,:,j,i] = interpolate.interp1d(
                            self.p_p, self.ug_p[t,:,j,i], fill_value='extrapolate')(self.p[t,:,j,i])
                        self.vg[t,:,j,i] = interpolate.interp1d(
                            self.p_p, self.vg_p[t,:,j,i], fill_value='extrapolate')(self.p[t,:,j,i])


        elif (method == '4th'):

            s = Slice(istart, iend, jstart, jend)

            # Calculate advective tendencies
            self.dtthl_advec_mean = (
                -self.u[s(0,0)] * fd.grad4c(
                    self.thl[s(0,-2)], self.thl[s(0,-1)], self.thl[s(0,+1)], self.thl[s(0,+2)], dx) \
                -self.v[s(0,0)] * fd.grad4c(
                    self.thl[s(-2,0)], self.thl[s(-1,0)], self.thl[s(+1,0)], self.thl[s(+2,0)], dy)
                                    ).mean(axis=(2,3))

            self.dtqt_advec_mean =  (
                -self.u[s(0,0)] * fd.grad4c(
                    self.qt[s(0,-2)], self.qt[s(0,-1)], self.qt[s(0,+1)], self.qt[s(0,+2)], dx) \
                -self.v[s(0,0)] * fd.grad4c(
                    self.qt[s(-2,0)], self.qt[s(-1,0)], self.qt[s(+1,0)], self.qt[s(+2,0)], dy)
                                    ).mean(axis=(2,3))

            self.dtu_advec_mean = (
                -self.u[s(0,0)] * fd.grad4c(
                    self.u[s(0,-2)], self.u[s(0,-1)], self.u[s(0,+1)], self.u[s(0,+2)], dx) \
                -self.v[s(0,0)] * fd.grad4c(
                    self.u[s(-2,0)], self.u[s(-1,0)], self.u[s(+1,0)], self.u[s(+2,0)], dy)
                                  ).mean(axis=(2,3))

            self.dtv_advec_mean = (
                -self.u[s(0,0)] * fd.grad4c(
                    self.v[s(0,-2)], self.v[s(0,-1)], self.v[s(0,+1)], self.v[s(0,+2)], dx) \
                -self.v[s(0,0)] * fd.grad4c(
                    self.v[s(-2,0)], self.v[s(-1,0)], self.v[s(+1,0)], self.v[s(+2,0)], dy)
                                  ).mean(axis=(2,3))

            # Geostrophic wind (gradient geopotential height on constant pressure levels)
            vg_p_mean = (
                utils.grav / self.fc * fd.grad4c(
                    self.z_p[s(0,-2)], self.z_p[s(0,-1)], self.z_p[s(0,+1)], self.z_p[s(0,+2)], dx)
                        ).mean(axis=(2,3))
            ug_p_mean = (
               -utils.grav / self.fc * fd.grad4c(
                    self.z_p[s(-2,0)], self.z_p[s(-1,0)], self.z_p[s(+1,0)], self.z_p[s(+2,0)], dy)
                        ).mean(axis=(2,3))


        # Interpolate geostrophic wind onto model grid.
        # Use Scipy's interpolation, as it can extrapolate (in case ps > 1000 hPa)
        self.ug_mean = np.zeros_like(self.p_mean)
        self.vg_mean = np.zeros_like(self.p_mean)

        for t in range(self.ntime):
            self.ug_mean[t,:] = interpolate.interp1d(
                    self.p_p, ug_p_mean[t,:], fill_value='extrapolate')(self.p_mean[t,:])
            self.vg_mean[t,:] = interpolate.interp1d(
                    self.p_p, vg_p_mean[t,:], fill_value='extrapolate')(self.p_mean[t,:])

        # Momentum tendency coriolis
        self.dtu_coriolis_mean = +self.fc * (self.v_mean - self.vg_mean)
        self.dtv_coriolis_mean = -self.fc * (self.u_mean - self.ug_mean)

        # Total momentum tendency
        self.dtu_total_mean = self.dtu_advec_mean + self.dtu_coriolis_mean
        self.dtv_total_mean = self.dtv_advec_mean + self.dtv_coriolis_mean

        # Calculate root fraction for low and high vegetation
        # Root-fraction coefficients from ECMWF documentation:
        # ar = np.array([ 5.558, 10.739,  6.706,  7.066,  5.99 ,  7.344,  8.235,  4.372,
        #                 8.992,  5.558,  4.372, -1.   ,  7.344, -1.   , -1.   ,  6.326,
        #                 6.326,  4.453,  4.453, -1.   ])
        # br = np.array([ 2.614,  2.608,  2.175,  1.953,  1.955,  1.303,  1.627,  0.978,
        #                 8.992,  2.614,  0.978, -1.   ,  1.303, -1.   , -1.   ,  1.567,
        #                 1.567,  1.631,  1.631, -1.   ])

        # # Calculate half level soil depths
        # zh = np.zeros(self.z_theta_soil.size+1)
        # for k in range(zh.size-2, -1, -1):
        #     zh[k] = zh[k+1] - 2*(zh[k+1] - self.z_soil[::-1][k])

        # def calc_root_fraction(index):
        #     rf = np.zeros_like(self.z_soil)
        #     for k in range(1, rf.size):
        #         rf[k] = 0.5 * (np.exp(ar[index] * zh[k+1]) + \
        #                        np.exp(br[index] * zh[k+1]) - \
        #                        np.exp(ar[index] * zh[k  ]) - \
        #                        np.exp(br[index] * zh[k  ]));
        #     rf[0] = 1.-rf.sum()
        #     return rf[::-1]

        # if gridpoint_is_land:
        #     self.root_frac_low_nn  = calc_root_fraction(self.veg_type_low_nn-1)
        #     self.root_frac_high_nn = calc_root_fraction(self.veg_type_high_nn-1)
        # else:
        #     self.root_frac_low_nn = np.zeros(4)-1
        #     self.root_frac_high_nn = np.zeros(4)-1


    def get_les_input(self, z):
        """
        Interpolate variables required for LES onto model grid,
        and return xarray.Dataset with all possible LES input
        """

        def interp_z(array, z):
            out = np.empty((self.ntime, z.size))
            for t in range(self.ntime):
                # print('self.z_mean[t,:]', self.z_mean[t,:].shape)
                # print('array[t,:]', array[t,:].shape)
                out[t,:] = np.interp(z, self.z_mean[t,:], array[t,:])
            return out

        def add_ds_var(ds, name, data, dims, long_name, units):
            if dims is not None:
                ds[name] = (dims, data)
            else:
                ds[name] = data
            ds[name].attrs['long_name'] = long_name
            ds[name].attrs['units'] = units

        #
        # Create xarray Dataset
        #

        ds = xr.Dataset(
                coords = {
                    'time': self.datetime,
                    'z': z,
                    'zs': self.z_theta_soil,
                    'lev': np.arange(self.nhalf),
                    'lay': np.arange(self.nfull)
                })

        ds['z'].attrs['long_name'] = 'full level height LES'
        ds['z'].attrs['units'] = 'm'

        ds['zs'].attrs['long_name'] = 'full level depth soil'
        ds['zs'].attrs['units'] = 'm'

        variables = {
                'thl': ('liquid water potential temperature', 'K'),
                'qt': ('total specific humidity', 'kg kg-1'),
                'u': ('zonal wind component', 'm s-1'),
                'v': ('meridional wind component', 'm s-1'),
                'wls': ('vertical wind component', 'm s-1'),
                'p': ('air pressure', 'Pa'),
                'dqtdx_advec': ('tendency total specific humidity in x-direction', 'kg kg-1 m-1'),
                'dqtdy_advec': ('tendency total specific humidity in y-direction', 'kg kg-1 m-1'),
                'dtthl_advec': ('advective tendency liquid water potential temperature', 'K s-1'),
                'dtqt_advec': ('advective tendency total specific humidity', 'kg kg-1 s-1'),
                'dtu_advec': ('advective tendency zonal wind', 'm s-2'),
                'dtv_advec': ('advective tendency meridional wind', 'm s-2'),
                'ug': ('geostrophic wind component zonal wind', 'm s-1'),
                'vg': ('geostrophic wind component meridional wind', 'm s-1'),
                # 'o3': ('ozone volume mixing ratio', 'ppmv'),
                }

        #
        # Interpolate LES input to LES grid, and add to dataset
        #
        for var in variables.keys():
            var_era5 = '{}_mean'.format(var)
            if hasattr(self, var_era5):
                # print('Interpolating variable "{}"...'.format(var))
                data  = interp_z(getattr(self, var_era5), z)
                attrs = variables[var]
                add_ds_var(ds, var, data, ('time', 'z'), attrs[0], attrs[1])
            else:
                error('Can\'t interpolate variable \"{}\"...'.format(var))

        #
        # Add other input variables, which don't require interpolation
        #
        # Time:
        add_ds_var(ds, 'time_sec', self.time_sec, ('time'), 'seconds since start of experiment', 's')

        # Radiation background profiles
        add_ds_var(ds, 'z_lay', self.z_mean, ('time', 'lay'), 'Full level heights radiation', 'm')
        add_ds_var(ds, 'z_lev', self.zh_mean, ('time', 'lev'), 'Half level heights radiation', 'm')

        add_ds_var(ds, 'p_lay', self.p_mean, ('time', 'lay'), 'full level pressure radiation', 'Pa')
        add_ds_var(ds, 'p_lev', self.ph_mean, ('time', 'lev'), 'half level pressure radiation', 'Pa')

        add_ds_var(ds, 't_lay', self.T_mean, ('time', 'lay'), 'full level temperature radiation', 'K')
        add_ds_var(ds, 't_lev', self.Th_mean, ('time', 'lev'), 'half level temperature radiation', 'K')

        h2o_lay = self.qt_mean / (ep - ep * self.qt_mean)
        add_ds_var(ds, 'h2o_lay', h2o_lay, ('time', 'lay'), 'moisture volume mixing ratio', '')
        # add_ds_var(ds, 'o3_lay', self.o3_mean, ('time', 'lay'), 'ozone volume mixing ratio radiation', 'ppmv')

        # Soil variables
        add_ds_var(ds, 't_soil', self.T_soil_mean, ('time', 'zs'), 'soil temperature', 'K')
        add_ds_var(ds, 'theta_soil', self.theta_soil_mean, ('time', 'zs'), 'soil moisture content', 'm3 m-3')
        # add_ds_var(ds, 'type_soil', self.soil_type_nn, None, 'ECMWF soil type (Fortran indexing!)', '-')

        # Surface/vegetation:
        # add_ds_var(ds, 'type_low_veg',  self.veg_type_low_nn,  None, 'ECMWF low vegetation type (Fortran indexing!)', '-')
        # add_ds_var(ds, 'type_high_veg', self.veg_type_high_nn, None, 'ECMWF high vegetation type (Fortran indexing!)', '-')

        # add_ds_var(ds, 'root_frac_low_veg',  self.root_frac_low_nn,  ('zs'), 'root fraction low vegetation', '-')
        # add_ds_var(ds, 'root_frac_high_veg', self.root_frac_high_nn, ('zs'), 'root fraction high vegetation', '-')

        # add_ds_var(ds, 'lai_low_veg',  self.lai_low_mean,  ('time'), 'LAI low vegetation', '-')
        # add_ds_var(ds, 'lai_high_veg', self.lai_high_mean, ('time'), 'LAI high vegetation', '-')

        # add_ds_var(ds, 'c_low_veg',  self.cveg_low_mean,  ('time'), 'fraction low vegetation', '-')
        # add_ds_var(ds, 'c_high_veg', self.cveg_high_mean, ('time'), 'fraction high vegetation', '-')

        # add_ds_var(ds, 'z0m', self.z0m_mean, ('time'), 'roughness length momentum', 'm')
        # add_ds_var(ds, 'z0h', self.z0h_mean, ('time'), 'roughness length scalars', 'm')
        add_ds_var(ds, 'z0', self.z0_mean, ('time'), 'roughness length', 'm')

        # Time varying surface properties:
        add_ds_var(ds, 'ps', self.ps_mean, ('time'), 'surface pressure', 'Pa')
        # add_ds_var(ds, 'sst', self.sst_mean, ('time'), 'sea surface temperature', 'K')
        add_ds_var(ds, 'ts', self.Ts_mean, ('time'), 'surface (skin) temperature', 'K')
        add_ds_var(ds, 'wth', self.wths_mean, ('time'), 'surface sensible heat flux', 'K m s-1')
        add_ds_var(ds, 'wq', self.wqs_mean, ('time'), 'surface latent heat flux', 'kg kg-1 m s-1')

        # Misc
        ds.attrs['fc'] = self.fc
        ds.attrs['central_lon'] = self.settings['central_lon']
        ds.attrs['central_lat'] = self.settings['central_lat']
        ds.attrs['area'] = f'{self.area} spatial average'
        ds.attrs['source'] = 'ERA5 + (LS)²D'
        ds.attrs['description'] = 'Generated by (LS)²D: https://github.com/LS2D & https://pypi.org/project/ls2d)'
        ds.attrs['reference'] = 'van Stratum et al. (2023). The benefits and challenges of downscaling a global reanalysis with doubly-periodic large-eddy simulations. JAMES, https://doi.org/10.1029/2023MS003750'

        return ds
