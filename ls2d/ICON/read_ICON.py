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
from scipy.interpolate import interp1d

# LS2D modules
import ls2d.src.spatial_tools as spatial
import ls2d.src.finite_difference as fd
from ls2d.src.messages import *

import ls2d.ICON.ICON_tools as ICON_tools
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
        Open all NetCDF files required for start->end period.

        Format is selected via settings['ICON_format'] (default: 'reglatlon'):
          - 'averaged'     : single time-averaged file from ICON_averaged/
          - 'reglatlon'    : time-varying remapped regular lat/lon files from ICON_reglatlon/
          - 'unstructured' : time-varying native unstructured grid files from ICON_unstructured/

        For 'reglatlon', additional sub-options control which file variants are used:
          - settings['ml_sliced']        (bool, default False)
                True  → model level files from ICON_reglatlon/sliced/
                False → model level files from ICON_reglatlon/
          - settings['pl_sliced']        (bool, default False)
                True  → pressure level files from ICON_reglatlon/sliced/pressure_levels_output/
                False → pressure level files depend on pl_terrain_aware (see below)
          - settings['pl_terrain_aware'] (bool, default True; ignored when pl_sliced=True)
                True  → terrain-aware files from ICON_reglatlon/pressure_levels_terrain_output/ (plx_ prefix)
                False → standard files from ICON_reglatlon/pressure_levels_output/             (pl_ prefix)

        settings['case_name'] should be the experiment-specific filename prefix,
        e.g. 'CLOUDLAB_MIP_input_130_'.
        """
        header('Reading ICON from {} to {}'.format(self.start, self.end))

        # Normalize trailing slash
        path = self.settings['ICON_path']
        if path[-1] != '/':
            path += '/'
        self.settings['ICON_path'] = path

        case = self.settings['case_name']
        fmt  = self.settings.get('ICON_format', 'reglatlon')
        message(f'ICON format: {fmt}')

        def check_files(files):
            file_missing = False
            for f in files:
                if not os.path.exists(f):
                    error('File "{}" does not exist...'.format(f), exit=False)
                    file_missing = True
            return file_missing

        if fmt == 'averaged':
            # Single time-averaged file; filename pattern differs from per-timestep files.
            date_str = f'{self.start.year:04d}{self.start.month:02d}{self.start.day:02d}'
            av_file  = os.path.join(path, 'ICON_averaged', f'{case}{date_str}_input_averaged.nc')

            message(f'Opening averaged file: {av_file}')
            if check_files([av_file]):
                error('Required ICON averaged file is missing.')

            self.fma = xr.open_dataset(av_file)
            self.fpa = None

        elif fmt == 'reglatlon':
            an_dates = ICON_tools.get_required_analysis(self.start, self.end)

            # --- Model level files ---
            ml_sliced = self.settings.get('ml_sliced', False)
            ml_subdir = os.path.join('ICON_reglatlon', 'sliced') if ml_sliced else 'ICON_reglatlon'
            ml_prefix = f'rmp_{case}'

            an_model_files = [ICON_tools.file_path(
                d.year, d.month, d.day, d.hour, d.minute,
                path, ml_subdir, ml_prefix, False) for d in an_dates]

            message(f'Model level: {ml_subdir}/ (sliced={ml_sliced})')

            # --- Pressure level files ---
            pl_sliced        = self.settings.get('pl_sliced', False)
            pl_terrain_aware = self.settings.get('pl_terrain_aware', True)  # default: terrain-aware

            if pl_sliced:
                pl_subdir = os.path.join('ICON_reglatlon', 'sliced', 'pressure_levels_output')
                pl_prefix = f'pl_rmp_{case}'
                message(f'Pressure level: {pl_subdir}/ (sliced=True)')
            elif pl_terrain_aware:
                pl_subdir = os.path.join('ICON_reglatlon', 'pressure_levels_terrain_output')
                pl_prefix = f'plx_rmp_{case}'
                message(f'Pressure level: {pl_subdir}/ (terrain-aware=True)')
            else:
                pl_subdir = os.path.join('ICON_reglatlon', 'pressure_levels_output')
                pl_prefix = f'pl_rmp_{case}'
                message(f'Pressure level: {pl_subdir}/ (terrain-aware=False)')

            an_pres_files = [ICON_tools.file_path(
                d.year, d.month, d.day, d.hour, d.minute,
                path, pl_subdir, pl_prefix, False) for d in an_dates]

            files_missing  = check_files(an_model_files)
            files_missing += check_files(an_pres_files)
            if files_missing:
                error('One or more required ICON files are missing.')

            self.fma = xr.open_mfdataset(an_model_files, combine='by_coords')
            self.fpa = xr.open_mfdataset(an_pres_files,  combine='by_coords')

        elif fmt == 'unstructured':
            an_dates = ICON_tools.get_required_analysis(self.start, self.end)

            # Unstructured files use the bare case prefix (no 'rmp_' prepended).
            an_model_files = [ICON_tools.file_path(
                d.year, d.month, d.day, d.hour, d.minute,
                path, 'ICON_unstructured', case, False) for d in an_dates]

            message('Model level: ICON_unstructured/ (native grid, no separate pressure-level files)')
            if check_files(an_model_files):
                error('One or more required ICON files are missing.')

            self.fma = xr.open_mfdataset(an_model_files, combine='by_coords')
            self.fpa = None

        else:
            error(f'Unknown ICON_format "{fmt}". Valid options: averaged, reglatlon, unstructured.')
 



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
        self.u  = get_variable(self.fma, 'u',    s3d)  # Zonal wind
        self.v  = get_variable(self.fma, 'v',    s3d)  # Meridional wind
        self.w  = get_variable(self.fma, 'w',    s3d)  # Vertical velocity
        self.T  = get_variable(self.fma, 'temp', s3d)  # Temperature
        self.qc = get_variable(self.fma, 'qc',   s3d)  # Cloud water specific humidity
        self.qi = get_variable(self.fma, 'qi',   s3d)  # Cloud ice specific humidity
        self.qr = get_variable(self.fma, 'qr',   s3d)  # Rain specific humidity
        self.qs = get_variable(self.fma, 'qs',   s3d)  # Snow specific humidity
        self.qv  = get_variable(self.fma, 'qv',   s3d) # Water vapor specific humidity
        
        self.p  = get_variable(self.fma, 'pres', s3d) # Pressure at full levels  
        self.zifc = get_variable(self.fma, 'z_ifc',s3d) # Geometric height at half levels a.s.l
        self.topoc =  get_variable(self.fma, 'topography_c',s2d) # Terrain height a.s.l
        self.zh = self.zifc 

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

        self.z_p = get_variable(self.fpa, 'z_ifc', s3d)       # Geopotential height on pressure levels (m)
        self.p_p = get_variable(self.fpa, 'plev_3', s1d)      # Pressure levels (Pa)


    def calc_derived_data(self):
        """
        Calculate derived properties; conversion model levels to pressure/height,
        prognostic variables used by LES, etc.
        """

        # self.ql  = self.qc + self.qi + self.qr + self.qs  # Total liquid/solid specific humidity (kg kg-1)
        self.ql  = self.qc
        self.qt  = self.qv + self.ql                       # Total specific humidity (kg kg-1)
        self.Tv  = utils.calc_virtual_temp(
                self.T, self.qv, self.qc, self.qi, self.qr, self.qs)  # Virtual temp on full levels (K)

        # Calculate half level pressure and heights
        self.ph  = np.zeros((self.ntime, self.nhalf, self.nlat, self.nlon))  # Half level pressure (Pa)
        self.z  = np.zeros((self.ntime, self.nfull, self.nlat, self.nlon))  # Full level geometric height a.s.l (m)
        
        self.ph[:,1:-1,:,:] = 0.5 * (self.p[:,:-1,:,:] + self.p[:,1:,:,:])
        self.ph[:,-1,:,:] = self.p[:,-1,:,:] + 0.5 * (self.p[:,-1,:,:] - self.p[:,-2,:,:])
        self.ph[:,0,:,:] = self.p[:,0,:,:] - 0.5 * (self.p[:,1,:,:] - self.p[:,0,:,:])

        self.z = 0.5 * (self.zh[:, :-1, :, :] + self.zh[:, 1:, :, :])

        # Other derived quantities
        self.exn  = utils.calc_exner(self.p)  # Exner on full model levels (-)
        self.th   = (self.T / self.exn)  # Potential temperature (K)
        self.thl  = self.th - (utils.Lv * self.qc) / (utils.cpd * self.exn)   # Liquid water potential temperature (K)
    
        self.rho  = self.p / (utils.Rd * self.Tv)  # Density at full levels (kg m-3)
        self.U    = (self.u**2. + self.v**2)**0.5  # Absolute horizontal wind (m s-1)
        # TODO: use large scale W from era5
        self.wls  = self.w
        
        self.Tvs  = utils.calc_virtual_temp(self.Ts, self.qvs)  # Estimate surface Tv using lowest model q (...)
        self.rhos = self.ps / (utils.Rd * self.Tvs)  # Surface density (kg m-3)
        self.exns = utils.calc_exner(self.ps)  # Exner at surface (-)
        self.wts = self.H / (self.rhos * utils.cpd)  # Surface kinematic heat flux (K m s-1)
        self.wthls = self.wts / self.exns  # Surface kinematic heat flux (K m s-1)
        self.wqs =  self.LH / (self.rhos * utils.Lv)# Surface kinematic moisture flux (kg kg-1 m s-1)
        self.fc = 2 * 7.2921e-5 * np.sin(np.deg2rad(self.settings['central_lat']))  # Coriolis parameter

        self.ths = self.Ts / self.exns
        self.thls = self.ths - (utils.Lv * 0.0) / (utils.cpd * self.exns)   # Surface liquid water potential temperature (K)

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

        
    def _interp_vars_to_les_grid(self, dict_arrays, var_list, vars_half,
                                   z_asl_full_sub, z_asl_half_sub, topo_sub,
                                   grid, n_mean_lat, n_mean_lon):
        """
        Interpolate a dict of native-grid arrays to the LES z-grid,
        column-by-column with terrain offset, then horizontally average.
        Results are stored as self.<var>_mean.
        """
        for var in var_list:
            is_half = var in vars_half
            n_target = len(grid.zh) if is_half else len(grid.z)

            setattr(self, f'{var}_mean', np.zeros((self.ntime, n_target)))
            var_array_3d = dict_arrays[var]

            for t in range(self.ntime):
                var_interp = np.full((n_mean_lat, n_mean_lon, n_target), np.nan)

                z_3d = z_asl_half_sub[t] if is_half else z_asl_full_sub[t]
                v_3d = var_array_3d[t]
                z_target_3d = (self.zh_agl_homo[t] + topo_sub[t]) if is_half \
                               else (self.z_agl_homo[t] + topo_sub[t])

                for j in range(n_mean_lat):
                    for i in range(n_mean_lon):
                        z_col = z_3d[:, j, i]
                        v_col = v_3d[:, j, i]
                        z_tgt = z_target_3d[:, j, i]

                        valid = ~np.isnan(z_col) & ~np.isnan(v_col)
                        if not np.any(valid):
                            continue

                        sort_idx = np.argsort(z_col[valid])
                        z_s = z_col[valid][sort_idx]
                        v_s = v_col[valid][sort_idx]
                        var_interp[j, i, :] = np.interp(z_tgt, z_s, v_s)

                getattr(self, f'{var}_mean')[t, :] = np.nanmean(var_interp, axis=(0, 1))


    def get_mean_profiles(self, grid, n_av_lon=0, n_av_lat=0):
        """
        Compute spatially-averaged vertical profiles on the LES grid.

        Interpolates state variables from the native ICON grid to the target
        LES z-grid column-by-column (terrain-relative), then takes the
        horizontal mean over the averaging domain.  Also computes half-level
        and surface/soil mean profiles used by get_les_input().
        """
        header('Calculating mean profiles on LES grid')

        # ── Domain centre and averaging bounds ──────────────────────────────
        self.i = np.abs(self.lons - self.settings['central_lon']).argmin()
        self.j = np.abs(self.lats - self.settings['central_lat']).argmin()

        dlon = (1 + 2*n_av_lon) * float(self.lons[1] - self.lons[0])
        dlat = (1 + 2*n_av_lat) * float(self.lats[1] - self.lats[0])
        self.area = f'{dlon:.5f}°×{dlat:.5f}°'
        message(f'Averaging ICON over a {self.area} spatial area.')

        istart = self.i - n_av_lon;  iend = self.i + n_av_lon + 1
        jstart = self.j - n_av_lat;  jend = self.j + n_av_lat + 1

        # Store geometry for potential reuse by callers
        self._istart, self._iend = istart, iend
        self._jstart, self._jend = jstart, jend

        center4d = np.s_[:, :, jstart:jend, istart:iend]
        center3d = np.s_[:, jstart:jend, istart:iend]
        self._center4d = center4d
        self._center3d = center3d

        # ── Height and terrain slice ─────────────────────────────────────────
        z_asl_full_sub = self.z[center4d]
        z_asl_half_sub = self.zh[center4d]
        topo_sub       = self.topoc[center3d]
        self._z_asl_full_sub = z_asl_full_sub
        self._z_asl_half_sub = z_asl_half_sub
        self._topo_sub       = topo_sub

        n_mean_lat = z_asl_full_sub.shape[2]
        n_mean_lon = z_asl_full_sub.shape[3]
        self._n_mean_lat = n_mean_lat
        self._n_mean_lon = n_mean_lon

        # ── LES target grid broadcast over space ─────────────────────────────
        self.z_agl  = np.tile(grid.z,  (self.ntime, 1))
        self.zh_agl = np.tile(grid.zh, (self.ntime, 1))
        self.z_agl_homo  = np.broadcast_to(
            self.z_agl[:, :, None, None],
            (self.ntime, len(grid.z),  n_mean_lat, n_mean_lon))
        self.zh_agl_homo = np.broadcast_to(
            self.zh_agl[:, :, None, None],
            (self.ntime, len(grid.zh), n_mean_lat, n_mean_lon))

        # ── State variable interpolation ─────────────────────────────────────
        vars_full = ['p', 'T', 'thl', 'qt', 'qc', 'qi', 'ql', 'qv', 'u', 'v', 'rho']
        vars_half = ['ph', 'w']
        dict_state = {var: getattr(self, var)[center4d] for var in vars_full + vars_half}

        self._interp_vars_to_les_grid(
            dict_state, vars_full + vars_half, vars_half,
            z_asl_full_sub, z_asl_half_sub, topo_sub,
            grid, n_mean_lat, n_mean_lon)

        # ── Half-level profiles (for radiation) ──────────────────────────────
        for src_var, dst_var in [('T', 'Th'), ('ql', 'qlh'), ('qv', 'qvh')]:
            src = getattr(self, f'{src_var}_mean')
            out = np.zeros_like(self.zh_agl)
            out[:, 1:-1] = 0.5 * (src[:, 1:] + src[:, :-1])
            out[:, 0]    = src[:, 0] + \
                ((out[:, 1] - src[:, 0]) / (self.zh_agl[:, 1] - self.z_agl[:, 0])) * \
                (self.zh_agl[:, 0] - self.z_agl[:, 0])
            out[:, -1]   = src[:, -1] + \
                ((src[:, -1] - out[:, -2]) / (self.z_agl[:, -1] - self.zh_agl[:, -2])) * \
                (self.zh_agl[:, -1] - self.z_agl[:, -1])
            if dst_var in ('qlh', 'qvh'):
                out = np.maximum(out, 0.0)
            setattr(self, f'{dst_var}_mean', out)

        # ── Surface and soil means ────────────────────────────────────────────
        for var in ['T_soil', 'theta_soil']:
            setattr(self, f'{var}_mean', getattr(self, var)[center4d].mean(axis=(2, 3)))
        for var in ['ps', 'Ts', 'thls', 'qvs', 'wts', 'wthls', 'wqs', 'rhos', 'z0']:
            setattr(self, f'{var}_mean', getattr(self, var)[center3d].mean(axis=(1, 2)))

    def get_les_input(self, z):
        """
        Package the already interpolated forcings into an xarray.Dataset
        """
        def add_ds_var(ds, name, data, dims, long_name, units):
            if dims is not None:
                ds[name] = (dims, data)
            else:
                ds[name] = data
            ds[name].attrs['long_name'] = long_name
            ds[name].attrs['units'] = units

        ds = xr.Dataset(
                coords = {
                    'time': self.datetime,
                    'z': z,  
                    'zh': self.zh_agl[0, :],  
                    'zs': self.z_theta_soil
                })

        ds['z'].attrs['long_name'] = 'full level height LES'
        ds['z'].attrs['units'] = 'm'
        ds['zh'].attrs['long_name'] = 'half level height LES'
        ds['zh'].attrs['units'] = 'm'
        ds['zs'].attrs['long_name'] = 'full level depth soil'
        ds['zs'].attrs['units'] = 'm'

        # =====================================================================
        # EXPLICIT MAPPING: 'output_var': ('source_prefix', 'long_name', 'units', ('dims'))
        # =====================================================================
        variables = {
                'thl': ('thl', 'liquid water potential temperature', 'K', ('time', 'z')),
                'qt':  ('qt', 'total specific humidity', 'kg kg-1', ('time', 'z')),
                'qv':  ('qv', 'vapor specific humidity', 'kg kg-1', ('time', 'z')),
                'ql':  ('ql', 'liquid specific humidity', 'kg kg-1', ('time', 'z')),
                'u':   ('u', 'zonal wind component', 'm s-1', ('time', 'z')),
                'v':   ('v', 'meridional wind component', 'm s-1', ('time', 'z')),
                
                # 'w' is calculated on half levels ('zh') and mapped to DALES 'wls'
                'w': ('w', 'vertical wind component', 'm s-1', ('time', 'zh')),
                'wls': ('wls', 'LS vertical wind component', 'm s-1', ('time', 'zh')),
                
                'p':   ('p', 'air pressure', 'Pa', ('time', 'z')),
                'dtthl_advec': ('dtthl_advec', 'advective tendency liquid water potential temp', 'K s-1', ('time', 'z')),
                'dtqt_advec':  ('dtqt_advec', 'advective tendency total specific humidity', 'kg kg-1 s-1', ('time', 'z')),
                'dtu_advec':   ('dtu_advec', 'advective tendency zonal wind', 'm s-2', ('time', 'z')),
                'dtv_advec':   ('dtv_advec', 'advective tendency meridional wind', 'm s-2', ('time', 'z')),
                'ug': ('ug', 'geostrophic wind component zonal wind', 'm s-1', ('time', 'z')),
                'vg': ('vg', 'geostrophic wind component meridional wind', 'm s-1', ('time', 'z')),
                }

        for out_var, (src_var, long_name, units, dims) in variables.items():
            src_attr = f'{src_var}_mean'
            if hasattr(self, src_attr):
                data = getattr(self, src_attr)
                add_ds_var(ds, out_var, data, dims, long_name, units)
            else:
                warning(f'Variable "{src_attr}" not found in averaged data.')

        add_ds_var(ds, 'time_sec', self.time_sec, ('time'), 'seconds since start of experiment', 's')

        # Radiation and manual variables (Dimensions are explicitly set to 'z' or 'zh' here)
        add_ds_var(ds, 'z_lay', self.z_agl, ('time', 'z'), 'Full level heights radiation', 'm')
        add_ds_var(ds, 'z_lev', self.zh_agl, ('time', 'zh'), 'Half level heights radiation', 'm')
        add_ds_var(ds, 'p_lay', self.p_mean, ('time', 'z'), 'full level pressure radiation', 'Pa')
        add_ds_var(ds, 'p_lev', self.ph_mean, ('time', 'zh'), 'half level pressure radiation', 'Pa')
        add_ds_var(ds, 't_lay', self.T_mean, ('time', 'z'), 'full level temperature radiation', 'K')
        add_ds_var(ds, 't_lev', self.Th_mean, ('time', 'zh'), 'half level temperature radiation', 'K')
        add_ds_var(ds, 'qv_lay', self.qv_mean, ('time', 'z'), 'full level qv radiation', 'kg kg-1')
        add_ds_var(ds, 'qv_lev', self.qvh_mean, ('time', 'zh'), 'half level qv radiation', 'kg kg-1')
        add_ds_var(ds, 'ql_lay', self.ql_mean, ('time', 'z'), 'full level ql radiation', 'kg kg-1')
        add_ds_var(ds, 'ql_lev', self.qlh_mean, ('time', 'zh'), 'half level ql radiation', 'kg kg-1')

        h2o_lay = self.qt_mean / (ep - ep * self.qt_mean)
        add_ds_var(ds, 'h2o_lay', h2o_lay, ('time', 'z'), 'moisture volume mixing ratio', '')

        # Soil & Surface variables
        add_ds_var(ds, 't_soil', self.T_soil_mean, ('time', 'zs'), 'soil temperature', 'K')
        add_ds_var(ds, 'theta_soil', self.theta_soil_mean, ('time', 'zs'), 'soil moisture content', 'm3 m-3')
        add_ds_var(ds, 'z0', self.z0_mean, ('time'), 'roughness length', 'm')
        add_ds_var(ds, 'ps', self.ps_mean, ('time'), 'surface pressure', 'Pa')
        add_ds_var(ds, 'ts', self.Ts_mean, ('time'), 'surface (skin) temperature', 'K')
        add_ds_var(ds, 'thls', self.thls_mean, ('time'), 'surface (skin) liquid water potential temperature', 'K')
        add_ds_var(ds, 'qvs', self.qvs_mean, ('time'), 'surface specific humidity', 'kg kg-1')
        add_ds_var(ds, 'wth', self.wthls_mean, ('time'), 'surface sensible heat flux', 'K m s-1')
        add_ds_var(ds, 'wq', self.wqs_mean, ('time'), 'surface latent heat flux', 'kg kg-1 m s-1')

        ds.attrs['fc'] = self.fc
        ds.attrs['central_lon'] = self.settings['central_lon']
        ds.attrs['central_lat'] = self.settings['central_lat']
        ds.attrs['area'] = f'{self.area} spatial average'
        ds.attrs['source'] = 'ICON + (LS)²D'

        return ds