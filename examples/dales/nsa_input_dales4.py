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
from datetime import datetime
from collections import OrderedDict as odict
import argparse
import sys, os
import glob

import subprocess
import shutil

import xarray as xr
import numpy as np
from scipy.interpolate import interp1d

# LS2D & custom modules
sys.path.append('/Users/yunpeichu/LS2D')
sys.path.append('/Users/yunpeichu/py/')
import ls2d
import dales_ls2d_tools as dlt
from thermo import get_theta_l, get_ql

# ── Command-line overrides (used by the pipeline; fall back to defaults) ─────
def _parse_args():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument('--start_date', default=None,
                   help='Start datetime ISO format, e.g. 2022-11-20T05:00:00')
    p.add_argument('--end_date', default=None,
                   help='End datetime ISO format, e.g. 2022-11-20T09:00:00')
    p.add_argument('--out_dir', default=None,
                   help='Override output directory')
    return p.parse_args()

_args = _parse_args()

_start_date = (
    datetime.fromisoformat(_args.start_date)
    if _args.start_date else datetime(year=2022, month=11, day=25, hour=6)
)
_end_date = (
    datetime.fromisoformat(_args.end_date)
    if _args.end_date else datetime(year=2022, month=11, day=25, hour=9)
)
# ─────────────────────────────────────────────────────────────────────────────

central_lon = -156.60899353027344
central_lat = 71.322998046875
#
# Download ERA5 and generate LES initialisation and forcings
settings = {
#      **arctic_bounds,
    'central_lon' : central_lon,
    'central_lat' : central_lat,
    'area_size'   : 1,
    'case_name'   : 'NSA',
    'era5_path'   : '/Users/yunpeichu/LS2D/data',
    'era5_expver' : 1,
    'start_date'  : _start_date,
    'end_date'    : _end_date,
    'write_log'   : True,
    'data_source' : 'CDS'
    }

# Download required ERA5 files:
ls2d.download_era5(settings)

# Read ERA5 data, and calculate derived properties (thl, etc.):
era = ls2d.Read_era5(settings)

# Calculate initial conditions and large-scale forcings for LES:
era.calculate_forcings(n_av=0, method='2nd')

# Define vertical grid LES:
grid = ls2d.grid.Grid_stretched_capped(kmax=296, dz_start=5, k_T=210, s=0.05, k_M=256, dz_end=45)
# grid.plot()

expnr = 1
out_dir = (
    _args.out_dir
    or f'/Users/yunpeichu/LS2D/results/{settings.get("case_name", "les_input")}/run_era5'
)
os.makedirs(out_dir, exist_ok=True)

# Interpolate ERA5 variables and forcings onto LES grid.
# In addition, `get_les_input` returns additional variables needed to init LES.
les_input = era.get_les_input(grid.z)

# Save les_input (xarray.Dataset) to a compressed NetCDF file
# --- Modified path ---
nc_fname = os.path.join(out_dir, f"{settings.get('case_name', 'les_input')}_les_input.nc")

# Compute surface liquid water potential temperature using thermo.get_theta_l
# ts in K, ps in Pa, q_c=0 (no liquid water at surface)
thl_s = get_theta_l(les_input['ts'].values, les_input['ps'].values, q_c=0.0)
les_input['thl_s'] = (('time',), thl_s)
les_input['thl_s'].attrs['long_name'] = 'surface liquid water potential temperature'
les_input['thl_s'].attrs['units'] = 'K'

encoding = {v: {'zlib': True, 'complevel': 4} for v in les_input.data_vars}
les_input.to_netcdf(nc_fname, encoding=encoding)
# era5_backrad_input = xr.open_dataset('/Users/yunpeichu/work_dales/mpc_seed/era5/run_001/ERA5.backrad.inp.nc')
print(f"Saved LES input to {nc_fname}")
print(les_input)

# DALES specific initialisation.

# Settings:
tau_nudge = 10800    # Nudging time scale atmosphere
init_tke = 0.1       # Initial SGS-TKE

docstring = '(LS)2D case ARM-NSA: {} to {}'.format(
        settings['start_date'].isoformat(), settings['end_date'].isoformat())

# =====================================================================
# Read ARM interpolated sonde for profile initialisation
# (nsainterpolatedsondeC1.c1: 1-min time resolution, (time, height) grid)
# =====================================================================
sonde_dir = '/Users/yunpeichu/Arctic_data/ARM-NSA/nsainterpolatedsondeC1.c1'
_date_str = settings['start_date'].strftime('%Y%m%d')
sonde_files = sorted(glob.glob(os.path.join(sonde_dir, f'*.{_date_str}.*.nc')))
if not sonde_files:
    raise FileNotFoundError(f'No interpolated sonde file found for {_date_str} in {sonde_dir}')
best_sonde = sonde_files[0]
print(f' - Using sonde: {os.path.basename(best_sonde)}')

import netCDF4 as _nc4
_ds = _nc4.Dataset(best_sonde)

# Select time index closest to start_date (time in seconds since midnight)
_t        = np.ma.filled(_ds['time'][:], np.nan)
_t_target = (settings['start_date'] - settings['start_date'].replace(
                hour=0, minute=0, second=0, microsecond=0)).total_seconds()
_tidx     = int(np.argmin(np.abs(_t - _t_target)))
print(f' - Sonde time index {_tidx}: {_t[_tidx]/3600:.2f} UTC (target {_t_target/3600:.2f} UTC)')

# Height: km -> m AGL (station altitude stored as scalar 'alt')
_station_alt = float(_ds['alt'][:])                            # m AMSL
_alt = np.ma.filled(_ds['height'][:], np.nan) * 1e3 - _station_alt  # m AGL

# Profile at selected time
_temp = np.ma.filled(_ds['temp'][_tidx, :], np.nan)   # C
_temp_K = _temp + 273.15                                # K
_qt  = np.ma.filled(_ds['sh'][_tidx, :],             np.nan)   # g/g == kg/kg (dimensionless ratio)
_u   = np.ma.filled(_ds['u_wind'][_tidx, :],         np.nan)   # m/s
_v   = np.ma.filled(_ds['v_wind'][_tidx, :],         np.nan)   # m/s
_p   = np.ma.filled(_ds['bar_pres'][_tidx, :],       np.nan) * 1000.0   # kPa -> Pa
_rh  = np.ma.filled(_ds['rh'][_tidx, :],             np.nan)   # % -> fraction
_ds.close()

_ql = get_ql(_qt, _rh, _temp, _p)
_thl = get_theta_l(_temp_K, _p, _ql) 

# Remove invalid levels (height already ascending)
_ok = np.isfinite(_alt) & np.isfinite(_temp) & np.isfinite(_qt) & \
      np.isfinite(_u)   & np.isfinite(_v) & np.isfinite(_p) & np.isfinite(_rh) & \
      np.isfinite(_thl)
_alt, _temp, _qt, _u, _v, _p, _rh , _thl = _alt[_ok], _temp[_ok], _qt[_ok], _u[_ok], _v[_ok], _p[_ok], _rh[_ok], _thl[_ok]

sonde_t = interp1d(_alt, _temp, bounds_error=False,
                     fill_value=(_temp[0], _temp[-1]))(grid.z)
sonde_qt  = interp1d(_alt, _qt,  bounds_error=False,
                     fill_value=(_qt[0],  _qt[-1]))(grid.z)
sonde_u   = interp1d(_alt, _u,   bounds_error=False,
                     fill_value=(_u[0],   _u[-1]))(grid.z)
sonde_v   = interp1d(_alt, _v,   bounds_error=False,
                     fill_value=(_v[0],   _v[-1]))(grid.z)
sonde_rh   = interp1d(_alt, _rh,  bounds_error=False,
                     fill_value=(_rh[0],  _rh[-1]))(grid.z)
sonde_p   = interp1d(_alt, _p,   bounds_error=False,
                     fill_value=(_p[0],   _p[-1]))(grid.z)

sonde_ql = get_ql(sonde_qt, sonde_rh, sonde_t, sonde_p)
sonde_thl = get_theta_l(sonde_t, sonde_p, sonde_ql)
# =====================================================================

#
# Write initial profiles to `prof.inp.expnr`.
#
output = odict(
    [
        ('z (m)', grid.z),
        ('thl (K)', sonde_thl),
        ('qt (kg kg-1)', sonde_qt),
        ('u (m s-1)', sonde_u),
        ('v (m s-1)', sonde_v),
        ('tke (m2 s-2)', np.ones(grid.kmax) * init_tke),
    ]
)
dlt.write_profiles(os.path.join(out_dir, 'prof.inp.{0:03d}'.format(expnr)), output, grid.kmax, docstring)

#
# Write initial scalar profiles to `scalar.inp.expnr`.
#
zero = np.zeros(grid.kmax)
output = odict([('z (m)', grid.z), ('qr (kg kg-1)', zero), ('nr (kg kg-1)', zero)])

dlt.write_profiles(os.path.join(out_dir, 'scalar.inp.{0:03d}'.format(expnr)), output, grid.kmax, docstring)

#
# Write large-scale forcings to `ls_flux.inp.expnr`.
#
output_sfc = odict(
    [
        ('time', les_input.time_sec.values),
        ('p_s', les_input.ps.values),
        ('T_s', les_input.ts.values),  # Not sure if this works..
        ('qt_s', np.zeros_like(les_input.time_sec)),
    ]
)

output_ls = odict(
    [
        ('time', les_input.time_sec.values),
        ('z', grid.z),
        ('ug', les_input.ug.values),
        ('vg', les_input.vg.values),
        ('wls', les_input.wls.values),
        ('dqtdt', les_input.dtqt_advec.values),
        ('dthldt', les_input.dtthl_advec.values),
        ('dudt', les_input.dtu_advec.values),
        ('dvdt', les_input.dtv_advec.values),
    ]
)

dlt.write_forcings(os.path.join(out_dir, 'ls_flux.inp.{0:03d}'.format(expnr)), output_sfc, output_ls, docstring)

#
# Write nudging profiles to `nudge.inp.expnr`.
#
output = odict(
    [
        ('z (m)', grid.z),
        ('factor (-)', np.ones_like(les_input.u.values)),
        ('u (m s-1)', les_input.u.values),
        ('v (m s-1)', les_input.v.values),
        ('w (m s-1)', np.zeros_like(les_input.u.values)),
        ('thl (K)', les_input.thl.values),
        ('qt (kg kg-1)', les_input.qt.values),
    ]
)

dlt.write_time_profiles(
    os.path.join(out_dir, 'nudge.inp.{0:03d}'.format(expnr)),
    les_input.time_sec.values,
    output,
    grid.kmax,
    docstring,
)

#
# Also create non-time dependent file (lscale.inp), required by DALES (why?)
#
zero = np.zeros_like(grid.z)

output = odict(
    [
        ('height', grid.z),
        ('ug', zero),
        ('vg', zero),
        ('wfls', zero),
        ('dqtdxls', zero),
        ('dqtdyls', zero),
        ('dqtdtls', zero),
        ('dthldt', zero),
    ]
)

dlt.write_profiles(os.path.join(out_dir, 'lscale.inp.{0:03d}'.format(expnr)), output, grid.kmax, docstring)

# =====================================================================
# Interpolate sonde profiles onto ERA5 model level heights
# for use as the initial profile in scm_in.nc (replaces prof.inp role)
# era.z_mean[0,:] is height in m AMSL (bottom-to-top); NSA station ~9m,
# so AMSL ≈ AGL to within 9 m.
# =====================================================================
_era_z = era.z_mean[0, :]   # (nf,) ERA5 full-level heights at t=0

def _interp_to_era(alt_agl, src):
    """Interpolate sonde variable onto ERA5 height levels; clamp outside range."""
    return interp1d(alt_agl, src, bounds_error=False,
                    fill_value=(src[0], src[-1]))(_era_z)

era_init_t = _interp_to_era(_alt, _temp)
era_init_thl = _interp_to_era(_alt, _thl)
era_init_qt  = _interp_to_era(_alt, _qt)
era_init_u   = _interp_to_era(_alt, _u)
era_init_v   = _interp_to_era(_alt, _v)

# =====================================================================
# Read time-varying albedo from ERA5 forecast-albedo file (nsafal.nc).
# fal is dimensionless (0-1); shape (valid_time, latitude, longitude).
# =====================================================================
_fal_path = '/Users/yunpeichu/Arctic_data/ERA5/nsafal.nc'
_ds_fal   = _nc4.Dataset(_fal_path)

_fal_lat  = _ds_fal.variables['latitude'][:]    # degrees_north
_fal_lon  = _ds_fal.variables['longitude'][:]   # degrees_east
_fal_time = _ds_fal.variables['valid_time'][:]  # seconds since 1970-01-01
_fal      = _ds_fal.variables['fal'][:]         # (valid_time, lat, lon)

# Nearest grid point to domain centre
_j_fal = int(np.argmin(np.abs(_fal_lat - central_lat)))
_i_fal = int(np.argmin(np.abs(_fal_lon - central_lon)))
print(f' - Albedo grid point: lat={_fal_lat[_j_fal]:.4f}, lon={_fal_lon[_i_fal]:.4f}')

# Match ERA5 datetimes to nearest nsafal time steps (both in epoch seconds)
_epoch    = datetime(1970, 1, 1)
_era_sec  = np.array([(d - _epoch).total_seconds() for d in era.datetime])
_t_fal    = np.array([int(np.argmin(np.abs(_fal_time - t))) for t in _era_sec])

albedo_ts = np.asarray(_fal[_t_fal, _j_fal, _i_fal], dtype='f8')
albedo_0  = float(albedo_ts[0])

_ds_fal.close()
print(f' - Albedo (fal): min={albedo_ts.min():.4f}, max={albedo_ts.max():.4f}, t=0={albedo_0:.4f}')
# =====================================================================
# Read ARM ground IR skin temperature at start-time from nsagndirtC1.b1.
# Use the 1-min observation nearest to the run start hour (default 06:00 UTC).
# Falls back to ERA5 Ts_mean if the file is absent or QC fails.
# =====================================================================
_gndirt_dir  = '/Users/yunpeichu/Arctic_data/ARM-NSA/nsagndirtC1.b1'
_gndirt_file = os.path.join(_gndirt_dir, f'nsagndirtC1.b1.{_date_str}.000000.nc')
arm_t_skin = None
if os.path.exists(_gndirt_file):
    _ds_gnd = _nc4.Dataset(_gndirt_file)
    _gnd_time = np.ma.filled(_ds_gnd['time'][:], np.nan)  # seconds since midnight
    _gnd_target = (settings['start_date'].hour * 3600
                   + settings['start_date'].minute * 60
                   + settings['start_date'].second)
    _gnd_tidx = int(np.argmin(np.abs(_gnd_time - _gnd_target)))
    _sfc_ir = float(np.ma.filled(_ds_gnd['sfc_ir_temp'][_gnd_tidx], np.nan))
    _qc_ir  = int(np.ma.filled(_ds_gnd['qc_sfc_ir_temp'][_gnd_tidx], -1))
    _ds_gnd.close()
    if np.isfinite(_sfc_ir) and _sfc_ir > 0 and _qc_ir == 0:
        arm_t_skin = _sfc_ir
        print(f' - ARM skin temperature at t={_gnd_time[_gnd_tidx]/3600:.2f} UTC: {arm_t_skin:.2f} K')
    else:
        print(f' - ARM skin temperature QC failed or missing (qc={_qc_ir}, val={_sfc_ir:.2f}); using ERA5')
else:
    print(f' - ARM gndirt file not found for {_date_str}; using ERA5 skin temperature')
# =====================================================================
# Arctic CCN concentration: 50 /cm³ = 5e7 /m³.
# This sets both the initial droplet number (nc0 in namoption) and the
# CCN reservoir (n_ccn in scm_in). Keep them consistent.
N_CCN = 50e6   # /m³  (50 /cm³, clean Arctic background)

#
# Write scm_in.nc for DALES testbed mode (ltestbed=.true.).
# sea_ice_frac and albedo from ERA5; t=0 profiles from ARM interpolated sonde.
# n_ccn presence in scm_in.nc automatically sets ltb_setccn=.true. in DALES.
#
# Placeholders (no observational source available):
#   n_ccn: clean Arctic 50 /cm³ constant profile (no CAMS data)
#   fradSWnet/LWnet: zeroed; radiation diagnosed online by DALES
#
dlt.create_scm_in(
    era,
    os.path.join(out_dir, 'scm_in.nc'),
    freeze_2step=True,
    albedo=albedo_ts,
    n_ccn=N_CCN,
    init_thl=era_init_thl,
    init_qt=era_init_qt,
    init_u=era_init_u,
    init_v=era_init_v,
    t_skin=arm_t_skin,
)

#
# Write radiation background profiles to `backrad.inp.expnr`.
#
dlt.create_backrad(
    les_input['p_lay'].mean(axis=0).values,
    les_input['t_lay'].mean(axis=0).values,
    les_input['h2o_lay'].mean(axis=0).values,
    o3=les_input['o3_lay'].mean(axis=0).values,
    expnr=expnr,
    output_dir=out_dir,
    fmt='text',
)

# =====================================================================
# Generate namoption.001 with day-specific values substituted into
# the reference template at /Users/yunpeichu/work_dales/NSA/namoption.001
# =====================================================================
import re

NAMOPTION_TEMPLATE = '/Users/yunpeichu/work_dales/NSA/namoption.001'

def _update_namelist_value(text, key, new_value):
    """Replace `key = <old_value>` with `key = <new_value>` in Fortran namelist text."""
    pattern = r'(?m)(^\s*' + re.escape(key) + r'\s*=\s*)([^\n,/]+)'
    if not re.search(pattern, text):
        print(f'  WARNING: key "{key}" not found in namoption template')
        return text
    return re.sub(pattern, lambda m: m.group(1) + str(new_value), text)

def write_namoption(out_path, start_date, end_date, grid, era, albedo, n_ccn, siconc=None,
                    expnr=1, template_path=NAMOPTION_TEMPLATE):
    """Write namoption.XXX with day-specific values for a DALES testbed run."""
    with open(template_path, 'r') as f:
        text = f.read()

    runtime = int((end_date - start_date).total_seconds())
    xday    = int(start_date.strftime('%j'))     # day of year (1-366)
    xtime   = float(start_date.hour) + start_date.minute / 60.0
    xlat    = float(central_lat)
    xlon    = float(central_lon)
    ps_0    = float(era.ps_mean[0])
    z0m_0   = float(era.z0m_mean[0])
    thls_0 = get_theta_l(era.Ts_mean[0], ps_0, q_c=0)

    substitutions = {
        'iexpnr'   : expnr,
        'runtime'  : runtime,
        'kmax'     : grid.kmax,
        'khigh'    : grid.kmax,   # namfielddump
        'xlat'     : f'{xlat:.4f}',
        'xlon'     : f'{xlon:.4f}',
        'xday'     : xday,
        'xtime'    : f'{xtime:.1f}',
        'ps'       : f'{ps_0:.3f}',
        'thls'     : f'{thls_0:.2f}',
        'albedoav' : f'{albedo:.4f}',
        'z0'       : f'{z0m_0:.6f}',
        'nc0'      : f'{N_CCN:.1f}',   # initial cloud droplet number, consistent with n_ccn in scm_in
    }

    for key, value in substitutions.items():
        text = _update_namelist_value(text, key, value)

    # Update expnr in the output filename itself
    fname = os.path.join(out_path, f'namoption.{expnr:03d}')
    with open(fname, 'w') as f:
        f.write(text)
    print(f' - Saved {fname}')

write_namoption(
    out_dir,
    settings['start_date'],
    settings['end_date'],
    grid,
    era,
    albedo=albedo_0,
    n_ccn=N_CCN,
    expnr=expnr,
)

# Force exit to avoid spurious "Error in sys.excepthook" messages
# during Python 3.13 interpreter shutdown with netCDF4.
os._exit(0)


