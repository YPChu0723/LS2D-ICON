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
import sys,os

import subprocess
import shutil

import xarray as xr
import numpy as np

# LS2D & custom modules
sys.path.append('/Users/yunpeichu/LS2D')
import ls2d
import dales_ls2d_tools as dlt

#
# Download ERA5 and generate LES initialisation and forcings
settings = {
#      **arctic_bounds,
    'central_lon' : 7.8735,
    'central_lat' : 47.0705,
    'area_size'   : 1,
    'case_name'   : 'mpcseed',
    'era5_path'   : '/Users/yunpeichu/LS2D/data',
    'era5_expver' : 1,
    'start_date'  : datetime(year=2023, month=1, day=26, hour=0),
    'end_date'    : datetime(year=2023, month=1, day=26, hour=12),
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
grid = ls2d.grid.Grid_three_stage(kmax=156, dz0=10, z_stretch_start=800, stretch_factor=0.015, dz_max=20)
# grid.plot()

expnr = 1
# --- MODIFICATION START ---
out_dir = f'/Users/yunpeichu/LS2D/results/{settings.get("case_name", "les_input")}/run_era5'
# Create directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)
# --- MODIFICATION END ---

# Interpolate ERA5 variables and forcings onto LES grid.
# In addition, `get_les_input` returns additional variables needed to init LES.
les_input = era.get_les_input(grid.z)

# Save les_input (xarray.Dataset) to a compressed NetCDF file
# --- Modified path ---
nc_fname = os.path.join(out_dir, f"{settings.get('case_name', 'les_input')}_les_input.nc")

encoding = {v: {'zlib': True, 'complevel': 4} for v in les_input.data_vars}
les_input.to_netcdf(nc_fname, encoding=encoding)
# era5_backrad_input = xr.open_dataset('/Users/yunpeichu/work_dales/mpc_seed/era5/run_001/ERA5.backrad.inp.nc')
print(f"Saved LES input to {nc_fname}")
print(les_input)

# DALES specific initialisation.

# Settings:

tau_nudge = 10800    # Nudging time scale atmosphere
init_tke = 0.1       # Initial SGS-TKE

docstring = '(LS)2D case cloudlab: {} to {}'.format(
        settings['start_date'].isoformat(), settings['end_date'].isoformat())

#
# Write initial profiles to `prof.inp.expnr`.
#
output = odict(
    [
        ('z (m)', grid.z),
        ('thl (K)', les_input.thl[0, :].values),
        ('qt (kg kg-1)', les_input.qt[0, :].values),
        ('u (m s-1)', les_input.u[0, :].values),
        ('v (m s-1)', les_input.v[0, :].values),
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
        ('T_s', np.zeros_like(les_input.time_sec)),  # Not sure if this works..
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

#
# Write radiation background profiles to `backrad.inp.expnr`.
#
dlt.create_backrad(
    les_input['p_lay'].mean(axis=0),
    les_input['t_lay'].mean(axis=0),
    les_input['h2o_lay'].mean(axis=0),
)

# Force exit to avoid spurious "Error in sys.excepthook" messages
# during Python 3.13 interpreter shutdown with netCDF4.
os._exit(0)


