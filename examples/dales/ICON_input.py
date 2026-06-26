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
import sys, os
import xarray as xr

# Third party modules
import numpy as np

# LS2D & custom modules
sys.path.append('/Users/yunpeichu/LS2D-ICON/ls2d')
import ls2d
import dales_ls2d_tools as dlt

# ============================================================
# 1. Settings
# ============================================================
settings = {
    'central_lat' : 47.0705,
    'central_lon' : 7.8735,
    'area_size'   : 1,
    'case_name'   : 'CLOUDLAB_MIP_input_130_',
    'ICON_path'   : '/Users/yunpeichu/CLOUDLAB_MIP/MIP_data/CLOUDLAB_MIP_input',
    'ICON_format' : 'reglatlon',
    'era5_path'      : '/Users/yunpeichu/LS2D-ICON/data/era5',
    'era5_case_name' : 'cloudlab',
    'start_date'  : datetime(year=2023, month=1, day=26, hour=0, minute=0),
    'end_date'    : datetime(year=2023, month=1, day=26, hour=6, minute=0),
    'write_log'   : True,
    'data_source' : 'CDS',
}

expnr = 19
out_dir = f'/Users/yunpeichu/LS2D-ICON/results/mpcseed/run_site_based_0{expnr}'
os.makedirs(out_dir, exist_ok=True)

tau_nudge = 10800    # Nudging time scale (s)
init_tke  = 1e-05    # Initial SGS-TKE (m2 s-2)

docstring = '(LS)2D case cloudlab: {} to {}'.format(
        settings['start_date'].isoformat(), settings['end_date'].isoformat())

# ============================================================
# 2. LES vertical grid
# ============================================================
# grid = ls2d.grid.Grid_stretched_capped(kmax=296, dz_start=5, k_T=210, s=0.02, k_M=280, dz_end=20)
grid = ls2d.grid.Grid_stretched_capped(kmax=156, dz_start=10, k_T=100, s=0.02, k_M=140, dz_end=25)

# ============================================================
# 3. ICON: load data and compute mean profiles on LES grid
#    → prof.inp, scalar.inp, nudge.inp, backrad.inp
# ============================================================
icon = ls2d.Read_ICON(settings)
icon.get_mean_profiles(grid=grid, n_av_lat=0, n_av_lon=0)

#
# prof.inp — initial thermodynamic and wind profiles
#
output = odict([
        ('z (m)',        grid.z),
        ('thl (K)',      icon.thl_mean[0, :]),
        ('qt (kg kg-1)', icon.qt_mean[0, :]),
        ('u (m s-1)',    icon.u_mean[0, :]),
        ('v (m s-1)',    icon.v_mean[0, :]),
        ('tke (m2 s-2)', np.ones(grid.kmax) * init_tke)])

dlt.write_profiles(
        os.path.join(out_dir, 'prof.inp.{0:03d}'.format(expnr)),
        output, grid.kmax, docstring)

#
# scalar.inp — initial scalar profiles (cloud droplet number concentration)
#
output = odict([
        ('z (m)',    grid.z),
        ('Nc (m-3)', np.zeros(grid.kmax))])

dlt.write_profiles(
        os.path.join(out_dir, 'scalar.inp.{0:03d}'.format(expnr)),
        output, grid.kmax, docstring)

#
# nudge.inp — time-varying nudging profiles (state variables only)
#
output = odict([
        ('z (m)',        grid.z),
        ('factor (-)',   np.ones_like(icon.u_mean)),
        ('u (m s-1)',    icon.u_mean),
        ('v (m s-1)',    icon.v_mean),
        ('w (m s-1)',    icon.w_mean),
        ('thl (K)',      icon.thl_mean),
        ('qt (kg kg-1)', icon.qt_mean)])

dlt.write_time_profiles(
        os.path.join(out_dir, 'nudge.inp.{0:03d}'.format(expnr)),
        icon.time_sec, output, grid.kmax, docstring)

# Save ICON state profiles to NetCDF
icon_les = icon.get_les_input(grid.z)
nc_fname = os.path.join(out_dir, f"{settings['case_name']}icon_les.nc")
encoding = {v: {'zlib': True, 'complevel': 4} for v in icon_les.data_vars}
icon_les.to_netcdf(nc_fname, encoding=encoding)
print(f"Saved ICON LES state profiles to {nc_fname}")

# ============================================================
# 4. ERA5: large-scale forcings → ls_flux.inp, lscale.inp
# ============================================================
era5_settings = {
    'central_lat' : settings['central_lat'],
    'central_lon' : settings['central_lon'],
    'start_date'  : settings['start_date'],
    'end_date'    : settings['end_date'],
    'era5_path'   : settings['era5_path'],
    'case_name'   : settings['era5_case_name'],
    'data_source' : settings.get('data_source', 'CDS'),
    'write_log'   : settings.get('write_log', True),
}
era5 = ls2d.Read_era5(era5_settings)
era5.calculate_forcings(n_av_lat=0, n_av_lon=0.5, method='2nd')

#
# backrad.inp — radiation background profile (ERA5 full model levels)
#
_p_era5_full  = era5.p_mean[0, :]
_T_era5_full  = era5.T_mean[0, :]
_qv_era5_full = era5.qv_mean[0, :]
assert _p_era5_full.size == era5.nfull, 'backrad must use ERA5 full levels'
_keep = np.concatenate(([True], np.diff(_p_era5_full) < 0))
if not _keep.all():
    print(' - backrad: dropping {} non-monotonic pressure level(s) at indices {}'
          .format((~_keep).sum(), np.where(~_keep)[0].tolist()))
dlt.create_backrad(
        _p_era5_full[_keep],
        _T_era5_full[_keep],
        _qv_era5_full[_keep],
        expnr=expnr, output_dir=out_dir, fmt='nc')

era5_les = era5.get_les_input(grid.z, zh=grid.zh)

#
# ls_flux.inp — time-varying surface fluxes and large-scale forcings
#
output_sfc = odict([
        ('time',   icon_les.time_sec.values),
        ('p_s',    icon_les.ps.values),
        ('wthl_s', icon_les.wth.values),
        ('wqt_s',  icon_les.wq.values),
        ('thls',    icon_les.thls.values),
        ('qt_s',   icon_les.qvs.values)])

output_ls = odict([
        ('time',   era5_les.time_sec.values),
        ('z',      grid.z),
        ('ug',     era5_les.ug.values),
        ('vg',     era5_les.vg.values),
        ('wls',    era5_les.wls.values),
        ('dqtdt',  era5_les.dtqt_advec.values),
        ('dthldt', era5_les.dtthl_advec.values),
        ('dudt',   era5_les.dtu_advec.values),
        ('dvdt',   era5_les.dtv_advec.values)])

dlt.write_forcings(
        os.path.join(out_dir, 'ls_flux.inp.{0:03d}'.format(expnr)),
        output_sfc, output_ls, docstring)

#
# lscale.inp — time-invariant large-scale profile (required by DALES)
#
zero = np.zeros(grid.kmax)
output = odict([
        ('height',    grid.z),
        ('ug',        era5_les.ug[0, :].values),
        ('vg',        era5_les.vg[0, :].values),
        ('wfls',      era5_les.wls[0, :].values),
        ('dqtdxls',   zero),
        ('dqtdyls',   zero),
        ('dqtdtls',   zero),
        ('dthldt',    zero)])

dlt.write_profiles(
        os.path.join(out_dir, 'lscale.inp.{0:03d}'.format(expnr)),
        output, grid.kmax, docstring)

import os
os._exit(0)