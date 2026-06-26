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

from collections import OrderedDict as odict
import matplotlib.pyplot as pl
import netCDF4 as nc4
import numpy as np
import datetime
import os


# ---------------------------
# "Private" help functions
# ---------------------------
def _get_or_default(dict, name, shape, default_value):
    if name in dict:
        return dict[name]
    else:
        print(' - No input for "{}", defaulting values at zero'.format(name))
        return default_value * np.ones(shape)


# ---------------------------
# Function to write the DALES input
# ---------------------------
def write_profiles(file_name, variables, nlev, docstring=''):
    """
    Write the prof.inp.xxx input profiles for DALES
    """

    print(' - Saving {}'.format(file_name))

    f = open(file_name, 'w')

    # Write header (description file)
    if docstring == '':
        f.write('DALES\n')
    else:
        f.write('{}\n'.format(docstring))

    # Write header (column names)
    for var in variables.keys():
        f.write('{0:^17s} '.format(var))
    f.write('\n')

    # Write data
    for k in range(nlev):
        for var in variables.keys():
            f.write('{0:+1.10E} '.format(variables[var][k]))
        f.write('\n')

    f.close()


def write_time_profiles(file_name, time, variables, nlev, docstring=''):
    """
    Write time varying input profiles for DALES
    """

    print(' - Saving {}'.format(file_name))

    f = open(file_name, 'w')

    # Write header (description file)
    if docstring == '':
        f.write('DALES\n')
    else:
        f.write('{}\n'.format(docstring))

    # Write time dependent profiles
    for t in range(time.size):
        f.write('\n')

        # Write header (column names)
        for var in variables.keys():
            f.write('{0:^17s} '.format(var))
        f.write('\n')

        # Write time
        f.write('# {0:1.8E}\n'.format(time[t]))

        # Write data
        for k in range(nlev):
            for var in variables.keys():
                if len(variables[var].shape) == 1:
                    f.write('{0:+1.10E} '.format(variables[var][k]))
                else:
                    f.write('{0:+1.10E} '.format(variables[var][t, k]))
            f.write('\n')

    f.close()


def write_dummy_forcings(file_name, n_scalars, z, docstring):
    """
    Write dummy forcings
    """

    print('Saving {}'.format(file_name))

    f = open(file_name, 'w')

    # Write header (description file)
    if docstring == '':
        f.write('DALES\n\n')
    else:
        f.write('{}\n\n'.format(docstring))

    # Surface fluxes (zero)
    f.write('{0:^15s} '.format('time'))
    for i in range(n_scalars):
        f.write('{0:>10s}{1:<8d}'.format('sv', i + 1))
    f.write('\n')

    for time in [0, 1e6]:
        f.write('{0:+1.10E} '.format(time))
        for i in range(n_scalars):
            f.write('{0:+1.10E} '.format(0))
        f.write('\n')

    # Atmospheric forcings
    f.write('\n')

    for time in [0, 1e6]:
        f.write('# {0:+1.10E}\n'.format(time))
        for k in range(z.size):
            f.write('{0:+1.10E} '.format(z[k]))
            for i in range(n_scalars):
                f.write('{0:+1.10E} '.format(0))
            f.write('\n')

    f.close()


def write_forcings(file_name, timedep_sfc, timedep_atm, docstring=''):
    """
    Write the ls_flux.inp.xxx files
    """

    print(' - Saving {}'.format(file_name))

    f = open(file_name, 'w')

    # Always write something; DALES expects three line header
    if docstring == '':
        f.write('DALES time dependent input\n')
    else:
        f.write('{}\n'.format(docstring))

    # Write surface variables
    f.write(
        '{0:^15s} {1:^15s} {2:^15s} {3:^15s} {4:^15s} {5:^15s}\n'.format(
            'time', 'wthl_s', 'wqt_s', 'T_s', 'qt_s', 'p_s'
        )
    )
    f.write(
        '{0:^15s} {1:^15s} {2:^15s} {3:^15s} {4:^15s} {5:^15s}\n'.format(
            '(s)', '(K m s-1)', '(kg kg-1 m s-1)', '(K)', '(kg kg-1)', '(Pa)'
        )
    )

    if timedep_sfc is None:
        # Write a large initial time, so DALES will disable the surface timedep
        f.write('{0:+1.8E} {1:+1.8E} {2:+1.8E} {3:+1.8E} {4:+1.8E} {5:+1.8E}\n'.format(1e16, -1, -1, -1, -1, -1))
    else:
        nt = timedep_sfc['time'].size
        time = _get_or_default(timedep_sfc, 'time', nt, 0)
        wthls = _get_or_default(timedep_sfc, 'wthl_s', nt, 0)
        wqts = _get_or_default(timedep_sfc, 'wqt_s', nt, 0)
        Ts = _get_or_default(timedep_sfc, 'T_s', nt, 0)
        qts = _get_or_default(timedep_sfc, 'qt_s', nt, 0)
        ps = _get_or_default(timedep_sfc, 'p_s', nt, 0)

        for t in range(nt):
            f.write(
                '{0:+1.8E} {1:+1.8E} {2:+1.8E} {3:+1.8E} {4:+1.8E} {5:+1.8E}\n'.format(
                    time[t], wthls[t], wqts[t], Ts[t], qts[t], ps[t]
                )
            )

    if timedep_atm is not None:
        time = timedep_atm['time']
        z = timedep_atm['z']
        nt = time.size
        nlev = z.size

        ug = _get_or_default(timedep_atm, 'ug', [nt, nlev], 0)
        vg = _get_or_default(timedep_atm, 'vg', [nt, nlev], 0)
        wls = _get_or_default(timedep_atm, 'wls', [nt, nlev], 0)
        dxq = _get_or_default(timedep_atm, 'dqtdx', [nt, nlev], 0)
        dyq = _get_or_default(timedep_atm, 'dqtdy', [nt, nlev], 0)
        dtq = _get_or_default(timedep_atm, 'dqtdt', [nt, nlev], 0)
        dtth = _get_or_default(timedep_atm, 'dthldt', [nt, nlev], 0)
        dtu = _get_or_default(timedep_atm, 'dudt', [nt, nlev], 0)
        dtv = _get_or_default(timedep_atm, 'dvdt', [nt, nlev], 0)

        # Write atmospheric data
        for t in range(nt):
            f.write('\n')
            # Write header:
            f.write(
                '{0:^19s} {1:^19s} {2:^19s} {3:^19s} {4:^19s} {5:^19s} {6:^19s} {7:^19s} {8:^19s} {9:^19s}\n'.format(
                    'z (m)',
                    'u_g (m s-1)',
                    'v_g (m s-1)',
                    'w_ls (m s-1)',
                    'dqtdx (kg kg-1 m-1)',
                    'dqtdy (kg kg m-1)',
                    'dqtdt (kg kg-1 s-1)',
                    'dthldt (K s-1)',
                    'dudt (m s-2)',
                    'dvdt (m s-2)',
                )
            )

            # Write current time:
            f.write('# {0:1.8E}\n'.format(time[t]))

            # Write profiles:
            for k in range(nlev):
                f.write(
                    '{0:+1.12E} {1:+1.12E} {2:+1.12E} {3:+1.12E} {4:+1.12E} {5:+1.12E} {6:+1.12E} {7:+1.12E} {8:+1.12E} {9:+1.12E}\n'.format(
                        z[k],
                        ug[t, k],
                        vg[t, k],
                        wls[t, k],
                        dxq[t, k],
                        dyq[t, k],
                        dtq[t, k],
                        dtth[t, k],
                        dtu[t, k],
                        dtv[t, k],
                    )
                )

    f.close()


def create_backrad(p, T, q, o3=None, lwc=None, expnr=1, output_dir='.', fmt='text'):
    """
    Create the background profiles for DALES radiation schemes.

    fmt='text' : text file for modradfull / d4stream (backrad.inp.001)
                 First line: Tsurf ns
                 Then ns lines: p(Pa)  T(K)  q(kg/kg)  o3(ppmv)  lwc
    fmt='nc'   : NetCDF file for modradrrtmg (backrad.inp.001.nc)
    """

    if fmt == 'text':
        fname = os.path.join(output_dir, 'backrad.inp.{0:03d}'.format(expnr))
        print(' - Saving {}'.format(fname))

        ns = p.size
        Tsurf = float(T[0])  # index 0 = lowest level (surface), array is bottom-to-top

        # d4stream_setup expects pressure increasing with index (TOA first, surface last).
        # LS2D stores arrays bottom-to-top, so reverse before writing.
        p_out   = p[::-1]
        T_out   = T[::-1]
        q_out   = q[::-1]
        o3_data  = (o3[::-1]  if o3  is not None else np.zeros(ns))
        lwc_data = (lwc[::-1] if lwc is not None else np.zeros(ns))

        with open(fname, 'w') as f:
            f.write('{:.4f}  {:d}\n'.format(Tsurf, ns))
            for k in range(ns):
                f.write('{:.4f}  {:.4f}  {:.6E}  {:.6E}  {:.6E}\n'.format(
                    float(p_out[k]), float(T_out[k]), float(q_out[k]),
                    float(o3_data[k]), float(lwc_data[k])))

    elif fmt == 'nc':
        fname = os.path.join(output_dir, 'backrad.inp.{0:03d}.nc'.format(expnr))
        print(' - Saving {}'.format(fname))

        nc_file = nc4.Dataset(fname, 'w')
        nc_file.createDimension('lev', p.size)

        p_var = nc_file.createVariable('lev', 'f4', ('lev'))
        T_var = nc_file.createVariable('T', 'f4', ('lev'))
        q_var = nc_file.createVariable('q', 'f4', ('lev'))

        p_var[:] = p
        T_var[:] = T
        q_var[:] = q

        if o3 is not None:
            o3_var = nc_file.createVariable('o3', 'f4', ('lev'))
            o3_var[:] = o3

        nc_file.close()

    else:
        raise ValueError('Unknown fmt="{}"; use "text" or "nc".'.format(fmt))


def create_scm_in(era, file_path, albedo=0.1, sea_ice_frac=None, t_skin_seaice=None, n_ccn=1e8,
                  time_slice=None, freeze_2step=False,
                  init_thl=None, init_qt=None, init_u=None, init_v=None,
                  t_skin=None):
    """
    Generate a scm_in.nc file for DALES testbed mode (ltestbed=.true.) from ERA5 data.

    The scm_in file is read by modtestbed.f90 in DALES and provides time-varying
    large-scale forcing, nudging targets, surface boundary conditions, and radiation
    background profiles on the native ERA5 model levels.

    Parameters
    ----------
    era : Read_era5
        ERA5 object after calculate_forcings() has been called.
    file_path : str
        Output path for scm_in.nc.
    albedo : float
        Surface albedo (ERA5 does not provide a time-mean albedo; default 0.1).
    sea_ice_frac : float or array-like or None
        Sea ice fraction (0-1). Scalar or (ntime,) array. If None, the variable
        is omitted and modtestbed will not set the sea ice fraction.
    t_skin_seaice : float or array-like or None
        Sea ice skin temperature [K]. Scalar or (ntime,) array. If None, the
        general skin temperature (t_skin) is used. Requires sea_ice_frac != None.
    n_ccn : float
        Initial CCN concentration [/m3] (uniform profile). Default 1e8.
    time_slice : slice, array-like of int, or None
        Subset of time steps to write. ``None`` (default) writes all steps.
        Examples: ``slice(0, 4)`` for the first 4 steps; ``[0, 2, 4]`` for
        specific indices.
    freeze_2step : bool
        If ``True``, write exactly 2 time steps (``time=[0, 86400]``) with
        identical forcing profiles derived from the first selected ERA5 time
        step.  This produces a steady-state forcing file suitable for runs up
        to 86 400 s (1 day).  ``time_slice`` is still honoured when picking
        the source step (first index is used); the output ``time`` and
        ``second`` coordinates are always ``[0, 86400]``.
    init_thl : array-like of shape (nf,) or None
        Liquid water potential temperature [K] on ERA5 full levels (bottom-to-top,
        same grid as ``era.z_mean``). If provided, overrides the ERA5 temperature
        profile at the first selected time step.  Intended for sonde initialisation.
        The conversion to absolute temperature (T = thl * exner) assumes ql=0,
        which is appropriate for radiosonde data.
    init_qt : array-like of shape (nf,) or None
        Total water specific humidity [kg/kg] on ERA5 full levels.  For sonde
        data with no liquid water this equals water vapour q.
    init_u : array-like of shape (nf,) or None
        Zonal wind [m/s] on ERA5 full levels for the initial time step.
    init_v : array-like of shape (nf,) or None
        Meridional wind [m/s] on ERA5 full levels for the initial time step.

    Notes
    -----
    Sign convention for surface fluxes in scm_in (IFS/GCM convention):
      upward = negative.  modtestbed.f90 flips the sign internally.

    The temperature advective tendency is derived from the thl tendency as:
      tadv = Pi * dtthl_advec + Lv/cp * dqc_advec  (ice term neglected)
    so that modtestbed.f90 correctly recovers the original thl tendency via:
      dthl/dt = tadv/Pi - Lv*ladv/(Pi*cp),  with ladv = dqc_advec.

    Ozone is converted from ppmv (LS2D) to kg/kg (scm_in) using molar masses of
    O3 (47.9982) and dry air (28.9644).
    """

    # Physical constants (matching IFS and DALES defaults)
    grav    = 9.80665
    cpd     = 1004.709
    Lv      = 2.5008e6
    Rd      = 287.0597
    pref0   = 1e5
    o3_conv = 47.9982 / 28.9644 / 1e6   # ppmv -> kg/kg

    # --- Time selection ---
    if freeze_2step:
        # Resolve source index pool from time_slice, then take only the first.
        if time_slice is None:
            _src = np.arange(era.ntime)
        elif isinstance(time_slice, slice):
            _src = np.arange(era.ntime)[time_slice]
        else:
            _src = np.asarray(time_slice, dtype=int)
        t_idx = _src[:1]   # single source row for the frozen profile
        nt = 2
    else:
        if time_slice is None:
            t_idx = np.arange(era.ntime)
        elif isinstance(time_slice, slice):
            t_idx = np.arange(era.ntime)[time_slice]
        else:
            t_idx = np.asarray(time_slice, dtype=int)
        nt = len(t_idx)

    def _ts(arr):
        """Select time steps (axis 0); scalars are returned unchanged.
        In freeze_2step mode the single source row is duplicated to 2 rows."""
        a = np.asarray(arr)
        if a.ndim == 0:
            return a
        row = a[t_idx[0]]
        if freeze_2step:
            return np.stack([row, row], axis=0)
        return a[t_idx]

    nf = era.nfull   # ERA5 full model levels (137 for L137)
    nh = era.nhalf   # ERA5 half model levels (138 for L137)

    # --- Initial profile arrays (possibly overridden by sonde at first time step) ---

    T_local = np.zeros((nt, nf), dtype='f8')
    q_local = np.zeros((nt, nf), dtype='f8')
    u_local = np.zeros((nt, nf), dtype='f8')
    v_local = np.zeros((nt, nf), dtype='f8')

    if any(x is not None for x in (init_thl, init_qt, init_u, init_v)):
        if init_thl is not None:
            # T = t * exner  (assumes ql=0, valid for sonde data)
            T_local[:, :] = np.asarray(init_thl, dtype='f8')
        if init_qt is not None:
            # For sonde (no liquid water): qt == qv, written as 'q' in scm_in
            q_local[:, :] = np.asarray(init_qt, dtype='f8')
        if init_u is not None:
            u_local[:, :] = np.asarray(init_u, dtype='f8')
        if init_v is not None:
            v_local[:, :] = np.asarray(init_v, dtype='f8')

    # --- Derived fields ---
    exner_mean = (era.p_mean  / pref0) ** (Rd / cpd)
    exns_mean  = (era.ps_mean / pref0) ** (Rd / cpd)

    # Pressure velocity omega [Pa/s] from w [m/s]:  omega = -rho * g * w
    omega = -era.wls_mean * era.rho_mean * grav

    # Ozone: ppmv -> kg/kg
    o3_kgkg = era.o3_mean * o3_conv

    # Temperature advective tendency [K/s] from liquid water pot. temperature tendency.
    # dT/dt_adv = Pi * dthl/dt_adv + Lv/cp * dqc/dt_adv  (ice term neglected)
    # modtestbed recovers thl tendency as: dthl/dt = tadv/Pi - Lv*ladv/(Pi*cp)
    # So setting tadv = dT/dt_adv and ladv = dqc/dt_adv is fully consistent.
    tadv = (era.dtthl_advec_mean * exner_mean
            + (Lv / cpd) * era.dtqc_advec_mean)

    # Surface fluxes (IFS convention: upward = negative, modtestbed flips sign)
    sfc_sens_flx = -era.wths_mean * era.rhos_mean * cpd * exns_mean  # W/m2
    sfc_lat_flx  = -era.wqs_mean  * Lv                               # W/m2

    # Time metadata
    epoch     = datetime.datetime(1970, 1, 1)
    base_time = np.array([(d - epoch).total_seconds() for d in era.datetime], dtype='f8')
    date_arr  = np.array([int(d.strftime('%Y%m%d')) for d in era.datetime], dtype='i4')

    lat_val = float(era.lats[era.j])
    lon_val = float(era.lons[era.i])

    # --- Create NetCDF file ---
    print(' - Saving {}'.format(file_path))
    nc = nc4.Dataset(file_path, 'w')

    # Dimensions
    nc.createDimension('time',   nt)
    nc.createDimension('nlev',   nf)
    nc.createDimension('nlevp1', nh)

    def _v(name, dims, data, long_name, units, dtype='f4'):
        """Create a compressed NetCDF variable with attributes."""
        var = nc.createVariable(name, dtype, dims, zlib=True, complevel=4)
        var.long_name = long_name
        var.units     = units
        var[:]        = np.asarray(data, dtype=dtype)

    # --- Coordinate / dimension variables ---
    _time_coord = np.array([0, 86400], dtype='i4') if freeze_2step else _ts(era.time_sec)
    _v('time',   ('time',),   _time_coord,            'time', 'seconds since 00 UTC on first day', dtype='i4')
    _v('nlev',   ('nlev',),   np.arange(nf),      'model full levels',  '', dtype='i4')
    _v('nlevp1', ('nlevp1',), np.arange(nh),      'model half levels',  '', dtype='i4')
    # --- Scalar time-series variables ---
    _v('base_time',    ('time',), _ts(base_time),           'epoch time',                         'seconds since 1-1-1970 00:00', dtype='f8')
    _v('date',         ('time',), _ts(date_arr),            'date',                               'yyyymmdd', dtype='i4')
    _v('second',       ('time',), _time_coord,              'seconds since start of sequence',    's')
    _v('lat',          ('time',), np.full(nt, lat_val),     'latitude',                           'degrees North')
    _v('lon',          ('time',), np.full(nt, lon_val),     'longitude',                          'degrees East')
    _v('lat_grid',     ('time',), np.full(nt, lat_val),     'latitude of closest IFS gridpoint',  'degrees North')
    _v('lon_grid',     ('time',), np.full(nt, lon_val),     'longitude of closest IFS gridpoint', 'degrees East')
    _v('ps',           ('time',), _ts(era.ps_mean),         'surface pressure',                   'Pa')
    if t_skin is not None:
        _t_skin_arr = np.full(nt, float(t_skin), dtype='f4')
    else:
        _t_skin_arr = np.asarray(_ts(era.Ts_mean), dtype='f4')
    _v('t_skin',       ('time',), _t_skin_arr,              'skin temperature',                   'K')
    _v('sfc_sens_flx', ('time',), _ts(sfc_sens_flx),        'surface sensible heat flux',          'W/m2')
    _v('sfc_lat_flx',  ('time',), _ts(sfc_lat_flx),         'surface latent heat flux',            'W/m2')
    _v('mom_rough',    ('time',), _ts(era.z0m_mean),        'roughness length for momentum',       'm')
    _v('heat_rough',   ('time',), _ts(era.z0h_mean),        'roughness length for heat',           'm')
    _alb = np.asarray(albedo, dtype='f4')
    if _alb.ndim == 0:
        _alb = np.full(nt, float(albedo), dtype='f4')
    elif len(_alb) == era.ntime:
        _alb = _ts(_alb)   # handles both normal mode and freeze_2step
    else:
        _alb = np.full(nt, float(_alb.flat[0]), dtype='f4')
    _v('albedo',       ('time',), _alb,                     'albedo',                             '0-1')
    # _v('open_sst',     ('time',), _ts(era.sst_mean),        'open sea surface temperature',        'K')
    # _v('t_skin_ocean', ('time',), _ts(era.sst_mean),        'skin temperature - ocean',            'K')
    # _v('orog',         ('time',), np.zeros(nt),             'orography - surface geopotential',   'm2/s2')

    # if sea_ice_frac is not None:
    #     _sif_arr = np.asarray(sea_ice_frac, dtype='f4')
    #     if _sif_arr.ndim == 1 and len(_sif_arr) == era.ntime:
    #         _sif_arr = _sif_arr[t_idx]
    #     sif = np.broadcast_to(_sif_arr, (nt,)).copy()
    #     _v('sea_ice_frct',  ('time',), sif, 'sea ice fraction',        '0-1')
    #     if t_skin_seaice is None:
    #         tsi = _ts(era.Ts_mean)
    #     else:
    #         _tsi_arr = np.asarray(t_skin_seaice, dtype='f4')
    #         if _tsi_arr.ndim == 1 and len(_tsi_arr) == era.ntime:
    #             _tsi_arr = _tsi_arr[t_idx]
    #         tsi = np.broadcast_to(_tsi_arr, (nt,)).copy()
    #     _v('t_skin_seaice', ('time',), tsi, 'skin temperature - sea ice', 'K')

    # --- Full-level profile variables (time, nlev) ---
    _v('height_f',       ('time', 'nlev'), _ts(era.z_mean),              'full level height',   'm')
    _v('pressure_f',     ('time', 'nlev'), _ts(era.p_mean),              'full level pressure', 'Pa')
    # _v('gz_f',           ('time', 'nlev'), _ts(era.z_mean) * grav,       'geopotential height', 'm2/s2')
    _v('u',              ('time', 'nlev'), _ts(era.u_mean),              'zonal wind (domain averaged)',          'm/s')
    _v('v',              ('time', 'nlev'), _ts(era.v_mean),              'meridional wind (domain averaged)',     'm/s')
    _v('u_local',        ('time', 'nlev'), u_local,                        'zonal wind (at domain midpoint)',       'm/s')
    _v('v_local',        ('time', 'nlev'), v_local,                        'meridional wind (at domain midpoint)',  'm/s')
    _v('t',              ('time', 'nlev'), _ts(era.T_mean),              'temperature (domain averaged)',         'K')
    _v('t_local',        ('time', 'nlev'), T_local,                        'temperature (at domain midpoint)',      'K')
    _v('q',              ('time', 'nlev'), _ts(era.q_mean),              'water vapor mixing ratio (domain averaged)',         'kg/kg')
    _v('q_local',        ('time', 'nlev'), q_local,                        'water vapor specific humidity (at domain midpoint)', 'kg/kg')
    _v('ql',             ('time', 'nlev'), _ts(era.qc_mean),             'liquid water mixing ratio (domain averaged)',         'kg/kg')
    _v('ql_local',       ('time', 'nlev'), _ts(era.qc_mean),             'liquid water specific humidity (at domain midpoint)', 'kg/kg')
    _v('qi',             ('time', 'nlev'), _ts(era.qi_mean),             'ice water mixing ratio (domain averaged)',            'kg/kg')
    _v('qi_local',       ('time', 'nlev'), _ts(era.qi_mean),             'ice water specific humidity (at domain midpoint)',    'kg/kg')
    _v('omega',          ('time', 'nlev'), _ts(omega),                   'large-scale pressure velocity (domain averaged)',     'Pa/s')
    _v('ug',             ('time', 'nlev'), _ts(era.ug_mean),             'geostrophic wind - zonal component',       'm/s')
    _v('vg',             ('time', 'nlev'), _ts(era.vg_mean),             'geostrophic wind - meridional component',  'm/s')
    _v('tadv',           ('time', 'nlev'), _ts(tadv),                    'tendency in temperature due to large-scale horizontal advection',    'K/s')
    _v('qadv',           ('time', 'nlev'), _ts(era.dtqt_advec_mean),     'tendency in water vapor due to large-scale horizontal advection',    'kg/kg/s')
    _v('uadv',           ('time', 'nlev'), _ts(era.dtu_advec_mean),      'tendency in zonal wind due to large-scale horizontal advection',     'm/s2')
    _v('vadv',           ('time', 'nlev'), _ts(era.dtv_advec_mean),      'tendency in meridional wind due to large-scale horizontal advection','m/s2')
    _v('ladv',           ('time', 'nlev'), _ts(era.dtqc_advec_mean),     'tendency in liquid water spec hum due to large-scale horizontal advection', 'kg/kg/s')
    _v('iadv',           ('time', 'nlev'), _ts(era.dtqi_advec_mean),     'tendency in frozen water due to large-scale horizontal advection',           'kg/kg/s')
    # _v('aadv',           ('time', 'nlev'), np.zeros((nt, nf)),           'tendency in cloud fraction due to large-scale horizontal advection',         '1/s')
    _v('o3',             ('time', 'nlev'), _ts(o3_kgkg),                 'ozone mass mixing ratio (domain averaged)', 'kg/kg')
    _v('n_ccn',          ('time', 'nlev'), np.full((nt, nf), n_ccn),     'CCN concentration (initial)',              '/m3')
    # d_cl and d_ci intentionally omitted: DALES uses its internal defaults (defdcl/defdci).
    # _v('cloud_fraction', ('time', 'nlev'), np.zeros((nt, nf)),           'cloud fraction (domain averaged)',          '0-1')
    # _v('cc_local',       ('time', 'nlev'), np.zeros((nt, nf)),           'cloud fraction (at domain midpoint)',       '0-1')

    # --- Half-level profile variables (time, nlevp1) ---
    _v('height_h',   ('time', 'nlevp1'), _ts(era.zh_mean),       'half level height',               'm')
    _v('pressure_h', ('time', 'nlevp1'), _ts(era.ph_mean),       'half level pressure',             'Pa')
    # _v('fradSWnet',  ('time', 'nlevp1'), np.zeros((nt, nh)),     'radiative flux - net short wave', 'W/m2')
    # _v('fradLWnet',  ('time', 'nlevp1'), np.zeros((nt, nh)),     'radiative flux - net long wave',  'W/m2')

    # Soil variables intentionally omitted: NSA is ocean/sea-ice surface;
    # including h_soil would trigger ltb_soildata=.true. in modtestbed which
    # then requires field_capacity/wilting_point global attributes.

    nc.close()
