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

import matplotlib.pyplot as pl
import numpy as np
import sys

#
# Vertical grids
#
class _Grid:
    def __init__(self, kmax, dz0):
        self.kmax = kmax
        self.dz0  = dz0

        self.z = np.zeros(kmax)
        self.zh = np.zeros(kmax+1)
        self.dz = np.zeros(kmax)
        self.zsize = None

    def plot(self, logx=False, logy=False):

        fig, ax = pl.subplots(figsize=(5,8))
        ax.set_title("zsize={:.1f} m".format(self.zsize))
        ax.plot(np.arange(self.kmax), self.z/1000, color='k', linestyle='-', label=r'$z$')
        ax.set_xlabel(r'Vertical grid point (-)')
        ax.set_ylabel(r'height $z$ (km)')
        ax.tick_params(axis='x', labelcolor='k')
        ax.set_ylim(0,2)
        ax.set_xlim(0,self.kmax)
        ax.grid()

        ax2 = ax.twinx()
        ax2.set_ylabel(r'Grid spacing $\Delta z$ (m)', color='green')
        ax2.plot(np.arange(self.kmax), self.dz, color='green', linestyle='--', label=r'$\Delta z$')
        ax2.tick_params(axis='x', labelcolor='green')
        ax2.set_ylim(0,30)

        if logx:
            for ax in fig.axes:
                ax.set_xscale('log')
        if logy:
            for ax in fig.axes:
                ax.set_yscale('log')

        pl.tight_layout()


class Grid_equidist(_Grid):
    def __init__(self, kmax, dz0):
        _Grid.__init__(self, kmax, dz0)

        self.zsize = kmax * dz0
        self.z[:]  = np.arange(dz0/2, self.zsize, dz0)
        self.zh[:] = np.arange(0, self.zsize+0.1, dz0)
        self.dz[:] = dz0


class Grid_stretched(_Grid):
    def __init__(self, kmax, dz0, nloc1, nbuf1, dz1, nloc2=None, nbuf2=None, dz2=None):
        _Grid.__init__(self, kmax, dz0)

        double_stretched = nloc2 is not None and nbuf2 is not None and dz2 is not None

        dn = 1./kmax
        n = np.linspace(dn, 1.-dn, kmax)

        nloc1 *= dn
        nbuf1 *= dn

        if double_stretched:
            nloc2 *= dn
            nbuf2 *= dn

        dzdn1 = dz0/dn
        dzdn2 = dz1/dn

        if double_stretched:
            dzdn3  = dz2/dn
            dzdn = dzdn1 + 0.5*(dzdn2-dzdn1)*(1. + np.tanh((n-nloc1)/nbuf1)) \
                         + 0.5*(dzdn3-dzdn2)*(1. + np.tanh((n-nloc2)/nbuf2))
        else:
            dzdn = dzdn1 + 0.5*(dzdn2-dzdn1)*(1. + np.tanh((n-nloc1)/nbuf1))

        self.dz[:] = dzdn*dn

        stretch = np.zeros(self.dz.size)
        self.z[0]  = 0.5*self.dz[0]
        stretch[0] = 1.

        for k in range(1, self.kmax):
              self.z [k] = self.z[k-1] + 0.5*(self.dz[k-1]+self.dz[k])
              stretch[k] = self.dz[k]/self.dz[k-1]

        self.zsize = self.z[kmax-1] + 0.5*self.dz[kmax-1]

        self.zh[1:-1] = 0.5 * (self.z[1:] + self.z[:-1])
        self.zh[0] = 0
        self.zh[-1] = self.zsize


class Grid_linear_stretched(_Grid):
    def __init__(self, kmax, dz0, alpha):
        _Grid.__init__(self, kmax, dz0)

        self.dz[:] = dz0 * (1 + alpha)**np.arange(kmax)
        self.zh = np.zeros(kmax+1)
        self.zh[1:] = np.cumsum(self.dz)
        self.z[:] = 0.5 * (self.zh[1:] + self.zh[:-1])

        self.zsize = self.z[kmax-1] + 0.5*self.dz[kmax-1]

        # Re-calculate zh as center between z, to stay in line with MicroHH definition.
        self.zh[1:-1] = 0.5 * (self.z[1:] + self.z[:-1])
        self.zh[0] = 0
        self.zh[-1] = self.zsize


class Grid_stretched_manual(_Grid):
    def __init__(self, kmax, dz0, heights, factors):
        _Grid.__init__(self, kmax, dz0)

        self.z[0]  = dz0/2.
        self.dz[0] = dz0

        def index(z, goal):
            return np.where(z-goal>0)[0][0]-1

        for k in range(1, kmax):
            self.dz[k] = self.dz[k-1] * factors[index(heights, self.z[k-1])]
            self.z[k] = self.z[k-1] + self.dz[k]

        self.zsize = self.z[kmax-1] + 0.5*self.dz[kmax-1]

        self.zh[1:-1] = 0.5 * (self.z[1:] + self.z[:-1])
        self.zh[0] = 0
        self.zh[-1] = self.zsize


class Grid_three_stage(_Grid):
    def __init__(self, kmax, dz0, z_stretch_start, stretch_factor, dz_max):
        
        _Grid.__init__(self, kmax, dz0)

        k_t = int(np.round(z_stretch_start / dz0))
        self.dz[:k_t] = dz0

        # 2. Stretch until we hit dz_max OR run out of grid points
        stretch_index = k_t 

        
        while stretch_index < kmax and ((dz0 * (1 + stretch_factor)**(stretch_index -(k_t + 1))) < dz_max):
            self.dz[stretch_index] = dz0 * (1 + stretch_factor)**(stretch_index - (k_t + 1))
            stretch_index += 1

        if stretch_index < kmax:
            self.dz[stretch_index:] = dz_max


        self.z = np.zeros(kmax)
        self.z[0] = dz0 * 0.5
        for k in range(1, kmax):
            self.z[k] = self.z[k-1] + self.dz[k]
        
        self.zsize = self.z[kmax-1] + 0.5*self.dz[kmax-1]

        self.zh[1:-1] = 0.5 * (self.z[1:] + self.z[:-1])
        self.zh[0] = 0
        self.zh[-1] = self.zsize


class Grid_stretched_capped(_Grid):
    """
    Three-stage vertical grid following the formula in Appendix A of the paper:

        Δz[k] = dz_start                          if k <= k_T
                 dz_start * (1+s)^(k-(k_T+1))     if k_T < k < k_M
                 dz_end                            if k >= k_M

        z[0] = dz_start / 2
        z[k] = z[k-1] + dz[k]                     for k >= 1

    Parameters
    ----------
    kmax : int
        Total number of vertical levels.
    dz_start : float
        Uniform fine grid spacing for k <= k_T (m).
    k_T : int
        Last level index with uniform fine spacing (0-based).
    s : float
        Stretching factor (dimensionless, e.g. 0.0125).
    k_M : int
        First level index with uniform coarse spacing (0-based).
    dz_end : float
        Uniform coarse grid spacing for k >= k_M (m).

    Example (paper parameters — kmax=297 gives zsize ≈ 11840 m):
        Grid_stretched_capped(kmax=297, dz_start=10, k_T=120, s=0.0125, k_M=260, dz_end=185)
    """

    def __init__(self, kmax, dz_start, k_T, s, k_M, dz_end):
        _Grid.__init__(self, kmax, dz_start)

        for k in range(kmax):
            if k <= k_T:
                self.dz[k] = dz_start
            elif k < k_M:
                self.dz[k] = dz_start * (1 + s) ** (k - (k_T + 1))
            else:
                self.dz[k] = dz_end

        self.z[0] = dz_start / 2
        for k in range(1, kmax):
            self.z[k] = self.z[k - 1] + self.dz[k]

        self.zsize = self.z[kmax - 1] + 0.5 * self.dz[kmax - 1]

        self.zh[1:-1] = 0.5 * (self.z[1:] + self.z[:-1])
        self.zh[0] = 0
        self.zh[-1] = self.zsize

if __name__ == '__main__':
    """
    For debug/testing.
    """

    grid4 = Grid_three_stage(kmax=156, dz0=10, z_stretch_start=800, stretch_factor=0.015, dz_max=20)
    grid5 = Grid_stretched_capped(kmax=156, dz_start=10, k_T=100, s=0.02, k_M=140, dz_end=25)
    # grid5 = Grid_stretched_capped(kmax=296, dz_start=5, k_T=210, s=0.02, k_M=280, dz_end=20)
    grid4.plot()
    pl.savefig('/Users/yunpeichu/visualization/ls2d_plots/grid_three_stage.png')
    grid5.plot()
    print(grid5.dz)
    pl.savefig('/Users/yunpeichu/visualization/ls2d_plots/grid_stretched_capped.png')
