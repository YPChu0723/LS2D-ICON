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
import pkg_resources
import os

# Third party modules
import numpy as np

class utils:
    """
    Various tools to calculate e.g. properties of the vertical IFS/ERA grid,
    or the thermodynamics as used by IFS. Wrapped in a class, to prevent
    mixing up differences in methods/constants/.. between IFS and other models
    """
    def __init__(self):

        # Constants (see IFS part IV, chapter 12)
        self.grav  = 9.80665
        self.Rd    = 287.0597
        self.Rv    = 461.5250
        self.eps   = self.Rv/self.Rd-1.
        self.cpd   = 1004.7090
        self.Lv    = 2.5008e6


    def calc_virtual_temp(self, T, qv, ql=0, qi=0, qr=0, qs=0):
        """
        Calculate the virtual temperature
        Equation: Tv = T * ([Rv/Rd-1]*qv - ql - qi - qr - qs)
        See IFS part IV, eq. 12.6
        Keyword arguments:
            T -- absolute temperature (K)
            q* -- specific humidities (kg kg-1):
                qv = vapor
                ql = liquid (optional)
                qi = ice    (optional)
                qr = rain   (optional)
                qs = snow   (optional)
        """

        return T * (1+self.eps*qv - ql - qi - qr - qs)

    def calc_exner(self, p):
        return (p/1e5)**(self.Rd/self.cpd)



if __name__ == "__main__":
    """ Test / example, only executed if script is called directly """

    import matplotlib.pyplot as pl
    pl.ion()
    pl.close('all')
