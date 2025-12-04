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
import os

# Third party modules

# LS2D modules
from ls2d.src.messages import *


def file_path(year, month, day, hour, minute, path, case, ftype, return_dir=True):
    """
    Return saving path of files in format `path/yyyy/mm/dd/type.nc`
    """

    folder = os.path.join(path, case)
    suffix = f'{year:04d}{month:02d}{day:02d}_input_ml_{year:04d}{month:02d}{day:02d}T{hour:02d}{minute:02d}00Z'
    file = os.path.join(folder, ftype+suffix + '.nc')

    if return_dir:
        return folder, file
    else:
        return file


import datetime

def get_required_analysis(start, end, freq=0.5):
    """
    Returns a list of time steps between start and end with a step size of 'freq' hours.
    Output format: [[year, month, day, hour, minute], ...]
    """
    
    # 1. Define the step size (freq is in hours)
    # e.g., freq=0.5 becomes 30 minutes
    step = datetime.timedelta(hours=freq)

    # 2. Ensure we start fresh from the given start time
    current = start
    
    # 3. List to store the results
    result_times = []

    # 4. Iterate from start to end (inclusive)
    while current <= end:
        # Create the list of [year, month, day, hour, minute]
        result_times.append(current)
        
        # Move to the next time step
        current += step

    return result_times


def get_required_forecast(start, end):

    # One day datetime offset
    one_day = datetime.timedelta(days=1)

    # Forecast runs through midnight, so last analysis = last day
    last_forecast = datetime.datetime(end.year, end.month, end.day)

    # If start time is before 06 UTC, include previous day for the forecast files
    if start.hour > 6:
        first_forecast = datetime.datetime(start.year, start.month, start.day)
    else:
        first_forecast = datetime.datetime(start.year, start.month, start.day) - one_day

    # Create list of datetime objects:
    dates = [first_forecast + i*one_day for i in range((last_forecast-first_forecast).days + 1)]

    return dates


def lower_to_hour(time):
    time_out = datetime.datetime(time.year, time.month, time.day, time.hour)
    if time.minute != 0 or time.second != 0:
        warning('Changed date/time from {} to {}'.format(time, time_out))
    return time_out
