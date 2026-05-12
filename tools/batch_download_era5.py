"""
Batch-download ERA5 data for all days in a month.

Works in two phases:
  Phase 1 (submit):  Submit CDS requests for all days x 3 file types
                     (model_an, pressure_an, surface_an) in a single call.
  Phase 2 (poll):    Poll CDS at regular intervals until every NetCDF file
                     is present on disk.

Once this script exits successfully, run run_november.py for downscaling.
That script detects existing ERA5 files and skips the download step,
going straight to generating the LES/DALES forcing files.

Usage
-----
    python batch_download_era5.py [--year YEAR] [--month MONTH]
    python batch_download_era5.py --year 2022 --month 11
    python batch_download_era5.py --year 2022 --month 11 --poll-interval 300

Defaults: year=2022, month=11, start-hour=5, end-hour=9.
"""

import argparse
import calendar
import datetime
import os
import sys
import time

sys.path.insert(0, '/Users/yunpeichu/LS2D')

import ls2d
from ls2d.ecmwf import era_tools

# ── Settings ─────────────────────────────────────────────────────────────────
# These MUST match nsa_input.py so that run_november.py finds the same files.
SETTINGS_BASE = {
    'central_lon': -156.60899353027344,
    'central_lat': 71.322998046875,
    'area_size':   1,
    'case_name':   'NSA',
    'era5_path':   '/Users/yunpeichu/LS2D/data',
    'era5_expver': 1,
    'write_log':   False,   # Print CDS messages to console, not log files
    'data_source': 'CDS',
}

FTYPES = ['model_an', 'pressure_an', 'surface_an']
# ─────────────────────────────────────────────────────────────────────────────


def collect_required_dates(year, month, start_hour, end_hour):
    """Return sorted list of unique analysis dates needed for every day of *month*."""
    _, n_days = calendar.monthrange(year, month)
    all_dates = set()
    for day in range(1, n_days + 1):
        start = datetime.datetime(year, month, day, start_hour)
        end   = datetime.datetime(year, month, day, end_hour)
        for date in era_tools.get_required_analysis(start, end):
            all_dates.add(date)
    return sorted(all_dates)


def find_missing_files(dates, era5_path, case_name):
    """Return list of (date, ftype) pairs whose .nc file is not yet on disk."""
    missing = []
    for date in dates:
        for ftype in FTYPES:
            _, era_file = era_tools.era5_file_path(
                date.year, date.month, date.day,
                era5_path, case_name, ftype,
            )
            if not os.path.isfile(era_file):
                missing.append((date, ftype))
    return missing


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--year',          type=int, default=2022)
    p.add_argument('--month',         type=int, default=11)
    p.add_argument('--start-hour',    type=int, default=5,
                   help='Simulation start hour UTC (default: 5)')
    p.add_argument('--end-hour',      type=int, default=9,
                   help='Simulation end hour UTC (default: 9)')
    p.add_argument('--poll-interval', type=int, default=300,
                   help='Seconds between CDS status polls (default: 300 = 5 min)')
    p.add_argument('--max-wait',      type=int, default=18000,
                   help='Maximum total wait in seconds (default: 18000 = 5 h)')
    return p.parse_args()


def main():
    args = parse_args()

    _, n_days = calendar.monthrange(args.year, args.month)
    required_dates = collect_required_dates(
        args.year, args.month, args.start_hour, args.end_hour)
    n_total = len(required_dates) * len(FTYPES)

    print(f'ERA5 batch download  –  {args.year}-{args.month:02d}  '
          f'({args.start_hour:02d}:00–{args.end_hour:02d}:00 UTC)')
    print(f'{len(required_dates)} unique days × {len(FTYPES)} file types '
          f'= {n_total} files total')

    # Build settings with a month-spanning date range so that download_era5()
    # computes and queues all required dates in a single call.
    settings = SETTINGS_BASE.copy()
    settings['start_date'] = datetime.datetime(args.year, args.month, 1,
                                               args.start_hour)
    settings['end_date']   = datetime.datetime(args.year, args.month, n_days,
                                               args.end_hour)

    # ── Phase 1: submit all CDS requests ─────────────────────────────────────
    print(f'\n{"─"*60}')
    print('Phase 1 – Submitting / checking CDS requests …')
    ls2d.download_era5(settings, exit_when_waiting=False)

    # ── Phase 2: poll until every file is on disk ─────────────────────────────
    print(f'\n{"─"*60}')
    print('Phase 2 – Polling until all files are downloaded …')
    print(f'(poll interval: {args.poll_interval}s, max wait: {args.max_wait}s)\n')

    deadline = time.time() + args.max_wait
    attempt  = 0

    while True:
        # Normalize era5_path the same way download_era5 does (adds trailing /)
        era5_path = settings['era5_path']

        missing = find_missing_files(required_dates, era5_path, settings['case_name'])
        n_done  = n_total - len(missing)
        print(f'  Status: {n_done}/{n_total} files present')

        if not missing:
            break

        if time.time() >= deadline:
            print(f'\nTimeout reached after {args.max_wait}s.')
            print(f'{len(missing)} file(s) still missing:')
            for date, ftype in missing:
                _, fp = era_tools.era5_file_path(
                    date.year, date.month, date.day,
                    era5_path, settings['case_name'], ftype)
                print(f'  {fp}')
            print('\nCheck https://cds.climate.copernicus.eu/requests?tab=all '
                  'for request status.')
            sys.exit(1)

        attempt += 1
        print(f'  Waiting {args.poll_interval}s before attempt {attempt} …\n',
              flush=True)
        time.sleep(args.poll_interval)

        # Re-call download_era5: it will check pickle status, download
        # completed files, and skip files that are already on disk.
        print(f'{"─"*60}')
        print(f'Phase 2 – Poll attempt {attempt} …')
        ls2d.download_era5(settings, exit_when_waiting=False)

    print(f'\n{"═"*60}')
    print('All ERA5 files downloaded successfully.')
    print(f'\nNext step – run downscaling for all days in {args.year}-{args.month:02d}:')
    print(f'    python run_november.py --year {args.year} --month {args.month}')
    print('The downscaling script will detect the existing ERA5 files and')
    print('skip the CDS download step, going straight to forcing generation.')
    print(f'{"═"*60}')


if __name__ == '__main__':
    main()
