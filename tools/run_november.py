"""
Simple daily LS2D forcing pipeline – November 2022, ARM-NSA site.

For each day in the specified month, runs nsa_input.py with a fixed 05:00–09:00 UTC
window and organises outputs as:

  /Users/yunpeichu/LS2D/results/NSA/run_era5/YYYYMMDD/   (LS2D output)
  /Users/yunpeichu/work_dales/NSA/input/YYYYMMDD/         (copy for DALES)

Experiment number is always 001.

Usage
-----
    python run_november.py [--year YEAR] [--month MONTH] [--dry-run]

Defaults: year=2022, month=11 (all days in that month).
"""

import argparse
import calendar
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

NSA_INPUT_SCRIPT = '/Users/yunpeichu/LS2D/examples/dales/nsa_input.py'
LS2D_OUT_ROOT    = '/Users/yunpeichu/LS2D/results/NSA_2/run_era5'
DALES_INPUT_ROOT = '/Users/yunpeichu/work_dales/NSA_2/input'

START_HOUR = 6   # 06:00 UTC
END_HOUR   = 9   # 09:00 UTC

# Files that must exist to consider a run successful.
# When LS2D only submits a CDS request (data not yet ready) these are absent.
SUCCESS_FILES   = ['scm_in.nc']
RETRY_WAIT_SEC  = 300   # wait 5 minutes between retries
MAX_RETRIES     = 60    # give up after ~5 hours total


def run_until_complete(cmd, out_dir):
    """
    Repeatedly call *cmd* until all SUCCESS_FILES appear in *out_dir*.

    LS2D sometimes submits a CDS queue request and exits without producing
    output.  On the next call the queued data may be ready and LS2D will
    download it and generate the files.  We keep retrying with RETRY_WAIT_SEC
    pauses until the files exist or MAX_RETRIES is exhausted.

    Returns True on success, False on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        print(f'  [attempt {attempt}/{MAX_RETRIES}] running nsa_input.py …',
              flush=True)
        rc = subprocess.run(cmd).returncode

        missing = [f for f in SUCCESS_FILES
                   if not os.path.isfile(os.path.join(out_dir, f))]
        if not missing:
            print('  Output files found – run complete.', flush=True)
            return True

        if attempt < MAX_RETRIES:
            print(f'  Output incomplete (missing: {missing}).')
            print(f'  CDS request probably still queued. '
                  f'Waiting {RETRY_WAIT_SEC}s before retry …', flush=True)
            time.sleep(RETRY_WAIT_SEC)

    print(f'  Gave up after {MAX_RETRIES} attempts.')
    return False


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--year',  type=int, default=2022)
    p.add_argument('--month', type=int, default=11)
    p.add_argument('--dry-run', action='store_true', dest='dry_run',
                   help='Print what would happen without running anything')
    return p.parse_args()


def main():
    args = parse_args()
    _, n_days = calendar.monthrange(args.year, args.month)

    for day in range(1, n_days + 1):
        start_dt   = datetime(args.year, args.month, day, START_HOUR)
        end_dt     = datetime(args.year, args.month, day, END_HOUR)
        datestr    = start_dt.strftime('%Y%m%d')
        out_dir    = os.path.join(LS2D_OUT_ROOT, datestr)
        dales_dir  = os.path.join(DALES_INPUT_ROOT, datestr)

        print(f'\n{"─"*60}')
        print(f'{datestr}  {START_HOUR:02d}:00–{END_HOUR:02d}:00 UTC')
        print(f'  LS2D  output : {out_dir}')
        print(f'  DALES input  : {dales_dir}')

        # Skip days whose output files are already present.
        already_done = all(
            os.path.isfile(os.path.join(out_dir, f)) for f in SUCCESS_FILES
        )
        if already_done:
            print('  Already complete – skipping.')
            continue

        if args.dry_run:
            continue

        # ── Step 1: run nsa_input.py (with CDS retry logic) ─────────────
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            sys.executable,
            NSA_INPUT_SCRIPT,
            '--start_date', start_dt.strftime('%Y-%m-%dT%H:%M:%S'),
            '--end_date',   end_dt.strftime('%Y-%m-%dT%H:%M:%S'),
            '--out_dir',    out_dir,
        ]
        ok = run_until_complete(cmd, out_dir)
        if not ok:
            print(f'  ERROR: failed to obtain output for {datestr}; skipping copy.')
            continue

        # ── Step 2: copy generated files to DALES input dir ─────────────
        os.makedirs(dales_dir, exist_ok=True)
        for fname in os.listdir(out_dir):
            src = os.path.join(out_dir, fname)
            dst = os.path.join(dales_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        print(f'  Copied to {dales_dir}')


if __name__ == '__main__':
    main()

