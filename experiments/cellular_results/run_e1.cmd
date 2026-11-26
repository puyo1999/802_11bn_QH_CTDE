@echo off
rem Full E1: throughput objective (200 drops, 3 seeds), then max-weight (200 drops, 1 seed).
rem Jobs are cached in e1_jobs\, so rerunning this file resumes an interrupted run.
cd /d "%~dp0\..\.."
.venv\Scripts\python.exe -u experiments\cellular_gap.py e1 --drops 200 --seeds 3 > experiments\cellular_results\e1_full.log 2>&1
.venv\Scripts\python.exe -u experiments\cellular_gap.py e1 --drops 200 --seeds 1 --objective maxweight > experiments\cellular_results\e1_maxweight.log 2>&1
echo done > experiments\cellular_results\e1_done.flag
