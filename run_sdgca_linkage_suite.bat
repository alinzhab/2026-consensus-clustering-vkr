@echo off
cd /d "%~dp0"
C:\Users\tikho\anaconda3\python.exe python_port\run_sdgca_linkage_suite.py --runs 5 --m 20 > results\sdgca_linkage_full_suite_stdout.log 2> results\sdgca_linkage_full_suite_stderr.log
