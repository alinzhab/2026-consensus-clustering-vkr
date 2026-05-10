@echo off
REM Запуск пакета SDGCA с разными linkage-методами.
REM Использует .venv проекта, если он создан; иначе python из PATH.
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

if not exist "results" mkdir results

%PYTHON% consensus_lab\run_sdgca_linkage_suite.py --runs 5 --m 20 > results\sdgca_linkage_full_suite_stdout.log 2> results\sdgca_linkage_full_suite_stderr.log
