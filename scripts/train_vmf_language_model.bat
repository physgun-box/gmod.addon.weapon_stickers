@echo off
setlocal

set SCRIPT_DIR=%~dp0
set REPO_ROOT=%~dp0..

python "%REPO_ROOT%\scripts\train_vmf_language_model.py" %*
