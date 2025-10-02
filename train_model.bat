@echo off
setlocal
cd /d %~dp0
python train_vmf_model.py %*
