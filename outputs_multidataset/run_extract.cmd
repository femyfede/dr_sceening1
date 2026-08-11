@echo off
cd /d "C:\Users\FEDELIKA MAXIMUS\OneDrive\Desktop\drfinal"
venv\Scripts\python.exe outputs_multidataset\run_extract.py --workers 5 --only-missing >> outputs_multidataset\extract_stdout.log 2>> outputs_multidataset\extract_stderr.log
venv\Scripts\python.exe outputs_multidataset\run_extract.py --merge >> outputs_multidataset\extract_stdout.log 2>> outputs_multidataset\extract_stderr.log
venv\Scripts\python.exe outputs_multidataset\multidataset\train.py --config outputs_multidataset\multidataset\config.yaml >> outputs_multidataset\extract_stdout.log 2>> outputs_multidataset\extract_stderr.log
