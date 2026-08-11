@echo off
cd /d "C:\Users\FEDELIKA MAXIMUS\OneDrive\Desktop\drfinal"
venv\Scripts\python.exe -m streamlit run interface\app.py --server.port 8501 --server.headless true --server.address 127.0.0.1 --server.fileWatcherType none --server.runOnSave false >> interface\streamlit_stdout.log 2>> interface\streamlit_stderr.log
