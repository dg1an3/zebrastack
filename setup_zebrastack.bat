set PYTHON_VERSION=3.8.6
wget -P %TEMP% https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe
%TEMP%\python-%PYTHON_VERSION%-amd64.exe
pip install -r requirements.txt

ECHO Download unpack CXR8 data