@echo off
setlocal EnableDelayedExpansion

REM --- CONFIGURATION ---
set SCRIPT_DIR=%~dp0
set REPO_URL=https://github.com/somethington/tpp_solver/archive/refs/heads/main.zip
set REPO_ZIP=%SCRIPT_DIR%tpp_solver.zip
set REPO_DIR=%SCRIPT_DIR%tpp_solver-main
set REPO_LAST_MODIFIED=%SCRIPT_DIR%repo_last_modified.txt
set PYTHON_VERSION=3.11.0
set PYTHON_EMBED_ZIP=python-%PYTHON_VERSION%-embed-amd64.zip
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/%PYTHON_EMBED_ZIP%
set PYTHON_DIR=%SCRIPT_DIR%python-3.11-embed

REM --- UV Configuration ---
set UV_VERSION=0.5.7
set UV_ZIP=%SCRIPT_DIR%uv-x86_64-pc-windows-msvc.zip
set UV_URL=https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip
set UV_DIR=%SCRIPT_DIR%.tools
set UV_EXE=%UV_DIR%\uv.exe

REM --- Install UV ---
if not exist "%UV_DIR%" mkdir "%UV_DIR%"
if not exist "%UV_EXE%" (
    echo Downloading uv %UV_VERSION%...
    powershell -Command ^
        "try { Invoke-WebRequest -Uri '%UV_URL%' -OutFile '%UV_ZIP%' } catch { Write-Error 'Failed to download uv.'; exit 1 }"
    if ERRORLEVEL 1 (
        echo Failed to download uv. Please check the URL or your internet connection.
        exit /b 1
    )

    echo Extracting uv...
    powershell -Command ^
        "try { Expand-Archive -Path '%UV_ZIP%' -DestinationPath '%UV_DIR%' -Force } catch { Write-Error 'Failed to extract uv.'; exit 1 }"
    if ERRORLEVEL 1 (
        echo Failed to extract uv. Cleaning up...
        del "%UV_ZIP%"
        exit /b 1
    )

    del "%UV_ZIP%"
    echo Cleaned up uv ZIP file.
)

REM --- Test UV installation ---
if exist "%UV_EXE%" (
    "%UV_EXE%" --version
    if ERRORLEVEL 1 (
        echo Failed to verify uv installation.
        exit /b 1
    )
) else (
    echo uv installation failed. File not found: %UV_EXE%.
    exit /b 1
)

REM --- Check Repository Update ---
set DOWNLOAD_REPO=1
if exist "%REPO_LAST_MODIFIED%" (
    echo Checking if repository has been updated...
    powershell -Command ^
        "try { $response = Invoke-WebRequest -Uri '%REPO_URL%' -Method HEAD; $localModified = Get-Content '%REPO_LAST_MODIFIED%'; if ($response.Headers['Last-Modified'] -eq $localModified) { exit 0 } else { exit 1 } } catch { exit 1 }"
    if ERRORLEVEL 0 (
        echo Repository is up-to-date.
        set DOWNLOAD_REPO=0
    ) else (
        echo Repository has been updated. Preparing to download new version...
    )
)

REM --- Download and Extract Repository ---
if !DOWNLOAD_REPO! EQU 1 (
    if exist "%REPO_ZIP%" del "%REPO_ZIP%"
    echo Downloading repository as a ZIP...
    powershell -Command "Invoke-WebRequest -Uri \"%REPO_URL%\" -OutFile \"%REPO_ZIP%\""
    if ERRORLEVEL 1 (
        echo Failed to download repository.
        exit /b 1
    )

    echo Extracting the repository...
    powershell -Command ^
        "try { Expand-Archive -Path '%REPO_ZIP%' -DestinationPath '%SCRIPT_DIR%' -Force } catch { exit 1 }"
    if ERRORLEVEL 1 (
        echo Failed to extract repository.
        del "%REPO_ZIP%"
        exit /b 1
    )

    del "%REPO_ZIP%"
    echo Cleaned up repository ZIP file.

    REM --- Save Last-Modified Header ---
    powershell -Command ^
        "try { Invoke-WebRequest -Uri '%REPO_URL%' -Method HEAD | Select-Object -ExpandProperty Headers | ForEach-Object { $_.'Last-Modified' } | Set-Content '%REPO_LAST_MODIFIED%' } catch { exit 1 }"
) else (
    echo Skipping repository download and extraction.
)

REM --- Set VENV_DIR inside the repository directory ---
set VENV_DIR=%REPO_DIR%\.venv
echo Virtual Environment Directory: %VENV_DIR%

cd "%REPO_DIR%"

REM --- Create Virtual Environment ---
if not exist "%VENV_DIR%" (
    echo Creating virtual environment with uv...
    "%UV_EXE%" venv "%VENV_DIR%"
    if ERRORLEVEL 1 (
        echo Failed to create virtual environment. Ensure uv is functioning and paths are correct.
        exit /b 1
    )
) else (
    echo Virtual environment directory already exists. Skipping creation.
)

REM --- Activate Virtual Environment ---
call "%VENV_DIR%\Scripts\activate.bat"
if ERRORLEVEL 1 (
    echo Failed to activate virtual environment.
    exit /b 1
)

REM --- Install Requirements ---
if exist "requirements.txt" (
    echo Installing requirements using uv...
    "%UV_EXE%" pip install -r requirements.txt
    if ERRORLEVEL 1 (
        echo Failed to install requirements.
        exit /b 1
    )
) else (
    echo No requirements.txt found. Skipping.
)

REM --- Run Application ---
echo Running streamlit...
python -m streamlit run tpp_solver_mt.py

REM --- Deactivate Virtual Environment ---
deactivate

endlocal
