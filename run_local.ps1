$ErrorActionPreference = "Stop"
$AppDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $AppDirectory
py -m streamlit run streamlit_app.py

