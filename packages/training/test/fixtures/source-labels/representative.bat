@echo off
rem representative comment
set "NAME=world"
set DIRNAME=%~dp0
set APP_HOME=%DIRNAME%
if "%NAME%"=="world" (
  echo hello ^
    %NAME%
)
