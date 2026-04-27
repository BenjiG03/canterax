@ECHO OFF

set SPHINXBUILD=python -m sphinx
set SOURCEDIR=source
set BUILDDIR=_build

if "%1" == "clean" (
    rmdir /s /q "%BUILDDIR%"
    goto end
)

%SPHINXBUILD% -b html "%SOURCEDIR%" "%BUILDDIR%\html"

:end
