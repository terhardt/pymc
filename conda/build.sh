#!/bin/bash
if [ "$(uname)" == "Darwin" ]; then
    export LDFLAGS="-static-libgfortran -Wall -undefined dynamic_lookup -bundle -arch x86_64"
fi

$PYTHON setup.py build
$PYTHON setup.py install
