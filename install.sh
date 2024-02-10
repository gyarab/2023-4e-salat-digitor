#! /bin/bash
mkdir build 2> /dev/null
cp digitor.json build/
cd build
cmake ..
make


