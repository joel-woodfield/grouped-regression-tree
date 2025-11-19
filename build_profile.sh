#!/bin/bash

cmake -S . -B build_profile -DBUILD_PROFILING_TOOL=ON -DCMAKE_BUILD_TYPE=Release

cmake --build build_profile --config Release
