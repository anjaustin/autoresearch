#!/bin/bash
gcc -O3 -mavx2 -mfma -march=native -fopenmp -shared -fPIC \
    -o softchip/ghost_engine.so softchip/ghost_engine.c -lm
