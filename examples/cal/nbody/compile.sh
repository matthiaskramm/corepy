#!/bin/bash
#exec gcc nbody.c -o nbody -lX11 -lXi -lglut -lGL -lGLU
exec gcc -shared -Wall -fPIC -O3 glrender.c -o glrender.so -lglut -lGL -lGLU
