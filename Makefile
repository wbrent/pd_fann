# Makefile for [fann]

# specify a location for Pd if desired
# PDDIR = /home/yourname/somedirectory/pd-0.51-4

lib.name = fann

# specify the location and name of libfann.a
ldlibs = -L/usr/local/lib -lfann

# specify the location of FFTW header file
cflags = -Iinclude -I/usr/local/include

$(lib.name).class.sources = ./fann.c

datafiles = $(lib.name)-help.pd

# provide the path to pd-lib-builder
PDLIBBUILDER_DIR=./pd-lib-builder/
include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder
