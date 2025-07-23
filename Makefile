# Makefile for [fann]

# specify a location for Pd if desired
# PDDIR = /home/yourname/somedirectory/pd-0.55-2

lib.name = fann

# ensure linking to static FANN library by specifying the location libfann.a
ldlibs = /usr/local/lib/libfann.a

define forLinux
  # enable openMP
  ldlibs += -fopenmp
endef

# specify the location of header files
cflags = -Iinclude -I/usr/local/include

$(lib.name).class.sources = ./fann.c

datafiles = $(lib.name)-help.pd

# provide the path to pd-lib-builder
PDLIBBUILDER_DIR=./pd-lib-builder/
include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder
