SOURCES = fann.c

# ****SUPPLY THE LOCATION OF PD SOURCE****
pd_src = /Applications/Pd-0.52-2.app/Contents/Resources

CFLAGS = -DPD -I$(pd_src)/src -I/usr/local/include -Wall -W -g


UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  EXTENSION = pd_darwin
  OS = macosx
  LIBDIR = /usr/local/lib
  OPT_CFLAGS = -O3 -ftree-vectorize -Wshadow -Wstrict-prototypes -Wno-unused -Wno-parentheses -Wno-switch
  FAT_FLAGS = -arch x86_64
  CFLAGS += -fPIC $(FAT_FLAGS)
  LDFLAGS += -bundle -undefined dynamic_lookup $(FAT_FLAGS)
  LIBS += -lc -L$(LIBDIR) -lfann
  STRIP = strip -x
 endif
ifeq ($(UNAME),Linux)
  EXTENSION = pd_linux
  OS = linux
  LIBDIR = /usr/local/lib
  OPT_CFLAGS = -O6 -funroll-loops -fomit-frame-pointer
  CFLAGS += -fPIC
  LDFLAGS += -Wl,--export-dynamic -shared -fPIC
  LIBS += -lc -lfann
  STRIP = strip --strip-unneeded -R .note -R .comment
endif
# ifeq (MINGW,$(findstring MINGW,$(UNAME)))
#   CC = gcc
#   EXTENSION = dll
#   OS = windows
#   LIBDIR = "C:\MinGW\lib\fftw-3.3.5-dll32"
#   OPT_CFLAGS = -O3 -funroll-loops -fomit-frame-pointer \
#     -march=pentium4 -mfpmath=sse -msse -msse2
#   WINDOWS_HACKS = -D'O_NONBLOCK=1'
#   CFLAGS += -mms-bitfields $(WINDOWS_HACKS) \
#     -I"C:\MinGW\lib\fftw-3.3.5-dll32"
#   LDFLAGS += -static-libgcc -s -shared \
#     -Wl,--enable-auto-import $(pd_src)/bin/pd.dll
#   LIBS += -L$(LIBDIR) -L$(pd_src)/bin -lpd -lfann \
#     -lwsock32 -lkernel32 -luser32 -lgdi32
#   STRIP = strip --strip-unneeded -R .note -R .comment
# endif

CFLAGS += $(OPT_CFLAGS)


all: $(SOURCES:.c=.o)
	$(CC) $(LDFLAGS) -o $(SOURCES:.c=.$(EXTENSION)) $(SOURCES:.c=.o) $(LIBS)
	chmod a-x $(SOURCES:.c=.$(EXTENSION))
	$(STRIP) $(SOURCES:.c=.$(EXTENSION))
	rm -f -- $(SOURCES:.c=.o)


.PHONY: clean

clean:
	-rm -f -- $(SOURCES:.c=.o)
	-rm -f -- $(SOURCES:.c=.$(EXTENSION))
