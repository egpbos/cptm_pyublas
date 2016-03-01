# inspired by:
# - http://stackoverflow.com/q/13868903/1199693
# - http://stackoverflow.com/a/2484343/1199693

# compiler
CXX       := c++
# linker (usually the same)
LD        := c++

SOURCES   := crunch tests gibbs_inner
INCPATH   := -I/usr/include/python2.7 -I/home/patrick/.local/lib/python2.7/site-packages/PyUblas-2013.1-py2.7-linux-x86_64.egg/pyublas/include/
LIBPATH   := -L/usr/lib/x86_64-linux-gnu
LIBS      :=-lpython2.7 -lboost_python -lboost_system
WARNING=-Wall
OPT=-O3

# necessary for building shared library (which Python modules are, apparently):
OSHARED=-fPIC
BSHARED=-shared

# flags for object files:
OFLAGS=$(OPT) $(OSHARED) $(WARNING) $(INCPATH) --std=c++11

# flags for the output binary:
BFLAGS=$(BSHARED) $(LIBPATH) $(LIBS)

VPATH = src
BUILDDIR = build

# Make.
# SRC       := $(addsuffix .cpp,$(SOURCES))
OBJ       := $(addprefix $(BUILDDIR)/,$(addsuffix .o,$(SOURCES)))


$(BUILDDIR)/%.o: %.cpp
	@$(CXX) $(OFLAGS) -c $< -o $@
	@echo "CXX $<"
# the @ prevents the command from being printed before being executed

all: checkdirs build_module

build_module:	$(OBJ)
	@$(LD) $(OBJ) $(BFLAGS) -o crunch.so
	@echo "linking"

checkdirs: $(BUILDDIR)

$(BUILDDIR):
	@mkdir -p $@

clean:
	@rm -rf $(BUILDDIR)