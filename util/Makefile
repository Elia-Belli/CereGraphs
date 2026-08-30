# Builds util/analyze -- a native binary (not containerized, unlike
# cslc/cs_python), so it must be compiled locally after every fresh clone
# rather than committed: a binary built on one machine's glibc/libstdc++
# routinely refuses to run on another (see docs/ERRORS.md). `make` picks up
# an existing up-to-date `analyze` and does nothing, so it's safe/cheap to
# call unconditionally from any script that needs it -- no separate
# timestamp check required.
#
# Usage:
#   make -C util        # build (or rebuild if analyze.cpp/mmio.c changed)
#   make -C util clean  # remove the binary and object files

CXX := g++
CC := gcc
CXXFLAGS := -Wall -g -I include -std=c++17 -O3
CFLAGS := -Wall -g -O3

.PHONY: all clean

all: analyze

analyze: analyze.o mmio.o
	$(CXX) $(CXXFLAGS) -o $@ $^

analyze.o: analyze.cpp mmio.h include/argparse/argparse.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

mmio.o: mmio.c mmio.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f analyze analyze.o mmio.o
