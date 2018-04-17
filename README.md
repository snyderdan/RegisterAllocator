RegisterAllocator

This program was written for CS415 - Compilers

The primary function of this program is to allocate physical registers
given ILOC assembly which uses virtual registers. It does this by analyzing
the code and executing the specified allocation algorithm before printing
the output to standard output.

It has 4 different allocation algorithms available:
 1) Simple top down allocation
 2) Top down allocation accounting for live ranges
 3) Bottom up allocation
 4) Modified bottom up allocation

All algorithms run in O(n) time. 

usage: alloc.py [-h] [-v] k {b,s,t,o} file

CS415 Project 1 -- local register allocation

positional arguments:
  k              Number of physical registers
  {b,s,t,o}      Allocation algorithm
  file           ILOC input file

optional arguments:
  -h, --help     show this help message and exit
  -v, --verbose  Verbose output - print allocation information
