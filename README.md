```
 _______ __     __ __                       _______ __
|     __|  |--.|__|  |--.--.--.--.--.---.-.|     __|  |_.----.-----.---.-..--------.
|__     |     ||  |  _  |  |  |  |  |  _  ||__     |   _|   _|  -__|  _  ||        |
|_______|__|__||__|_____|_____|___  |___._||_______|____|__| |_____|___._||__|__|__|
                              |_____|
```
# Bandwidth Benchmark

## Dependencies

* [ROCm][]
* [libnuma][]
* [pthreads][]

## Compiling

* `cd src`
* `make` or `make rocm` to build for AMD
* `make cuda` to build for NVIDIA

## Usage

The command line syntex is:

```
./shibuya <SIZE>     size of arrays in megabytes
          <TIME>     test duration in seconds
          <STREAM_1>
          <STREAM_2>
          ...
```

where streams are define as:

```
<C|D><#>-<C|M|A|T|D|H>-<N|D><#>-<N|D><#>-<N|D><#>-<#>
 Core     Copy        NUMA     NUMA     NUMA
   Device   Multiply    Device   Device   Device
              Add
                Triad
                  Dot
                    hipMemcpy

 core or device executing the operation
          type of operation
                      location for array A
                               location for array B
                                        location for array C (only for add and triad)
                                                 # of the core controlling the device (only for device streams)
```

### Examples of Streams

* `C0-C-N0-N0`
  * **Core 0** performs a **copy** (`B[i]=A[i]`) with A and B in **NUMA node 0**.
* `C2-M-N0-N2`
  * **Core 2** performs a **multiplication** (`B[i]=alpha*A[i]`) with A in **NUMA node 0** and B in **NUMA node 2**.
* `D0-C-D0-D0-0`
  * **Device 0** performs a **copy** (`B[i]=A[i]`) with A and B in the **memory of device 0**.
    The execution is controlled by core 0.
* `D0-T-D0-D0-D1-8`
  * **Device 0** performs a **triad** (`C[i]=alpha*A[i]+B[i]`) with A and B in the **memory of device 0** and C in the **memory of device 1**.
    The execution is controlled by core 8.

### Examples of command lines

* `./shibuya 128 3 C0-C-N0-N0 C1-C-N0-N0 C2-C-N0-N0 C3-C-N0-N0`
  * four cores simultaneously making copies of size 128 MB for 3 seconds in NUMA node 0
* `./shibuya 1024 5 D0-H-D0-D1-0 D1-H-D1-D2-1 D2-H-D2-D3-2 D3-H-D3-D0-3`
  * four devices simultaneously making copies, using `hipMemcpy()`, of size 1 GB for 5 seconds in a round-robin fashion

### Checking the Topology

The topology of the system can be checked using [hwloc][].
To build hwloc from sources run:

```
git clone git@github.com:open-mpi/hwloc.git
cd hwloc
./autogen.sh
./configure --prefix=/opt/hwloc
make -j 32
sudo make install
```

Run:
* `/opt/hwloc/bin/lstopo topo.svg` to print the topology to an SVG file,
* `/opt/hwloc/bin/lstopo -f --no-io topo.svg` to exclude IO from the diagram,
* `/opt/hwloc/bin/lstopo -f --no-io --no-smt topo.svg` to exclude IO and show physical cores only.

SVG files can be opened with [Inkscape].
On Linux they can also easily be converted to raster images using [ImageMagic], e.g., `convert topo.svg topo.png`.

## Important Remarks

* The `H` operation uses `hipMemcpy()` to copy the data from the source to the destination.
  If executed by a GPU device, host memory is page-locked.
  If executed by a CPU core, host memory is not page-locked.

* The benchmark reports the aggregate bandwidth.
  E.g., for the copy operation (`B[i]=A[i]`) it reports the bandwidth of reading A + the bandwidth of writing B.
  If the same memory is the source and the destination then combined read/write bandwidth of that memory is reported.
  If the source and the destination are different, then the result is the bandwidth of the bottleneck ×2
  — either the slower memory or the interconnect.

## Help

Jakub Kurzak (<jakurzak@amd.com>)

[ROCm]: https://github.com/RadeonOpenCompute/ROCm
[libnuma]: https://linux.die.net/man/3/numa
[pthreads]: https://linux.die.net/man/7/pthreads
[hwloc]: https://www.open-mpi.org/projects/hwloc/
[Inkscape]: https://inkscape.org/
[ImageMagic]: https://imagemagick.org/index.php
