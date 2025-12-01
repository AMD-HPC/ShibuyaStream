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

## Building

```
git clone git@github.com:AMD-HPC/ShibuyaStream.git
cd ShibuyaStream
mkdir build
cd build
cmake ..
make -j
```

Need be, set `CMAKE_PREFIX_PATH` and `CMAKE_MODULE_PATH`, e.g.:

```
export CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake:${CMAKE_PREFIX_PATH}
export CMAKE_MODULE_PATH=/opt/rocm/share/rocm/cmake:${CMAKE_MODULE_PATH}
```

## Usage

The command line syntax is:

```
./shibuya  SIZE      size of arrays in megabytes
           TIME      test duration in seconds
          [STREAM_1]
          [STREAM_2]
          ...
```

where streams are defined as:

```
 core or device
 |        type of operation
 |        |             location of A
 |        |             |        location of B
 |        |             |        |        location of C (only for add and triad)
 |        |             |        |        |        the core controlling the device
 |        |             |        |        |        |       (only for device streams)
<C|D><#>-<C|M|A|T|D|H>-<N|D><#>-<N|D><#>-<N|D><#>-<#>
 Core     Copy          NUMA     NUMA     NUMA
   Device   Multiply      Device   Device   Device
              Add
                Triad
                  Dot
                    hipMemcpy
```

### Examples of Streams

* `C0-C-N0-N0`
  * **Core 0** performs a **copy** (`B[i] = A[i]`) with A and B in **NUMA node 0**.
* `C2-M-N0-N2`
  * **Core 2** performs a **multiplication** (`B[i] = alpha*A[i]`) with A in **NUMA node 0** and B in **NUMA node 2**.
* `D0-C-D0-D0-0`
  * **Device 0** performs a **copy** (`B[i] = A[i]`) with A and B in the **memory of device 0**.
    The execution is controlled by core 0.
* `D0-T-D0-D0-D1-8`
  * **Device 0** performs a **triad** (`C[i] = alpha*A[i]+B[i]`) with A and B in the **memory of device 0** and C in the **memory of device 1**.
    The execution is controlled by core 8.

### Examples of command lines

* `./shibuya 128 3 C0-C-N0-N0 C1-C-N0-N0 C2-C-N0-N0 C3-C-N0-N0`
  * four cores simultaneously making copies of size 128 MB for 3 seconds in NUMA node 0
* `./shibuya 1024 5 D0-H-D0-D1-0 D1-H-D1-D2-1 D2-H-D2-D3-2 D3-H-D3-D0-3`
  * four devices simultaneously making copies, using `hipMemcpy()`, of size 1 GB for 5 seconds in a round-robin fashion

### Environment Settings

#### AVX Options

* `export SHIBUYA_AVX=1` to use AVX instructions for host streams.
* `export SHIBUYA_AVX_NON_TEMPORAL=1` to use AVX instructions with non-temporal hint for host streams.

If both `SHIBUYA_AVX` and `SHIBUYA_AVX_NON_TEMPORAL` are set, non-temporal stores are used.\
Make sure to `unset SHIBUYA_AVX_NON_TEMPORAL` to use AVX instructions **without** the non-temporal hint.

#### Device Options

* `export SHIBUYA_DEVICE_NON_TEMPORAL=1` to use LLVM non-temporal memory access builtins in device kernels.

By default, in device kernels, each work item touches one element of each input array.\
Use `SHIBUYA_DEVICE_ELEMENTS_PER_ITEM` to change the default value, e.g.:

* `export SHIBUYA_DEVICE_ELEMENTS_PER_ITEM=4` for each work item to touch 4 consecutive elements.

Supported values of `SHIBUYA_DEVICE_ELEMENTS_PER_ITEM` are 1, 2, 4, and 8.

By default, in device kernels, each work group touches one contiguous chunk of each input array.\
Use `SHIBUYA_DEVICE_CHUNKS_PER_GROUP` to change the default value, e.g.:

* `export SHIBUYA_DEVICE_CHUNKS_PER_GROUP=4` for each work group to touch 4 discontiguous chunks.

Supported values of `SHIBUYA_DEVICE_CHUNKS_PER_GROUP` are 1, 2, 4, and 8.

#### Output Options

When printing results, by default, the bandwidth is reported at one second intervals.\
Use `SHIBUYA_OUTPUT_INTERVAL` to change the default value, e.g.:

* `export SHIBUYA_OUTPUT_INTERVAL=10` to set the interval to 10 seconds.
* `export SHIBUYA_OUTPUT_INTERVAL=0.1` to set the interval to one tenth of a second.

#### Correctness Testing

ShibuyaStream is primarily a bandwidth benchmark.\
By default, correctness is not checked when measuring bandwidth.\
Only a simple sanity check is done after bandwidth measurements are finished.\
More rigorous correctness testing can be enabled using the `SHIBUYA_STRINGENT` flag.

* `export SHIBUYA_STRINGENT=1` to switch from measuring bandwidth to stringent correctness testing,
* `unset SHIBUYA_STRINGENT` to switch from testing correctness to measuring bandwidth.

Bandwidth is not measured when testing correctness, as the measurement would not be accurate.\
The stringent correctness test only supports the copy operation (`C`).

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

> **Note**\
> The `H` operation uses `hipMemcpy()` to copy the data from the source to the destination.
> * If executed by a GPU device, host memory is page-locked.
> * If executed by a CPU core, host memory is not page-locked.

> **Note**\
> The benchmark reports the aggregate bandwidth.
> E.g., for the copy operation (`B[i]=A[i]`) it reports the bandwidth of reading A + the bandwidth of writing B.
> If the same memory is the source and the destination then combined read/write bandwidth of that memory is reported.
> If the source and the destination are different, then the result is the bandwidth of the bottleneck ×2
> — either the slower memory or the interconnect.

> **Note**\
> Config is printed to stderr, while performance is printed to stdout. This makes it easy to, e.g.:
> * `2>/dev/null` to discard stderr,
> * `1>results.csv` to send stdout to a file,
> * `&>results.txt` to send stdout and stderr to the same file (overwrite),
> * `&>>results.txt` to send stdout and stderr to the same file (append).

> **Warning**\
> Do not use the same CPU core to execute a CPU stream and control a GPU stream.\
> E.g. if a CPU stream starts with `C0-` make sure that no GPU stream ends with `-0`.\
> Also, do not use the same CPU core to control more then one GPU.\
> I.e., if one GPU stream ends with `-0`, make sure no other does.

> **Warning**\
> Do not launch more then one stream to the same GPU.\
> I.e., if one stream starts with `D0-` make sure no other stream starts with `D0-`.

> **Warning: OLCF Frontier**\
> By default, Frontier uses low-noise mode which prevents users from running on core 0:\
> https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#low-noise-mode-layout \
> Unless you disable the low-noise mode using `-S 0`, attempts to use core 0 will fail.\
> Specifically, the call to `pthread_setaffinity_np()` will fail.

## Help

Jakub Kurzak (<jakurzak@amd.com>)

[ROCm]: https://github.com/RadeonOpenCompute/ROCm
[libnuma]: https://linux.die.net/man/3/numa
[pthreads]: https://linux.die.net/man/7/pthreads
[hwloc]: https://www.open-mpi.org/projects/hwloc/
[Inkscape]: https://inkscape.org/
[ImageMagic]: https://imagemagick.org/index.php
