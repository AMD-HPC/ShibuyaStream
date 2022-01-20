//------------------------------------------------------------------------------
/// \file
/// \brief      main ShibuyaStream driver routines
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#include "Report.h"
#include "HostStream.h"
#include "DeviceStream.h"

#include <iostream>
#include <thread>

#include <unistd.h>
#include <numa.h>

//------------------------------------------------------------------------------
/// \brief
///     Print minimal system info.
///     Sets up and launches the run.
///     Prints the performance results.
///
/// \todo
///     Template precision.
///     Add support for vector types.
///
void run(int argc, char** argv)
{
    ASSERT(numa_available() != -1, "NUMA not available.");
    int num_cpus = numa_num_configured_cpus();
    fprintf(stderr, "%3d CPUs\n", num_cpus);

    int numa_nodes = numa_num_configured_nodes();
    fprintf(stderr, "%3d NUMA nodes\n", numa_nodes);

    struct bitmask* bitmask;
    bitmask = numa_get_mems_allowed();
    for (int i = 0; i < numa_num_possible_nodes(); ++i) {
        if (numa_bitmask_isbitset(bitmask, i)) {
            long long free_size;
            long long node_size = numa_node_size64(i, &free_size);
            fprintf(stderr, "\t%2d: %lld, %lld\n", i, node_size, free_size);
        }
    }

    int num_gpus;
    HIP_CALL(hipGetDeviceCount(&num_gpus), "Getting the device count failed.");
    fprintf(stderr, "%3d GPUs\n", num_gpus);

    /// \todo Print command-line syntax.
    ASSERT(argc > 3, "Invalid command line.");
    // size in MB
    std::size_t array_size = std::atol(argv[1])*1024*1024;
    std::size_t array_length = array_size/sizeof(double);
    // duration in seconds
    double test_duration = std::atof(argv[2]);

    double alpha = 1.0f;
    std::vector<Stream<double>*> streams(argc-3);
    for (int i = 3; i < argc; ++i)
        streams[i-3] = Stream<double>::make(argv[i], array_length,
                                            test_duration, alpha);

    fprintf(stderr, "%3ld streams\n", streams.size());
    for (auto const& stream : streams) {
        stream->printInfo();
        fprintf(stderr, "\n");
    }

    std::vector<std::thread> threads(streams.size());
    for (int i = 0; i < streams.size(); ++i)
        threads[i] = std::thread([&, i] {
            streams[i]->run();
            streams[i]->test();
        });
    for (auto& thread : threads)
        thread.join();

    double max_time = 0.0;
    for (auto const& stream : streams) {
        double end_time = stream->maxTime();
        if (end_time > max_time) {
            max_time = end_time;
        }
    }

    // Set output interval.
    double interval = 1.0;
    if (std::getenv("SHIBUYA_OUTPUT_INTERVAL") != nullptr) {
        interval = std::atof(std::getenv("SHIBUYA_OUTPUT_INTERVAL"));
        ASSERT(interval > 0.0);
    }

    fprintf(stderr, "%lf max time\n", max_time);
    fprintf(stderr, "%lf interval\n", interval);
    fflush(stderr);
    usleep(100);

    Report report(max_time, interval);
    for (auto const& stream : streams)
        report.addTimeline(*stream);
    report.print();
}

//------------------------------------------------------------------------------
/// \brief
///     Launches the run inside a `try` block.
///     Caches and reports exceptions.
///
///     Usage: ./shibuya size time strea1 stream2 ...
///
///     - size: size in megabytes
///     - time: duration in seconds
///     - stream1 stream2 ...: communication streams (see README for details)
///
int main(int argc, char** argv)
{
    try {
        run(argc, argv);
    }
    catch (Exception& e) {
        std::cerr << std::endl << e.what() << std::endl << std::endl;
        exit(EXIT_FAILURE);
    }
    exit(EXIT_SUCCESS);
}
