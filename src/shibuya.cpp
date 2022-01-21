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
    // Print number of CPUs and NUMA nodes.
    ASSERT(numa_available() != -1, "NUMA not available.");
    fprintf(stderr, "\033[38;5;66m");
    int num_cpus = numa_num_configured_cpus();
    fprintf(stderr, "%3d CPU%s\n", num_cpus, num_cpus > 1 ? "s" : "");
    int num_nodes = numa_num_configured_nodes();
    fprintf(stderr, "%3d NUMA node%s\n", num_nodes, num_nodes > 1 ? "s" : "");

    // Print NUMA nodes' info.
    struct bitmask* bitmask;
    bitmask = numa_get_mems_allowed();
    for (int i = 0; i < numa_num_possible_nodes(); ++i) {
        if (numa_bitmask_isbitset(bitmask, i)) {
            long long free_size;
            long long node_size = numa_node_size64(i, &free_size);
            fprintf(stderr, "\t%2d: %lld, %lld\n", i, node_size, free_size);
        }
    }

    // Print number of GPUs.
    int num_gpus;
    HIP_CALL(hipGetDeviceCount(&num_gpus), "Getting the device count failed.");
    fprintf(stderr, "%3d GPU%s\n", num_gpus, num_gpus > 1 ? "s" : "");

    // Set length of arrays.
    ASSERT(argc > 3, "Invalid command line.");
    std::size_t array_size = std::atol(argv[1])*1024*1024;
    std::size_t array_length = array_size/sizeof(double);

    // Set test duration.
    double test_duration = std::atof(argv[2]);

    // Create streams.
    double alpha = 1.0f;
    std::vector<Stream<double>*> streams(argc-3);
    for (int i = 3; i < argc; ++i)
        streams[i-3] = Stream<double>::make(argv[i], array_length,
                                            test_duration, alpha);

    // Print streams' info.
    fprintf(stderr, "%3ld stream%s\n", streams.size(),
            streams.size() > 1 ? "s" : "");
    for (auto const& stream : streams) {
        stream->printInfo();
        fprintf(stderr, "\n");
    }

    // Launch the run, test, join thrads.
    std::vector<std::thread> threads(streams.size());
    for (int i = 0; i < streams.size(); ++i)
        threads[i] = std::thread([&, i] {
            streams[i]->run();
            streams[i]->test();
        });
    for (auto& thread : threads)
        thread.join();

    // Find min, max, end time.
    double min_time = std::numeric_limits<double>::infinity();
    double max_time = 0.0;
    double end_time = 0.0;
    for (auto const& stream : streams) {
        if (stream->minTime() < min_time) min_time = stream->minTime();
        if (stream->maxTime() > max_time) max_time = stream->maxTime();
        if (stream->endTime() > end_time) end_time = stream->endTime();
    }
    fprintf(stderr, "min time:\t%lf\n", min_time);
    fprintf(stderr, "max time:\t%lf\n", max_time);
    fprintf(stderr, "end time:\t%lf\n", end_time);
    fflush(stderr);
    usleep(100);

    // Set output interval.
    double interval = 1.0;
    if (std::getenv("SHIBUYA_OUTPUT_INTERVAL") != nullptr) {
        interval = std::atof(std::getenv("SHIBUYA_OUTPUT_INTERVAL"));
        ASSERT(interval > 0.0);
    }

    // Print performance report.
    fprintf(stderr, "\033[0m");
    Report report(end_time, interval);
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
