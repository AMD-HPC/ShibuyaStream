//------------------------------------------------------------------------------
/// \file
/// \brief      Report class declaration and inline routines
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include "Stream.h"

//------------------------------------------------------------------------------
/// \brief
///     Generates a symmary of the execution.
///
class Report {
public:
    /// \brief
    ///     Creates the Report object.
    ///
    /// \param[in] duration
    ///     duration of the iteration in seconds
    ///
    /// \param[in] interval
    ///     the sampling interval for the report
    ///
    Report(double duration, double interval)
        : duration_(duration), interval_(interval) {}

    ~Report() {}

    template <typename T>
    void addTimeline(Stream<T> const& stream);

    void print();

private:
    double duration_; ///< duration in seconds
    double interval_; ///< sampling interval

    std::vector<std::string> labels_;             ///< vector of streams' labels
    std::vector<std::vector<double>> bandwidths_; ///< bandwidths of each stream
};

//------------------------------------------------------------------------------
/// \brief
///     Adds a sampled timeline of a stream to the Report.
///     Samples the bandwidth of the stream using interval_.
///
template <typename T>
inline
void
Report::addTimeline(Stream<T> const& stream)
{
    labels_.push_back(stream.label_);
    std::vector<double> sampled_bw;
    double time = 0.0;
    for (std::size_t i = 0; i < stream.timestamps_.size(); ++i) {
        while (time < stream.timestamps_[i]) {
            sampled_bw.push_back(stream.bandwidths_[i]);
            time += interval_;
        }
    }
    bandwidths_.push_back(sampled_bw);
}

//------------------------------------------------------------------------------
/// \brief
///     Prints the performance report.
///     Prints the labels of all streams.
///     Prints the sampled bandwidths of all streams in columns.
///     Computes the aggregate bandwidth and prints as the last column.
///
inline
void
Report::print()
{
    printf("time,");
    for (auto const& label : labels_)
        printf("%s,", label.c_str());
    printf("total\n");

    double time = 0.0;
    for (std::size_t sample = 0; sample < bandwidths_[0].size(); ++sample) {
        printf("%lf", time);
        double total_bandwidth = 0.0;
        for (int stream = 0; stream < bandwidths_.size(); ++stream) {
            if (sample < bandwidths_[stream].size()) {
                total_bandwidth += bandwidths_[stream][sample];
                printf(",%lf", bandwidths_[stream][sample]);
            }
            else {
                printf(",%lf", 0.0);
            }
        }
        printf(",%lf", total_bandwidth);
        time += interval_;
        printf("\n");
    }
}
