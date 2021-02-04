
#pragma once

#include "Stream.h"

//------------------------------------------------------------------------------
// \class Report
// \brief Generates execution summaries.
class Report {
public:
    Report(double duration, double interval)
        : duration_(duration), interval_(interval) {}

    ~Report() {}

    template <typename T>
    void addTimeline(Stream<T> const& stream)
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

    void print() {
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

private:
    double duration_;
    double interval_;
    std::vector<std::string> labels_;
    std::vector<std::vector<double>> bandwidths_;
};
