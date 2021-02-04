
#pragma once

#include "Workload.h"
#include "DeviceArray.h"

#include <chrono>
#include <vector>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__NVCC__)
#include <cuda_runtime.h>
#include "hip2cuda.h"
#endif

//------------------------------------------------------------------------------
// \class Stream
// \brief parent class for HostStream and DeviceStream
template <typename T>
class Stream {
    friend class Report;

public:
    static void enablePeerAccess(int from, int to)
    {
        static std::map<std::tuple<int, int>, bool> peer_access;
        auto from_to = peer_access.find({from, to});
        if (from_to == peer_access.end()) {
            CALL_HIP(hipSetDevice(from));
            CALL_HIP(hipDeviceEnablePeerAccess(to, 0));
            peer_access[{from, to}] = true;
            printf("\tpeer access from %d to %d\n", from, to);
        }
    }

    static Stream<T>* make(std::string const& string,
                           std::size_t size, double duration, T alpha);

    Stream(std::string label,
           Workload workload,
           std::size_t length,
           double duration,
           T alpha,
           Array<T>* a,
           Array<T>* b,
           Array<T>* c = nullptr)
        : label_(label), workload_(workload), length_(length),
          size_(length*sizeof(T)), duration_(duration),
          alpha_(alpha), a_(a), b_(b), c_(c) {}

    virtual ~Stream()
    {
        delete a_;
        delete b_;
        if (workload_.type() == Workload::Type::Add ||
            workload_.type() == Workload::Type::Triad)
            delete c_;
    }

    void run()
    {
        setAffinity();
        double timestamp;
        do {
            auto start = std::chrono::high_resolution_clock::now();
            dispatch();
            auto stop = std::chrono::high_resolution_clock::now();
            timestamp =
                std::chrono::duration_cast<
                    std::chrono::duration<double>>(stop-beginning_).count();
            timestamps_.push_back(timestamp);
            double time =
                std::chrono::duration_cast<
                    std::chrono::duration<double>>(stop-start).count();
            bandwidths_.push_back(bandwidth(time));
        }
        while (timestamp < duration_);
    }

    void test()
    {
        switch (workload_.type()) {
            case Workload::Type::Hip:
            case Workload::Type::Copy:
            case Workload::Type::Mul:
            case Workload::Type::Add:
                a_->init(0.0, 1.0);
                b_->init(length_, -1.0);
                break;
            case Workload::Type::Triad:
                alpha_ = 2.0;
                a_->init(0.0, 0.5);
                b_->init(length_, -1.0);
                break;
            case Workload::Type::Dot:
                a_->init(0.2, 0.0);
                b_->init(5.0, 0.0);
                dot_sum_ = 0.0;
                break;
            default: assert(false);
        }
        if (workload_.type() == Workload::Type::Add ||
            workload_.type() == Workload::Type::Triad)
            c_->init(0.0, 0.0);
        setAffinity();
        dispatch();
        switch (workload_.type()) {
            case Workload::Type::Hip:   b_->check(0.0, 1.0); break;
            case Workload::Type::Copy:  b_->check(0.0, 1.0); break;
            case Workload::Type::Mul:   b_->check(0.0, alpha_); break;
            case Workload::Type::Add:   c_->check(length_, 0.0); break;
            case Workload::Type::Triad: c_->check(length_, 0.0); break;
            case Workload::Type::Dot:   assert(dot_sum_ == length_); break;
            default: assert(false);
        }
    }

    virtual void printInfo() = 0;

    double maxTime() { return timestamps_[timestamps_.size()-1]; }

    double minInterval()
    {
        double min_interval = timestamps_[0];
        for (int i = 1; i < timestamps_.size(); ++i) {
            double interval = timestamps_[i]-timestamps_[i-1];
            if (interval < min_interval) {
                min_interval = interval;
            }
        }
        return min_interval;
    }

    void printStats()
    {
        for (std::size_t i = 0; i < timestamps_.size(); ++i)
            printf("\t%10lf\t%lf\n", timestamps_[i], bandwidths_[i]);
    }

protected:
    void printWorkload() { workload_.print(); }

    void printArrays()
    {
        a_->printInfo();
        b_->printInfo();
        if (workload_.type() == Workload::Type::Add ||
            workload_.type() == Workload::Type::Triad)
            c_->printInfo();
    }

    void setCoreAffinity(int core_id)
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        int retval = pthread_setaffinity_np(pthread_self(),
                                            sizeof(cpu_set_t), &cpuset);
        assert(retval == 0);
    }

    std::string label_;
    Workload workload_;
    std::size_t length_; ///< length in elements
    std::size_t size_;   ///< size in bytes
    T alpha_;
    Array<T>* a_;
    Array<T>* b_;
    Array<T>* c_;
    T dot_sum_;

    double duration_;                ///< desired duration in seconds
    std::vector<double> timestamps_; ///< timestamps in seconds
    std::vector<double> bandwidths_; ///< bandwidths in GBPS

private:
    static void scanString(std::string const& string,
                           char& hardware_char, int& hardware_id,
                           char& workload_char,
                           char& a_char, int& a_id,
                           char& b_char, int& b_id,
                           char& c_char, int& c_id,
                           int& host_core_id);

    virtual void setAffinity() = 0;

    virtual void copy()  = 0;
    virtual void mul()   = 0;
    virtual void add()   = 0;
    virtual void triad() = 0;
    virtual void dot()   = 0;

    void hip()
    {
        auto a = this->a_->host_ptr();
        if (typeid(this->a_) == typeid(DeviceArray<T>))
            a = this->a_->device_ptr();

        auto b = this->b_->host_ptr();
        if (typeid(this->b_) == typeid(DeviceArray<T>))
            b = this->b_->device_ptr();

        hipMemcpy(b, a, this->size_, hipMemcpyDefault);
        hipDeviceSynchronize();
    }

    void dispatch()
    {
        switch (workload_.type()) {
            case Workload::Type::Hip:   hip();   break;
            case Workload::Type::Copy:  copy();  break;
            case Workload::Type::Mul:   mul();   break;
            case Workload::Type::Add:   add();   break;
            case Workload::Type::Triad: triad(); break;
            case Workload::Type::Dot:   dot();   break;
            default: assert(false);
        }
    }

    double bandwidth(double time)
    {
        switch (workload_.type()) {
            case Workload::Type::Hip:   return 2.0*size_/time/1e9;
            case Workload::Type::Copy:  return 2.0*size_/time/1e9;
            case Workload::Type::Mul:   return 2.0*size_/time/1e9;
            case Workload::Type::Add:   return 3.0*size_/time/1e9;
            case Workload::Type::Triad: return 3.0*size_/time/1e9;
            case Workload::Type::Dot:   return 2.0*size_/time/1e9;
            default: assert(false);
        }
        return 0.0; // suppressing NVCC warning
    }

    static std::chrono::high_resolution_clock::time_point beginning_;
};

template <typename T>
std::chrono::high_resolution_clock::time_point Stream<T>::beginning_ =
    std::chrono::high_resolution_clock::now();
