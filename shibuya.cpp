/*
    Jakub Kurzak
    AMD Research
    2020
*/
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <numa.h>
#include <unistd.h>
#include <pthread.h>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#if defined(__NVCC__)
#include <cuda_runtime.h>
#endif

#if defined(__NVCC__)
    #define hipDeviceSynchronize    cudaDeviceSynchronize
    #define hipFree                 cudaFree
    #define hipHostGetDevicePointer cudaHostGetDevicePointer
    #define hipHostMalloc           cudaMallocHost
    #define hipHostRegister         cudaHostRegister
    #define hipHostRegisterMapped   cudaHostRegisterMapped
    #define hipHostUnregister       cudaHostUnregister
    #define hipMalloc               cudaMalloc
    #define hipMemcpy               cudaMemcpy
    #define hipMemcpyDefault        cudaMemcpyDefault
    #define hipSetDevice            cudaSetDevice
    #define hipSuccess              cudaSuccess
#endif

#define CALL_HIP(call) assert(call == hipSuccess)

//------------------------------------------------------------------------------
// \class Workload
// \brief type of streaming workload
class Workload {
public:
    enum class Type {Copy, Mul, Add, Triad, Dot, Hip};

    Workload() {};
    Workload(char letter)
    {
        static std::map<char, Type> types {
            {'H', Type::Hip}, // hipMemcpy
            {'C', Type::Copy},
            {'M', Type::Mul},
            {'A', Type::Add},
            {'T', Type::Triad},
            {'D', Type::Dot}
        };
        type_ = types[letter];
    }

    std::string const& name()
    {
        static std::map<Type, std::string> names {
            {Type::Hip,   "hip"}, // hipMemcpy
            {Type::Copy,  "copy"},
            {Type::Mul,   "mul"},
            {Type::Add,   "add"},
            {Type::Triad, "triad"},
            {Type::Dot,   "dot"}
        };
        return names[type_];
    }

    Type type() { return type_; }
    void print() { fprintf(stderr, "\t%5s", name().c_str()); };

private:
    Type type_;
};

//------------------------------------------------------------------------------
// \class Array
// \brief parent class for HostArray and DeviceArray
template <typename T>
class Array {
public:
    static Array<T>* make(char type, int id, std::size_t length);

    Array(std::size_t length): length_(length), size_(length*sizeof(T)) {}
    virtual ~Array() {}

    T* device_ptr() { return device_ptr_; }
    virtual T* host_ptr() = 0;
    virtual void registerMem() = 0;
    virtual void unregisterMem() = 0;
    virtual void printInfo() = 0;
    virtual void init(T start, T step) = 0;
    virtual void check(T start, T step) = 0;

protected:
    std::size_t size_;   ///< size in bytes
    std::size_t length_; ///< length in elements
    T* device_ptr_;      ///< device ptr. (device mem. or registered host mem.)
};

//------------------------------------------------------------------------------
// \class HostArray
// \brief array in host memory
template <typename T>
class HostArray: public Array<T> {
public:
    HostArray(int numa_id, std::size_t size)
        : Array<T>(size), numa_id_(numa_id)
    {
        this->host_ptr_ = (T*)numa_alloc_onnode(this->size_, numa_id_);
        assert(this->host_ptr_ != nullptr);
    }

    ~HostArray()
    {
        numa_free(this->host_ptr_, this->size_);
    }

    T* host_ptr() override { return host_ptr_; }

    void registerMem() override {
        CALL_HIP(hipHostRegister(this->host_ptr_,
                                 this->size_,
                                 hipHostRegisterMapped));
        CALL_HIP(hipHostGetDevicePointer((void**)(&this->device_ptr_),
                                         (void*)this->host_ptr_, 0));
    }

    void unregisterMem() override {
        CALL_HIP(hipHostUnregister(this->host_ptr_));
    }

    void printInfo() override { fprintf(stderr, "\tnode%6d", numa_id_); }

    void init(T start, T step) override
    {
        T val = start;
        for (std::size_t i = 0; i < this->length_; ++i) {
            host_ptr_[i] = val;
            val += step;
        }
    }

    void check(T start, T step) override
    {
        T val = start;
        for (std::size_t i = 0; i < this->length_; ++i) {
            assert(host_ptr_[i] == val);
            val += step;
        }
    }

private:
    int numa_id_; ///< NUMA node number
    T* host_ptr_; ///< host pointer to host memory
};

//------------------------------------------------------------------------------
// \class DeviceArray
// \brief array in device memory
#if defined(__NVCC__)
    template <typename T>
    __global__ void init_kernel(T* a, T start, T step)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        a[i] = start + step*i;
    }

    template <typename T>
    __global__ void check_kernel(T* a, T start, T step)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        assert(a[i] == start + step*i);
    }
#endif

template <typename T>
class DeviceArray: public Array<T> {
public:
    DeviceArray(int device_id, std::size_t size)
        : Array<T>(size), device_id_(device_id)
    {
        CALL_HIP(hipSetDevice(device_id_));
        CALL_HIP(hipMalloc(&this->device_ptr_, this->size_));
    }

    ~DeviceArray()
    {
        CALL_HIP(hipSetDevice(device_id_));
        CALL_HIP(hipFree(this->device_ptr_));
    }

    T* host_ptr() override { return(this->device_ptr_); }
    void registerMem() override {}
    void unregisterMem() override {}
    void printInfo() override { fprintf(stderr, "\tdevice%4d", device_id_); }

    void init(T start, T step) override
    {
        CALL_HIP(hipSetDevice(device_id_));
        init_kernel<<<dim3(this->length_/group_size_),
                      dim3(group_size_),
                      0, 0>>>(this->device_ptr_, start, step);
        hipDeviceSynchronize();
    }

    void check(T start, T step) override
    {
        CALL_HIP(hipSetDevice(device_id_));
        check_kernel<<<dim3(this->length_/group_size_),
                       dim3(group_size_),
                       0, 0>>>(this->device_ptr_, start, step);
        hipDeviceSynchronize();
    }

private:
    static const int group_size_ = 256;

#if defined(__HIPCC__)
    static __global__ void init_kernel(T* a, T start, T step)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        a[i] = start + step*i;
    }

    static __global__ void check_kernel(T* a, T start, T step)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        assert(a[i] == start + step*i);
    }
#endif

    int device_id_;
};

//------------------------------------------------------------------------------
// \brief Creates either a HostArray or a DeviceArray.
template <typename T>
Array<T>*
Array<T>::make(char type, int id, size_t length)
{
    switch (type) {
        case 'N': return new HostArray<T>(id, length);
        case 'D': return new DeviceArray<T>(id, length);
        default: assert(false);
    }
    return nullptr; // suppressing NVCC warning
}

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

//------------------------------------------------------------------------------
// \class HostStream
// \brief stream for a CPU core
template <typename T>
class HostStream: public Stream<T> {
public:
    HostStream(std::string label,
               int core_id,
               Workload workload,
               std::size_t length,
               double duration,
               T alpha,
               Array<T>* a,
               Array<T>* b,
               Array<T>* c = nullptr)
        : Stream<T>(label, workload, length, duration, alpha, a, b, c),
          core_id_(core_id) {}

    ~HostStream() {}

    void printInfo() override
    {
        fprintf(stderr, "\tcore%6d", core_id_);
        this->printWorkload();
        this->printArrays();
    }

private:
    void setAffinity() override { this->setCoreAffinity(core_id_); }

    void copy() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        for (std::size_t i = 0; i < this->length_; ++i)
            b[i] = a[i];
    }

    void mul() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        T alpha = this->alpha_;
        for (std::size_t i = 0; i < this->length_; ++i)
            b[i] = alpha*a[i];
    }

    void add() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        T* c = this->c_->host_ptr();
        for (std::size_t i = 0; i < this->length_; ++i)
            c[i] = a[i]+b[i];
    }

    void triad() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        T* c = this->c_->host_ptr();
        T alpha = this->alpha_;
        for (std::size_t i = 0; i < this->length_; ++i)
            c[i] = alpha*a[i] + b[i];
    }

    void dot() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        T sum = 0.0;
        for (std::size_t i = 0; i < this->length_; ++i)
            sum += a[i]*b[i];

        this->dot_sum_ = sum;
    }

    int core_id_;
};

//------------------------------------------------------------------------------
// \class DeviceStream
// \brief stream for a GPU device
#if defined(__NVCC__)
    template <typename T>
    __global__ void copy_kernel(T const* a, T* b)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        b[i] = a[i];
    }

    template <typename T>
    __global__ void mul_kernel(T alpha, T const* a, T* b)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        b[i] = alpha*a[i];
    }

    template <typename T>
    __global__ void add_kernel(T const* a, T const* b, T* c)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        c[i] = a[i]+b[i];
    }

    template <typename T>
    __global__ void triad_kernel(T alpha, T const* a, T const* b, T* c)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        c[i] = alpha*a[i] + b[i];
    }

    template <typename T>
    __global__ void dot_kernel(T const* a, T* b,
                                      std::size_t length, T* dot_sums)
    {
        const int group_size_ = 1024;
        __shared__ T sums[group_size_];

        std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        const int thx = threadIdx.x;

        sums[thx] = 0.0;
        for (; i < length; i += blockDim.x*gridDim.x)
            sums[thx] += a[i]*b[i];

        for (int offset = blockDim.x/2; offset > 0; offset /= 2) {
            __syncthreads();
            if (thx < offset) {
                sums[thx] += sums[thx+offset];
            }
        }

        if (thx == 0)
            dot_sums[blockIdx.x] = sums[0];
    }
#endif

template <typename T>
class DeviceStream: public Stream<T> {
public:
    DeviceStream(std::string label,
                 int device_id,
                 int host_core_id,
                 Workload workload,
                 std::size_t length,
                 double duration,
                 T alpha,
                 Array<T>* a,
                 Array<T>* b,
                 Array<T>* c = nullptr)
        : Stream<T>(label, workload, length, duration, alpha, a, b, c),
          device_id_(device_id), host_core_id_(host_core_id)
    {
        this->a_->registerMem();
        this->b_->registerMem();
        if (workload.type() == Workload::Type::Add ||
            workload.type() == Workload::Type::Triad)
            this->c_->registerMem();

        CALL_HIP(hipHostMalloc(&dot_sums_, sizeof(T)*dot_num_groups_));
    }

    ~DeviceStream()
    {
        this->a_->unregisterMem();
        this->b_->unregisterMem();
        if (this->workload_.type() == Workload::Type::Add ||
            this->workload_.type() == Workload::Type::Triad)
            this->c_->unregisterMem();
    }

    void printInfo() override
    {
        fprintf(stderr, "\tdevice%4d", device_id_);
        this->printWorkload();
        this->printArrays();
        fprintf(stderr, "\thost core%4d", host_core_id_);
    }

private:
    static const int group_size_ = 1024;
    static const int dot_num_groups_ = 256;

    void setAffinity() override
    {
        // Set the hosting thread.
        this->setCoreAffinity(host_core_id_);
        // Set the streaming device.
        CALL_HIP(hipSetDevice(device_id_));
    }

    void copy() override
    {
        copy_kernel<<<dim3(this->length_/group_size_),
                      dim3(group_size_),
                      0, 0>>>(
            this->a_->device_ptr(),
            this->b_->device_ptr());
        hipDeviceSynchronize();
    }

    void mul() override
    {
        mul_kernel<<<dim3(this->length_/group_size_),
                     dim3(group_size_),
                     0, 0>>>(
            this->alpha_,
            this->a_->device_ptr(),
            this->b_->device_ptr());
        hipDeviceSynchronize();
    }

    void add() override
    {
        add_kernel<<<dim3(this->length_/group_size_),
                     dim3(group_size_),
                     0, 0>>>(
            this->a_->device_ptr(),
            this->b_->device_ptr(),
            this->c_->device_ptr());
        hipDeviceSynchronize();
    }

    void triad() override
    {
        triad_kernel<<<dim3(this->length_/group_size_),
                       dim3(group_size_),
                       0, 0>>>(
            this->alpha_,
            this->a_->device_ptr(),
            this->b_->device_ptr(),
            this->c_->device_ptr());
        hipDeviceSynchronize();
    }

    void dot() override
    {
        dot_kernel<<<dim3(dot_num_groups_),
                     dim3(group_size_),
                     0, 0>>>(
            this->a_->device_ptr(),
            this->b_->device_ptr(),
            this->length_,
            this->dot_sums_);
        hipDeviceSynchronize();

        this->dot_sum_ = 0.0;
        for (int i = 0; i < dot_num_groups_; ++i)
            this->dot_sum_ += this->dot_sums_[i];
    }

#if defined(__HIPCC__)
    static __global__ void copy_kernel(T const* a, T* b)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        b[i] = a[i];
    }

    static __global__ void mul_kernel(T alpha, T const* a, T* b)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        b[i] = alpha*a[i];
    }

    static __global__ void add_kernel(T const* a, T const* b, T* c)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        c[i] = a[i]+b[i];
    }

    static __global__ void triad_kernel(T alpha, T const* a, T const* b, T* c)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        c[i] = alpha*a[i] + b[i];
    }

    static __global__ void dot_kernel(T const* a, T* b,
                                      std::size_t length, T* dot_sums)
    {
        __shared__ T sums[group_size_];

        std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        const int thx = threadIdx.x;

        sums[thx] = 0.0;
        for (; i < length; i += blockDim.x*gridDim.x)
            sums[thx] += a[i]*b[i];

        for (int offset = blockDim.x/2; offset > 0; offset /= 2) {
            __syncthreads();
            if (thx < offset) {
                sums[thx] += sums[thx+offset];
            }
        }

        if (thx == 0)
            dot_sums[blockIdx.x] = sums[0];
    }
#endif

    int device_id_;
    int host_core_id_;
    T* dot_sums_;
};

//------------------------------------------------------------------------------
// \brief Creates either a HostStream or a DeviceStream.
template <typename T>
Stream<T>*
Stream<T>::make(std::string const& label,
                std::size_t length, double duration, T alpha)
{
    assert(length > 0);

    char hardware_char;
    int  hardware_id;
    char workload_char;
    char a_char;
    int  a_id;
    char b_char;
    int  b_id;
    char c_char;
    int  c_id;
    int  host_core_id;
    scanString(label,
               hardware_char, hardware_id,
               workload_char,
               a_char, a_id,
               b_char, b_id,
               c_char, c_id,
               host_core_id);

    if (hardware_char == 'D') {
        if (a_char == 'D' && hardware_id != a_id)
            enablePeerAccess(hardware_id, a_id);
        if (b_char == 'D' && hardware_id != b_id)
            enablePeerAccess(hardware_id, b_id);
        if (workload_char == 'A' || workload_char == 'T')
            if (c_char == 'D' && hardware_id != c_id)
                enablePeerAccess(hardware_id, c_id);
    }

    Array<T>* a = Array<T>::make(a_char, a_id, length);
    Array<T>* b = Array<T>::make(b_char, b_id, length);
    Array<T>* c;
    if (workload_char == 'A' || workload_char == 'T')
        c = Array<T>::make(c_char, c_id, length);

    switch (hardware_char) {
        case 'C':
            return new HostStream<T>(label,
                                     hardware_id,
                                     Workload(workload_char),
                                     length, duration,
                                     alpha, a, b, c);
        case 'D':
            return new DeviceStream<T>(label,
                                       hardware_id, host_core_id,
                                       Workload(workload_char),
                                       length, duration,
                                       alpha, a, b, c);
        default:
            assert(false);
    }
    return nullptr; // suppressing NVCC warning
}

//------------------------------------------------------------------------------
// \brief Scans command line definition of a stream.
template <typename T>
void
Stream<T>::scanString(std::string const& string,
                      char& hardware_char, int& hardware_id,
                      char& workload_char,
                      char& a_location_char, int& a_location_id,
                      char& b_location_char, int& b_location_id,
                      char& c_location_char, int& c_location_id,
                      int& host_core_id)
{
    // Scan hardware type and number.
    std::size_t pos = 0;
    hardware_char = string[pos++];
    std::size_t end = string.find('-', pos);
    hardware_id = std::stoi(string.substr(pos, end-pos));

    // Scan workload type.
    pos = end+1;
    workload_char = string[pos];
    pos += 2;

    // Scan array a location.
    a_location_char = string[pos++];
    end = string.find('-', pos);
    a_location_id = std::stoi(string.substr(pos, end-pos));
    pos += 2;

    // Scan array b location.
    b_location_char = string[pos++];
    end = string.find('-', pos);
    b_location_id = std::stoi(string.substr(pos, end-pos));
    pos += 2;

    // if add or triad
    if (workload_char == 'A' || workload_char == 'T') {
        // Scan array c location.
        c_location_char = string[pos++];
        end = string.find('-', pos);
        c_location_id = std::stoi(string.substr(pos, end-pos));
        pos += 2;
    }

    // if device stream
    if (hardware_char == 'D') {
        host_core_id = std::stoi(string.substr(pos, end-pos));
    }
}

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

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    assert(numa_available() != -1);
    int num_cpus = numa_num_configured_cpus();
    fprintf(stderr, "%3d CPUs\n", num_cpus);

    int numa_nodes = numa_num_configured_nodes();
    fprintf(stderr, "%3d NUMA nodes\n", numa_nodes);

    struct bitmask* bitmask;
    bitmask = numa_get_mems_allowed();
    for (int i = 0; i < numa_num_possible_nodes(); ++i) {
        if (numa_bitmask_isbitset(bitmask, i)) {
            long free_size;
            long node_size = numa_node_size(i, &free_size);
            fprintf(stderr, "\t%2d: %ld, %ld\n", i, node_size, free_size);
        }
    }

    int num_gpus;
    CALL_HIP(hipGetDeviceCount(&num_gpus));
    fprintf(stderr, "%3d GPUs\n", num_gpus);

    assert(argc > 3);
    // size in MB;
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

    double min_interval = std::numeric_limits<double>::infinity();
    for (auto const& stream : streams) {
        double interval;
        interval = stream->minInterval();
        interval = std::log10(interval);
        interval = std::floor(interval);
        interval = std::pow(10.0, interval);
        if (interval < min_interval)
            min_interval = interval;
    }
    fprintf(stderr, "%lf min interval\n", min_interval);

    double max_time = 0.0;
    for (auto const& stream : streams) {
        double end_time = stream->maxTime();
        if (end_time > max_time) {
            max_time = end_time;
        }
    }
    fprintf(stderr, "%lf max time\n", max_time);

    // for (auto const& stream : streams)
    //     stream->printStats();

    fflush(stderr);
    usleep(100);

    Report report(max_time, min_interval);
    for (auto const& stream : streams)
        report.addTimeline(*stream);
    report.print();

    return (EXIT_SUCCESS);
}
