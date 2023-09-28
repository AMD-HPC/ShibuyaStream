//------------------------------------------------------------------------------
/// \file
/// \brief      implementations of Stream methods
/// \date       2020-2023
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#include "AVXHostStream.h"
#include "NonTemporalAVXHostStream.h"
#include "DeviceStream.h"
#include "NonTemporalDeviceStream.h"

template <typename T>
std::chrono::high_resolution_clock::time_point Stream<T>::beginning_ =
    std::chrono::high_resolution_clock::now();

//------------------------------------------------------------------------------
/// \brief
///     Enables peer access from one device to another.
///     Only enables if the path not already enabled.
///     Stores a map of enabled connections.
///
/// \param[in] from
///     the device accessing the memory of another device
///
/// \param[in] to
///     the device being accessed
///
template <typename T>
void
Stream<T>::enablePeerAccess(int from, int to)
{
    static std::map<std::tuple<int, int>, bool> peer_access;
    auto from_to = peer_access.find({from, to});
    if (from_to == peer_access.end()) {
        HIP_CALL(hipSetDevice(from),
                 "Setting the device failed.");
        HIP_CALL(hipDeviceEnablePeerAccess(to, 0),
                 "Enabling of peer access failed.");
        peer_access[{from, to}] = true;
        fprintf(stderr, "\tpeer access from %d to %d\n", from, to);
    }
}

//------------------------------------------------------------------------------
/// \brief
///     Creates either a HostStream or a DeviceStream based on the label.
///     Enables peer access for inter-device streams.
///
/// \param[in] label
///     the string defining the stream, e.g., C0-C-N0-N0
///
/// \param[in] length
///     number of elements in the stream
///
/// \param[in] duration
///     the duration of the iteration in seconds
///
/// \param[in] alpha
///     the scaling factor for Mul and Triad
///
template <typename T>
Stream<T>*
Stream<T>::make(std::string const& label,
                std::size_t length, double duration, T alpha)
{
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
            return make_host(label,
                             hardware_id,
                             Workload(workload_char),
                             length, duration,
                             alpha, a, b, c);
        case 'D':
            return make_device(label,
                               hardware_id, host_core_id,
                               Workload(workload_char),
                               length, duration,
                               alpha, a, b, c);
        default:
            ERROR("Invalid device letter.");
    }
    return nullptr; // suppressing NVCC warning
}

template
Stream<double>*
Stream<double>::make(std::string const& label,
                     std::size_t length, double duration, double alpha);

//------------------------------------------------------------------------------
/// \brief
//
template <typename T>
Stream<T>*
Stream<T>::make_host(std::string const& label,
                     int hardware_id, Workload workload,
                     std::size_t length, double duration,
                     T alpha, Array<T>* a, Array<T>* b, Array<T>* c)
{
    if (std::getenv("SHIBUYA_AVX_NON_TEMPORAL") != nullptr)
        return new NonTemporalAVXHostStream<T>(label,
                                               hardware_id, workload,
                                               length, duration,
                                               alpha, a, b, c);
    else if (std::getenv("SHIBUYA_AVX") != nullptr)
        return new AVXHostStream<T>(label,
                                    hardware_id, workload,
                                    length, duration,
                                    alpha, a, b, c);
    else
        return new HostStream<T>(label,
                                 hardware_id, workload,
                                 length, duration,
                                 alpha, a, b, c);
}

//------------------------------------------------------------------------------
/// \brief
//
template <typename T>
Stream<T>*
Stream<T>::make_device(std::string const& label,
                       int hardware_id, int host_core_id,
                       Workload workload, std::size_t length, double duration,
                       T alpha, Array<T>* a, Array<T>* b, Array<T>* c)
{
    int elements_per_item = 1;
    if (std::getenv("SHIBUYA_DEVICE_ELEMENTS_PER_ITEM") != nullptr) {
        elements_per_item =
            std::atoi(std::getenv("SHIBUYA_DEVICE_ELEMENTS_PER_ITEM"));
    }

    int chunks_per_group = 1;
    if (std::getenv("SHIBUYA_DEVICE_CHUNKS_PER_GROUP") != nullptr) {
        chunks_per_group =
            std::atoi(std::getenv("SHIBUYA_DEVICE_CHUNKS_PER_GROUP"));
    }

#define NTDS(elements_per_item, chunks_per_group) \
    NonTemporalDeviceStream<                      \
        T, elements_per_item, chunks_per_group>(  \
            label, hardware_id, host_core_id,     \
            workload, length, duration,           \
            alpha, a, b, c)

#define DS(elements_per_item, chunks_per_group)  \
    DeviceStream<                                \
        T, elements_per_item, chunks_per_group>( \
            label, hardware_id, host_core_id,    \
            workload, length, duration,          \
            alpha, a, b, c)

    if (std::getenv("SHIBUYA_DEVICE_NON_TEMPORAL") != nullptr) {
        switch (elements_per_item) {
        case 1:
            switch (chunks_per_group) {
            case 1: return new NTDS(1, 1); break;
            case 2: return new NTDS(1, 2); break;
            case 4: return new NTDS(1, 4); break;
            case 8: return new NTDS(1, 8); break;
            default: ERROR("Invalid SHIBUYA_DEVICE_CHUNKS_PER_GROUP");
            }
            break;
        case 2:
            switch (chunks_per_group) {
            case 1: return new NTDS(2, 1); break;
            case 2: return new NTDS(2, 2); break;
            case 4: return new NTDS(2, 4); break;
            case 8: return new NTDS(2, 8); break;
            default: ERROR("Invalid SHIBUYA_DEVICE_CHUNKS_PER_GROUP");
            }
            break;
        case 4:
            switch (chunks_per_group) {
            case 1: return new NTDS(4, 1); break;
            case 2: return new NTDS(4, 2); break;
            case 4: return new NTDS(4, 4); break;
            case 8: return new NTDS(4, 8); break;
            default: ERROR("Invalid SHIBUYA_DEVICE_CHUNKS_PER_GROUP");
            }
            break;
        case 8:
            switch (chunks_per_group) {
            case 1: return new NTDS(8, 1); break;
            case 2: return new NTDS(8, 2); break;
            case 4: return new NTDS(8, 4); break;
            case 8: return new NTDS(8, 8); break;
            default: ERROR("Invalid SHIBUYA_DEVICE_CHUNKS_PER_GROUP");
            }
            break;
        default:  ERROR("Invalid SHIBUYA_DEVICE_NON_TEMPORAL");
        }
    }
    else {
        switch (elements_per_item) {
        case 1:
            switch (chunks_per_group) {
            case 1: return new DS(1, 1); break;
            case 2: return new DS(1, 2); break;
            case 4: return new DS(1, 4); break;
            case 8: return new DS(1, 8); break;
            default: ERROR("Invalid SHIBUYA_DEVICE_CHUNKS_PER_GROUP");
            }
            break;
        case 2:
            switch (chunks_per_group) {
            case 1: return new DS(2, 1); break;
            case 2: return new DS(2, 2); break;
            case 4: return new DS(2, 4); break;
            case 8: return new DS(2, 8); break;
            default: ERROR("Invalid SHIBUYA_DEVICE_CHUNKS_PER_GROUP");
            }
            break;
        case 4:
            switch (chunks_per_group) {
            case 1: return new DS(4, 1); break;
            case 2: return new DS(4, 2); break;
            case 4: return new DS(4, 4); break;
            case 8: return new DS(4, 8); break;
            default: ERROR("Invalid SHIBUYA_DEVICE_CHUNKS_PER_GROUP");
            }
            break;
        case 8:
            switch (chunks_per_group) {
            case 1: return new DS(8, 1); break;
            case 2: return new DS(8, 2); break;
            case 4: return new DS(8, 4); break;
            case 8: return new DS(8, 8); break;
            default: ERROR("Invalid SHIBUYA_DEVICE_CHUNKS_PER_GROUP");
            }
            break;
        default:  ERROR("Invalid SHIBUYA_DEVICE_NON_TEMPORAL");
        }
    }

#undef NTDS
#undef DS
}

//------------------------------------------------------------------------------
/// \brief
///     Executes the stream operation for the requested duration of time.
///     Stores the bandwidth and completion timestamp of each step.
///     Uses `std::chrono` to measure time.
///
template <typename T>
void
Stream<T>::run()
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

template
void
Stream<double>::run();

//------------------------------------------------------------------------------
/// \brief
///     Tests the correctness of the stream operation.
///     Initializes the input arrays to a sequence of consecutive numbers.
///     Checks if the output array contains a sequence of consecutive numbers.
///
/// \remark
///     The three parts: initialization, execution, and validation are executed
///     by different kernels. Therefore, correctness only requires coarse-grain
///     coherence.
///
template <typename T>
void
Stream<T>::test()
{
    switch (workload_.type()) {
        case Workload::Type::Hip:
        case Workload::Type::Copy:
        case Workload::Type::Mul:
        case Workload::Type::Add:
            a_->init(T(0.0), T(1.0));
            b_->init(T(length_), T(-1.0));
            break;
        case Workload::Type::Triad:
            alpha_ = T(2.0);
            a_->init(T(0.0), T(0.5));
            b_->init(T(length_), T(-1.0));
            break;
        case Workload::Type::Dot:
            a_->init(T(0.2), T(0.0));
            b_->init(T(5.0), T(0.0));
            dot_sum_ = T(0.0);
            break;
        default: ERROR("Invalid workload type.");
    }
    if (workload_.type() == Workload::Type::Add ||
        workload_.type() == Workload::Type::Triad)
        c_->init(T(0.0), T(0.0));
    setAffinity();
    dispatch();
    switch (workload_.type()) {
        case Workload::Type::Hip:   b_->check(T(0.0), T(1.0)); break;
        case Workload::Type::Copy:  b_->check(T(0.0), T(1.0)); break;
        case Workload::Type::Mul:   b_->check(T(0.0), T(alpha_)); break;
        case Workload::Type::Add:   c_->check(T(length_), T(0.0)); break;
        case Workload::Type::Triad: c_->check(T(length_), T(0.0)); break;
        case Workload::Type::Dot:
            ASSERT(dot_sum_ == length_, "Correctness check failed.");
            break;
        default: ERROR("Invalid workload type.");
    }
}

template
void
Stream<double>::test();

//------------------------------------------------------------------------------
/// \brief
///
template <typename T>
void
Stream<T>::stress()
{
    switch (workload_.type()) {
        case Workload::Type::Mul:
            ERROR("Multiply (M) not supported by SHIBUYA_STRINGENT.");
            return;
        case Workload::Type::Add:
            ERROR("Add (A) not supported by SHIBUYA_STRINGENT.");
            return;
        case Workload::Type::Triad:
            ERROR("Triad (T) not supported by SHIBUYA_STRINGENT.");
            return;
        case Workload::Type::Dot:
            ERROR("Dot (D) not supported by SHIBUYA_STRINGENT.");
            return;
        default: break;
    }

    setAffinity();
    double timestamp;
    double count = 0.0;
    do {
        a_->init(T(count), T(1.0));
        dispatch();
        b_->check(T(count), T(1.0));
        ++count;

        auto stop = std::chrono::high_resolution_clock::now();
        timestamp =
            std::chrono::duration_cast<
                std::chrono::duration<double>>(stop-beginning_).count();
    }
    while (timestamp < duration_);
}

template
void
Stream<double>::stress();

//------------------------------------------------------------------------------
/// \brief
///     Scans the label, i.e., the command line definition of the stream.
///
/// \returns
///     individual components of the stream's definition
///
/// \todo
///     Implement syntax checks.
///
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

    // Scan array `a` location.
    a_location_char = string[pos++];
    end = string.find('-', pos);
    a_location_id = std::stoi(string.substr(pos, end-pos));
    pos += 2;

    // Scan array `b` location.
    b_location_char = string[pos++];
    end = string.find('-', pos);
    b_location_id = std::stoi(string.substr(pos, end-pos));
    pos += 2;

    // if Add or Triad
    if (workload_char == 'A' || workload_char == 'T') {
        // Scan array 'c' location.
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
