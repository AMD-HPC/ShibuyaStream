
#include "HostStream.h"
#include "DeviceStream.h"

//------------------------------------------------------------------------------
template <typename T>
void
Stream<T>::enablePeerAccess(int from, int to)
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

template
Stream<double>*
Stream<double>::make(std::string const& label,
                     std::size_t length, double duration, double alpha);

//------------------------------------------------------------------------------
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
template <typename T>
void
Stream<T>::test()
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

template
void
Stream<double>::test();

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
