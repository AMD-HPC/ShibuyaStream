
#include "HostStream.h"
#include "DeviceStream.h"

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
