//------------------------------------------------------------------------------
/// \file
/// \brief      Workload class declaration and inline routines
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include <map>

//------------------------------------------------------------------------------
/// \brief
///     Represents the type of streaming workload.
///
class Workload {
public:
    enum class Type {
        Copy,  ///< `b[i] = a[i]`;
        Mul,   ///< `b[i] = alpha*a[i];`
        Add,   ///< `c[i] = a[i]+b[i];`
        Triad, ///< `c[i] = alpha*a[i] + b[i];`
        Dot,   ///< `sum += a[i]*b[i];`
        Hip    ///< `hipMemcpy()`
    };

    Workload() = default;
    /// Creates the workload object of the given type.
    Workload(char letter)
    {
        static std::map<char, Type> types {
            {'H', Type::Hip},
            {'C', Type::Copy},
            {'M', Type::Mul},
            {'A', Type::Add},
            {'T', Type::Triad},
            {'D', Type::Dot}
        };
        type_ = types[letter];
    }

    /// Returns a string with the name of the workload.
    std::string const& name()
    {
        static std::map<Type, std::string> names {
            {Type::Hip,   "hip"},
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
