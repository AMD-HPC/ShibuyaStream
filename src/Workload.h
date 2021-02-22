
#pragma once

#include <map>

//------------------------------------------------------------------------------
/// \class Workload
/// \brief type of streaming workload
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
