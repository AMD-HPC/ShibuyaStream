//------------------------------------------------------------------------------
/// \file
/// \brief      exception handling
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include <cstdio>
#include <exception>
#include <string>

#if defined(USE_HIP)
    #include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
    #include <cuda_runtime.h>
    #include "hip2cuda.h"
#endif

//------------------------------------------------------------------------------
/// \brief
///     Implements the base class for exception handling.
///
class Exception : public std::exception {
public:
    Exception() : std::exception() {}

    Exception(std::string const& msg,
              const char* func, const char* file, int line)
        : std::exception(),
          msg_(msg+"\n"+func+"() | "+file+" | L:"+std::to_string(line)),
          func_(func), file_(file), line_(line) {}

    virtual char const* what() const noexcept override
    {
        return msg_.c_str();
    }

protected:
    void what(std::string const& msg,
              const char* func, const char* file, int line)
    {
        msg_ = msg+"\n"+func+"() | "+file+" | "+std::to_string(line)+"\033[0m";
    }

    std::string msg_;
    std::string func_;
    std::string file_;
    int line_;
};

/// Report errors.
#define ERROR(msg) \
{ \
    throw Exception(std::string("\033[38;5;200mERROR:\033[38;5;255m ")+ \
                    msg, __func__, __FILE__, __LINE__); \
}

//------------------------------------------------------------------------------
/// \brief
///     Implements exception handling for the ERROR_IF macro.
///
class TrueConditionException : public Exception {
public:
    TrueConditionException(const char* condition,
                           const char* func,
                           const char* file,
                           int line)
        : Exception(std::string("\033[38;5;200mERROR:\033[38;5;255m ")+
                    "Condition '"+condition+"' is true.",
                    func, file, line) {}

    TrueConditionException(const char* condition,
                           const char* description,
                           const char* func,
                           const char* file,
                           int line)
        : Exception(std::string("\033[38;5;200mERROR:\033[38;5;255m ")+
                    description+" Condition '"+condition+"' is true.",
                    func, file, line) {}
};

/// Checks error conditions.
#define ERROR_IF(condition, ...) \
{ \
    if (condition) \
        throw TrueConditionException(#condition, ##__VA_ARGS__, \
                                     __func__, __FILE__, __LINE__); \
}

//------------------------------------------------------------------------------
/// \brief
///     Implements exception handling for the ASSERT macro.
///
class FalseConditionException : public Exception {
public:
    FalseConditionException(const char* assertion,
                            const char* func,
                            const char* file,
                            int line)
        : Exception(std::string("\033[38;5;200mERROR:\033[38;5;255m ")+
                    "Assertion '"+assertion+"' is false.",
                    func, file, line) {}

    FalseConditionException(const char* assertion,
                            const char* description,
                            const char* func,
                            const char* file,
                            int line)
        : Exception(std::string("\033[38;5;200mERROR:\033[38;5;255m ")+
                    description+" Assertion '"+assertion+"' is false.",
                    func, file, line) {}
};

/// Checks assertions.
#define ASSERT(assertion, ...) \
{ \
    if (!(assertion)) \
        throw FalseConditionException(#assertion, ##__VA_ARGS__, \
                                      __func__, __FILE__, __LINE__); \
}

//------------------------------------------------------------------------------
/// \brief
///     Implements exception handling for the HIP_CALL macro.
///
class HIPException : public Exception {
public:
    HIPException(const char* call,
                 hipError_t code,
                 const char* func,
                 const char* file,
                 int line)
        : Exception()
    {
        const char* name = hipGetErrorName(code);
        const char* string = hipGetErrorString(code);
        what(std::string("\033[38;5;200mHIP ERROR:\033[38;5;255m ")+
             call+" returned "+name+" ("+string+").",
             func, file, line);
    }

    HIPException(const char* call,
                 hipError_t code,
                 const char* description,
                 const char* func,
                 const char* file,
                 int line)
        : Exception()
    {
        char const* name = hipGetErrorName(code);
        char const* string = hipGetErrorString(code);
        what(std::string("\033[38;5;200mHIP ERROR:\033[38;5;255m ")+
             description+" \n"+call+" returned "+name+" ("+string+").",
             func, file, line);
    }
};

/// Checks for errors in HIP calls.
#define HIP_CALL(call, ...) \
{ \
    hipError_t code = call; \
    if (code != hipSuccess) \
        throw HIPException(#call, code, ##__VA_ARGS__, \
                           __func__, __FILE__, __LINE__); \
}
