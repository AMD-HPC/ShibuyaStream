
#include <cstdio>
#include <exception>
#include <string>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__NVCC__)
#include <cuda_runtime.h>
#include "hip2cuda.h"
#endif

//------------------------------------------------------------------------------
class Exception : public std::exception {
public:
    Exception(const char* func, const char* file, int line)
        : std::exception(),
          func_(func), file_(file), line_(line) {}

    Exception(std::string const& msg,
              const char* func, const char* file, int line)
        : std::exception(),
          msg_(msg), func_(func), file_(file), line_(line) {}

    virtual char const* what() const noexcept override
    {
        return msg_.c_str();
    }

protected:
    std::string msg_;
    std::string func_;
    std::string file_;
    int line_;
};

/// Throws Exception with a message.
#define ERROR(msg) \
{ \
    throw Exception(msg, __func__, __FILE__, __LINE__); \
}

//------------------------------------------------------------------------------
class TrueConditionException : public Exception {
public:
    TrueConditionException(const char* condition,
                           const char* description,
                           const char* func,
                           const char* file,
                           int line)
        : Exception(func, file, line)
    {
        msg_ = std::string("ShibuyaStream ERROR: ")+
               description+" Condition '"+condition+"' is true.";
    }
};

/// Throws TrueConditionException if condition is true.
#define ERROR_IF(condition, description) \
{ \
    if (condition) \
        throw TrueConditionException(#condition, description, \
                                     __func__, __FILE__, __LINE__); \
}

//------------------------------------------------------------------------------
class FalseConditionException : public Exception {
public:
    FalseConditionException(const char* assertion,
                            const char* description,
                            const char* func,
                            const char* file,
                            int line)
        : Exception(func, file, line)
    {
        msg_ = std::string("ShibuyaStream ERROR: ")+
               +description+" Assertion '"+assertion+"' is false.";
    }
};

/// Throws FalseConditionException if assertion is false.
#define ASSERT(assertion, description) \
{ \
    if (!(assertion)) \
        throw FalseConditionException(#assertion, description, \
                                     __func__, __FILE__, __LINE__); \
}

//------------------------------------------------------------------------------
class HipException : public Exception {
public:
    HipException(const char* call,
                 hipError_t code,
                 const char* description,
                 const char* func,
                 const char* file,
                 int line)
        : Exception(func, file, line)
    {
        const char* name = hipGetErrorName(code);
        const char* string = hipGetErrorString(code);
        msg_ = std::string("ShibuyaStream ERROR: ")+
               description+" "+call+" returned '"+name+"' ("+string+").";
    }
};

/// Throws HipException if call does not return hipSuccess.
#define HIP_CALL(call, description) \
{ \
    if (call != hipSuccess) \
        throw HipException(#call, call, description, \
                           __func__, __FILE__, __LINE__); \
}
