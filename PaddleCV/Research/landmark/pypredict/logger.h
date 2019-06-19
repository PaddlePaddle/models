#pragma once

#include <iostream>
#include <sstream>

// compatiable with glog
enum {
  INFO = 0,
  WARNING = 1,
  ERROR = 2,
  FATAL = 3,
};

struct NullStream : std::ostream {
  NullStream() : std::ios(0), std::ostream(0) {}
};

class Logger {
 public:
  Logger(const char *filename, int lineno, int loglevel) {
    static const char *log_levels[] = {"INFO ", "WARN ", "ERROR", "FATAL"};

    static NullStream nullstream;
    _loglevel = loglevel;
    _logstream = (_loglevel >= getloglevel()) ? &std::cerr : &nullstream;
    (*_logstream) << log_levels[_loglevel] << ":" << filename << "[" << lineno
                  << "]";
  }
  static inline int &getloglevel() {
    // default initialized with glog env
    static int globallevel = getgloglevel();
    return globallevel;
  }
  static inline void setloglevel(int loglevel) { getloglevel() = loglevel; }
  static int getgloglevel() {
    char *env = getenv("GLOG_minloglevel");
    int level = WARNING;
    if (env != NULL) {
      int num = 0;
      std::istringstream istr(env);
      istr >> num;
      if (!istr.fail()) {
        level = num;
      }
    }
    return level;
  }
  ~Logger() { *_logstream << std::endl; }
  std::ostream &getstream() { return *_logstream; }

 protected:
  int _loglevel;
  std::ostream *_logstream;
};

#define LOG(loglevel) Logger(__FILE__, __LINE__, loglevel).getstream()
