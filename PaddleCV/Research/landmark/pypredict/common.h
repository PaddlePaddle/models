#pragma once

#include <unistd.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

static inline bool file_exist(const std::string &file_name) {
  return ((access(file_name.c_str(), 0)) != -1) ? true : false;
}

template <class T>
static inline bool str2num(const std::string &str, T &num) {
  std::istringstream istr(str);
  istr >> num;
  return !istr.fail();
};

template <class T>
static bool strs2nums(const std::vector<std::string> &strs,
                      std::vector<T> &nums) {
  nums.resize(strs.size());
  for (size_t i = 0; i < strs.size(); i++) {
    if (!str2num(strs[i], nums[i])) {
      nums.clear();
      return false;
    }
  }

  return true;
};

template <class T>
static inline std::string num2str(T a) {
  std::stringstream istr;
  istr << a;
  return istr.str();
}
