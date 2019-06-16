/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

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
