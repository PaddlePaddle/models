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

#include <map>
#include <string>

typedef std::map<std::string, std::string> Map;
typedef Map::const_iterator MapIter;

class ConfParserBase {
 public:
  ConfParserBase() {}

  bool load(const std::string &file_name);

  bool load_from_string(const std::string &str);

  bool get_conf_float(const std::string &key, float &value) const;

  bool get_conf_uint(const std::string &key, unsigned int &value) const;

  bool get_conf_int(const std::string &key, int &value) const;

  bool get_conf_str(const std::string &key, std::string &value) const;

  bool exist(const char *name) const;

  void map_clear();

 private:
  bool parse_line(const std::string &line);

  Map _map;
};

class ConfParser {
 public:
  ConfParser() : _conf(NULL){};
  ~ConfParser();

  bool init(const std::string &conf_file);

  bool get_uint(const std::string &prefix,
                const std::string &key,
                unsigned int &value) const;

  bool get_uints(const std::string &prefix,
                 const std::string &key,
                 std::vector<unsigned int> &values) const;

  bool get_int(const std::string &prefix,
               const std::string &key,
               int &value) const;

  bool get_ints(const std::string &prefix,
                const std::string &key,
                std::vector<int> &values) const;

  bool get_float(const std::string &prefix,
                 const std::string &key,
                 float &value) const;

  bool get_floats(const std::string &prefix,
                  const std::string &key,
                  std::vector<float> &values) const;

  bool get_string(const std::string &prefix,
                  const std::string &key,
                  std::string &value) const;

  bool get_strings(const std::string &prefix,
                   const std::string &key,
                   std::vector<std::string> &values) const;

 public:
  ConfParserBase *_conf;
};
