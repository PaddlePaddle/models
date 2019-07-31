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
