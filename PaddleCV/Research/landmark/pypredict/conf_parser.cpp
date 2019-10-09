#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include "logger.h"

#include "common.h"
#include "conf_parser.h"

std::string join_string(const std::string &prefix, const std::string &str) {
  if (prefix.empty() && str.empty()) {
    return "";
  } else if (prefix.empty()) {
    return str;
  } else if (str.empty()) {
    return prefix;
  }

  return prefix + str;
}

bool read_text_file(const std::string &file_name, std::string &str) {
  LOG(INFO) << "read_text_file!";

  if (!file_exist(file_name)) {
    LOG(FATAL) << "file: " << file_name << "is not exist!";
    return false;
  }

  std::ifstream ifs(file_name.c_str(), std::ios::binary);
  if (!ifs) {
    LOG(FATAL) << "fail to open " << file_name;
    return false;
  }

  std::stringstream ss;
  ss << ifs.rdbuf();
  str = ss.str();

  return true;
}

std::vector<std::string> split_str(const std::string &str,
                                   const std::string &sep,
                                   bool suppress_blanks) {
  std::vector<std::string> array;
  size_t position = 0;
  size_t last_position = 0;

  last_position = position = 0;
  while (position + sep.size() <= str.size()) {
    if (str[position] == sep[0] && str.substr(position, sep.size()) == sep) {
      if (!suppress_blanks || position - last_position > 0) {
        array.push_back(str.substr(last_position, position - last_position));
      }
      last_position = position = position + sep.size();
    } else {
      position++;
    }
  }

  if (!suppress_blanks || last_position - str.size()) {
    array.push_back(str.substr(last_position));
  }

  return array;
}

void strip(std::string &s) {
  if (s.empty()) {
    return;
  }

  s.erase(remove_if(s.begin(), s.end(), isspace), s.end());

  if (s.size() == 1 &&
      (s[0] == ' ' || s[0] == '\t' || s[0] == '\n' || s[0] == '\r')) {
    s = "";
  }

  int begin = -1;
  int end = 0;
  for (size_t i = 0; i < s.length(); i++) {
    if (!(s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) {
      begin = i;
      break;
    }
  }

  if (begin < 0) {
    s = "";
    return;
  }

  for (int i = s.length() - 1; i >= 0; i--) {
    if (!(s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) {
      end = i;
      break;
    }
  }

  if (((int)s.size()) != end - begin + 1) {
    s = s.substr(begin, end - begin + 1);
  }
}

bool ConfParserBase::load(const std::string &file_name) {
  std::string str;
  if (!read_text_file(file_name, str)) {
    LOG(FATAL) << "fail to read " << file_name;
    return false;
  }

  load_from_string(str);

  return true;
}

bool ConfParserBase::load_from_string(const std::string &str) {
  map_clear();
  std::vector<std::string> lines = split_str(str, "\n", true);

  int count = 0;
  for (size_t i = 0; i < lines.size(); i++) {
    if (parse_line(lines[i])) {
      count++;
    }
  }

  return (count > 0);
}

bool ConfParserBase::get_conf_float(const std::string &key,
                                    float &value) const {
  MapIter it = _map.find(key);
  if (it == _map.end()) {
    return false;
  }

  float temp = 0;
  if (!str2num(it->second, temp)) {
    LOG(WARNING) << "failure to convert " << it->second << " to float";
    return false;
  }

  value = temp;

  return true;
}

bool ConfParserBase::get_conf_uint(const std::string &key,
                                   unsigned int &value) const {
  MapIter it = _map.find(key);
  if (it == _map.end()) {
    LOG(WARNING) << "fail to get: " << key;
    return false;
  }

  unsigned int temp = 0;
  if (!str2num(it->second, temp)) {
    LOG(ERROR) << "fail to convert " << it->second << " to float";
    return false;
  }

  value = temp;

  return true;
}

bool ConfParserBase::get_conf_int(const std::string &key, int &value) const {
  MapIter it = _map.find(key);
  if (it == _map.end()) {
    return false;
  }

  int temp = 0;
  if (!str2num(it->second, temp)) {
    LOG(ERROR) << "fail to convert " << it->second << " to float";
    return false;
  }

  value = temp;

  return true;
}

bool ConfParserBase::get_conf_str(const std::string &key,
                                  std::string &value) const {
  MapIter it = _map.find(key);
  if (it == _map.end()) {
    LOG(WARNING) << "fail to get: " << key;
    return false;
  } else {
    value = it->second;
  }

  return true;
}

bool ConfParserBase::exist(const char *name) const {
  return _map.find(name) != _map.end();
}

void ConfParserBase::map_clear() { _map.clear(); }

bool ConfParserBase::parse_line(const std::string &line) {
  std::string strip_line = line;
  strip(strip_line);
  if (strip_line.empty() || strip_line[0] == '#' || strip_line[0] == ';') {
    return false;
  }

  std::basic_string<char>::size_type index_pos = strip_line.find(':');
  if (index_pos == std::string::npos) {
    LOG(ERROR) << "wrong setting format of line: " << line;
    return false;
  }

  std::string key = strip_line.substr(0, index_pos);
  std::string value =
      strip_line.substr(index_pos + 1, strip_line.size() - index_pos - 1);
  if (!_map.insert(std::pair<std::string, std::string>(key, value)).second) {
    LOG(WARNING) << "value already exist for key: " << key;
    return false;
  }

  return true;
}

ConfParser::~ConfParser() {
  if (NULL != _conf) {
    delete _conf;
    _conf = NULL;
  }
}

bool ConfParser::init(const std::string &conf_file) {
  _conf = new ConfParserBase();
  if (!_conf->load(conf_file)) {
    LOG(FATAL) << "fail to laod conf file: " << conf_file;
    return false;
  }

  return true;
}

bool ConfParser::get_uint(const std::string &prefix,
                          const std::string &key,
                          unsigned int &value) const {
  std::string pre_key = join_string(prefix, key);
  if (!_conf->get_conf_uint(pre_key, value)) {
    return false;
  }

  return true;
}

bool ConfParser::get_uints(const std::string &prefix,
                           const std::string &key,
                           std::vector<unsigned int> &values) const {
  std::vector<std::string> str_values;
  get_strings(prefix, key, str_values);

  return strs2nums(str_values, values);
}

bool ConfParser::get_int(const std::string &prefix,
                         const std::string &key,
                         int &value) const {
  std::string pre_key = join_string(prefix, key);
  if (!_conf->get_conf_int(pre_key, value)) {
    return false;
  }

  return true;
}

bool ConfParser::get_ints(const std::string &prefix,
                          const std::string &key,
                          std::vector<int> &values) const {
  std::vector<std::string> str_values;
  get_strings(prefix, key, str_values);

  return strs2nums(str_values, values);
}

bool ConfParser::get_float(const std::string &prefix,
                           const std::string &key,
                           float &value) const {
  std::string pre_key = join_string(prefix, key);
  if (!_conf->get_conf_float(pre_key, value)) {
    return false;
  }

  return true;
}

bool ConfParser::get_floats(const std::string &prefix,
                            const std::string &key,
                            std::vector<float> &values) const {
  std::vector<std::string> str_values;
  get_strings(prefix, key, str_values);

  return strs2nums(str_values, values);
}

bool ConfParser::get_string(const std::string &prefix,
                            const std::string &key,
                            std::string &value) const {
  std::string pre_key = join_string(prefix, key);
  if (!_conf->get_conf_str(pre_key, value)) {
    return false;
  }

  return true;
}

bool ConfParser::get_strings(const std::string &prefix,
                             const std::string &key,
                             std::vector<std::string> &values) const {
  std::string pre_key = join_string(prefix, key);
  std::string value;
  if (!_conf->get_conf_str(pre_key, value)) {
    return false;
  }

  std::vector<std::string> split_value = split_str(value, ",", true);
  values.swap(split_value);

  return true;
}
