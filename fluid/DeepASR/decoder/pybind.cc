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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "post_decode_faster.h"

namespace py = pybind11;

PYBIND11_MODULE(post_decode_faster, m) {
  m.doc() = "Decoder for Deep ASR model";

  py::class_<Decoder>(m, "Decoder")
      .def(py::init<std::string, std::string, std::string, kaldi::BaseFloat>())
      .def("decode",
           (std::vector<std::string> (Decoder::*)(std::string)) &
               Decoder::decode,
           "Decode for the probability matrices in specifier "
           "and return the transcriptions.")
      .def(
          "decode",
          (std::string (Decoder::*)(
              std::string, const std::vector<std::vector<kaldi::BaseFloat>>&)) &
              Decoder::decode,
          "Decode one input probability matrix "
          "and return the transcription.");
}
