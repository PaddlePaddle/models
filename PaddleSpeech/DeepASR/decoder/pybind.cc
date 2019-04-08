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

#include "post_latgen_faster_mapped.h"

namespace py = pybind11;

PYBIND11_MODULE(post_latgen_faster_mapped, m) {
  m.doc() = "Decoder for Deep ASR model";

  py::class_<Decoder>(m, "Decoder")
      .def(py::init<std::string,
                    std::string,
                    std::string,
                    std::string,
                    size_t,
                    kaldi::BaseFloat>())
      .def("decode_from_file",
           (void (Decoder::*)(std::string, size_t)) & Decoder::decode_from_file,
           "Decode for the probability matrices in specifier "
           "and print the transcriptions.")
      .def(
          "decode",
          (std::string (Decoder::*)(
              std::string, const std::vector<std::vector<kaldi::BaseFloat>>&)) &
              Decoder::decode,
          "Decode one input probability matrix "
          "and return the transcription.")
      .def("decode_batch",
           (std::vector<std::string> (Decoder::*)(
               std::vector<std::string>,
               const std::vector<std::vector<std::vector<kaldi::BaseFloat>>>&,
               size_t num_processes)) &
               Decoder::decode_batch,
           "Decode one batch of probability matrices "
           "and return the transcriptions.");
}
