/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <vector>
#include "base/kaldi-common.h"
#include "base/timer.h"
#include "decoder/decodable-matrix.h"
#include "decoder/faster-decoder.h"
#include "fstext/fstext-lib.h"
#include "hmm/transition-model.h"
#include "lat/kaldi-lattice.h"  // for {Compact}LatticeArc
#include "tree/context-dep.h"
#include "util/common-utils.h"


class Decoder {
public:
  Decoder(std::string word_syms_filename,
          std::string fst_in_filename,
          std::string logprior_rxfilename);
  ~Decoder();

  std::vector<std::string> decode(std::string posterior_rspecifier);

private:
  fst::SymbolTable *word_syms;
  fst::VectorFst<fst::StdArc> *decode_fst;
  kaldi::FasterDecoder *decoder;
  kaldi::Vector<kaldi::BaseFloat> logprior;

  kaldi::Int32VectorWriter *words_writer;
  kaldi::Int32VectorWriter *alignment_writer;

  bool binary;
  kaldi::BaseFloat acoustic_scale;
  bool allow_partial;
};
