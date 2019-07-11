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

#include <string>
#include <vector>
#include "base/kaldi-common.h"
#include "base/timer.h"
#include "decoder/decodable-matrix.h"
#include "decoder/decoder-wrappers.h"
#include "fstext/kaldi-fst-io.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"
#include "util/common-utils.h"

class Decoder {
public:
  Decoder(std::string trans_model_in_filename,
          std::string word_syms_filename,
          std::string fst_in_filename,
          std::string logprior_in_filename,
          size_t beam_size,
          kaldi::BaseFloat acoustic_scale);
  ~Decoder();

  // Interface to accept the scores read from specifier and print
  // the decoding results directly
  void decode_from_file(std::string posterior_rspecifier,
                        size_t num_processes = 1);

  // Accept the scores of one utterance and return the decoding result
  std::string decode(
      std::string key,
      const std::vector<std::vector<kaldi::BaseFloat>> &log_probs);

  // Accept the scores of utterances in batch and return the decoding results
  std::vector<std::string> decode_batch(
      std::vector<std::string> key,
      const std::vector<std::vector<std::vector<kaldi::BaseFloat>>>
          &log_probs_batch,
      size_t num_processes = 1);

private:
  // For decoding one utterance
  std::string decode_internal(kaldi::LatticeFasterDecoder *decoder,
                              std::string key,
                              kaldi::Matrix<kaldi::BaseFloat> &loglikes);

  std::string DecodeUtteranceLatticeFaster(kaldi::LatticeFasterDecoder *decoder,
                                           kaldi::DecodableInterface &decodable,
                                           std::string utt,
                                           double *like_ptr);

  fst::SymbolTable *word_syms;
  fst::Fst<fst::StdArc> *decode_fst;
  std::vector<kaldi::LatticeFasterDecoder *> decoder_pool;
  kaldi::Vector<kaldi::BaseFloat> logprior;
  kaldi::TransitionModel trans_model;
  kaldi::LatticeFasterDecoderConfig config;

  kaldi::CompactLatticeWriter compact_lattice_writer;
  kaldi::LatticeWriter lattice_writer;
  kaldi::Int32VectorWriter *words_writer;
  kaldi::Int32VectorWriter *alignment_writer;

  bool binary;
  bool determinize;
  kaldi::BaseFloat acoustic_scale;
  bool allow_partial;
};
