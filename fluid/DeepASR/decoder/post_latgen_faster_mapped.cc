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

#include "post_latgen_faster_mapped.h"

using namespace kaldi;
typedef kaldi::int32 int32;
using fst::SymbolTable;
using fst::Fst;
using fst::StdArc;

Decoder::Decoder(std::string trans_model_in_filename,
                 std::string word_syms_filename,
                 std::string fst_in_filename,
                 std::string logprior_in_filename,
                 kaldi::BaseFloat acoustic_scale) {
  const char *usage =
      "Generate lattices using neural net model.\n"
      "Usage: post-latgen-faster-mapped [options] <trans-model> "
      "<fst-in|fsts-rspecifier> <logprior> <posts-rspecifier>"
      " <lattice-wspecifier> [ <words-wspecifier> [<alignments-wspecifier>] "
      "]\n";
  ParseOptions po(usage);
  allow_partial = false;
  this->acoustic_scale = acoustic_scale;
  LatticeFasterDecoderConfig config;

  config.Register(&po);
  int32 beam = 11;
  po.Register("beam", &beam, "Beam size");
  po.Register("acoustic-scale",
              &acoustic_scale,
              "Scaling factor for acoustic likelihoods");
  po.Register("word-symbol-table",
              &word_syms_filename,
              "Symbol table for words [for debug output]");
  po.Register("allow-partial",
              &allow_partial,
              "If true, produce output even if end state was not reached.");

  // int argc = 2;
  // char *argv[] = {"post-latgen-faster-mapped", "--beam=11"};
  // po.Read(argc, argv);

  std::ifstream is_logprior(logprior_in_filename);
  logprior.Read(is_logprior, false);

  {
    bool binary;
    Input ki(trans_model_in_filename, &binary);
    this->trans_model.Read(ki.Stream(), binary);
  }

  this->determinize = config.determinize_lattice;

  this->word_syms = NULL;
  if (word_syms_filename != "") {
    if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename))) {
      KALDI_ERR << "Could not read symbol table from file "
                << word_syms_filename;
    }
  }

  // Input FST is just one FST, not a table of FSTs.
  this->decode_fst = fst::ReadFstKaldiGeneric(fst_in_filename);

  this->decoder = new LatticeFasterDecoder(*decode_fst, config);

  std::string lattice_wspecifier =
      "ark:|gzip -c > mapped_decoder_data/lat.JOB.gz";
  if (!(determinize ? compact_lattice_writer.Open(lattice_wspecifier)
                    : lattice_writer.Open(lattice_wspecifier)))
    KALDI_ERR << "Could not open table for writing lattices: ";
  // << lattice_wspecifier;

  words_writer = new Int32VectorWriter("");
  alignment_writer = new Int32VectorWriter("");
}

Decoder::~Decoder() {
  if (!this->word_syms) delete this->word_syms;
  delete this->decode_fst;
  delete this->decoder;
  delete words_writer;
  delete alignment_writer;
}


std::vector<std::string> Decoder::decode(std::string posterior_rspecifier) {
  std::vector<std::string> ret;

  try {
    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    // int num_success = 0, num_fail = 0;

    KALDI_ASSERT(ClassifyRspecifier(fst_in_filename, NULL, NULL) ==
                 kNoRspecifier);
    SequentialBaseFloatMatrixReader posterior_reader("ark:" +
                                                     posterior_rspecifier);

    Timer timer;
    timer.Reset();

    {
      for (; !posterior_reader.Done(); posterior_reader.Next()) {
        std::string utt = posterior_reader.Key();
        Matrix<BaseFloat> &loglikes(posterior_reader.Value());
        KALDI_LOG << utt << " " << loglikes.NumRows() << " x "
                  << loglikes.NumCols();
        ret.push_back(decode(utt, loglikes));
      }
    }

    double elapsed = timer.Elapsed();
    return ret;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    // ret.push_back("error");
    return ret;
  }
}


std::string Decoder::decode(
    std::string key,
    const std::vector<std::vector<kaldi::BaseFloat>> &log_probs) {
  size_t num_frames = log_probs.size();
  size_t dim_label = log_probs[0].size();

  kaldi::Matrix<kaldi::BaseFloat> loglikes(
      num_frames, dim_label, kaldi::kSetZero, kaldi::kStrideEqualNumCols);
  for (size_t i = 0; i < num_frames; ++i) {
    memcpy(loglikes.Data() + i * dim_label,
           log_probs[i].data(),
           sizeof(kaldi::BaseFloat) * dim_label);
  }

  return decode(key, loglikes);
}


std::string Decoder::decode(std::string key,
                            kaldi::Matrix<kaldi::BaseFloat> &loglikes) {
  std::string decoding_result;
  if (loglikes.NumRows() == 0) {
    KALDI_WARN << "Zero-length utterance: " << key;
    // num_fail++;
  }
  KALDI_ASSERT(loglikes.NumCols() == logprior.Dim());

  loglikes.ApplyLog();
  loglikes.AddVecToRows(-1.0, logprior);

  DecodableMatrixScaledMapped matrix_decodable(
      trans_model, loglikes, acoustic_scale);
  double like;

  if (DecodeUtteranceLatticeFaster(*decoder,
                                   matrix_decodable,
                                   trans_model,
                                   word_syms,
                                   key,
                                   acoustic_scale,
                                   determinize,
                                   allow_partial,
                                   alignment_writer,
                                   words_writer,
                                   &compact_lattice_writer,
                                   &lattice_writer,
                                   &like)) {
    // tot_like += like;
    // frame_count += loglikes.NumRows();
    // num_success++;
    decoding_result = "succeed!";
  } else {  // else num_fail++;
    decoding_result = "fail!";
  }
  return decoding_result;
}
