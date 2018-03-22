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

#include "post_decode_faster.h"

using namespace kaldi;
typedef kaldi::int32 int32;
using fst::SymbolTable;
using fst::VectorFst;
using fst::StdArc;

Decoder::Decoder(std::string word_syms_filename,
                 std::string fst_in_filename,
                 std::string logprior_rxfilename) {
  const char *usage =
      "Decode, reading log-likelihoods (of transition-ids or whatever symbol "
      "is on the graph) as matrices.";

  ParseOptions po(usage);
  binary = true;
  acoustic_scale = 1.5;
  allow_partial = true;
  FasterDecoderOptions decoder_opts;
  decoder_opts.Register(&po, true);  // true == include obscure settings.
  po.Register("binary", &binary, "Write output in binary mode");
  po.Register("allow-partial",
              &allow_partial,
              "Produce output even when final state was not reached");
  po.Register("acoustic-scale",
              &acoustic_scale,
              "Scaling factor for acoustic likelihoods");

  word_syms = NULL;
  if (word_syms_filename != "") {
    word_syms = fst::SymbolTable::ReadText(word_syms_filename);
    if (!word_syms)
      KALDI_ERR << "Could not read symbol table from file "
                << word_syms_filename;
  }

  std::ifstream is_logprior(logprior_rxfilename);
  logprior.Read(is_logprior, false);

  // It's important that we initialize decode_fst after loglikes_reader, as it
  // can prevent crashes on systems installed without enough virtual memory.
  // It has to do with what happens on UNIX systems if you call fork() on a
  // large process: the page-table entries are duplicated, which requires a
  // lot of virtual memory.
  decode_fst = fst::ReadFstKaldi(fst_in_filename);

  decoder = new FasterDecoder(*decode_fst, decoder_opts);
}


Decoder::~Decoder() {
  if (!word_syms) delete word_syms;
  delete decode_fst;
  delete decoder;
}

std::vector<std::string> Decoder::decode(std::string posterior_rspecifier) {
  SequentialBaseFloatMatrixReader posterior_reader(posterior_rspecifier);
  std::vector<std::string> decoding_results;

  BaseFloat tot_like = 0.0;
  kaldi::int64 frame_count = 0;
  int num_success = 0, num_fail = 0;

  Timer timer;
  for (; !posterior_reader.Done(); posterior_reader.Next()) {
    std::string key = posterior_reader.Key();
    Matrix<BaseFloat> loglikes(posterior_reader.Value());

    if (loglikes.NumRows() == 0) {
      KALDI_WARN << "Zero-length utterance: " << key;
      num_fail++;
      continue;
    }
    KALDI_ASSERT(loglikes.NumCols() == logprior.Dim());

    loglikes.ApplyLog();
    loglikes.AddVecToRows(-1.0, logprior);

    DecodableMatrixScaled decodable(loglikes, acoustic_scale);
    decoder->Decode(&decodable);

    VectorFst<LatticeArc> decoded;  // linear FST.

    if ((allow_partial || decoder->ReachedFinal()) &&
        decoder->GetBestPath(&decoded)) {
      num_success++;
      if (!decoder->ReachedFinal())
        KALDI_WARN << "Decoder did not reach end-state, outputting partial "
                      "traceback.";

      std::vector<int32> alignment;
      std::vector<int32> words;
      LatticeWeight weight;
      frame_count += loglikes.NumRows();

      GetLinearSymbolSequence(decoded, &alignment, &words, &weight);

      if (word_syms != NULL) {
        std::string res;
        for (size_t i = 0; i < words.size(); i++) {
          std::string s = word_syms->Find(words[i]);
          res += s;
          if (s == "")
            KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
        }
        decoding_results.push_back(res);
      }
      BaseFloat like = -weight.Value1() - weight.Value2();
      tot_like += like;
    } else {
      num_fail++;
      KALDI_WARN << "Did not successfully decode utterance " << key
                 << ", len = " << loglikes.NumRows();
    }
  }

  double elapsed = timer.Elapsed();
  KALDI_LOG << "Time taken [excluding initialization] " << elapsed
            << "s: real-time factor assuming 100 frames/sec is "
            << (elapsed * 100.0 / frame_count);
  KALDI_LOG << "Done " << num_success << " utterances, failed for " << num_fail;
  KALDI_LOG << "Overall log-likelihood per frame is "
            << (tot_like / frame_count) << " over " << frame_count
            << " frames.";
  return decoding_results;
}
