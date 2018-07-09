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
#include <limits>
#include "ThreadPool.h"

using namespace kaldi;
typedef kaldi::int32 int32;
using fst::SymbolTable;
using fst::Fst;
using fst::StdArc;

Decoder::Decoder(std::string trans_model_in_filename,
                 std::string word_syms_filename,
                 std::string fst_in_filename,
                 std::string logprior_in_filename,
                 size_t beam_size,
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

  config.Register(&po);
  int32 beam = 11;
  po.Register("acoustic-scale",
              &acoustic_scale,
              "Scaling factor for acoustic likelihoods");
  po.Register("word-symbol-table",
              &word_syms_filename,
              "Symbol table for words [for debug output]");
  po.Register("allow-partial",
              &allow_partial,
              "If true, produce output even if end state was not reached.");

  int argc = 2;
  char *argv[] = {(char *)"post-latgen-faster-mapped",
                  (char *)("--beam=" + std::to_string(beam_size)).c_str()};

  po.Read(argc, argv);

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

  kaldi::LatticeFasterDecoder *decoder =
      new LatticeFasterDecoder(*decode_fst, config);
  decoder_pool.emplace_back(decoder);

  std::string lattice_wspecifier =
      "ark:|gzip -c > mapped_decoder_data/lat.JOB.gz";
  if (!(determinize ? compact_lattice_writer.Open(lattice_wspecifier)
                    : lattice_writer.Open(lattice_wspecifier)))
    KALDI_ERR << "Could not open table for writing lattices: "
              << lattice_wspecifier;

  words_writer = new Int32VectorWriter("");
  alignment_writer = new Int32VectorWriter("");
}

Decoder::~Decoder() {
  if (!this->word_syms) delete this->word_syms;
  delete this->decode_fst;
  for (size_t i = 0; i < decoder_pool.size(); ++i) {
    delete decoder_pool[i];
  }
  delete words_writer;
  delete alignment_writer;
}


void Decoder::decode_from_file(std::string posterior_rspecifier,
                               size_t num_processes) {
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
    double elapsed = 0.0;

    for (size_t n = decoder_pool.size(); n < num_processes; ++n) {
      kaldi::LatticeFasterDecoder *decoder =
          new LatticeFasterDecoder(*decode_fst, config);
      decoder_pool.emplace_back(decoder);
    }
    elapsed = timer.Elapsed();
    ThreadPool thread_pool(num_processes);

    while (!posterior_reader.Done()) {
      timer.Reset();
      std::vector<std::future<std::string>> que;
      for (size_t i = 0; i < num_processes && !posterior_reader.Done(); ++i) {
        std::string utt = posterior_reader.Key();
        Matrix<BaseFloat> &loglikes(posterior_reader.Value());
        que.emplace_back(thread_pool.enqueue(std::bind(
            &Decoder::decode_internal, this, decoder_pool[i], utt, loglikes)));
        posterior_reader.Next();
      }
      timer.Reset();
      for (size_t i = 0; i < que.size(); ++i) {
        std::cout << que[i].get() << std::endl;
      }
    }

  } catch (const std::exception &e) {
    std::cerr << e.what();
  }
}

inline kaldi::Matrix<kaldi::BaseFloat> vector2kaldi_mat(
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
  return loglikes;
}

std::vector<std::string> Decoder::decode_batch(
    std::vector<std::string> keys,
    const std::vector<std::vector<std::vector<kaldi::BaseFloat>>>
        &log_probs_batch,
    size_t num_processes) {
  ThreadPool thread_pool(num_processes);
  std::vector<std::string> decoding_results;  //(keys.size(), "");

  for (size_t n = decoder_pool.size(); n < num_processes; ++n) {
    kaldi::LatticeFasterDecoder *decoder =
        new LatticeFasterDecoder(*decode_fst, config);
    decoder_pool.emplace_back(decoder);
  }

  size_t index = 0;
  while (index < keys.size()) {
    std::vector<std::future<std::string>> res_in_que;
    for (size_t t = 0; t < num_processes && index < keys.size(); ++t) {
      kaldi::Matrix<kaldi::BaseFloat> loglikes =
          vector2kaldi_mat(log_probs_batch[index]);
      res_in_que.emplace_back(
          thread_pool.enqueue(std::bind(&Decoder::decode_internal,
                                        this,
                                        decoder_pool[t],
                                        keys[index],
                                        loglikes)));
      index++;
    }
    for (size_t i = 0; i < res_in_que.size(); ++i) {
      decoding_results.emplace_back(res_in_que[i].get());
    }
  }
  return decoding_results;
}

std::string Decoder::decode(
    std::string key,
    const std::vector<std::vector<kaldi::BaseFloat>> &log_probs) {
  kaldi::Matrix<kaldi::BaseFloat> loglikes = vector2kaldi_mat(log_probs);
  return decode_internal(decoder_pool[0], key, loglikes);
}


std::string Decoder::decode_internal(
    LatticeFasterDecoder *decoder,
    std::string key,
    kaldi::Matrix<kaldi::BaseFloat> &loglikes) {
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
  return this->DecodeUtteranceLatticeFaster(
      decoder, matrix_decodable, key, &like);
}


std::string Decoder::DecodeUtteranceLatticeFaster(
    LatticeFasterDecoder *decoder,
    DecodableInterface &decodable,  // not const but is really an input.
    std::string utt,
    double *like_ptr) {  // puts utterance's like in like_ptr on success.
  using fst::VectorFst;
  std::string ret = utt + ' ';

  if (!decoder->Decode(&decodable)) {
    KALDI_WARN << "Failed to decode file " << utt;
    return ret;
  }
  if (!decoder->ReachedFinal()) {
    if (allow_partial) {
      KALDI_WARN << "Outputting partial output for utterance " << utt
                 << " since no final-state reached\n";
    } else {
      KALDI_WARN << "Not producing output for utterance " << utt
                 << " since no final-state reached and "
                 << "--allow-partial=false.\n";
      return ret;
    }
  }

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  {  // First do some stuff with word-level traceback...
    VectorFst<LatticeArc> decoded;
    if (!decoder->GetBestPath(&decoded))
      // Shouldn't really reach this point as already checked success.
      KALDI_ERR << "Failed to get traceback for utterance " << utt;

    std::vector<int32> alignment;
    std::vector<int32> words;
    GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
    num_frames = alignment.size();
    // if (alignment_writer->IsOpen()) alignment_writer->Write(utt, alignment);
    if (word_syms != NULL) {
      for (size_t i = 0; i < words.size(); i++) {
        std::string s = word_syms->Find(words[i]);
        ret += s + ' ';
      }
    }
    likelihood = -(weight.Value1() + weight.Value2());
  }

  // Get lattice, and do determinization if requested.
  Lattice lat;
  decoder->GetRawLattice(&lat);
  if (lat.NumStates() == 0)
    KALDI_ERR << "Unexpected problem getting lattice for utterance " << utt;
  fst::Connect(&lat);
  if (determinize) {
    CompactLattice clat;
    if (!DeterminizeLatticePhonePrunedWrapper(
            trans_model,
            &lat,
            decoder->GetOptions().lattice_beam,
            &clat,
            decoder->GetOptions().det_opts))
      KALDI_WARN << "Determinization finished earlier than the beam for "
                 << "utterance " << utt;
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &clat);
    // disable output lattice temporarily
    // compact_lattice_writer.Write(utt, clat);
  } else {
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &lat);
    // lattice_writer.Write(utt, lat);
  }
  return ret;
}
