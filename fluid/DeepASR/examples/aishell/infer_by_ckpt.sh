decode_to_path=./decoding_result.txt

export CUDA_VISIBLE_DEVICES=2,3,4,5
python -u ../../infer_by_ckpt.py --batch_size 96  \
                        --checkpoint checkpoints/deep_asr.pass_20.checkpoint \
                        --infer_feature_lst data/test_feature.lst  \
                        --mean_var data/global_mean_var \
                        --frame_dim 80  \
                        --class_num 3040 \
                        --num_threads 24  \
                        --beam_size 11 \
                        --decode_to_path $decode_to_path \
                        --trans_model mapped_decoder_data/exp/tri5a/final.mdl \
                        --log_prior mapped_decoder_data/logprior \
                        --vocabulary mapped_decoder_data/exp/tri5a/graph/words.txt \
                        --graphs mapped_decoder_data/exp/tri5a/graph/HCLG.fst \
                        --acoustic_scale 0.059 \
                        --parallel
