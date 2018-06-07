export CUDA_VISIBLE_DEVICES=0,1
python -u ../../infer_by_ckpt.py --batch_size 64  \
                        --checkpoint deep_asr.pass_20.checkpoint \
                        --infer_feature_lst data/test_feature.lst  \
                        --infer_label_lst data/test_label.lst  \
                        --mean_var data/aishell/global_mean_var \
                        --frame_dim 80  \
                        --class_num 3040 \
                        --post_matrix_path post_matrix.decoded \
                        --target_trans data/text.test \
                        --trans_model mapped_decoder_data/exp/tri5a/final.mdl \
                        --log_prior mapped_decoder_data/logprior \
                        --vocabulary mapped_decoder_data/exp/tri5a/graph/words.txt \
                        --graphs mapped_decoder_data/exp/tri5a/graph/HCLG.fst \
                        --acoustic_scale 0.059 \
                        --parallel
