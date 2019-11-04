decode_to_path=./decoding_result.txt

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u ../../infer_by_ckpt.py --batch_size 96  \
                        --checkpoint checkpoints/deep_asr.latest.checkpoint \
                        --infer_feature_lst data/test_feature.lst  \
                        --mean_var data/global_mean_var \
                        --device CPU \
                        --frame_dim 80  \
                        --class_num 3040 \
                        --num_threads 24  \
                        --beam_size 11 \
                        --decode_to_path $decode_to_path \
                        --trans_model aux/final.mdl \
                        --log_prior aux/logprior \
                        --vocabulary aux/graph/words.txt \
                        --graphs aux/graph/HCLG.fst \
                        --acoustic_scale 0.059 \
                        --parallel
