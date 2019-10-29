#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from itertools import chain

from paddle import fluid
import paddle.fluid.dygraph as dg

from tqdm import tqdm
from eval_model import eval_model, save_states


def train_model(model, loader, criterion, optimizer, clipper, writer, args,
                hparams):
    assert fluid.framework.in_dygraph_mode(
    ), "this function must be run within dygraph guard"

    local_rank = dg.parallel.Env().local_rank

    # amount of shifting when compute losses
    linear_shift = hparams.outputs_per_step
    mel_shift = hparams.outputs_per_step

    global_step = 0
    global_epoch = 0
    ismultispeaker = model.n_speakers > 1
    checkpoint_dir = os.path.join(args.output, "checkpoints")
    tensorboard_dir = os.path.join(args.output, "log")

    for epoch in range(hparams.nepochs):
        epoch_loss = 0.
        for step, inputs in tqdm(enumerate(loader())):

            if len(inputs) == 9:
                (text, input_lengths, mel, linear, text_positions,
                 frame_positions, done, target_lengths, speaker_ids) = inputs
            else:
                (text, input_lengths, mel, linear, text_positions,
                 frame_positions, done, target_lengths) = inputs
                speaker_ids = None

            model.train()
            if not (args.train_seq2seq_only or args.train_postnet_only):
                results = model(text, input_lengths, mel, speaker_ids,
                                text_positions, frame_positions)
                mel_outputs, linear_outputs, alignments, done_hat = results
            elif args.train_seq2seq_only:

                if speaker_ids is not None:
                    speaker_embed = model.speaker_embedding(speaker_ids)
                else:
                    speaker_embed = None
                results = model.seq2seq(text, input_lengths, mel, speaker_embed,
                                        text_positions, frame_positions)
                mel_outputs, alignments, done_hat, decoder_states = results
                if model.r > 1:
                    mel_outputs = fluid.layers.transpose(mel_outputs,
                                                         [0, 3, 2, 1])
                    mel_outputs = fluid.layers.reshape(
                        mel_outputs,
                        [mel_outputs.shape[0], -1, 1, model.mel_dim])
                    mel_outputs = fluid.layers.transpose(mel_outputs,
                                                         [0, 3, 2, 1])

                linear_outputs = None
            else:
                assert (
                    model.use_decoder_state_for_postnet_input is False
                ), "when train only the converter, you have no decoder states"

                if speaker_ids is not None:
                    speaker_embed = model.speaker_embedding(speaker_ids)
                else:
                    speaker_embed = None
                linear_outputs = model.converter(mel, speaker_embed)
                alignments = None
                mel_outputs = None
                done_hat = None

            if not args.train_seq2seq_only:
                n_priority_freq = int(hparams.priority_freq /
                                      (hparams.sample_rate * 0.5) *
                                      model.linear_dim)
                linear_mask = fluid.layers.sequence_mask(
                    target_lengths, maxlen=linear.shape[-1], dtype="float32")
                linear_mask = linear_mask[:, linear_shift:]
                linear_predicted = linear_outputs[:, :, :, :-linear_shift]
                linear_target = linear[:, :, :, linear_shift:]
                lin_l1_loss = criterion.l1_loss(
                    linear_predicted,
                    linear_target,
                    linear_mask,
                    priority_bin=n_priority_freq)
                lin_div = criterion.binary_divergence(
                    linear_predicted, linear_target, linear_mask)
                lin_loss = criterion.binary_divergence_weight * lin_div \
                    + (1 - criterion.binary_divergence_weight) * lin_l1_loss
                if writer is not None and local_rank == 0:
                    writer.add_scalar("linear_loss",
                                      float(lin_loss.numpy()), global_step)
                    writer.add_scalar("linear_l1_loss",
                                      float(lin_l1_loss.numpy()), global_step)
                    writer.add_scalar("linear_binary_div_loss",
                                      float(lin_div.numpy()), global_step)

            if not args.train_postnet_only:
                mel_lengths = target_lengths // hparams.downsample_step
                mel_mask = fluid.layers.sequence_mask(
                    mel_lengths, maxlen=mel.shape[-1], dtype="float32")
                mel_mask = mel_mask[:, mel_shift:]
                mel_predicted = mel_outputs[:, :, :, :-mel_shift]
                mel_target = mel[:, :, :, mel_shift:]
                mel_l1_loss = criterion.l1_loss(mel_predicted, mel_target,
                                                mel_mask)
                mel_div = criterion.binary_divergence(mel_predicted, mel_target,
                                                      mel_mask)
                mel_loss = criterion.binary_divergence_weight * mel_div \
                    + (1 - criterion.binary_divergence_weight) * mel_l1_loss
                if writer is not None and local_rank == 0:
                    writer.add_scalar("mel_loss",
                                      float(mel_loss.numpy()), global_step)
                    writer.add_scalar("mel_l1_loss",
                                      float(mel_l1_loss.numpy()), global_step)
                    writer.add_scalar("mel_binary_div_loss",
                                      float(mel_div.numpy()), global_step)

                done_loss = criterion.done_loss(done_hat, done)
                if writer is not None and local_rank == 0:
                    writer.add_scalar("done_loss",
                                      float(done_loss.numpy()), global_step)

                if hparams.use_guided_attention:
                    decoder_length = target_lengths.numpy() / (
                        hparams.outputs_per_step * hparams.downsample_step)
                    attn_loss = criterion.attention_loss(alignments,
                                                         input_lengths.numpy(),
                                                         decoder_length)
                    if writer is not None and local_rank == 0:
                        writer.add_scalar("attention_loss",
                                          float(attn_loss.numpy()), global_step)

            if not (args.train_seq2seq_only or args.train_postnet_only):
                if hparams.use_guided_attention:
                    loss = lin_loss + mel_loss + done_loss + attn_loss
                else:
                    loss = lin_loss + mel_loss + done_loss
            elif args.train_seq2seq_only:
                if hparams.use_guided_attention:
                    loss = mel_loss + done_loss + attn_loss
                else:
                    loss = mel_loss + done_loss
            else:
                loss = lin_loss

            if writer is not None and local_rank == 0:
                writer.add_scalar("loss", float(loss.numpy()), global_step)

            if isinstance(optimizer._learning_rate,
                          fluid.optimizer.LearningRateDecay):
                current_lr = optimizer._learning_rate.step().numpy()
            else:
                current_lr = optimizer._learning_rate
            if writer is not None and local_rank == 0:
                writer.add_scalar("learning_rate", current_lr, global_step)

            epoch_loss += loss.numpy()[0]

            if (local_rank == 0 and global_step > 0 and
                    global_step % hparams.checkpoint_interval == 0):
                save_states(global_step, writer, mel_outputs, linear_outputs,
                            alignments, mel, linear,
                            input_lengths.numpy(), checkpoint_dir)
                step_path = os.path.join(
                    checkpoint_dir, "checkpoint_{:09d}".format(global_step))
                dg.save_dygraph(model.state_dict(), step_path)
                dg.save_dygraph(optimizer.state_dict(), step_path)

            if (local_rank == 0 and global_step > 0 and
                    global_step % hparams.eval_interval == 0):
                eval_model(global_step, writer, model, checkpoint_dir,
                           ismultispeaker)

            if args.use_data_parallel:
                loss = model.scale_loss(loss)
                loss.backward()
                model.apply_collective_grads()
            else:
                loss.backward()

            if not (args.train_seq2seq_only or args.train_postnet_only):
                param_list = model.parameters()
            elif args.train_seq2seq_only:
                if ismultispeaker:
                    param_list = chain(model.speaker_embedding.parameters(),
                                       model.seq2seq.parameters())
                else:
                    param_list = model.seq2seq.parameters()
            else:
                if ismultispeaker:
                    param_list = chain(model.speaker_embedding.parameters(),
                                       model.seq2seq.parameters())
                else:
                    param_list = model.converter.parameters()

            optimizer.minimize(
                loss, grad_clip=clipper, parameter_list=param_list)

            if not (args.train_seq2seq_only or args.train_postnet_only):
                model.clear_gradients()
            elif args.train_seq2seq_only:
                if ismultispeaker:
                    model.speaker_embedding.clear_gradients()
                model.seq2seq.clear_gradients()
            else:
                if ismultispeaker:
                    model.speaker_embedding.clear_gradients()
                model.converter.clear_gradients()

            global_step += 1

        average_loss_in_epoch = epoch_loss / (step + 1)
        print("Epoch loss: {}".format(average_loss_in_epoch))
        if writer is not None and local_rank == 0:
            writer.add_scalar("average_loss_in_epoch", average_loss_in_epoch,
                              global_epoch)
        global_epoch += 1
