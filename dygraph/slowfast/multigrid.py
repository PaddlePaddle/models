"""Functions for multigrid training."""

import numpy as np


class MultigridSchedule(object):
    """
    This class defines multigrid training schedule and update cfg accordingly.
    """

    def init_multigrid(self, cfg):
        """
        Update cfg based on multigrid settings.
        Args:
            cfg (configs): configs that contains training and multigrid specific
                hyperparameters.
        Returns:
            cfg (configs): the updated cfg.
        """
        self.schedule = None
        # We may modify cfg.TRAIN.BATCH_SIZE, cfg.DATA.NUM_FRAMES, and
        # cfg.DATA.TRAIN_CROP_SIZE during training, so we store their original
        # value in cfg and use them as global variables.
        cfg.MULTIGRID.default_batch_size = cfg.TRAIN.batch_size  # total bs,64
        cfg.MULTIGRID.default_temporal_size = cfg.DATA.num_frames  # 32
        cfg.MULTIGRID.default_crop_size = cfg.DATA.train_crop_size  # 224

        if cfg.MULTIGRID.LONG_CYCLE:
            self.schedule = self.get_long_cycle_schedule(cfg)
            cfg.OPTIMIZER.steps = [0] + [s[-1] for s in self.schedule]
            # Fine-tuning phase.
            cfg.OPTIMIZER.steps[-1] = (
                cfg.OPTIMIZER.steps[-2] + cfg.OPTIMIZER.steps[-1]) // 2
            cfg.OPTIMIZER.lrs = [
                cfg.OPTIMIZER.gamma**s[0] * s[1][0] for s in self.schedule
            ]
            # Fine-tuning phase.
            cfg.OPTIMIZER.lrs = cfg.OPTIMIZER.lrs[:-1] + [
                cfg.OPTIMIZER.lrs[-2],
                cfg.OPTIMIZER.lrs[-1],
            ]

            cfg.OPTIMIZER.max_epoch = self.schedule[-1][-1]

        elif cfg.MULTIGRID.SHORT_CYCLE:
            cfg.OPTIMIZER.steps = [
                int(s * cfg.MULTIGRID.epoch_factor) for s in cfg.OPTIMIZER.steps
            ]
            cfg.OPTIMIZER.max_epoch = int(cfg.OPTIMIZER.max_epoch *
                                          cfg.OPTIMIZER.max_epoch)
        return cfg

    def update_long_cycle(self, cfg, cur_epoch):
        """
        Before every epoch, check if long cycle shape should change. If it
            should, update cfg accordingly.
        Args:
            cfg (configs): configs that contains training and multigrid specific
                hyperparameters.
            cur_epoch (int): current epoch index.
        Returns:
            cfg (configs): the updated cfg.
            changed (bool): whether to change long cycle shape at this epoch
        """
        base_b, base_t, base_s = get_current_long_cycle_shape(self.schedule,
                                                              cur_epoch)
        if base_s != cfg.DATA.train_crop_size or base_t != cfg.DATA.num_frames:

            cfg.DATA.num_frames = base_t
            cfg.DATA.train_crop_size = base_s
            cfg.TRAIN.batch_size = base_b * cfg.MULTIGRID.default_batch_size  #change bs

            bs_factor = (float(cfg.TRAIN.batch_size / cfg.TRAIN.num_gpus) /
                         cfg.MULTIGRID.bn_base_size)

            if bs_factor == 1:  #single bs == bn_base_size (== 8)
                cfg.TRAIN.bn_norm_type = "batchnorm"
            else:
                cfg.TRAIN.bn_norm_type = "sub_batchnorm"
                cfg.TRAIN.bn_num_splits = int(bs_factor)

            cfg.MULTIGRID.long_cycle_sampling_rate = cfg.DATA.sampling_rate * (
                cfg.MULTIGRID.default_temporal_size // base_t)
            print("Long cycle updates:")
            print("\tbn_norm_type: {}".format(cfg.TRAIN.bn_norm_type))
            if cfg.TRAIN.bn_norm_type == "sub_batchnorm":
                print("\tbn_num_splits: {}".format(cfg.TRAIN.bn_num_splits))
            print("\tTRAIN.batch_size: {}".format(cfg.TRAIN.batch_size))
            print("\tDATA.NUM_FRAMES x LONG_CYCLE_SAMPLING_RATE: {}x{}".format(
                cfg.DATA.num_frames, cfg.MULTIGRID.long_cycle_sampling_rate))
            print("\tDATA.train_crop_size: {}".format(cfg.DATA.train_crop_size))
            return cfg, True
        else:
            return cfg, False

    def get_long_cycle_schedule(self, cfg):
        """
        Based on multigrid hyperparameters, define the schedule of a long cycle.
        Args:
            cfg (configs): configs that contains training and multigrid specific
                hyperparameters.
        Returns:
            schedule (list): Specifies a list long cycle base shapes and their
                corresponding training epochs.
        """

        steps = cfg.OPTIMIZER.steps

        default_size = float(cfg.DATA.num_frames * cfg.DATA.train_crop_size
                             **2)  # 32 * 224 * 224  C*H*W
        default_iters = steps[-1]  # 196

        # Get shapes and average batch size for each long cycle shape.
        avg_bs = []
        all_shapes = []
        #        for t_factor, s_factor in cfg.MULTIGRID.long_cycle_factors:
        for item in cfg.MULTIGRID.long_cycle_factors:
            t_factor, s_factor = item["value"]
            base_t = int(round(cfg.DATA.num_frames * t_factor))
            base_s = int(round(cfg.DATA.train_crop_size * s_factor))
            if cfg.MULTIGRID.SHORT_CYCLE:
                shapes = [
                    [
                        base_t,
                        cfg.MULTIGRID.default_crop_size *
                        cfg.MULTIGRID.short_cycle_factors[0],
                    ],
                    [
                        base_t,
                        cfg.MULTIGRID.default_crop_size *
                        cfg.MULTIGRID.short_cycle_factors[1],
                    ],
                    [base_t, base_s],
                ]  #first two is short_cycle, last is the base long_cycle
            else:
                shapes = [[base_t, base_s]]

            # (T, S) -> (B, T, S)
            shapes = [
                [int(round(default_size / (s[0] * s[1] * s[1]))), s[0], s[1]]
                for s in shapes
            ]
            avg_bs.append(np.mean([s[0] for s in shapes]))
            all_shapes.append(shapes)

        # Get schedule regardless of cfg.MULTIGRID.epoch_factor.
        total_iters = 0
        schedule = []
        for step_index in range(len(steps) - 1):
            step_epochs = steps[step_index + 1] - steps[step_index]

            for long_cycle_index, shapes in enumerate(all_shapes):
                #ensure each of 4 sequences run the same num of iters
                cur_epochs = (step_epochs * avg_bs[long_cycle_index] /
                              sum(avg_bs))

                # get cur_iters from cur_epochs
                cur_iters = cur_epochs / avg_bs[long_cycle_index]
                total_iters += cur_iters
                schedule.append((step_index, shapes[-1], cur_epochs))

        iter_saving = default_iters / total_iters  # ratio between default iters and real iters

        final_step_epochs = cfg.OPTIMIZER.max_epoch - steps[-1]

        # We define the fine-tuning phase to have the same amount of iteration
        # saving as the rest of the training.
        #final_step_epochs / iter_saving make fine-tune having the same iters as training
        ft_epochs = final_step_epochs / iter_saving * avg_bs[-1]

        #        schedule.append((step_index + 1, all_shapes[-1][2], ft_epochs))
        schedule.append((step_index + 1, all_shapes[-1][-1], ft_epochs))

        # Obtrain final schedule given desired cfg.MULTIGRID.epoch_factor.
        x = (cfg.OPTIMIZER.max_epoch * cfg.MULTIGRID.epoch_factor /
             sum(s[-1] for s in schedule))

        final_schedule = []
        total_epochs = 0
        for s in schedule:
            epochs = s[2] * x
            total_epochs += epochs
            final_schedule.append((s[0], s[1], int(round(total_epochs))))
        print_schedule(final_schedule)
        return final_schedule


def print_schedule(schedule):
    """
    Log schedule.
    """
    print(
        "Long_cycle_index\tBase_shape(bs_factor,temporal_size,crop_size)\tEpochs"
    )
    for s in schedule:
        print("{}\t\t\t{}\t\t\t\t\t{}".format(s[0], s[1], s[2]))


def get_current_long_cycle_shape(schedule, epoch):
    """
    Given a schedule and epoch index, return the long cycle base shape.
    Args:
        schedule (configs): configs that contains training and multigrid specific
            hyperparameters.
        cur_epoch (int): current epoch index.
    Returns:
        shapes (list): A list describing the base shape in a long cycle:
            [batch size relative to default,
            number of frames, spatial dimension].
    """
    for s in schedule:
        if epoch < s[-1]:
            return s[1]
    return schedule[-1][1]
