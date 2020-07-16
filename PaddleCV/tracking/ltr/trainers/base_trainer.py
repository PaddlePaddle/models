import os
import glob
from paddle import fluid
from paddle.fluid import dygraph
import pickle


class BaseTrainer:
    """Base trainer class. Contains functions for training and saving/loading chackpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}

    def update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(
                self.settings.env.workspace_dir)
            self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir,
                                                'checkpoints')
            if not os.path.exists(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None

    def train(self, max_epochs, load_latest=False, fail_safe=True):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
            fail_safe - Bool indicating whether the training to automatically restart in case of any crashes.
        """

        num_tries = 10
        for i in range(num_tries):
            try:
                if load_latest:
                    self.load_checkpoint()

                for epoch in range(self.epoch + 1, max_epochs + 1):
                    self.epoch = epoch
                    self.train_epoch()

                if self._checkpoint_dir:
                    self.save_checkpoint()
            except:
                print('Training crashed at epoch {}'.format(self.epoch))
                if fail_safe:
                    load_latest = True
                    print('Restarting training from last epoch ...')
                else:
                    raise

        print('Finished training!')

    def train_epoch(self):
        raise NotImplementedError

    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""
        actor_type = type(self.actor).__name__
        net_type = type(self.actor.net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net_info': getattr(self.actor.net, 'info', None),
            'constructor': getattr(self.actor.net, 'constructor', None),
            'stats': self.stats,
            'settings': self.settings
        }

        directory = '{}/{}/{}_ep{:04d}'.format(self._checkpoint_dir,
                                               self.settings.project_path,
                                               net_type, self.epoch)
        if not os.path.exists(directory):
            os.makedirs(directory)

        fluid.save_dygraph(self.actor.net.state_dict(), directory)
        fluid.save_dygraph(self.optimizer.state_dict(), directory)
        with open(os.path.join(directory, '_custom_state.pickle'), 'wb') as f:
            pickle.dump(state, f)

    def load_checkpoint(self, checkpoint=None):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net_type = type(self.actor.net).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(
                glob.glob('{}/{}/{}_ep*'.format(self._checkpoint_dir,
                                                self.settings.project_path,
                                                net_type)))
            if checkpoint_list:
                checkpoint_path = os.path.splitext(checkpoint_list[-1])[0]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}/{}_ep{:04d}'.format(
                self._checkpoint_dir, self.settings.project_path, net_type,
                checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # paddle load network
        net_params, opt_params = fluid.load_dygraph(checkpoint_path)
        self.actor.net.load_dict(net_params)
        self.optimizer.set_dict(opt_params)

        # paddle load state
        current_state = pickle.load(
            open(os.path.join(checkpoint_path, '_custom_state.pickle'), 'rb'))

        print("\nload checkpoint done !! Current states are as follows:")
        for key, value in current_state.items():
            print(key, value)
        self.epoch = current_state['epoch']
        self.stats = current_state['stats']

        return True
