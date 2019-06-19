# Intro

This directory contains examples for the new configuration design

**NOTE**: this directory is intended for internal communications during the migration and should be removed before release


# Rationale

To avoid common pitfalls of current configuration design, which is simply a giant, loosely defined dictionary, i.e., misspelled or displaced keys may lead to errors in model training workflow.


# Design

The proposed design utilizes some of python's reflection mechanism to extract configuration schematics from class definition. To be specific, it extracts information from class constructor arguments, including name, docstring, default value, data type (if type hint is given).


## API

most of the functionality is exposed in `core/workspace.py`

-   `serializable`: this decorator make a class directly serializable in yaml config file, by taking advantage of pyyaml's serialization mechanism **NOTE**: these classes should be "trivially serializable", meaning they should have the constructor arguments and nothing else attached to `self` as attributes, see examples bellow
-   `register`: this decorator register a class as configurable module it understands several special annotations if given in the class definition
    -   `__category__`: for better organization of modules
    -   `__inject__`: a list constructor arguments, which are intended to take configurable modules as input, module instances will be created at runtime an passed in. The configuration value can be a class name string, a serialized object, a config key pointing to a serialized object, or a dict (in which case the constructor needs to handle it, see example)
    -   `__op__`: a time saver annotation for quickly wrapping a paddle operator into a callable object, use with `__append_doc__` (extract docstring from target paddle operator, skip documenting the arguments) to further simplify the process
-   `create`: constructs a module instance according to global configuration
-   `load_config` and `merge_config`: for loading yaml file and merge config settings from command line


## example

```python
from ppdet.core.workspace import register, serializable

@register
@serializable
class AnchorGenerator(object):
    # XXX wraps paddle operator
    __op__ = fluid.layers.anchor_generator
    # XXX docstring for args are extracted from paddle OP
    __append_doc__ = True

    def __init__(self,
                 stride=[16.0, 16.0],
                 anchor_sizes=[32, 64, 128, 256, 512],
                 aspect_ratios=[0.5, 1., 2.],
                 variance=[1., 1., 1., 1.]):
        super(AnchorGenerator, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.variance = variance
        self.stride = stride


@register
@serializable
class RPNTargetAssign(object):
    __op__ = fluid.layers.rpn_target_assign
    __append_doc__ = True

    def __init__(self,
                 rpn_batch_size_per_im=256,
                 rpn_straddle_thresh=0.,
                 rpn_fg_fraction=0.5,
                 rpn_positive_overlap=0.7,
                 rpn_negative_overlap=0.3,
                 use_random=True):
        super(RPNTargetAssign, self).__init__()
        self.rpn_batch_size_per_im = rpn_batch_size_per_im
        self.rpn_straddle_thresh = rpn_straddle_thresh
        self.rpn_fg_fraction = rpn_fg_fraction
        self.rpn_positive_overlap = rpn_positive_overlap
        self.rpn_negative_overlap = rpn_negative_overlap
        self.use_random = use_random


@register
@serializable
class GenerateProposals(object):
    __op__ = fluid.layers.generate_proposals
    __append_doc__ = True

    def __init__(self,
                 pre_nms_top_n=6000,
                 post_nms_top_n=1000,
                 nms_thresh=.5,
                 min_size=.1,
                 eta=1.):
        super(GenerateProposals, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        self.eta = eta


@register
class RPNHead(object):
    r"""
    RPN Head

    Args:
        anchor_generator (object): `AnchorGenerator` instance
        rpn_target_assign (object): `RPNTargetAssign` instance
        train_proposal (object): `GenerateProposals` instance for training
        test_proposal (object): `GenerateProposals` instance for testing
    """
    # XXX these are
    __inject__ = ['anchor_generator', 'rpn_target_assign',
                  'train_proposal', 'test_proposal']

    def __init__(self,
                 anchor_generator=AnchorGenerator().__dict__,
                 rpn_target_assign=RPNTargetAssign().__dict__,
                 train_prop=GenerateProposals(12000, 2000).__dict__,
                 test_prop=GenerateProposals().__dict__):
        super(RPNHead, self).__init__()
        self.anchor_generator = anchor_generator
        self.rpn_target_assign = rpn_target_assign
        self.train_proposal = train_proposal
        self.test_proposal = test_proposal
        # XXX if dict input is desired, they need to be handled here
        if isinstance(anchor_generator, dict):
            self.anchor_generator = AnchorGenerator(**anchor_generator)
        if isinstance(rpn_target_assign, dict):
            self.rpn_target_assign = RPNTargetAssign(**rpn_target_assign)
        if isinstance(train_proposal, dict):
            self.train_proposal = GenerateProposals(**train_prop)
        if isinstance(test_proposal, dict):
            self.test_proposal = GenerateProposals(**test_prop)

    # XXX demo purpose
    def forward(self, mode='train'):
        # ...
        anchor, anchor_var = self.anchor_generator(input=rpn_conv)
        # ...
        prop_op = self.train_proposal if mode == 'train' else self.test_proposal
        rpn_rois, rpn_roi_probs = prop_op(
            scores=rpn_cls_score_prob,
            bbox_deltas=rpn_bbox_pred,
            im_info=im_info,
            anchors=self.anchor,
            variances=self.anchor_var)
        # ...
```

the correspondent YAML snippet, NOTE this is the configuration in **FULL**, if any option use the default value, it can be omitted. In case of the above example, all arguments have default value, meaning nothing is required in the config file

```yaml
RPNHead:
  anchor_generator:
    anchor_sizes:
    - 32
    - 64
    - 128
    - 256
    - 512
    aspect_ratios:
    - 0.5
    - 1.0
    - 2.0
    stride:
    - 16.0
    - 16.0
    variance:
    - 1.0
    - 1.0
    - 1.0
    - 1.0
  rpn_target_assign:
    rpn_batch_size_per_im: 256
    rpn_fg_fraction: 0.5
    rpn_negative_overlap: 0.3
    rpn_positive_overlap: 0.7
    rpn_straddle_thresh: 0.0
    use_random: true
  test_proposal:
    eta: 1.0
    min_size: 0.1
    nms_thresh: 0.5
    post_nms_top_n: 1000
    pre_nms_top_n: 6000
  train_proposal:
    eta: 1.0
    min_size: 0.1
    nms_thresh: 0.5
    post_nms_top_n: 2000
    pre_nms_top_n: 12000
```

config file can also have serialized objects in it, denoted with `!`

```yaml
RPNHead:
  anchor_generator: !AnchorGenerator
    anchor_sizes:
    - 32
    - 64
    - 128
    - 256
    - 512
    aspect_ratios:
    - 0.5
    - 1.0
    - 2.0
    stride:
    - 16.0
    - 16.0
    variance:
    - 1.0
    - 1.0
    - 1.0
    - 1.0
  rpn_target_assign: !RPNTargetAssign
    rpn_batch_size_per_im: 256
    rpn_fg_fraction: 0.5
    rpn_negative_overlap: 0.3
    rpn_positive_overlap: 0.7
    rpn_straddle_thresh: 0.0
    use_random: true
  test_proposal: !GenerateProposals
    eta: 1.0
    min_size: 0.1
    nms_thresh: 0.5
    post_nms_top_n: 1000
    pre_nms_top_n: 6000
  train_proposal: !GenerateProposals
    eta: 1.0
    min_size: 0.1
    nms_thresh: 0.5
    post_nms_top_n: 2000
    pre_nms_top_n: 12000
```

to use the module

```python
from ppdet.core.worskspace import load_config, merge_config, create

load_config('some_config_file.yml')
merge_config(more_config_options_from_command_line)

rpn_head = create('RPNHead')
# use the created module!
```


# Requirements

Two python packages are used, both are optional

-   [typeguard](https://github.com/agronholm/typeguard) is used for type checking in Python 3
-   [docstring<sub>parser</sub>](https://github.com/rr-/docstring_parser) is needed for docstring parsing

To install

```shell
pip install typeguard http://github.com/willthefrog/docstring_parser/tarball/master
```


# Tooling

`tools/configure.py` is provided for simplifying the configuration process, it understands 4 subcommands

1.  `list`: list currently register modules by category, list modules in a single category with "&#x2013;category"
2.  `help`: get help for a given module, including module description, description for its options, example configuration and command line flags
3.  `analyze`: check a given configuration file for missing options (corresponding to required constructor arguments), extraneous options (if constructor does not accept `kwargs`), options with mismatch type (if type hint is given) and missing dependencies (a "injected" module that is not properly configured) it also highlights options with value overridden by the user, i.e., non default values
4.  `generate`: generate a configuration template for a given list of modules, by default it generates a full config file, if a "&#x2013;minimal" flag is given, it will generate a template that only contain non optional settings, for example, to generate a configuration for Faster R-CNN with ResNet backbone and FPN, run

    ```shell
    python configure.py generate FasterRCNN ResNet RPNHead RoIAlign BBoxAssigner BBoxHead FasterRCNNTrainFeed FasterRCNNTestFeed LearningRate OptimizerBuilder
    ```

    for a minimal version, run

    ```shell
    python configure.py generate FasterRCNN BBoxHead  --minimal
    ```
