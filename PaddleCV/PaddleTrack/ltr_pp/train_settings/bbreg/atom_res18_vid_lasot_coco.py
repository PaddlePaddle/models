import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph

import ltr_pp.actors as actors
import ltr_pp.data.transforms as dltransforms
from ltr_pp.data import processing, sampler, loader
from ltr_pp.dataset import ImagenetVID, MSCOCOSeq, Lasot, Got10k
from ltr_pp.models.bbreg.atom import atom_resnet50, atom_resnet18
from ltr_pp.trainers import LTRTrainer


def run(settings):
    # Most common settings are assigned in the settings struct
    settings.description = 'ATOM IoUNet with ResNet18 backbone and trained with vid, lasot, coco.'
    settings.print_interval = 1  # How often to print loss and other info
    settings.batch_size = 64  # Batch size
    settings.num_workers = 4  # Number of workers for image loading
    settings.normalize_mean = [0.485, 0.456, 0.406]  # Normalize mean (default pytorch ImageNet values)
    settings.normalize_std = [0.229, 0.224, 0.225]  # Normalize std (default pytorch ImageNet values)
    settings.search_area_factor = 5.0  # Image patch size relative to target size
    settings.feature_sz = 18  # Size of feature map
    settings.output_sz = settings.feature_sz * 16  # Size of input image patches

    # Settings for the image sample and proposal generation
    settings.center_jitter_factor = {'train': 0, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0, 'test': 0.5}
    settings.proposal_params = {'min_iou': 0.1, 'boxes_per_frame': 16, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}

    # Train datasets
    vid_train = ImagenetVID()
    lasot_train = Lasot(split='train')
    coco_train = MSCOCOSeq()

    # Validation datasets
    got10k_val = Got10k(split='val')

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = dltransforms.ToGrayscale(probability=0.05)

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = dltransforms.Compose([dltransforms.ToArrayAndJitter(0.2),
                                            dltransforms.Normalize(mean=settings.normalize_mean,
                                                                   std=settings.normalize_std)])

    # The augmentation transform applied to the validation set (individually to each image in the pair)
    transform_val = dltransforms.Compose([dltransforms.ToArray(),
                                          dltransforms.Normalize(mean=settings.normalize_mean,
                                                                 std=settings.normalize_std)])

    # Data processing to do on the training pairs
    data_processing_train = processing.ATOMProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      proposal_params=settings.proposal_params,
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)

    # Data processing to do on the validation pairs
    data_processing_val = processing.ATOMProcessing(search_area_factor=settings.search_area_factor,
                                                    output_sz=settings.output_sz,
                                                    center_jitter_factor=settings.center_jitter_factor,
                                                    scale_jitter_factor=settings.scale_jitter_factor,
                                                    mode='sequence',
                                                    proposal_params=settings.proposal_params,
                                                    transform=transform_val,
                                                    joint_transform=transform_joint)

    # The sampler for training
    dataset_train = sampler.ATOMSampler([vid_train, lasot_train, coco_train], [1, 1, 1],
                                        samples_per_epoch=1000 * settings.batch_size, max_gap=50,
                                        processing=data_processing_train)

    # The loader for training
    train_loader = loader.LTRLoader('train', dataset_train,
                                    training=True,
                                    batch_size=settings.batch_size,
                                    num_workers=4,
                                    stack_dim=1)

    # The sampler for validation
    dataset_val = sampler.ATOMSampler([got10k_val], [1, ],
                                      samples_per_epoch=500 * settings.batch_size, max_gap=50,
                                      processing=data_processing_val)

    # The loader for validation
    val_loader = loader.LTRLoader('val', dataset_val,
                                  training=False,
                                  batch_size=settings.batch_size,
                                  epoch_interval=5,
                                  num_workers=4,
                                  stack_dim=1)

    # creat network, set objective, creat optimizer, learning rate scheduler, trainer
    with dygraph.guard():
        # Create network
        net = atom_resnet18(backbone_pretrained=True)

        # Freeze backbone
        state_dicts = net.state_dict()
        for k in state_dicts.keys():
            if 'feature_extractor' in k and "running" not in k:
                state_dicts[k].stop_gradient = True

        # Set objective
        objective = fluid.layers.square_error_cost

        # Create actor, which wraps network and objective
        actor = actors.AtomActor(net=net, objective=objective)

        # Set to training mode
        actor.train()

        # define optimizer and learning rate
        gama = 0.2
        lr = 1e-3
        lr_scheduler = fluid.dygraph.PiecewiseDecay([15, 30, 45],
                                                    values=[lr, lr * gama, lr * gama * gama],
                                                    step=1000,
                                                    begin=0)

        optimizer = fluid.optimizer.Adam(
            parameter_list=net.bb_regressor.parameters(),
            learning_rate=lr_scheduler)

        trainer = LTRTrainer(actor, [train_loader, val_loader], optimizer, settings, lr_scheduler)
        trainer.train(40, load_latest=False, fail_safe=False)
