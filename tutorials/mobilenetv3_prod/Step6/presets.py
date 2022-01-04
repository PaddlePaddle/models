import paddle

from paddlevision.transforms import autoaugment, transforms


class ClassificationPresetTrain:
    def __init__(self,
                 crop_size,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 hflip_prob=0.5,
                 auto_augment_policy=None,
                 random_erase_prob=0.0):
        trans = [transforms.RandomResizedCrop(crop_size)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean, std=std),
        ])

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(self,
                 crop_size,
                 resize_size=256,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        mean = tuple([m * 255 for m in mean])
        std = tuple([s * 255 for s in std])
        self.transforms = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            # fix to support pt-quant
            paddle.vision.transforms.Transpose((2, 0, 1)),
            paddle.vision.transforms.Normalize(
                mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transforms(img)
