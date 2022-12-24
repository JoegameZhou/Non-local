from .kinetics import Kinetics, KineticsMultiCrop
import mindspore.dataset as ds
from mindspore import dtype as mstype
from mindspore.dataset.transforms import c_transforms
import mindspore.dataset.vision.py_transforms as py_trans
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore.communication.management import get_rank, get_group_size
import random
from .temporal_transforms import TemporalRandomCrop, LoopPadding, TemporalMultiCrop, TemporalCenterCrop
from .target_transforms import ClassLabel
from .spatial_transforms import GroupCornerCrop, GroupRandomCrop, SpatialTransformer, Stack, ToTorchFormatTensor, GroupNormalize, GroupRandomResizeCrop, GroupRandomHorizontalFlip, GroupColorJitter, GroupScale, GroupCenterCrop

def get_training_set(opt):
    def spatial_transform(img_group):
        #op1 = py_trans.Resize(256)
        #crop = py_trans.RandomCrop(224)
        op1 = GroupRandomResizeCrop([256, 320], 224)
        op2 = GroupRandomHorizontalFlip(is_flow=False)
        op3 = GroupColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
        op4 = Stack()
        op5 = ToTorchFormatTensor()
        op6 = GroupNormalize(
            mean=[.485, .456, .406],
            std=[.229, .224, .225]
            )
        #op = SpatialTransformer([256,320], 224, 0.05, 0.05, 0.05, 0.05, mean=[.485, .456, .406], std=[.229, .224, .225])
        return op6(op5(op4(op3(op2(op1(img_group))))))
        #return op(img_group)
    temporal_transform = TemporalRandomCrop(opt.sample_duration)
    target_transform = ClassLabel()

    if opt.dataset == 'kinetics':
        data_generator = Kinetics(
            root_path=opt.video_path,
            annotation_path=opt.annotation_path,
            subset='training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration
        )

    if not opt.distributed:
        dataset = ds.GeneratorDataset(data_generator, ["clip", "target"], shuffle=True,
                                            num_parallel_workers=opt.n_threads)
        #dataset = dataset.map(operations=type_cast_op, input_columns="target")
        dataset = dataset.batch(opt.batch_size, drop_remainder=True)
    else:
        rank_id = get_rank()
        rank_size = get_group_size()
        dataset = ds.GeneratorDataset(data_generator, ["clip", "target"], shuffle=True,
                                      num_parallel_workers=opt.n_threads, num_shards=rank_size, shard_id=rank_id)
        dataset = dataset.batch(opt.batch_size, drop_remainder=True)

    return dataset


def get_val_set(opt):
    def spatial_transform(img_group):
        
        op1 = GroupScale(256)
        op2 = GroupCenterCrop(256)
        op3 = Stack()
        op4 = ToTorchFormatTensor()
        op5 = GroupNormalize(
            mean=[.485, .456, .406],
            std=[.229, .224, .225]
        )
        return op5(op4(op3(op2(op1(img_group)))))
    temporal_transform = TemporalCenterCrop(opt.sample_duration)
    target_transform = ClassLabel()

    if opt.dataset == 'kinetics':
        data_generator = Kinetics(
            root_path=opt.video_path,
            annotation_path=opt.annotation_path,
            subset='validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)

    dataset = ds.GeneratorDataset(data_generator, ["clip", "target"], shuffle=False,
                                          num_parallel_workers=opt.n_threads)
    dataset = dataset.batch(opt.batch_size, drop_remainder=True)

    return dataset


def get_test_set(opt):
    def spatial_transform(img_group):
        
        op1 = GroupScale(256)
        op2 = GroupRandomCrop(256)
        op3 = Stack()
        op4 = ToTorchFormatTensor()
        op5 = GroupNormalize(
            mean=[.485, .456, .406],
            std=[.229, .224, .225]
        )
        return op5(op4(op3(op2(op1(img_group)))))
    temporal_transform = TemporalMultiCrop(opt.sample_duration, 10)
    target_transform = ClassLabel()

    if opt.dataset == 'kinetics':
        data_generator = KineticsMultiCrop(
            root_path=opt.video_path,
            annotation_path=opt.annotation_path,
            subset='validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)

    dataset = ds.GeneratorDataset(data_generator, ["clip", "target"], shuffle=False,
                                          num_parallel_workers=opt.n_threads)
    dataset = dataset.batch(opt.batch_size, drop_remainder=True)

    return dataset

class CornerCrop(object):

    def __init__(self, size, crop_position=None):
        self.size = size
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']
        self.crop_position = crop_position
        if crop_position is None:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]
        else:
            self.randomize = False


    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            th, tw = (self.size, self.size)
            x1 = int(round((image_width - tw) / 2.))
            y1 = int(round((image_height - th) / 2.))
            x2 = x1 + tw
            y2 = y1 + th
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = self.size
            y2 = self.size
        elif self.crop_position == 'tr':
            x1 = image_width - self.size
            y1 = 0
            x2 = image_width
            y2 = self.size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - self.size
            x2 = self.size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - self.size
            y1 = image_height - self.size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img
