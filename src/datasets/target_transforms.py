import mindspore
from mindspore import Tensor
from mindspore import dtype as mstype

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target):
        dst = []
        for t in self.transforms:
            dst.append(t(target))
        return dst


class ClassLabel(object):

    def __call__(self, target):
        #cast=mindspore.ops.Cast
        #return cast(target['label'],mindspore.float32)
        return target['label']

class VideoID(object):

    def __call__(self, target):
        return target['video_id']
