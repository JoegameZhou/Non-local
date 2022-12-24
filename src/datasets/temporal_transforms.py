import random
import numpy as np


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[:self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - self.size)
        #begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size*2, len(frame_indices))

        out = frame_indices[begin_index:end_index:2]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        #rand_end = max(0, len(frame_indices) - self.size - 1)
        rand_end = max(0, len(frame_indices) - self.size*2 + 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size*2, len(frame_indices))

        out = frame_indices[begin_index:end_index:2]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalMultiCrop(object):
    
    def __init__(self, size, K):
        self.size = size
        self.K = K
    def __call__(self, frame_indices):
        centers = [int(idx) for idx in np.linspace(self.size, len(frame_indices)-self.size, self.K)]
        clips = []
        for c in centers:
            begin = max(0, c-self.size)
            end = min(c+self.size, len(frame_indices))
            clip = frame_indices[begin:end:2]
            for index in clip:
                if len(clip) >= self.size:
                    break
                clip.append(index)
            clips.append(clip)
        return clips

