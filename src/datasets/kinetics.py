import numpy as np
from PIL import Image
import os
import math
import functools
import json
import copy
from .spatial_transforms import GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor, GroupNormalize

def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #     return accimage_loader
    # else:
    #     return pil_loader
    return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '0000{:d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('test/{}'.format(key))
            elif subset == 'training':
                label = value['annotations']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])
            else:
                label = value['annotations']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    not_exist = 0
    #max_len=0
    #if subset=='validation':
    #    max_len=len(video_names)
    #else:
    #    max_len=100000
    for i in range(len(video_names)):
        if i % 1000 == 0 or i == len(video_names)-1:
            print("\rdataset loading [{}/{}]".format(i+1-not_exist, len(video_names)), end="", flush=True)

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            not_exist += 1
            continue
        n_frames_file_path = os.path.join(video_path, 'n_frames')        
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i][:-14].split('/')[1] if subset is 'training' else video_names[i].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class

class Kinetics:
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=32,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        #print(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            clip = self.spatial_transform(clip)
            
        #clip = np.stack(clip, 0).transpose(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)


class KineticsMultiCrop(Kinetics):
    def __init__(self,
                  root_path,
                  annotation_path,
                  subset,
                  n_samples_for_each_video=1,
                  spatial_transform=None,
                  temporal_transform=None,
                  target_transform=None,
                  sample_duration=32,
                  get_loader=get_default_video_loader):
        #self.data, self.class_names = make_dataset(
        #     root_path, annotation_path, subset, n_samples_for_each_video,
        #     sample_duration)
        super(KineticsMultiCrop, self).__init__(root_path,
                                            annotation_path,
                                            subset,
                                            n_samples_for_each_video,
                                            spatial_transform,
                                            temporal_transform,
                                            target_transform,
                                            sample_duration,
                                            get_loader)

    def __getitem__(self, index):
        """
        Args:
             index (int): Index
        Returns:
             tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clips = [self.loader(path, clip) for clip in frame_indices]
        frames=[]
        if self.spatial_transform is not None:
            for clip in clips:
                crops_frames=[]
                for _ in range(3):
                    spatial_img = self.spatial_transform(clip)           #32,3,256,256
                    #spatial_img = np.stack(spatial_img, 0).transpose(1, 0, 2, 3)  # 3,32,256,256
                    crops_frames.append(spatial_img)      #3,3,32,256,256
                crops_frames = np.stack(crops_frames,0)
                frames.append(crops_frames)
            frames = np.stack(frames,0) #10,3,3,32,256,256
        #print(frames.shape)
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
 
        return frames, target
 
    def __len__(self):
         return len(self.data)
    


if __name__ == '__main__':
    # import mindspore.dataset.vision.py_transforms as py_trans
    # from src.datasets.dataset import CornerCrop
    # from src.datasets.temporal_transforms import TemporalRandomCrop, LoopPadding
    # from src.datasets.target_transforms import ClassLabel
    # import mindspore.dataset as ds
    #
    # def spatial_transform(img):
    #     crop_method = CornerCrop(224)
    #     op1 = py_trans.Resize(224)
    #     op2 = py_trans.ToTensor()
    #     return op2(crop_method(op1(img)))
    # temporal_transform = LoopPadding(32)
    # target_transform = ClassLabel()
    #
    # data_generator = Kinetics(
    #     root_path='/root/shixinyu/Non-Local-mindspore/data/Kinetics-400/jpg',
    #     annotation_path='/root/shixinyu/Non-Local-mindspore/data/Kinetics-400/kinetics.json',
    #     subset='validation',
    #     n_samples_for_each_video=3,
    #     spatial_transform=spatial_transform,
    #     temporal_transform=temporal_transform,
    #     target_transform=target_transform,
    #     sample_duration=32)
    # dataset = ds.GeneratorDataset(data_generator, ["clip", "target"], shuffle=False,
    #                                       num_parallel_workers=4)
    # dataset = dataset.batch(16, drop_remainder=True)
    #
    # for item in dataset.create_dict_iterator():
    #     print(item['clip'], item['target'])
    #     break

    # data = load_annotation_data('/root/shixinyu/Non-Local-mindspore/data/Kinetics-400/kinetics.json')
    data = load_annotation_data("/home/Non-Local-mindspore/data/Kinetics-400/kinetics.json")
    # for key, value in data['database'].items():
    #     print(value['subset'])
    video_names, annotations = get_video_names_and_annotations(data, 'validation')
    not_exist = 0
    #for i in range(len(video_names)):
        #video_path = os.path.join('/opt/npu/data/Kinetics400_img/val/', video_names[i])
        #video_path = os.path.join('/opt/npu/data/kinetics-400_img', video_names[i].replace(' ', '_')+'.mp4')
        #if not os.path.exists(video_path):
        #    print(video_path)
        #    not_exist += 1
    #print(not_exist)
    #print(len(video_names))
    #data = make_dataset('/opt/npu/data/Kinetics400_img/val/',
    #              '/home/Non-Local-mindspore/data/Kinetics-400/kinetics.json', 'validation',
    #              3, 32)
    #print(len(data))
