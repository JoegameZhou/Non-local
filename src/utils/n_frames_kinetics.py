from __future__ import print_function, division
import os
import sys
import subprocess


def class_process(dir_path, class_name):
  class_path = os.path.join(dir_path, class_name)
  if not os.path.isdir(class_path):
    return

  for file_name in os.listdir(class_path):
    video_dir_path = os.path.join(class_path, file_name)
    image_indices = []
    if not os.path.isdir(video_dir_path):
      continue
    for image_file_name in os.listdir(video_dir_path):
        indice = image_file_name.find('jpg')
        if image_file_name[0:indice-1].isdigit():
            image_indices.append(int(image_file_name[0:indice-1]))

    if len(image_indices) == 0:
      print('no image files', video_dir_path)
      n_frames = 0
    else:
      image_indices.sort(reverse=True)
      n_frames = image_indices[0]
      print(video_dir_path, n_frames)
    with open(os.path.join(video_dir_path, 'n_frames'), 'w') as dst_file:
      dst_file.write(str(n_frames))


if __name__=="__main__":
  dir_path = sys.argv[1]
  for class_name in os.listdir(dir_path):
    class_process(dir_path, class_name)

  # class_name = 'test'
  # class_process(dir_path, class_name)
