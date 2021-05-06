# from https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model

"""
Usage: python resize_image_tt.py --project <project_name>

the main difference between the orignal script xxxx.py and xxxxx_tt.py
is that the '_tt' version processes automatically the train and test
folders assuming the tree :

images/
   |---<project_name>/
            |----train/
            |      |-----*.jpg
            |      |-----*.xml
            |----test/
            |      |-----*.jpg
            |      |-----*.xml
            |----train_labels.csv
            |----test_labels.csv

The directory <project_name> is given by the option --project.

JLC v1.0 2020/07/19 initial revision of the '_tt' version.

"""

import os, argparse
from resize_images import rescale_images

def main(project, size):
    for folder in ['train', 'test']:
        image_path = os.path.join('images', project, folder)
        rescale_images(image_path, size)        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rescale images")
    parser.add_argument('-p', '--project', type=str, required=True, help='name of the directory under images/ where to find "train" and "test" images sub-directories')
    parser.add_argument('-s', '--size', type=int, nargs=2, required=True, metavar=('width', 'height'), help='Image size')
    args = parser.parse_args()
    main(args.project, args.size)
