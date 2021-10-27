"""
Usage: python generate_tfrecord_tt.py --project <project_name>

the version difference between the orignal script xxxx.py and the version xxxxx_tt.py
is that the '_tt' version processes automatically the train and test data assuming
the tree :

<project>
     |---images/
     |     |----train/
     |     |      |-----*.jpg
     |     |      |-----*.xml
     |     |----test/
     |     |      |-----*.jpg
     |     |      |-----*.xml
     |     |----train_labels.csv
     |     |----test_labels.cs
     |
     |---training/
     |     |---<pre-trained_net>/
     |     |           |
     |     |           
     |     |----train.record
     |     |----test.record
     |

The directory <project> is given by the option --project.

### Adapated from teh work of Gilbert Tanner :
### https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model
###
### JLC v1.0 2020/07/11 initial revision of the '_tt' version.

"""

import sys, os, io, argparse
import pandas as pd
import tensorflow.compat.v1 as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

if sys.version < '3.5':
    print("You need a version of Python >= 3.5 to run this programm, sorry.")
    sys.exit()

def check_tree(project):    
    ret = True
    for path in ('images', 'training'):
        target = os.path.join(project, path)
        if not os.path.isdir(target):
            print(f'missing directory {target}')
            ret = False
    return ret

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))

    for id, _ in enumerate(classes_text, start=1):
        classes.append(id)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):

    project = args.project
    if not check_tree(project): sys.exit()

    out_dir = os.path.join(project, 'training')
    img_dir = os.path.join(project, 'images')

    for folder in ('train', 'test'):
        csv_file = os.path.join(img_dir, folder+'_labels.csv')
        out_file = os.path.join(out_dir, folder+'.record')
        img_folder_dir = os.path.join(img_dir, folder)
        
        with tf.python_io.TFRecordWriter(out_file) as writer:
            examples = pd.read_csv(csv_file)
            grouped = split(examples, 'filename')
            for group in grouped:
                tf_example = create_tf_example(group, img_folder_dir)
                writer.write(tf_example.SerializeToString())

        print(f'Successfully created the TFRecord file: {out_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert CSV data to tfrecord format")
    parser.add_argument('-p', '--project', type=str, required=True,
                        help='name of project directory that contains the images/ and training/ subdirectories')
    args = parser.parse_args()

    tf.app.run()
