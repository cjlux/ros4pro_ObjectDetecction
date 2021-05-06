# from https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model

"""
Usage: python xml_to_csv.py --project <project_name>

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

JLC v1.0 2020/07/11 initial revision of the '_tt' version.

"""

import os, glob, argparse
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main(project):
    for folder in ['train', 'test']:
        image_path = os.path.join(os.getcwd(), 'images' , project, folder)
        xml_df = xml_to_csv(image_path)
        csv_path = os.path.join('images', project, folder +'_labels.csv')
        xml_df.to_csv(csv_path, index=None)
        print('Successfully converted xml data in file <{}>.'.format(csv_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert XML to CSV")
    parser.add_argument('-p', '--project', type=str, required=True, help='name of the sub-directory under images/')
    args = parser.parse_args()
    main(args.project)
