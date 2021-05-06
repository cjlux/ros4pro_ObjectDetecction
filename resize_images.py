# from https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model

from PIL import Image
import os
import argparse

def rescale_images(directory, size):
    print("resizing <{}> to {} :".format(os.path.join(directory,"*.jpg"),size)) 
    list_img = [img for img in os.listdir(directory) if img.lower().endswith(".jpg")]
    list_img.sort()
    for img in list_img:
        img_path = os.path.join(directory, img)
        print("\t{}".format(img))
        im = Image.open(img_path)
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save(directory+img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rescale images")
    parser.add_argument('-d', '--directory', type=str, required=True, help='Directory containing the images')
    parser.add_argument('-s', '--size', type=int, nargs=2, required=True, metavar=('width', 'height'), help='Image size')
    args = parser.parse_args()
    rescale_images(args.directory, args.size)
