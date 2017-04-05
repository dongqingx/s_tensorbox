#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Prepare data for train & test.


import json
import sys

def prepare_train_data(json_file):
    output_stream = open(json_file, "w")
    json_images = []

    with open("./train0001.json", "r") as json_stream:
        json_data = json.load(json_stream)
        for i in range(198):
            img_name = json_data[197 - i]['image_path'].split('/')[-1]
            img_num = int(img_name.split('.')[0])
            image_path = "train_img/" + str(img_num).zfill(4) + ".png"
            p_image_path = "train_img/" + str(img_num - 1).zfill(4) + ".png"
            pp_image_path = "train_img/" + str(img_num - 2).zfill(4) + ".png"
            f_image_path = "train_img/" + str(img_num + 1).zfill(4) + ".png"
            rects = json_data[197 - i]['rects']
            json_image = dict([("image_path", image_path),
                               ("p_image_path", p_image_path),
                               ("pp_image_path", pp_image_path),
                               ("f_image_path", f_image_path),
                               ("rects", rects)])

            json_images.append(json_image)

    with open("./train0201.json", "r") as json_stream:
        json_data = json.load(json_stream)
        for i in range(200):
            img_name = json_data[199 - i]['image_path'].split('/')[-1]
            img_num = int(img_name.split('.')[0])
            image_path = "train_img/" + str(img_num).zfill(4) + ".png"
            p_image_path = "train_img/" + str(img_num - 1).zfill(4) + ".png"
            pp_image_path = "train_img/" + str(img_num - 2).zfill(4) + ".png"
            f_image_path = "train_img/" + str(img_num + 1).zfill(4) + ".png"
            rects = json_data[199 - i]['rects']
            json_image = dict([("image_path", image_path),
                               ("p_image_path", p_image_path),
                               ("pp_image_path", pp_image_path),
                               ("f_image_path", f_image_path),
                               ("rects", rects)])

            json_images.append(json_image)

    with open("./train0401.json", "r") as json_stream:
        json_data = json.load(json_stream)
        for i in range(200):
            img_name = json_data[199 - i]['image_path'].split('/')[-1]
            img_num = int(img_name.split('.')[0])
            image_path = "train_img/" + str(img_num).zfill(4) + ".png"
            p_image_path = "train_img/" + str(img_num - 1).zfill(4) + ".png"
            pp_image_path = "train_img/" + str(img_num - 2).zfill(4) + ".png"
            f_image_path = "train_img/" + str(img_num + 1).zfill(4) + ".png"
            rects = json_data[199 - i]['rects']
            json_image = dict([("image_path", image_path),
                               ("p_image_path", p_image_path),
                               ("pp_image_path", pp_image_path),
                               ("f_image_path", f_image_path),
                               ("rects", rects)])

            json_images.append(json_image)

    with open("./train0601.json", "r") as json_stream:
        json_data = json.load(json_stream)
        for i in range(200):
            img_name = json_data[199 - i]['image_path'].split('/')[-1]
            img_num = int(img_name.split('.')[0])
            image_path = "train_img/" + str(img_num).zfill(4) + ".png"
            p_image_path = "train_img/" + str(img_num - 1).zfill(4) + ".png"
            pp_image_path = "train_img/" + str(img_num - 2).zfill(4) + ".png"
            f_image_path = "train_img/" + str(img_num + 1).zfill(4) + ".png"
            rects = json_data[199 - i]['rects']
            json_image = dict([("image_path", image_path),
                               ("p_image_path", p_image_path),
                               ("pp_image_path", pp_image_path),
                               ("f_image_path", f_image_path),
                               ("rects", rects)])

            json_images.append(json_image)

    with open("./train0801.json", "r") as json_stream:
        json_data = json.load(json_stream)
        for i in range(200):
            img_name = json_data[199 - i]['image_path'].split('/')[-1]
            img_num = int(img_name.split('.')[0])
            image_path = "train_img/" + str(img_num).zfill(4) + ".png"
            p_image_path = "train_img/" + str(img_num - 1).zfill(4) + ".png"
            pp_image_path = "train_img/" + str(img_num - 2).zfill(4) + ".png"
            f_image_path = "train_img/" + str(img_num + 1).zfill(4) + ".png"
            rects = json_data[199 - i]['rects']
            json_image = dict([("image_path", image_path),
                               ("p_image_path", p_image_path),
                               ("pp_image_path", pp_image_path),
                               ("f_image_path", f_image_path),
                               ("rects", rects)])

            json_images.append(json_image)

    with open("./train1001.json", "r") as json_stream:
        json_data = json.load(json_stream)
        for i in range(200):
            img_name = json_data[199 - i]['image_path'].split('/')[-1]
            img_num = int(img_name.split('.')[0])
            image_path = "train_img/" + str(img_num).zfill(4) + ".png"
            p_image_path = "train_img/" + str(img_num - 1).zfill(4) + ".png"
            pp_image_path = "train_img/" + str(img_num - 2).zfill(4) + ".png"
            f_image_path = "train_img/" + str(img_num + 1).zfill(4) + ".png"
            rects = json_data[199 - i]['rects']
            json_image = dict([("image_path", image_path),
                               ("p_image_path", p_image_path),
                               ("pp_image_path", pp_image_path),
                               ("f_image_path", f_image_path),
                               ("rects", rects)])

            json_images.append(json_image)

    with open("./train1201.json", "r") as json_stream:
        json_data = json.load(json_stream)
        for i in range(200):
            img_name = json_data[199 - i]['image_path'].split('/')[-1]
            img_num = int(img_name.split('.')[0])
            image_path = "train_img/" + str(img_num).zfill(4) + ".png"
            p_image_path = "train_img/" + str(img_num - 1).zfill(4) + ".png"
            pp_image_path = "train_img/" + str(img_num - 2).zfill(4) + ".png"
            f_image_path = "train_img/" + str(img_num + 1).zfill(4) + ".png"
            rects = json_data[199 - i]['rects']
            json_image = dict([("image_path", image_path),
                               ("p_image_path", p_image_path),
                               ("pp_image_path", pp_image_path),
                               ("f_image_path", f_image_path),
                               ("rects", rects)])

            json_images.append(json_image)

    with open("./train1401.json", "r") as json_stream:
        json_data = json.load(json_stream)
        for i in range(200):
            img_name = json_data[199 - i]['image_path'].split('/')[-1]
            img_num = int(img_name.split('.')[0])
            image_path = "train_img/" + str(img_num).zfill(4) + ".png"
            p_image_path = "train_img/" + str(img_num - 1).zfill(4) + ".png"
            pp_image_path = "train_img/" + str(img_num - 2).zfill(4) + ".png"
            f_image_path = "train_img/" + str(img_num + 1).zfill(4) + ".png"
            rects = json_data[199 - i]['rects']
            json_image = dict([("image_path", image_path),
                               ("p_image_path", p_image_path),
                               ("pp_image_path", pp_image_path),
                               ("f_image_path", f_image_path),
                               ("rects", rects)])

            json_images.append(json_image)

    with open("./train1601.json", "r") as json_stream:
        json_data = json.load(json_stream)
        for i in range(200):
            img_name = json_data[199 - i]['image_path'].split('/')[-1]
            img_num = int(img_name.split('.')[0])
            image_path = "train_img/" + str(img_num).zfill(4) + ".png"
            p_image_path = "train_img/" + str(img_num - 1).zfill(4) + ".png"
            pp_image_path = "train_img/" + str(img_num - 2).zfill(4) + ".png"
            f_image_path = "train_img/" + str(img_num + 1).zfill(4) + ".png"
            rects = json_data[199 - i]['rects']
            json_image = dict([("image_path", image_path),
                               ("p_image_path", p_image_path),
                               ("pp_image_path", pp_image_path),
                               ("f_image_path", f_image_path),
                               ("rects", rects)])

            json_images.append(json_image)

    with open("./train1801.json", "r") as json_stream:
        json_data = json.load(json_stream)
        for i in range(200):
            img_name = json_data[199 - i]['image_path'].split('/')[-1]
            img_num = int(img_name.split('.')[0])
            image_path = "train_img/" + str(img_num).zfill(4) + ".png"
            p_image_path = "train_img/" + str(img_num - 1).zfill(4) + ".png"
            pp_image_path = "train_img/" + str(img_num - 2).zfill(4) + ".png"
            f_image_path = "train_img/" + str(img_num + 1).zfill(4) + ".png"
            rects = json_data[199 - i]['rects']
            json_image = dict([("image_path", image_path),
                               ("p_image_path", p_image_path),
                               ("pp_image_path", pp_image_path),
                               ("f_image_path", f_image_path),
                               ("rects", rects)])

            json_images.append(json_image)

    output_stream.write(json.dumps(json_images, indent = 2))

def prepare_test_data(json_file):
    output_stream = open(json_file, "w")
    json_images = []
    '''
    with open("./test0010.json", "r") as json_stream:
        json_data = json.load(json_stream)
        for i in range(50):
            img_name = json_data[49 - i]['image_path'].split('/')[-1]
            img_num = int(img_name.split('.')[0])
            image_path = "test_img/" + str(img_num).zfill(4) + ".png"
            p_image_path = "test_img/" + str(img_num - 1).zfill(4) + ".png"
            pp_image_path = "test_img/" + str(img_num - 2).zfill(4) + ".png"
            f_image_path = "test_img/" + str(img_num + 1).zfill(4) + ".png"
            rects = json_data[49 - i]['rects']
            json_image = dict([("image_path", image_path),
                               ("p_image_path", p_image_path),
                               ("pp_image_path", pp_image_path),
                               ("f_image_path", f_image_path),
                               ("rects", rects)])

            json_images.append(json_image)

    with open("./test0510.json", "r") as json_stream:
        json_data = json.load(json_stream)
        for i in range(50):
            img_name = json_data[49 - i]['image_path'].split('/')[-1]
            img_num = int(img_name.split('.')[0])
            image_path = "test_img/" + str(img_num).zfill(4) + ".png"
            p_image_path = "test_img/" + str(img_num - 1).zfill(4) + ".png"
            pp_image_path = "test_img/" + str(img_num - 2).zfill(4) + ".png"
            f_image_path = "test_img/" + str(img_num + 1).zfill(4) + ".png"
            rects = json_data[49 - i]['rects']
            json_image = dict([("image_path", image_path),
                               ("p_image_path", p_image_path),
                               ("pp_image_path", pp_image_path),
                               ("f_image_path", f_image_path),
                               ("rects", rects)])

            json_images.append(json_image)

    with open("./test1010.json", "r") as json_stream:
        json_data = json.load(json_stream)
        for i in range(50):
            img_name = json_data[49 - i]['image_path'].split('/')[-1]
            img_num = int(img_name.split('.')[0])
            image_path = "test_img/" + str(img_num).zfill(4) + ".png"
            p_image_path = "test_img/" + str(img_num - 1).zfill(4) + ".png"
            pp_image_path = "test_img/" + str(img_num - 2).zfill(4) + ".png"
            f_image_path = "test_img/" + str(img_num + 1).zfill(4) + ".png"
            rects = json_data[49 - i]['rects']
            json_image = dict([("image_path", image_path),
                               ("p_image_path", p_image_path),
                               ("pp_image_path", pp_image_path),
                               ("f_image_path", f_image_path),
                               ("rects", rects)])

            json_images.append(json_image)

    with open("./test1510.json", "r") as json_stream:
        json_data = json.load(json_stream)
        for i in range(50):
            img_name = json_data[49 - i]['image_path'].split('/')[-1]
            img_num = int(img_name.split('.')[0])
            image_path = "test_img/" + str(img_num).zfill(4) + ".png"
            p_image_path = "test_img/" + str(img_num - 1).zfill(4) + ".png"
            pp_image_path = "test_img/" + str(img_num - 2).zfill(4) + ".png"
            f_image_path = "test_img/" + str(img_num + 1).zfill(4) + ".png"
            rects = json_data[49 - i]['rects']
            json_image = dict([("image_path", image_path),
                               ("p_image_path", p_image_path),
                               ("pp_image_path", pp_image_path),
                               ("f_image_path", f_image_path),
                               ("rects", rects)])

            json_images.append(json_image)

    with open("./test2010.json", "r") as json_stream:
        json_data = json.load(json_stream)
        for i in range(50):
            img_name = json_data[49 - i]['image_path'].split('/')[-1]
            img_num = int(img_name.split('.')[0])
            image_path = "test_img/" + str(img_num).zfill(4) + ".png"
            p_image_path = "test_img/" + str(img_num - 1).zfill(4) + ".png"
            pp_image_path = "test_img/" + str(img_num - 2).zfill(4) + ".png"
            f_image_path = "test_img/" + str(img_num + 1).zfill(4) + ".png"
            rects = json_data[49 - i]['rects']
            json_image = dict([("image_path", image_path),
                               ("p_image_path", p_image_path),
                               ("pp_image_path", pp_image_path),
                               ("f_image_path", f_image_path),
                               ("rects", rects)])

            json_images.append(json_image)

    with open("./test2510.json", "r") as json_stream:
        json_data = json.load(json_stream)
        for i in range(50):
            img_name = json_data[49 - i]['image_path'].split('/')[-1]
            img_num = int(img_name.split('.')[0])
            image_path = "test_img/" + str(img_num).zfill(4) + ".png"
            p_image_path = "test_img/" + str(img_num - 1).zfill(4) + ".png"
            pp_image_path = "test_img/" + str(img_num - 2).zfill(4) + ".png"
            f_image_path = "test_img/" + str(img_num + 1).zfill(4) + ".png"
            rects = json_data[49 - i]['rects']
            json_image = dict([("image_path", image_path),
                               ("p_image_path", p_image_path),
                               ("pp_image_path", pp_image_path),
                               ("f_image_path", f_image_path),
                               ("rects", rects)])

            json_images.append(json_image)
    '''
    with open("./test3010.json", "r") as json_stream:
        json_data = json.load(json_stream)
        for i in range(50):
            img_name = json_data[49 - i]['image_path'].split('/')[-1]
            img_num = int(img_name.split('.')[0])
            image_path = "test_img/" + str(img_num).zfill(4) + ".png"
            p_image_path = "test_img/" + str(img_num - 1).zfill(4) + ".png"
            pp_image_path = "test_img/" + str(img_num - 2).zfill(4) + ".png"
            f_image_path = "test_img/" + str(img_num + 1).zfill(4) + ".png"
            rects = json_data[49 - i]['rects']
            json_image = dict([("image_path", image_path),
                               ("p_image_path", p_image_path),
                               ("pp_image_path", pp_image_path),
                               ("f_image_path", f_image_path),
                               ("rects", rects)])

            json_images.append(json_image)

    output_stream.write(json.dumps(json_images, indent = 2))


if __name__ == "__main__":
    prepare_test_data("./val_boxes.json")
