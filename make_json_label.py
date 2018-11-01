import cv2
import numpy as np
import os
from shutil import copyfile
import json
from webcolors import name_to_rgb
import shutil
import sys
import pickle
from random import randint
import difflib
from block_seg import block_segment
from shutil import copyfile
import random

from load_image_models import predictSI

dict_label_color = {'shipper': "blue", 'consignee': "green", 'notify': "red", 'also_notify': "magenta", 'por': "yellow", 'pol': "cyan",
                    'pod': "navy", 'del': "pink", 'DESCRIPTION': "purple", 'vvd_name': "gray", 'Gross Weight': "lavender", 'Measurement': "orange"}

dict_label_color_2 = {'SHIPPER': "blue", 'CONSIGNEE': "green", 'NOTIFY': "red", 'ALSO_NOTIFY': "magenta", 'POR': "yellow", 'POL': "cyan",
                      'POD': "navy", 'DEL': "pink", 'DESCRIPTION': "purple", 'VESSEL': "gray", 'Gross Weight': "lavender", 'Measurement': "orange"}


def draw_boxes_to_img(file_path, file_name, sub_folder, list_boxes_info):

    try:
        img = cv2.imread(file_path + file_name)
        img_kv = img.copy()
        h, w, _ = img.shape
        for item in list_boxes_info:
            (key_name, box_color, xmin_k, ymin_k, xmax_k, ymax_k,
             xmin_v, ymin_v, xmax_v, ymax_v) = item
            cv2.rectangle(img, (xmin_k, ymin_k), (xmax_k, ymax_k),
                          color=box_color, thickness=cv2.FILLED)
            cv2.rectangle(img, (xmin_v, ymin_v), (xmax_v, ymax_v),
                          color=box_color, thickness=cv2.FILLED)

        opacity = 0.5
        cv2.addWeighted(img, opacity, img_kv,
                        1 - opacity, 0, img_kv)

        # cv2.imwrite("test_images/" + sub_folder + "-" + file_name, img_kv)
    except:
        print("can not load image")


def recreate_folder(folder):
    if os.path.isdir(folder) == True:
        shutil.rmtree(folder)
    os.makedirs(folder)


def main(flags):

    num_test = flags.num_test
    list_keys = ['shipper', 'del', 'por', 'pod', 'also_notify',
                 'notify', 'pol', 'consignee', 'vvd_name']

    dict_keys_similar = {'shipper': 'SHIPPER', 'del': 'DEL', 'por': 'POR', 'pod': 'POD', 'also_notify': 'ALSO_NOTIFY',
                         'notify': 'NOTIFY', 'pol': 'POL', 'consignee': 'CONSIGNEE', 'vvd_name': 'VESSEL'}

    recreate_folder("test_images")
    list_files_json = os.listdir("export")
    json_list = []
    count_file = 0

    random.shuffle(list_files_json)
    for file_json in list_files_json:
        if file_json.endswith(".json") == False:
            continue

        filename_json = "export/" + file_json
        with open(filename_json, 'r') as f:
            datastore = json.load(f)

        sub_folder = file_json.split(".")[0]
        folder_name = "si_3000/" + sub_folder + "/"

        random.shuffle(datastore)
        for index, item in enumerate(datastore):
            try:
                count_file += 1
                file_name = item["img-path"].split("/")[-1]
                print(file_name + "(" + str(count_file) + ")")
                score_list = predictSI(folder_name + file_name)
                form_type = score_list[0][0]
                form_score = score_list[0][1]
                if form_type == "Other" or form_score < 0.8:
                    continue

                json_dict = {}
                json_dict['file_name'] = file_name
                json_dict['folder_name'] = folder_name

                # # check if it is a table form or not
                # img = cv2.imread(folder_name + file_name)
                # bboxs = block_segment(folder_name, file_name)
                # if len(bboxs) < 9:
                #     print(file_name + " is not table form")
                #     continue

                json_dict["keys"] = {}
                list_boxes_info = []
                for key_name in list_keys:
                    key_content = item[key_name]["key"]
                    value_text = item[key_name]["value"]
                    pos_key = item[key_name]["key-posn"]
                    pos_value = item[key_name]["value-posn"]
                    box_color = name_to_rgb(dict_label_color[key_name])
                    k = dict_keys_similar[key_name]
                    list_boxes_info.append(
                        (k, box_color, pos_key['l'], pos_key['t'], pos_key['r'], pos_key['b'],
                            pos_value['l'], pos_value['t'], pos_value['r'], pos_value['b']))

                    json_dict["keys"][k] = {}
                    json_dict["keys"][k]["key_pos"] = {}
                    json_dict["keys"][k]["key_pos"]["xmin"] = str(
                        pos_key['l'])
                    json_dict["keys"][k]["key_pos"]["ymin"] = str(
                        pos_key['t'])
                    json_dict["keys"][k]["key_pos"]["xmax"] = str(
                        pos_key['r'])
                    json_dict["keys"][k]["key_pos"]["ymax"] = str(
                        pos_key['b'])
                    json_dict["keys"][k]["key_content"] = key_content.lower()

                    json_dict["keys"][k]["value_pos"] = {}
                    json_dict["keys"][k]["value_pos"]["xmin"] = str(
                        pos_value['l'])
                    json_dict["keys"][k]["value_pos"]["ymin"] = str(
                        pos_value['t'])
                    json_dict["keys"][k]["value_pos"]["xmax"] = str(
                        pos_value['r'])
                    json_dict["keys"][k]["value_pos"]["ymax"] = str(
                        pos_value['b'])
                    json_dict["keys"][k]["value_content"] = value_text.lower()

                # append to list json
                json_list.append(json_dict)

                # print(list_boxes_info)
                draw_boxes_to_img(folder_name,
                                  file_name, sub_folder, list_boxes_info)

                num_test -= 1
                if num_test == 0:
                    with open('test_label.json', 'w') as outfile:
                        json.dump(json_list, outfile)
                    return

            except Exception as e:
                print("error at " + str(e))

    with open('test_label.json', 'w') as outfile:
        json.dump(json_list, outfile)


def read_flags():
    """Returns flags"""
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--num_test", default=100, type=int, help="Number of testing files")
    flags = parser.parse_args()
    return flags


if __name__ == '__main__':
    flags = read_flags()
    main(flags)
