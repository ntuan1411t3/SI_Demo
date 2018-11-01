import cv2
import numpy as np
import os
from pytesseract import image_to_string
from classifier import keyword_detection, sentence_classifier
import shutil
from block_seg import block_segment
import math
from webcolors import name_to_rgb
import random
import time
import json


class Box_Info:
    def __init__(self, xmin, ymin, xmax, ymax, flag_key, textbox_content, percent_k, textbox_key, stroke_width):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.flag_key = flag_key
        self.textbox_content = textbox_content
        self.percent_k = percent_k
        self.textbox_key = textbox_key
        self.stroke_width = stroke_width


########################################
dict_label_color = {'SHIPPER': "blue", 'CONSIGNEE': "green", 'NOTIFY': "red", 'ALSO_NOTIFY': "magenta", 'POR': "yellow", 'POL': "cyan",
                    'POD': "navy", 'DEL': "pink", 'DESCRIPTION': "purple", 'VESSEL': "gray", 'Gross Weight': "lavender", 'Measurement': "orange"}

key_cluster_1 = ['SHIPPER', 'CONSIGNEE', 'NOTIFY', 'ALSO_NOTIFY']
key_cluster_2 = ['POL', 'POD', 'DEL', 'VESSEL', 'POR']


def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist


def crop_contours(img, cnt):

    x0, y0, w0, h0 = cv2.boundingRect(cnt)
    th1 = img[y0:y0+h0, x0:x0+w0]

    _, contours, _ = cv2.findContours(
        th1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    list_x = []
    list_y = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        list_x.append(x)
        list_x.append(x+w)
        list_y.append(y)
        list_y.append(y+h)

    x1 = min(list_x)
    y1 = min(list_y)
    x2 = max(list_x)
    y2 = max(list_y)
    return x0+x1, y0+y1, x2-x1, y2-y1


def take_character_boxes(image):
    # find contours
    _, contours, _ = cv2.findContours(
        image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    height, width = image.shape
    output = np.zeros((height, width), np.uint8)

    # loop in all the contour areas
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w < 200) and (h < 70):
            output[y:y+h, x:x+w] = image[y:y+h, x:x+w]

    return output, contours


def ocr_textbox(textbox_img):

    result = None
    ocr_result = ""
    try:
        ocr_result = image_to_string(
            textbox_img, config='-l eng --tessdata-dir "tessdata" --psm 13').lower()
    except Exception as e:
        print("ocr error: " + str(e))

    # flag key or not
    flag_key = False
    textbox_content = ocr_result
    percent_k = 0
    # detect key or not
    try:
        keyword = keyword_detection(ocr_result)
        label = keyword[0]
        percent = keyword[1]
        textbox_key = ocr_result
        if label != None:
            flag_key = True
            textbox_content = label
            percent_k = percent

        result = (flag_key, textbox_content, percent_k, textbox_key)

    except Exception as e:
        print("can not detect key: " + str(e))

    return result


def get_block_img_info(index_block, image):

    height, width, _ = image.shape

    output_img = image.copy()
    ocr_img = image.copy()
    image = binary_img(image)

    # find all text boxes
    thresh, _ = take_character_boxes(image)
    kernel2 = np.ones((1, 60), np.uint8)
    line_img = cv2.dilate(thresh, kernel2, iterations=1)
    _, contours, _ = cv2.findContours(
        line_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color_red = (0, 0, 255)
    # for each textbox

    count_key = 0
    list_boxes_info = []
    for index, cnt in enumerate(contours):
        # try crop text region
        x, y, w, h = crop_contours(thresh, cnt)
        if w < 10 or h < 10 or h > 100:
            continue
        cv2.rectangle(output_img, (x, y), (x + w, y + h),
                      color=color_red, thickness=2)

        # ocr text box
        xmin_ocr = x - 3
        if xmin_ocr < 0:
            xmin_ocr = 0

        ymin_ocr = y - 3
        if ymin_ocr < 0:
            ymin_ocr = 0

        xmax_ocr = x + w + 3
        if xmax_ocr > width:
            xmax_ocr = width - 1

        ymax_ocr = y + h + 3
        if ymax_ocr > height:
            ymax_ocr = height - 1

        textbox_img = ocr_img[ymin_ocr:ymax_ocr, xmin_ocr:xmax_ocr]
        h_ocr, w_ocr, _ = textbox_img.shape
        if h_ocr < 5 or w_ocr < 50:
            continue

        result_ocr = ocr_textbox(textbox_img)
        if result_ocr == None:
            continue
        (flag_key, textbox_content, percent_k, textbox_key) = result_ocr
        # print(str(index_block) + " : '" + textbox_content + "' : " +
        #       str(flag_key) + " : " + str(percent_k))

        # calculate stroke width
        stroke_width = get_stroke_width(textbox_img)
        box_info = Box_Info(x, y, x + w, y + h, flag_key,
                            textbox_content, percent_k, textbox_key, stroke_width)

        list_boxes_info.append(box_info)

        # for visual key
        if flag_key == True:
            count_key += 1
            cv2.putText(output_img, str(textbox_content), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color_red, thickness=2)

    return output_img, count_key, list_boxes_info


def draw_key_value_image(dict_key_value, folder_name, output_folder, file_name):

    img_result = cv2.imread(folder_name + file_name)
    img_result_copy = img_result.copy()

    for k, v in dict_key_value.items():
        for item in v:
            list_values = item[1]
            color_value = name_to_rgb(dict_label_color[k])
            for value in list_values:
                cv2.rectangle(
                    img_result, (value[0], value[1]), (value[2], value[3]), color=color_value, thickness=cv2.FILLED)

            cv2.putText(img_result, str(k) + " : " + str(item[0]), (item[2][0], item[2][1]), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=color_value, thickness=2)

    opacity = 0.5
    cv2.addWeighted(img_result, opacity, img_result_copy,
                    1 - opacity, 0, img_result_copy)
    # cv2.imwrite(output_folder + "result_" + file_name, img_result_copy)


def make_json_dict(file_name, folder_name, dict_key_value):

    json_dict = {}
    json_dict['file_name'] = file_name
    json_dict['folder_name'] = folder_name
    json_dict["keys"] = {}

    list_keys = ['SHIPPER', 'CONSIGNEE', 'NOTIFY',
                 'ALSO_NOTIFY', 'POL', 'POD', 'DEL', 'VESSEL', 'POR']
    # init dict
    for k in list_keys:
        json_dict["keys"][k] = {}
        json_dict["keys"][k]["key_pos"] = {}
        json_dict["keys"][k]["key_pos"]["xmin"] = str(0)
        json_dict["keys"][k]["key_pos"]["ymin"] = str(0)
        json_dict["keys"][k]["key_pos"]["xmax"] = str(0)
        json_dict["keys"][k]["key_pos"]["ymax"] = str(0)
        json_dict["keys"][k]["key_content"] = ""

        json_dict["keys"][k]["value_pos"] = {}
        json_dict["keys"][k]["value_pos"]["xmin"] = str(0)
        json_dict["keys"][k]["value_pos"]["ymin"] = str(0)
        json_dict["keys"][k]["value_pos"]["xmax"] = str(0)
        json_dict["keys"][k]["value_pos"]["ymax"] = str(0)
        json_dict["keys"][k]["value_content"] = ""

    for k, v in dict_key_value.items():

        if k not in list_keys:
            continue

        if len(v) > 0:
            (percent_k, list_values, key_pos, key_content) = v[0]
            list_xmin = []
            list_ymin = []
            list_xmax = []
            list_ymax = []
            value_text = []
            for value_info in list_values:
                (xmin, ymin, xmax, ymax, value_content) = value_info
                list_xmin.append(xmin)
                list_ymin.append(ymin)
                list_xmax.append(xmax)
                list_ymax.append(ymax)
                value_text.append(value_content)

            if len(list_values) > 0:
                xmin = min(list_xmin)
                ymin = min(list_ymin)
                xmax = max(list_xmax)
                ymax = max(list_ymax)
            else:
                xmin = 0
                ymin = 0
                xmax = 0
                ymax = 0

            (xmin_k, ymin_k, xmax_k, ymax_k) = key_pos
            json_dict["keys"][k] = {}
            json_dict["keys"][k]["key_pos"] = {}
            json_dict["keys"][k]["key_pos"]["xmin"] = str(xmin_k)
            json_dict["keys"][k]["key_pos"]["ymin"] = str(ymin_k)
            json_dict["keys"][k]["key_pos"]["xmax"] = str(xmax_k)
            json_dict["keys"][k]["key_pos"]["ymax"] = str(ymax_k)
            json_dict["keys"][k]["key_content"] = key_content

            json_dict["keys"][k]["value_pos"] = {}
            json_dict["keys"][k]["value_pos"]["xmin"] = str(xmin)
            json_dict["keys"][k]["value_pos"]["ymin"] = str(ymin)
            json_dict["keys"][k]["value_pos"]["xmax"] = str(xmax)
            json_dict["keys"][k]["value_pos"]["ymax"] = str(ymax)

            if len(value_text) == 0:
                value_text = ""
            else:
                value_text.reverse()
                value_text = '\n'.join(value_text)

            json_dict["keys"][k]["value_content"] = value_text

    return json_dict


def read_json_data(filename_json):
    with open(filename_json, 'r') as f:
        datastore = json.load(f)

    list_test = []

    for index, json_dict in enumerate(datastore):
        file_name = json_dict['file_name']
        folder_name = json_dict['folder_name']
        list_test.append((folder_name, file_name))

    return list_test


def get_values_free_key(key_info, list_boxes_info, xmin_block, ymin_block):

    list_k_value = []
    list_candidate = []

    for box_info in list_boxes_info:
        if box_info.ymax > key_info.ymax and box_info.xmax > key_info.xmin and box_info.xmin < key_info.xmax:
            list_candidate.append(box_info)

    sorted_list_candidate = sorted(
        list_candidate, key=lambda x: x.ymax)

    ymax_k_new = key_info.ymax
    y_range_under = 0
    for box_info in sorted_list_candidate:
        if box_info.flag_key == True or box_info.stroke_width == key_info.stroke_width or box_info.ymin - ymax_k_new > 100:
            y_range_under = box_info.ymin
            break

        k_value_box = (box_info.xmin + xmin_block, box_info.ymin + ymin_block, box_info.xmax + xmin_block,
                       box_info.ymax + ymin_block, box_info.textbox_content)
        list_k_value.append(k_value_box)
        ymax_k_new = box_info.ymax

    if len(list_k_value) > 0:
        return list_k_value

    # find value on the left
    list_candidate = []
    for box_info in list_boxes_info:
        if box_info.xmax > key_info.xmax and np.absolute(box_info.ymin - key_info.ymin) < 10:
            list_candidate.append(box_info)

    sorted_list_candidate = sorted(
        list_candidate, key=lambda x: x.xmax)

    if len(sorted_list_candidate) > 0:

        box_info = sorted_list_candidate[0]
        if box_info.flag_key == False and box_info.stroke_width != key_info.stroke_width and box_info.xmin - key_info.xmax <= 300:
            k_value_box = (box_info.xmin + xmin_block, box_info.ymin + ymin_block, box_info.xmax + xmin_block,
                           box_info.ymax + ymin_block, box_info.textbox_content)
            list_k_value.append(k_value_box)
            box_info_new = Box_Info(box_info.xmin, box_info.ymin, box_info.xmax, box_info.ymax, box_info.flag_key,
                                    box_info.textbox_content, box_info.percent_k, box_info.textbox_key, key_info.stroke_width)

            # continue go to underlines
            list_k_value_under = get_values_under_first_value(
                box_info_new, list_boxes_info, y_range_under, xmin_block, ymin_block)
            list_k_value += list_k_value_under

    return list_k_value


def get_values_under_first_value(key_info, list_boxes_info, y_range_under, xmin_block, ymin_block):

    list_k_value = []
    list_candidate = []

    for box_info in list_boxes_info:
        if box_info.ymax > key_info.ymax and box_info.xmax > key_info.xmin and box_info.xmin < key_info.xmax and box_info.ymax < y_range_under:
            list_candidate.append(box_info)

    sorted_list_candidate = sorted(
        list_candidate, key=lambda x: x.ymax)

    ymax_k_new = key_info.ymax
    for box_info in sorted_list_candidate:
        if box_info.flag_key == True or box_info.stroke_width == key_info.stroke_width or box_info.ymin - ymax_k_new > 100:
            break

        k_value_box = (box_info.xmin + xmin_block, box_info.ymin + ymin_block,
                       box_info.xmax + xmin_block, box_info.ymax + ymin_block, box_info.textbox_content)
        list_k_value.append(k_value_box)
        ymax_k_new = ymax

    return list_k_value


def get_stroke_width(img_crop):
    # gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold = 128 / 255
    # gray = gray / 255
    # tmp = gray[gray < threshold].flatten()
    # return round(tmp.sum() / len(tmp), 2)

    img_crop = binary_img(img_crop)
    stroke_width = 0
    kernel = np.ones((3, 3), np.uint8)
    num_white_origin = np.sum(img_crop == 255)

    while np.sum(img_crop == 255) >= 0.01 * num_white_origin:
        img_crop = cv2.erode(img_crop, kernel, iterations=1)
        stroke_width += 1
        if stroke_width == 10:
            return stroke_width
    return stroke_width


def key_value_detection(folder_name, file_name):
    img = cv2.imread(folder_name + file_name)
    bboxs = block_segment(folder_name, file_name)
    # check if it is not a table form
    if table_from_detection(len(bboxs)) == False:
        return None

    h, w, _ = img.shape
    list_block_info = []
    for index, bb in enumerate(bboxs):

        # ignore some boxes
        if bb[1] > int(3*h/5):
            continue

        img_box = img[bb[1]:bb[3], bb[0]:bb[2]]
        output_img, count_key, list_boxes_info = get_block_img_info(
            index, img_box)
        block_info = ((bb[0], bb[1], bb[2], bb[3]), list_boxes_info)
        list_block_info.append(block_info)
        # cv2.imwrite(output_folder + file_name.split(".")
        #             [0] + "_" + str(index) + "_" + str(count_key) + ".png", output_img)

    dict_key_value = find_key_value_block(list_block_info)
    return dict_key_value


def find_key_value_block(list_block_info):

    dict_key_value = {'SHIPPER': [], 'CONSIGNEE': [], 'NOTIFY': [],
                      'ALSO_NOTIFY': [], 'POR': [], 'POL': [],
                      'POD': [], 'DEL': [], 'DESCRIPTION': [],
                      'VESSEL': [], 'Gross Weight': [], 'Measurement': []}

    for index_block, block_info in enumerate(list_block_info):
        list_result_kv = get_kv_from_block(
            index_block, block_info, list_block_info)
        for kv in list_result_kv:
            (key_name, percent_key, list_k_value, key_pos, key_content) = kv
            dict_key_value[key_name].append(
                (percent_key, list_k_value, key_pos, key_content))

    dict_key_value_select(dict_key_value)
    return dict_key_value


def dict_key_value_select(dict_key_value):

    # if key is higher than value, it is ok
    for k, v in dict_key_value.items():
        list_new_v = []
        for item in v:
            if check_key_position(item) == True:
                list_new_v.append(item)
        dict_key_value[k] = list_new_v

    # if have muliple key, only the highest key is selected
    for k, v in dict_key_value.items():
        if len(v) > 1:
            sorted_v = sorted(v, key=lambda tup: (tup[2][1], tup[2][0]))
            highest_v = sorted_v[0]
            list_new_v = []
            list_new_v.append(highest_v)
            dict_key_value[k] = list_new_v

    return dict_key_value


def check_key_position(item):
    ymin_k = item[2][1]
    list_values = item[1]
    for value in list_values:
        if value[1] < ymin_k:
            return False
    return True


def get_kv_from_block(index_block, block_info, list_block_info):
    num_key, num_value = count_item_value(block_info)
    list_key_value = []
    (xmin_block, ymin_block, xmax_block, ymax_block) = block_info[0]
    list_boxes_info = block_info[1]

    if num_key == 1 and num_value > 0:
        key_name = ""
        key_content = ""
        list_k_value = []
        percent_key = 0

        for box_info in list_boxes_info:
            if box_info.flag_key == True:
                key_name = box_info.textbox_content
                key_content = box_info.textbox_key
                percent_key = box_info.percent_k
                key_pos = (xmin_block + box_info.xmin, ymin_block + box_info.ymin,
                           xmin_block + box_info.xmax, ymin_block + box_info.ymax)
            else:
                k_value_box = (xmin_block + box_info.xmin, ymin_block + box_info.ymin,
                               xmin_block + box_info.xmax, ymin_block + box_info.ymax, box_info.textbox_content)
                list_k_value.append(k_value_box)

        list_key_value.append(
            (key_name, percent_key, list_k_value, key_pos, key_content))
        return list_key_value

    if num_key > 1:
        for box_info in list_boxes_info:
            if box_info.flag_key == True:
                key_name = box_info.textbox_content
                key_content = box_info.textbox_key
                percent_key = box_info.percent_k
                key_pos = (xmin_block + box_info.xmin, ymin_block + box_info.ymin,
                           xmin_block + box_info.xmax, ymin_block + box_info.ymax)
                # list_k_value = find_value_by_key(
                #     box_info, list_boxes_info, xmin_block, ymin_block)
                list_k_value = get_values_free_key(
                    box_info, list_boxes_info, xmin_block, ymin_block)

                list_key_value.append(
                    (key_name, percent_key, list_k_value, key_pos, key_content))

        return list_key_value

    if num_key == 1 and num_value == 0:

        if is_key_no_value(block_info) == True:
            return list_key_value

        under_block, left_block = get_nearest_under_left_block(
            block_info, list_block_info)
        percent_key = 0
        key_name = ""
        key_content = ""
        box_info = list_boxes_info[0]
        if box_info.flag_key == True:
            key_name = box_info.textbox_content
            key_content = box_info.textbox_key
            percent_key = box_info.percent_k
            key_pos = (xmin_block + box_info.xmin, ymin_block + box_info.ymin,
                       xmin_block + box_info.xmax, ymin_block + box_info.ymax)

        list_k_value_u = []
        list_k_value_l = []
        if under_block != None:
            num_key_u, num_value_u = count_item_value(under_block)
            if num_key_u == 0 and num_value_u > 0:
                list_k_value_u = get_all_values_in_block(under_block)

        if left_block != None:
            num_key_l, num_value_l = count_item_value(left_block)
            if num_key_l == 0 and num_value_l > 0:
                list_k_value_l = get_all_values_in_block(left_block)

        list_k_value = []

        # check if key of cluster 1 or not ?
        len_u = len(list_k_value_u)
        len_l = len(list_k_value_l)
        if key_name in key_cluster_1:
            if len_u >= len_l:
                list_k_value = list_k_value_u
            else:
                list_k_value = list_k_value_l
        else:
            if len_u == 1 and check_colon(list_k_value_u) == False:
                list_k_value += list_k_value_u
            if len_l == 1 and check_colon(list_k_value_l) == False:
                list_k_value += list_k_value_l

        list_key_value.append(
            (key_name, percent_key, list_k_value, key_pos, key_content))
        return list_key_value

    return list_key_value


def is_key_no_value(block_info):
    (xmin_block, ymin_block, xmax_block, ymax_block) = block_info[0]
    list_boxes_info = block_info[1]
    for box_info in list_boxes_info:
        if (box_info.ymax - box_info.ymin) < 0.5 * (ymax_block - ymin_block):
            return True

    return False


def check_colon(list_k_value):
    for item in list_k_value:
        content = item[4]
        if content.endswith(':'):
            return True
    return False


def get_all_values_in_block(block_info):
    (xmin_block, ymin_block, xmax_block, ymax_block) = block_info[0]
    list_boxes_info = block_info[1]

    list_k_value = []
    for box_info in list_boxes_info:
        k_value_box = (xmin_block + box_info.xmin, ymin_block + box_info.ymin,
                       xmin_block + box_info.xmax, ymin_block + box_info.ymax, box_info.textbox_content)
        list_k_value.append(k_value_box)

    return list_k_value


def get_nearest_under_left_block(block_info, list_block_info):
    under_block = None
    left_block = None
    (xmin_block, ymin_block, xmax_block, ymax_block) = block_info[0]

    # find nearest under box
    list_candidate = []
    for block in list_block_info:
        (xmin_b, ymin_b, xmax_b, ymax_b) = block[0]
        if ymax_b > ymax_block and np.absolute(xmin_block - xmin_b) < 10:
            list_candidate.append(block)

    sorted_list_candidate = sorted(
        list_candidate, key=lambda tup: tup[0][3])

    if len(sorted_list_candidate) > 0:
        under_block = sorted_list_candidate[0]

    # find nearest left box
    list_candidate = []
    for block in list_block_info:
        (xmin_b, ymin_b, xmax_b, ymax_b) = block[0]
        if xmax_b > xmax_block and np.absolute(ymin_block - ymin_b) < 10:
            list_candidate.append(block)

    sorted_list_candidate = sorted(
        list_candidate, key=lambda tup: tup[0][2])

    if len(sorted_list_candidate) > 0:
        left_block = sorted_list_candidate[0]

    return under_block, left_block


def find_value_by_key(box_info_key, list_boxes_info, xmin_block, ymin_block):
    list_k_value = []
    list_candidate = []
    for box_info in list_boxes_info:
        # find a box below key
        if box_info.ymin > box_info_key.ymax and (box_info_key.xmin < box_info.xmax) and (box_info_key.xmax > box_info.xmin):
            list_candidate.append(box_info)

    sorted_list_candidate = sorted(list_candidate, key=lambda x: x.ymin)
    if len(sorted_list_candidate) > 0:
        box_info = sorted_list_candidate[0]
        if box_info.flag_key == False:
            k_value_box = (xmin_block + box_info.xmin, ymin_block + box_info.ymin,
                           xmin_block + box_info.xmax, ymin_block + box_info.ymax, box_info.textbox_content)
            list_k_value.append(k_value_box)

    # find value on same line
    if len(list_k_value) == 0:
        list_candidate = []
        for box_info in list_boxes_info:
            if box_info.xmin > box_info_key.xmax and np.absolute(box_info_key.ymin - box_info.ymin) < 10:
                list_candidate.append(box_info)

        sorted_list_candidate = sorted(list_candidate, key=lambda x: x.xmin)
        if len(sorted_list_candidate) > 0:
            box_info = sorted_list_candidate[0]
            if box_info.flag_key == False:
                k_value_box = (xmin_block + box_info.xmin, ymin_block + box_info.ymin,
                               xmin_block + box_info.xmax, ymin_block + box_info.ymax, box_info.textbox_content)
                list_k_value.append(k_value_box)

    return list_k_value


def count_item_value(block_info):
    num_key = 0
    num_value = 0
    (xmin_block, ymin_block, xmax_block, ymax_block) = block_info[0]
    list_boxes_info = block_info[1]

    for box_info in list_boxes_info:
        if box_info.flag_key == True:
            num_key += 1
        else:
            num_value += 1

    return num_key, num_value


def table_from_detection(num_block):
    if num_block > 5:
        return True
    return False


def binary_img(img):
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret1, th1 = cv2.threshold(
        gray1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th1


def recreate_folder(folder):
    if os.path.isdir(folder) == True:
        shutil.rmtree(folder)
    os.makedirs(folder)


def read_flags():
    """Returns flags"""
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--images_out", default="result_images", help="Image result folder")
    parser.add_argument("--json_test", default="test_label.json",
                        help="Json files for testing")
    parser.add_argument(
        "--json_predict", default="test_predict.json", help="Json files predicted")

    flags = parser.parse_args()
    return flags


def main(flags):
    output_folder = flags.images_out + "/"
    json_test_file = flags.json_test
    json_predict_file = flags.json_predict

    recreate_folder(output_folder)
    list_file_process = []
    json_list = []
    total_time = 0
    list_test = read_json_data(json_test_file)
    print(len(list_test))

    for index, item in enumerate(list_test):
        folder_name = item[0]
        file_name = item[1]
        # if "76552.png" not in file_name:
        #     continue
        print(file_name + "(" + str(index) + ")")
        start_time = time.time()
        try:
            list_file_process.append(file_name)
            dict_key_value = key_value_detection(folder_name, file_name)
            if dict_key_value == None:
                print(file_name + " is not a table form")
            else:
                json_dict = make_json_dict(
                    file_name, folder_name, dict_key_value)
                json_list.append(json_dict)
                draw_key_value_image(dict_key_value, folder_name,
                                     output_folder, file_name)

        except Exception as e:
            print("image error at " + file_name + " : " + str(e))

        if index % 5 == 0:
            with open(json_predict_file, 'w') as outfile:
                json.dump(json_list, outfile)
            print("json saved")

        new_time = (time.time() - start_time)
        total_time += new_time
        print("--- %s seconds ---" % new_time)

    with open(json_predict_file, 'w') as outfile:
        json.dump(json_list, outfile)
    print("json saved")
    # print(list_file_process)
    print("total time : --- %s hours ---" % (total_time / 3600))


if __name__ == "__main__":
    flags = read_flags()
    main(flags)
