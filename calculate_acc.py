import json
import numpy as np
import os
import difflib


def get_dict_by_file_name(file_name, folder_name, datalabel):
    for json_label in datalabel:
        if file_name == json_label['file_name'] and folder_name == json_label['folder_name']:
            return json_label
    return None


def main(flags):

    filename_json = flags.json_test
    filename_label_json = flags.json_predict

    with open(filename_json, 'r') as f:
        datatest = json.load(f)

    with open(filename_label_json, 'r') as f:
        datalabel = json.load(f)

    list_field_acc = []
    list_char_acc = []
    list_file_name = []

    for index_test, json_test in enumerate(datatest):
        try:
            file_name = json_test['file_name']
            folder_name = json_test['folder_name']
            json_label = get_dict_by_file_name(
                file_name, folder_name, datalabel)
            count_field_acc = 0
            count_char_acc = 0

            if json_label == None:
                continue

            for k, v in json_test["keys"].items():
                s_test = json_test["keys"][k]["value_content"]
                s_label = json_label["keys"][k]["value_content"]

                # calculate distance
                dist = difflib.SequenceMatcher(None, s_test, s_label).ratio()
                if dist >= 0.9:
                    count_field_acc += 1
                count_char_acc += dist

            c_acc = round(count_char_acc/9, 2)
            f_acc = round(count_field_acc/9, 2)
            list_char_acc.append(c_acc)
            list_field_acc.append(f_acc)
            list_file_name.append(file_name)

        except Exception as e:
            print("error at " + str(e))

    char_acc = round(sum(list_char_acc)/len(list_char_acc), 2)
    field_acc = round(sum(list_field_acc)/len(list_field_acc), 2)
    print("field acc = " + str(field_acc))
    print("char acc = " + str(char_acc))
    print("len files = " + str(len(list_file_name)))

    with open("result.txt", "w") as text_file:
        text_file.write("file_name\t field_acc\t char_acc\n")

        for index, item in enumerate(list_file_name):
            text_file.write(
                item + "," + str(list_field_acc[index]) + "," + str(list_char_acc[index]) + "\n")
        text_file.write(str(field_acc) + "," + str(char_acc))


def read_flags():
    """Returns flags"""
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--json_test", default="test_label.json",
                        help="Json files for testing")
    parser.add_argument(
        "--json_predict", default="test_predict.json", help="Json files predicted")

    flags = parser.parse_args()
    return flags


if __name__ == '__main__':
    flags = read_flags()
    main(flags)
