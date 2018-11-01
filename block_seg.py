import cv2
import numpy as np
import os
import imutils


def is_line3(arr, max_char, merge_threshold, density_threshold):
    trans = np.zeros(len(arr), dtype=int)
    trans[arr < density_threshold] = 1
    if np.sum(trans) < max_char:
        return []

    count = 0
    is_begin = True
    ret = []
    start = 0
    for i, a in enumerate(trans):
        if a == 1:
            if is_begin:
                start = i
                is_begin = False
            count += 1

        if a == 0 and not is_begin:
            if count >= max_char:
                ret.append([start, start + count])
            is_begin = True
            count = 0

    if count >= max_char:
        ret.append([start, start+count])

    if len(ret) == 0:
        return []

    fine_tune = []
    prev = ret[0]
    for cur in ret[1:]:
        if cur[0] - prev[1] < merge_threshold:
            prev = [prev[0], cur[1]]
        else:
            fine_tune.append(prev)
            prev = cur
    fine_tune.append(prev)
    return fine_tune


def find_row(r, c):
    for id, row in enumerate(rows[r]):

        if row[0] <= c and row[1] > c:
            # check it's intersec line or not
            try:
                inter_point = rows_intersect.get(r)[id]
            except:
                inter_point = -1
            return id, inter_point

    return -1, -1


def find_col(r, c):
    for id, col in enumerate(cols[c]):

        if col[0] <= r and col[1] > r:
            # check it's intersec line or not
            try:
                inter_point = cols_intersect.get(c)[id]
            except:
                inter_point = -1
            return id, inter_point

    return -1, -1


def find_row_intersect(r, c, isLeft, c_bound):  # return column point
    tmp = inter_lookup[1, inter_lookup[0] == r]
    try:
        if isLeft:
            result = tmp[(tmp <= c) & (c_bound <= tmp)][-1]
        else:
            result = tmp[(tmp >= c) & (c_bound >= tmp)][0]
    except:
        result = -1

    return result


def find_col_intersect(r, c, isUp, r_bound):  # return row point
    tmp = inter_lookup[0, inter_lookup[1] == c]
    try:
        if isUp:
            result = tmp[(tmp <= r) & (r_bound <= tmp)][-1]
        else:
            result = tmp[(tmp >= r) & (r_bound >= tmp)][0]
    except:
        result = -1

    return result


def find_near_col_intersect(r, c, isLeft):
    if isLeft:
        tmp = set(inter_lookup[1, (inter_lookup[1] < c) & (
            inter_lookup[1] >= c - near_threshold_row)])
        tmp = list(tmp)[::-1]
    else:
        tmp = set(inter_lookup[1, (inter_lookup[1] > c) & (
            inter_lookup[1] <= c + near_threshold_row)])

    for c_id in tmp:
        near_cols = np.array(cols[c_id])
        if np.sum((near_cols[:, 0] <= r) & (near_cols[:, 1] > r)):
            return c_id

    return -1


width = 700
height = 1000
min_width = 40
min_height = 20
density_threshold = 250

variance_threshold = 5
near_threshold_col = 20
boundary_ratio = 0.15
near_threshold_row = 10
special_row_threshold = 0.4
left_bound_const = int(width * boundary_ratio)
right_bound_const = int(width * (1-boundary_ratio))
up_bound_const = int(height * boundary_ratio)
down_bound_const = int(height * (1-boundary_ratio))
threshold_r = 5
out_folder = "block_segment"
inter_lookup = None
rows = None
cols = None
rows_intersect = None
cols_intersect = None
if not os.path.isdir(out_folder):
    os.mkdir(out_folder)


def binary_img(img):
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret1, th1 = cv2.threshold(
        gray1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th1


def block_segment(folder_name, file_name):
    global inter_lookup, rows, cols, rows_intersect, cols_intersect

    url = folder_name + file_name
    img_origin = cv2.imread(url)

    if img_origin is None:
        print("error image!")
        return
    if img_origin.shape[0] <= img_origin.shape[1]:
        print("image with height > width")

        #img_origin = np.transpose(img_origin, (1, 0, 2))

    img = cv2.resize(img_origin, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ret1, gray = cv2.threshold(
    #     gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # cv2.imwrite("block_char_test/binary.png", gray)

    padding_ratio = 0.005
    padding_w = int(padding_ratio * width)
    padding_h = int(padding_ratio*2 * height)
    gray[:padding_h] = 255
    gray[-padding_h:] = 255
    gray[:, :padding_w] = 255
    gray[:, -padding_w:] = 255

    # detect background color
    uniques, fres = np.unique(gray, return_counts=True)
    bg_colors = uniques[np.logical_and(
        uniques >= 200, uniques < density_threshold)]
    bg_colors_fres = fres[np.logical_and(
        uniques >= 200, uniques < density_threshold)]
    bg_colors = bg_colors[bg_colors_fres >= 0.005 * width * height]

    for bg_color in bg_colors:
        gray[gray == bg_color] = 255

    canny = np.zeros(gray.shape, dtype=np.uint8)  # for finding intersection
    empty = np.ones(gray.shape, dtype=np.uint8) * 255
    only_lines = np.ones(gray.shape, dtype=np.uint8) * 255
    # collect all lines

    rows = []
    rows_not_empty = []
    for i in range(gray.shape[0]):
        ret = is_line3(gray[i], min_width, 15, density_threshold)
        rows.append(ret)
        if ret:
            rows_not_empty.append(i)
            for j in range(len(ret)):
                canny[i, np.arange(ret[j][0], ret[j][1])] += 1
                only_lines[i, np.arange(ret[j][0], ret[j][1])] = 0

    cols = []
    for i in range(gray.shape[1]):
        ret = is_line3(gray[:, i], min_height, 10, density_threshold)
        cols.append(ret)
        for j in range(len(ret)):
            canny[np.arange(ret[j][0], ret[j][1]), i] += 1

    edges = cv2.Canny(only_lines, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is None:
        print("no line cases")
        return
    theta = np.average(lines[:, 0, 1])
    angle = 90 - theta / np.pi * 180
    print("angle =  %f" % angle)

    del only_lines, edges, lines
    if np.abs(angle) >= 0.1:

        img_origin = imutils.rotate_bound(img_origin, angle)
        img = cv2.resize(img_origin, (width, height),
                         interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        padding_ratio = 0.01
        padding_w = int(padding_ratio * width)
        padding_h = int(padding_ratio*2 * height)
        gray[:padding_h] = 255
        gray[-padding_h:] = 255
        gray[:, :padding_w] = 255
        gray[:, -padding_w:] = 255
        # detect background color
        uniques, fres = np.unique(gray, return_counts=True)
        bg_colors = uniques[np.logical_and(
            uniques >= 200, uniques < density_threshold)]
        bg_colors_fres = fres[np.logical_and(
            uniques >= 200, uniques < density_threshold)]
        bg_colors = bg_colors[bg_colors_fres >= 0.005 * width * height]
        for bg_color in bg_colors:
            gray[gray == bg_color] = 255

        # for finding intersection
        canny = np.zeros(gray.shape, dtype=np.uint8)
        empty = np.ones(gray.shape, dtype=np.uint8) * 255
        # collect all lines
        rows = []
        rows_not_empty = []
        for i in range(gray.shape[0]):
            ret = is_line3(gray[i], min_width, 15, density_threshold)
            rows.append(ret)
            if ret:
                rows_not_empty.append(i)
                for j in range(len(ret)):
                    canny[i, np.arange(ret[j][0], ret[j][1])] += 1

        cols = []
        for i in range(gray.shape[1]):
            ret = is_line3(gray[:, i], min_height, 10, density_threshold)
            cols.append(ret)
            for j in range(len(ret)):
                canny[np.arange(ret[j][0], ret[j][1]), i] += 1

    # merge lines and find candidate lines
    rs, cs = np.where(canny == 2)
    n_candidate = rs.shape[0]
    from collections import defaultdict
    rows_intersect = defaultdict(dict)
    cols_intersect = defaultdict(dict)

    # list here = tupple
    # find intersection lines
    for t in range(n_candidate):
        for id, row in enumerate(rows[rs[t]]):
            if row[0] <= cs[t] and row[1] > cs[t]:
                rows_intersect[rs[t]][id] = cs[t]
                break

    for t in range(n_candidate):
        for id, col in enumerate(cols[cs[t]]):
            if col[0] <= rs[t] and col[1] > rs[t]:
                cols_intersect[cs[t]][id] = rs[t]
                break

    inter_lookup = np.array([rs, cs])
    empty[rs, cs] = 0
    del rs, cs

    rows_candidate = []
    cols_candidate = []
    # find long rows that does not intersect, above half
    special_rows = []
    for direct in [1, -1]:
        for i in rows_not_empty:
            # move right
            curRow = i
            curLeft = rows[curRow][0][0]  # only care the first row
            if curLeft >= left_bound_const:
                continue

            curRight = rows[curRow][0][1]-1
            is_append = True
            while find_row(curRow - direct, curRight)[0] != -1:
                curRow -= direct
                pos, inter = find_row(curRow, curRight)
                if inter != -1:
                    is_append = False
                    break
                curRight = rows[curRow][pos][1] - 1

            if is_append and curRight - curLeft >= img.shape[1] * special_row_threshold:
                if direct < 0:
                    special_rows.append([i, curLeft, curRight+1])
                else:
                    special_rows.append([curRow, curLeft, curRight+1])

    special_rows = np.array(special_rows)
    if special_rows.shape[0]:
        special_rows = special_rows[np.argsort(special_rows[:, 0])]

    ############

    for direct in [1, -1]:
        if direct:
            # row
            for key, value in rows_intersect.items():
                for id, intersect in value.items():
                    # move left
                    curRow = key
                    curLeft = rows[curRow][id][0]
                    curRight = intersect

                    tmp = inter_lookup[1, inter_lookup[0] == key]
                    try:
                        inter = tmp[(tmp < curRight) & (curLeft <= tmp)][0]
                        if curRight >= inter + variance_threshold:
                            rows_candidate.append([key, inter, curRight])
                            curRight = inter
                    except:
                        inter = curRight

                    while find_row(curRow + direct, curLeft)[0] != -1:
                        curRow += direct
                        pos, next_inter = find_row(curRow, curLeft)

                        if next_inter != -1:
                            curLeft = find_row_intersect(
                                curRow, curLeft, True, rows[curRow][pos][0])
                            if curLeft != -1:
                                inter = curLeft
                                break
                        curLeft = rows[curRow][pos][0]

                    if curRight >= curLeft + min_width:

                        if direct > 0:
                            insertedRow = key
                        else:
                            insertedRow = curRow

                        # change curLeft value
                        intersect_left = find_near_col_intersect(
                            insertedRow, curLeft, True)

                        if intersect_left != -1 and inter - curLeft >= min_width:
                            curLeft = intersect_left

                        rows_candidate.append([insertedRow, curLeft, curRight])
                        # clean special row
                        if special_rows.shape[0]:
                            delete_ids = np.arange(special_rows.shape[0])[np.logical_and(special_rows[:, 0] <= max(
                                key, curRow) + variance_threshold, special_rows[:, 0] >= min(key, curRow) - variance_threshold)]
                            special_rows = np.delete(
                                special_rows, delete_ids, 0)

                    # move right
                    curRow = key
                    curLeft = intersect
                    curRight = rows[curRow][id][1]-1
                    inter = intersect
                    is_append = True
                    while find_row(curRow - direct, curRight)[0] != -1:
                        curRow -= direct
                        pos, next_inter = find_row(curRow, curRight)
                        if next_inter != -1:
                            curRight = find_row_intersect(
                                curRow, curRight, False, rows[curRow][pos][1] - 1)
                            if curRight != -1:
                                is_append = False
                                break

                        curRight = rows[curRow][pos][1] - 1
                    if is_append and curRight >= curLeft + min_width:
                        if direct < 0:
                            insertedRow = key
                        else:
                            insertedRow = curRow

                        # change curRight value
                        intersect_right = find_near_col_intersect(
                            insertedRow, curRight, False)

                        if intersect_right != -1 and curRight - inter >= min_width:
                            curRight = intersect_right

                        rows_candidate.append([insertedRow, curLeft, curRight])
                        # clean special row
                        if special_rows.shape[0]:
                            delete_ids = np.arange(special_rows.shape[0])[np.logical_and(special_rows[:, 0] <= max(
                                key, curRow) + variance_threshold, special_rows[:, 0] >= min(key, curRow) - variance_threshold)]
                            special_rows = np.delete(
                                special_rows, delete_ids, 0)

            # columns
            for key, value in cols_intersect.items():
                for id, intersect in value.items():
                    # move up
                    curCol = key
                    curUp = cols[curCol][id][0]
                    curDown = intersect
                    tmp = inter_lookup[0, inter_lookup[1] == key]
                    try:
                        inter = tmp[(tmp < curDown) & (curUp <= tmp)][0]
                        if curDown >= inter + variance_threshold:
                            cols_candidate.append([key, inter, curDown])
                            curDown = inter
                    except:
                        inter = curDown

                    while find_col(curUp, curCol - direct)[0] != -1:
                        curCol -= direct
                        pos, next_inter = find_col(curUp, curCol)

                        if next_inter != -1:
                            curUp = find_col_intersect(
                                curUp, curCol, True, cols[curCol][pos][0])
                            if curUp != -1:
                                inter = curUp
                                break

                        curUp = cols[curCol][pos][0]

                    curUps = []
                    # specialine
                    if special_rows.shape[0] and inter - curUp >= min_height:
                        curUps = special_rows[np.logical_and(
                            special_rows[:, 0] < curUp, special_rows[:, 0] >= curUp - near_threshold_col), 0]
                    if len(curUps):
                        curUp = curUps[-1]

                    if direct > 0:
                        cols_candidate.append([key, curUp, curDown])
                    else:
                        cols_candidate.append([curCol, curUp, curDown])

                    # move down
                    curCol = key
                    curUp = intersect
                    curDown = cols[curCol][id][1]-1
                    # tmp = inter_lookup[0,inter_lookup[1] == key]
                    # inter = tmp[(tmp <= curDown) & (curUp<= tmp)][-1]
                    inter = intersect
                    is_append = True
                    while find_col(curDown, curCol + direct)[0] != -1:
                        curCol += direct
                        pos, next_inter = find_col(curDown, curCol)
                        if next_inter != -1:
                            curDown = find_col_intersect(
                                curDown, curCol, False, cols[curCol][pos][1] - 1)
                            if curDown != -1:
                                is_append = False
                                break

                        curDown = cols[curCol][pos][1] - 1

                    if is_append:
                        curDowns = []
                        if special_rows.shape[0] and curDown - inter >= min_height:
                            curDowns = special_rows[np.logical_and(
                                special_rows[:, 0] > curDown, special_rows[:, 0] <= curDown + near_threshold_col), 0]
                        if len(curDowns):
                            curDown = curDowns[0]
                        if direct < 0:
                            cols_candidate.append([key, curUp, curDown])
                        else:
                            cols_candidate.append([curCol, curUp, curDown])

    del rows, cols

    if len(cols_candidate) == 0 or len(rows_candidate) == 0:
        print("only lines")
        return
    #######################################################
        # add vertical
    # find left boundary ,right boundary
    rows_candidate = np.array(rows_candidate)
    cols_candidate = np.array(cols_candidate)

    if len(special_rows):
        rows_candidate = np.append(rows_candidate, special_rows, 0)

    rows_candidate = np.unique(rows_candidate, axis=0)
    left_bound = np.min(rows_candidate[:, 1])
    right_bound = np.max(rows_candidate[:, 2])

    rows_candidate[rows_candidate[:, 1] <= left_bound_const, 1] = left_bound
    rows_candidate[rows_candidate[:, 2] >= right_bound_const, 2] = right_bound

    # expend horizon lines, add vertical lines
    left_ids = rows_candidate[rows_candidate[:, 1] == left_bound, 0]
    right_ids = rows_candidate[rows_candidate[:, 2] == right_bound, 0]
    # skip 1 max row only
    if left_ids.shape[0] > 1:
        cols_candidate = np.vstack(
            [cols_candidate, [left_bound, np.min(left_ids), np.max(left_ids)]])
    if right_ids.shape[0] > 1:
        cols_candidate = np.vstack(
            [cols_candidate, [right_bound, np.min(right_ids), np.max(right_ids)]])

        # add horizontal
    # find up boundary ,down boundary and draw horizontal lines
    # up_bound case
    up_bound = np.min(cols_candidate[:, 1])
    if np.min(rows_candidate[:, 0]) > up_bound + variance_threshold:
        cols_candidate[cols_candidate[:, 1] <=
                       up_bound + near_threshold_col, 1] = up_bound
        up_ids = set(cols_candidate[cols_candidate[:, 1] == up_bound, 0])
        up_ids = list(up_ids)
        if len(up_ids) > 1:
            rows_candidate = np.vstack(
                [rows_candidate, [up_bound, np.min(up_ids), np.max(up_ids)]])
        else:  # up_ids.shape[0] ==1

            rows_candidate = np.vstack(
                [rows_candidate, [up_bound, left_bound, right_bound]])
            cols_candidate = np.vstack(
                [cols_candidate, [left_bound, up_bound, np.min(left_ids)]])
            cols_candidate = np.vstack(
                [cols_candidate, [right_bound, up_bound, np.min(right_ids)]])

    # down_bound case
    down_bound = np.max(cols_candidate[:, 2])
    max_arg = np.argmax(rows_candidate[:, 0])
    if rows_candidate[max_arg, 0] < down_bound - variance_threshold \
            or rows_candidate[max_arg, 2] - rows_candidate[max_arg, 1] < img.shape[1] * special_row_threshold:
        cols_candidate[cols_candidate[:, 2] >=
                       down_bound - near_threshold_col, 2] = down_bound
        down_ids = set(cols_candidate[cols_candidate[:, 2] == down_bound, 0])
        down_ids = list(down_ids)
        if len(down_ids) > 1:
            rows_candidate = np.vstack(
                [rows_candidate, [down_bound, np.min(down_ids), np.max(down_ids)]])
        else:  # down_ids.shape[0] ==1
            rows_candidate = np.vstack(
                [rows_candidate, [down_bound, left_bound, right_bound]])
            cols_candidate = np.vstack(
                [cols_candidate, [left_bound, np.max(left_ids), down_bound]])
            cols_candidate = np.vstack(
                [cols_candidate, [right_bound, np.max(right_ids), down_bound]])

    ########################################################
    # remove dumplicated horizontal lines
    rows_filter = []
    rows_candidate = np.array(rows_candidate)
    rows_candidate = rows_candidate[np.argsort(rows_candidate[:, 0])]
    # print(rows_candidate)
    while rows_candidate.shape[0]:
        startRow = rows_candidate[0][0]
        curRow = rows_candidate[0][0]
        curLeft = rows_candidate[0][1]
        curRight = rows_candidate[0][2]
        rows_filter.append([curRow, curLeft, curRight])
        rows_candidate = np.delete(rows_candidate, 0, 0)

        candidate_id = np.where(rows_candidate[:, 0] == curRow + 1)[0]
        while candidate_id.shape[0]:
            candidate = rows_candidate[candidate_id]
            candidate_result = candidate_id[np.logical_and(
                candidate[:, 1] >= curLeft - variance_threshold, candidate[:, 2] <= curRight + variance_threshold)]
            if candidate_result.shape[0]:
                if curRow == startRow + 1:
                    second_length = rows_candidate[candidate_result[0]
                                                   ][2] - rows_candidate[candidate_result[0]][1]
                    if second_length < 0.8 * (curRight - curLeft):
                        break

                curRow += 1
                rows_candidate = np.delete(rows_candidate, candidate_result, 0)
            else:
                break
            candidate_id = np.where(rows_candidate[:, 0] == curRow + 1)[0]

        if curRow > startRow+8:
            rows_filter.append([curRow, curLeft, curRight])

    # draw lines
    for candidate in rows_filter:
        empty[candidate[0], np.arange(candidate[1], candidate[2]+2)] = 0
        #img[candidate[0],np.arange(candidate[1],candidate[2]+1),:] = [0, 0, 255]

    for candidate in cols_candidate:
        empty[np.arange(candidate[1]-1, candidate[2]+1), candidate[0]] = 0
        #img[np.arange(candidate[1],candidate[2]+1), candidate[0],:] = [0, 0, 255]

    (_, contours, _) = cv2.findContours(
        empty, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print(img_origin.shape)
    w_ratio = img_origin.shape[1] / width
    h_ratio = img_origin.shape[0] / height

    path = out_folder+'/'+os.path.basename(url).split(".")[0]
    if not os.path.isdir(path):
        os.mkdir(path)
    # cv2.imwrite(path+"/origin.png",img_origin)

    bboxs = np.zeros([0, 4], dtype=int)
    img_clone = img_origin.copy()
    areas = []
    for i, contour in enumerate(contours):
        mrect = cv2.boundingRect(contour)
        if mrect[0] < 1 or mrect[2] < 10 or mrect[3] < 5:
            continue
        bboxs = np.vstack([bboxs, mrect])
        areas.append(mrect[2]*mrect[3])
    bboxs = np.delete(bboxs, np.argmax(areas), 0)

    # #### remove cover box
    # cover = []
    # #### merger box
    bboxs[:, 2] += bboxs[:, 0]
    bboxs[:, 3] += bboxs[:, 1]
    bboxs = bboxs.astype(float)
    bboxs[:, 0] *= w_ratio
    bboxs[:, 1] *= h_ratio
    bboxs[:, 2] = bboxs[:, 2] * w_ratio + 2
    bboxs[:, 3] = bboxs[:, 3] * h_ratio + 2
    bboxs = bboxs.astype(int)

    i = 0
    for bb in bboxs:
        name = "/"+str(bb[0])+"_"+str(bb[1])+"_" + \
            str(bb[2])+"_"+str(bb[3])+".png"
        # cv2.imwrite(path+name, img_clone[bb[1]:bb[3], bb[0]:bb[2]])
        x_center = (bb[0] + bb[2]) // 2
        y_center = (bb[1] + bb[3]) // 2
        cv2.putText(img_origin, str(i), (x_center, y_center),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.rectangle(img_origin, (bb[0], bb[1]),
                      (bb[2], bb[3]), (0, 0, 255), 2)

        i += 1

    # cv2.imwrite(path+"/0.png", img_origin)
    # cv2.imwrite("block_char_test/"+file_name, img_origin)
    return bboxs
