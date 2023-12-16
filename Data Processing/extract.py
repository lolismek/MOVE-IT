import csv
import numpy as np

NO_DEVICES = 4
INTERVAL_ERROR = 100
SAMPLE_ERROR = 10

def merge_intervals(intervals):
    if len(intervals) == 0:
        return intervals

    ans = []
    intervals = sorted(intervals, key=lambda x: x[0])

    i = 1
    left_end = intervals[0][0]
    right_end = intervals[0][1]
    while i < len(intervals):
        if intervals[i][0] > right_end:
            ans.append([left_end, right_end])
            left_end = intervals[i][0]
            right_end = intervals[i][1]
        else:
            right_end = max(right_end, intervals[i][1])
        i += 1
    ans.append([left_end, right_end])

    return ans

def included(p1, p2):
    return p1[0] <= p2[0] and p2[1] <= p1[1]

def split_intervals(intervals, del_intervals):
    ans = []
    for interval in intervals:
        i = 0
        while i < len(del_intervals):
            if included(interval, del_intervals[i]):
                L = interval[0]
                if i > 0:
                    L = max(L, del_intervals[i - 1][1])
                R = interval[1]
                if i + 1 < len(del_intervals):
                    R = min(R, del_intervals[i + 1][0])
                ans.append([L, R])            
            i += 1

    if len(ans) == 0:
        ans = intervals

    return ans

def clean_intervals(intervals, expected_length):
    ans = []
    for el in intervals:
        if el[1] - el[0] >= expected_length:
            ans.append(el)
    
    return ans

def extract_samples(file_name, expected_pause, sample_sz, stride, jump, category, participant_id, debug):
    data_path = "data/" + file_name + ".csv"
    intervals_path = "data/intervale_" + file_name + ".csv"

    intervals = []

    with open(intervals_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:                
            x = float(row[0])
            y = float(row[1])
            x += INTERVAL_ERROR
            y -= INTERVAL_ERROR
            if x <= y:
                intervals.append([x, y])

    data = [[], [], [], [], []]
    last_timestamp = [0, 0, 0, 0, 0]

    del_intervals = []

    with open(data_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            aux = []
            for el in row:
                aux.append(float(el))
            aux[0] = int(aux[0])
            aux[1] = int(aux[1])

            if last_timestamp[aux[0]] != 0 and aux[1] - last_timestamp[aux[0]] > expected_pause + SAMPLE_ERROR:
                del_intervals.append([last_timestamp[aux[0]], aux[1]])

            data[aux[0]].append(aux)
            last_timestamp[aux[0]] = aux[1]

    del_intervals = merge_intervals(del_intervals)
    intervals = split_intervals(intervals, del_intervals)
    intervals = clean_intervals(intervals, sample_sz * expected_pause * jump)

    if debug:
        print(intervals)
        sum = 0
        for interv in intervals:
            if interv[1] - interv[0] >= 2000:
                sum += (interv[1] - interv[0]) / 1000
        print(sum)

    for device_index in range(1, NO_DEVICES + 1):
        new_data = []

        first_in_interval = True
        interval_index = 0

        for row in data[device_index]:
            if interval_index >= len(intervals):
                break
            if intervals[interval_index][0] <= row[1] and row[1] <= intervals[interval_index][1]:
                if first_in_interval:
                    new_data.append([row])
                    first_in_interval = False
                else:
                    # print("!!!")
                    # print(interval_index)
                    # print(len(new_data))
                    new_data[interval_index].append(row)
            elif row[1] > intervals[interval_index][0]:
                interval_index += 1
                first_in_interval = True

        data[device_index] = new_data

    buckets = []

    i = 0
    while i < len(intervals):
        lim = min(len(data[1][i]), len(data[2][i]), len(data[3][i]), len(data[4][i]))
        
        j = 0
        while j + sample_sz * jump < lim:
            k = 0
            while k < jump:
                bucket = [[], [], [], [], []]
                
                ind = 0
                iter = 0
                while ind + j + k < lim and iter < sample_sz:
                    for device in range(1, NO_DEVICES + 1):
                        bucket[device].append(data[device][i][ind + j + k])
                    iter += 1
                    ind += jump

                if len(bucket[1]) == sample_sz and len(bucket[2]) == sample_sz and len(bucket[3]) == sample_sz and len(bucket[4]) == sample_sz:
                    buckets.append(bucket)

                k += 1
            j += int(sample_sz * stride)
        i += 1

    data = []
    for bucket in buckets:
        row = []
        for device in range(1, NO_DEVICES + 1):
            i = 1
            while i <= 6:
                for sample in range(0, sample_sz):
                    row.append(bucket[device][sample][1 + i])
                i += 1
        row.append(category)
        row.append(participant_id)

        data.append(row)

    print(len(data))
    return data


#extract_samples("sarituri_costea", 2, 500, 0.5, 2, 1, 1, 0)

# extract_samples("alergare_alex", 1, 500, 0.5, 1, 1, 0) -> daca sunt pauzele mai mari de 2???
# de avut grija la expect_pause si jump in acest sens!!!