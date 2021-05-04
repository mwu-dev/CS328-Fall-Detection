#truncates the data given a threshold in milliseconds from the left and right
def cut_data(data, threshold = 1000):
    initialT = data[0][3]
    for i, val in enumerate(data):
        if val[3] - initialT >= threshold: #once the difference in time is 1 second
            data = data[i:]
            # print(f'Shaved off {i+1} lines for the first second of data.')
            break
    finalT = data[len(data)-1][3]
    for i, val in reversed(list(enumerate(data))):
        if finalT - val[3] >= threshold:
            # print(f'Shaved off {len(data) - i + 1} lines for the final second of data.')
            data = data[:i]
            break
    return data