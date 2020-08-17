
def read_raw_data(path1, path2 ):

    f = open(path1, 'r')
    net_data = []
    for line in f:

        if line[-1] == '\n':
            line = line[:-1]
        elif line[-2:] == '\r\n':
            line = line[:-2]
        net_data.append(line)
    f = open(path2, 'r')
    svm_data = []
    for line in f:

        if line[-1] == '\n':
            line = line[:-1]
        elif line[-2:] == '\r\n':
            line = line[:-2]
        svm_data.append(line)
    count = 0
    for i in net_data:
        for j in svm_data:
            if i == j:
                count += 1
    print count


read_raw_data("../../Results/net.txt", "../../Results/dsjtzs_txfzjh_preliminary_SVMA.txt")