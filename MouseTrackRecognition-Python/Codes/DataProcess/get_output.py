

class GetOutput:

    def __init__(self):
        return

    def process_output(self):
        output_path = "../../Results/svm_output.txt"
        illegal_data_path = "../../Results/illegal_data.txt"
        result_path = "../../Results/dsjtzs_txfzjh_preliminary_SVMB.txt"
        net_output = list()
        f = open(output_path, 'r')
        count = 0
        temp = 0
        for line in f:
            # print line
            if line[-1] == '\n':
                line = line[:-1]
            elif line[-2:] == '\r\n':
                line = line[:-2]
            if line == '0':
                temp += 1
            net_output.append(line)
            count += 1
        net_output = list(map(int, net_output))
        print temp
        f.close()

        f = open(illegal_data_path, 'r')
        illegal_data = f.readline()
        illegal_data = illegal_data[1:-2]
        illegal_data = illegal_data.split(', ')
        illegal_data = list(map(int, illegal_data))
        f.close()

        for j in illegal_data:
            net_output.insert(j, 1)

        f = open(result_path, 'a')
        for i in range(len(net_output)):
            if net_output[i] == 0:
                f.write(str(i + 1))
                f.write('\n')
        f.close()

        return

if __name__ == '__main__':
    go = GetOutput()
    go.process_output()
