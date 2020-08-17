import matplotlib.pyplot as plt


class DataProcess:

    raw_track_list = []
    raw_track_data_list = []
    output_train = list()
    target_point = list()

    def __init__(self):
        self.raw_track_data_list = list()
        self.raw_track_list = list()
        self.target_point = list()
        # self.output_train = list()
        return

    def read_raw_data(self, raw_data_path):

        f = open(raw_data_path, 'r')
        for line in f:
            track_data = []
            if line[-1] == '\n':
                line = line[:-1]
            elif line[-2:] == '\r\n':
                line = line[:-2]
            str = line.split(' ')
            str[1] = str[1][:-1].split(';')

            for track_str in str[1]:
                track_str = track_str.split(',')
                track_data.append(track_str)
            str[1] = track_data
            self.raw_track_data_list.append(track_data)
            str[2] = str[2].split(',')
            # print str[2]
            self.target_point.append(str[2])

            self.raw_track_list.append(str)
            if len(str[-1]) == 1:
                self.output_train.append(str[-1])
            # self.plot_track(str[1], str[2])
        # print self.output_train
        # print self.target_point
        return str

    @staticmethod
    def __plot_track(track_str, end):

        x = []
        y = []
        t = []

        for str in track_str:
            x.append(str[0])
            y.append(str[1])
            t.append(str[2])

        plt.subplot(111)
        plt.plot(x, y, '-', end[0], end[1], '*', c='red')

        plt.show()


if __name__ == '__main__':
    dp = DataProcess()
    dp.read_raw_data()
