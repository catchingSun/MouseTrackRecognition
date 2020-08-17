from __future__ import division
from process_data import *
import numpy as np
import math


class ConstructEigenvector:

    pdd = 0
    track_data_list = []
    illegal_data = []
    flag = 0
    target_point = []

    def __init__(self, raw_data_path):
        self.pdd = DataProcess()
        self.track_data_list = []
        self.illegal_data = []

        if raw_data_path == "../../Datas/RawDatas/dsjtzs_txfz_test1.txt":
            self.flag = 1
        self.pdd.read_raw_data(raw_data_path)
        self.target_point = self.pdd.target_point

        # self.target_point = list(map(int, self.target_point))
        # print self.target_point
        self.track_data_list = self.pdd.raw_track_data_list

    @staticmethod
    def __convert_strlist_to_intlist(every_track):
        temp = list()
        for j in every_track:
            temp.append(list(map(int, j)))
        return temp

    def construct_the_eigenvector(self):
        eigenvector = list()
        count = 0

        for i in self.track_data_list:

            if len(i) > 1:
                temp_vector = list()
                i = self.__convert_strlist_to_intlist(i)
                # temp0 = self.__calculate_first_five_direction(i)
                # for j in temp0:
                #     temp_vector.append(j)

                # temp1 = self.__calculate_inflection_points_num(i)
                # temp_vector.append(temp1)
                #
                temp2 = self.__calculate_internal_direction(i)
                for j in temp2:
                    temp_vector.append(j)
                #
                # temp3 = self.__calculate_move_distance(i)
                # for j in temp3:
                #     temp_vector.append(j)
                #
                # temp4 = self.__calculate_move_speed(i)
                # for j in temp4:
                #     temp_vector.append(j)
                # #
                # temp5 = self.__calculate_time(i)
                # for j in temp5:
                #     temp_vector.append(j)
                # #
                # tpt = self.target_point[count]
                # tpt = list(map(float, tpt))
                # temp6 = self.__calculate_end_to_target_dis(i, tpt)
                # for j in temp6:
                #     temp_vector.append(j)
                # #
                # temp7 = self.__calculate_move_speed_x(i)
                # for j in temp7:
                #     temp_vector.append(j)
                # #
                # temp8 = self.__calculate_move_speed_y(i)
                # for j in temp8:
                #     temp_vector.append(j)

                eigenvector.append(temp_vector)
            else:
                # print count
                # print i
                if self.flag == 0:
                    self.pdd.output_train.pop(count)
                else:
                    self.illegal_data.append(count)
            count += 1
        if self.flag == 1:
            path = "../../Results/illegal_dataA.txt"
            mse_file = open(path, 'w')
            mse_file.write(str(self.illegal_data))
            mse_file.write("\n")
            mse_file.close()
        # eigenvector = np.array(eigenvector)
        # print eigenvector.max(axis=0)
        return eigenvector

    # @staticmethod
    # def normalize_eigenvector(eigenvector):
    #     aevct = np.array(eigenvector)
    #     mean = aevct.mean(axis=0)
    #     print mean
    #     max = aevct.max(axis=0)
    #     print max
    #     min = aevct.min(axis=0)
    #     print min
    #     for i in range(len(aevct[0])):
    #         for j in range(len(aevct)):
    #             # print aevct[j][i]
    #             temp = math.atan(aevct[j][i]) * 2 / math.pi
    #             aevct[j][i] = temp
    #     # # nevct = (aevct - min) / (max - min)
    #     # # print aevct
    #     return aevct

    def __calculate_first_five_direction(self, every_track):
        # every_track = self.__convert_strlist_to_intlist(every_track)

        start_dirt = self.__calculate_direction(every_track[0][0],
                                                every_track[0][1], every_track[1][0], every_track[1][1])
        end_dirt = self.__calculate_direction(every_track[-2][0],
                                              every_track[-2][1], every_track[-1][0], every_track[-1][1])

        total_dis = 0
        total_time = every_track[-1][2] - every_track[0][2]

        for i in range(len(every_track) - 1):
            j = i + 1
            total_time += every_track[i][2]
            total_dis += self.__calculate_distance(every_track[j][0],
                                                   every_track[j][1], every_track[i][0], every_track[i][1])
        total_speed = total_dis / total_time
        # print start_dirt, end_dirt, total_dis, total_speed, total_time
        return start_dirt, end_dirt, total_dis, total_speed, total_time

    @staticmethod
    def __calculate_end_to_target_dis(every_track, target_point):
        tdis_x = target_point[0] - every_track[-1][0]
        tdis_y = target_point[1] - every_track[-1][1]
        return tdis_x, tdis_y

    @staticmethod
    def __calculate_inflection_points_num(every_track):
        iftp_num = 0
        for i in range(len(every_track) - 2):
            j = i + 1
            k = i + 2
            if (every_track[j][2] != every_track[i][2]) or (every_track[j][2] != every_track[k][2]):
                iftp_num += 1
        return iftp_num

    def __calculate_internal_direction(self, every_track):
        angle_num = [1] * 8
        for i in range(len(every_track) - 1):
            j = i + 1
            tmp_angle = self.__calculate_direction(every_track[i][0],
                                                   every_track[i][1], every_track[j][0], every_track[j][1])
            if (tmp_angle >= 0) and (tmp_angle < math.pi / 4):
                angle_num[0] += 1
            elif (tmp_angle >= math.pi / 4) and (tmp_angle < math.pi / 2):
                angle_num[1] += 1
            elif (tmp_angle >= math.pi / 2) and (tmp_angle < 3 * math.pi / 4):
                angle_num[2] += 1
            elif (tmp_angle >= 3 * math.pi / 4) and (tmp_angle < math.pi):
                angle_num[3] += 1
            elif (tmp_angle >= math.pi) and (tmp_angle < 5 * math.pi / 4):
                angle_num[4] += 1
            elif (tmp_angle >= 5 * math.pi / 4) and (tmp_angle < 3 * math.pi / 2):
                angle_num[5] += 1
            elif (tmp_angle >= 3 * math.pi / 2) and (tmp_angle < 7 * math.pi / 4):
                angle_num[6] += 1
            else:
                angle_num[7] += 1
        #     self.__calculate_direction
        return angle_num

    def __calculate_move_distance(self, every_track):
        internal_dis = list()
        for i in range(len(every_track) - 1):
            j = i + 1
            temp = self.__calculate_distance(every_track[j][0],
                                             every_track[j][1], every_track[i][0], every_track[i][1])
            internal_dis.append(temp)
        dis_array = np.array(internal_dis)
        dis_mean = np.mean(dis_array)
        dis_var = np.var(dis_array)
        return dis_mean, dis_var

    def __calculate_move_speed(self, every_track):
        internal_v = list()
        for i in range(len(every_track) - 1):
            temp_v = 0
            j = i + 1
            temp_d = self.__calculate_distance(every_track[j][0],
                                             every_track[j][1], every_track[i][0], every_track[i][1])
            temp_t = every_track[j][2] - every_track[i][2]
            if temp_t == 0 and temp_d == 0:
                temp_v == 0
            elif temp_t == 0 and temp_d != 0:
                temp_v == 10000
            else:
                temp_v = temp_d / temp_t
            internal_v.append(temp_v)
        v_array = np.array(internal_v)
        v_mean = np.mean(v_array)
        v_var = np.var(v_array)
        v_min = np.mean(v_array)
        v_max = np.max(v_array)
        return v_mean, v_var, v_min, v_max

    @staticmethod
    def __calculate_move_speed_x(every_track):
        internal_vx_component = list()
        for i in range(len(every_track) - 1):
            temp_v_x = 0
            j = i + 1
            temp_d_x = every_track[j][0] - every_track[i][0]
            temp_t = every_track[j][2] - every_track[i][2]

            if temp_t == 0 and temp_d_x == 0:
                temp_v_x == 0
            elif temp_t == 0 and temp_d_x != 0:
                temp_v_x == 10000
            else:
                temp_v_x = temp_d_x / temp_t
            internal_vx_component.append(temp_v_x)
        vx_array = np.array(internal_vx_component)
        vx_mean = np.mean(vx_array)
        vx_var = np.var(vx_array)
        vx_min = np.mean(vx_array)
        vx_max = np.max(vx_array)
        return vx_mean, vx_var, vx_min, vx_max

    @staticmethod
    def __calculate_move_speed_y(every_track):
        internal_vy_component = list()
        for i in range(len(every_track) - 1):
            temp_v_y = 0
            j = i + 1
            temp_d_y = every_track[j][1] - every_track[i][1]
            temp_t = every_track[j][2] - every_track[i][2]

            if temp_t == 0 and temp_v_y == 0:
                temp_v_y == 0
            elif temp_t == 0 and temp_v_y != 0:
                temp_v_y == 10000
            else:
                temp_v_y = temp_d_y / temp_t
            internal_vy_component.append(temp_v_y)
        vy_array = np.array(internal_vy_component)
        vy_mean = np.mean(vy_array)
        vy_var = np.var(vy_array)
        vy_min = np.mean(vy_array)
        vy_max = np.max(vy_array)
        return vy_mean, vy_var, vy_min, vy_max

    @staticmethod
    def __calculate_time(every_track):
        internal_t = list()
        for i in range(len(every_track) - 1):
            j = i + 1
            internal_t.append(every_track[j][2] - every_track[i][2])
        t_array = np.array(internal_t)
        t_mean = np.mean(t_array)
        t_var = np.var(t_array)
        t_min = np.mean(t_array)
        t_max = np.max(t_array)
        return t_mean, t_var, t_min, t_max

    @staticmethod
    def __calculate_direction(x1, y1, x2, y2):
        tdistance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        ty = y2 - y1
        if tdistance != 0:
            direction = math.asin(ty / tdistance)
        else :
            direction = 0
        return direction

    @staticmethod
    def __calculate_distance(x1, y1, x2, y2):
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return distance

if __name__ == '__main__':
    train_raw_data_path = 0
    ce = ConstructEigenvector(train_raw_data_path)
    eigenvector = ce.construct_the_eigenvector()
    nect = ce.normalize_eigenvector(eigenvector)
    # print nect