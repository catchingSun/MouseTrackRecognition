from construct_the_eigenvector import *
from process_data import *
from sklearn import svm


class SVMModel:

    eigenvector = list()
    test_data = list()
    output_train = list()

    def __init__(self):
        prd = DataProcess()
        train_raw_data_path = "../../Datas/RawDatas/dsjtzs_txfz_training.txt"
        cte = ConstructEigenvector(train_raw_data_path)
        self.eigenvector = cte.construct_the_eigenvector()
        # temp = cte.construct_the_eigenvector()
        # self.eigenvector = cte.normalize_eigenvector(temp)
        self.output_train = list(map(int, prd.output_train))
        # print self.eigenvector
        test_raw_data_path = "../../Datas/RawDatas/dsjtzs_txfz_test1.txt"
        cte1 = ConstructEigenvector(test_raw_data_path)
        self.test_data = cte1.construct_the_eigenvector()
        # temp = cte1.construct_the_eigenvector()
        # self.test_data = cte1.normalize_eigenvector(temp)
        return

    def model(self):
        clf = svm.SVC(kernel='linear')
        clf.fit(self.eigenvector, self.output_train)
        results = clf.predict(self.test_data)
        path = "../../Results/svmA_output.txt"
        rst_file = open(path, 'w')
        count = 0
        for i in results:
            rst_file.write(str(i))
            rst_file.write("\n")
            if i == 0:
                count += 1
        rst_file.close()
        print count
        return

if __name__ == "__main__":
    sm = SVMModel()
    sm.model()