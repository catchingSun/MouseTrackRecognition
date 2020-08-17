from construct_the_eigenvector import *
from process_data import *
from pybrain.structure import SoftmaxLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError


class NeurolNetworksModel:

    eigenvector = list()
    test_data = list()
    output_train = list()

    vct_len = 0

    def __init__(self):
        prd = DataProcess()
        train_raw_data_path = "../../Datas/RawDatas/dsjtzs_txfz_training.txt"
        cte = ConstructEigenvector(train_raw_data_path)
        self.eigenvector = cte.construct_the_eigenvector()
        self.output_train = np.array(list(map(int, prd.output_train)))
        self.eigenvector = np.array(self.eigenvector)
        # print self.eigenvector
        test_raw_data_path = "../../Datas/RawDatas/dsjtzs_txfz_test1.txt"
        cte1 = ConstructEigenvector(test_raw_data_path)
        self.test_data = cte1.construct_the_eigenvector()
        self.test_data = np.array(self.test_data)

        self.vct_len = len(self.eigenvector[0])

        # print self.test_data
        return

    def construct_net(self):
        data = self.consturt_train_data()
        trndata = data[0]
        tstdata = data[1]
        test_data = data[2]
        ds = data[3]
        print self.vct_len
        net = buildNetwork(self.vct_len, 30, 2, outclass=SoftmaxLayer)
        trainer = BackpropTrainer(net, trndata, momentum=0.01, verbose=True, weightdecay=0.001)
        err_train = trainer.trainUntilConvergence(maxEpochs=20000)
        tstresult = percentError(trainer.testOnClassData(), tstdata['target'])
        print tstresult
        out = net.activateOnDataset(test_data)
        out = out.argmax(axis=1)  # the highest output activation gives the class
        print out
        path = "../../Results/output.txt"
        mse_file = open(path, 'w')
        for i in out:
            mse_file.write(str(i))
            mse_file.write("\n")
        mse_file.close()

        return

    def consturt_train_data(self):

        # print len(self.output_train)
        # print len(self.eigenvector)
        ds = ClassificationDataSet(self.vct_len, 1, nb_classes=2)
        for i in range(len(self.output_train)):
            ds.appendLinked(self.eigenvector[i], self.output_train[i])
        # print ds
        # print ds
        ds.calculateStatistics()

        # split training, testing, validation data set (proportion 4:1)
        tstdata_temp, trndata_temp = ds.splitWithProportion(0.25)
        tstdata = ClassificationDataSet(self.vct_len, 1, nb_classes=2)
        for n in range(0, tstdata_temp.getLength()):
            tstdata.appendLinked(tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1])

        trndata = ClassificationDataSet(self.vct_len, 1, nb_classes=2)
        for n in range(0, trndata_temp.getLength()):
            trndata.appendLinked(trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1])
        # one hot encoding
        # print trndata
        testdata = ClassificationDataSet(self.vct_len, 1, nb_classes=2)
        test_data_temp = self.test_data
        for n in range(len(test_data_temp)):
            testdata.addSample(test_data_temp[n], [0])
        # print testdata
        trndata._convertToOneOfMany()
        tstdata._convertToOneOfMany()
        testdata._convertToOneOfMany()
        return trndata, tstdata, testdata, ds


if __name__ == '__main__':
    nnm = NeurolNetworksModel()
    nnm.construct_net()