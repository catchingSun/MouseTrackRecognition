package com.traveldata.leimly.predictmodel

import com.traveldata.leimly.getFeatures.{FeaturesExtract => TestFeaturesExtract}
import com.traveldata.leimly.predictmodel.CaseClass.TestFeatures
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.mllib.feature.{ChiSqSelector, Normalizer, StandardScaler}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.feature.{LabeledPoint => MLLabeledPoint}
import org.apache.spark.ml.linalg.{DenseVector => MLDenseVector}

import scala.reflect.io.{Path => ScalaPath}

//import org.apache.hadoop.fs.{FileSystem, Path}
/**
  * Created by lizm on 17-7-28.
  */

object Main {

  val model = new Model()

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("com.traveldata.leimly.getFeatures.DataProcess")
    //    val conf = new SparkConf().setMaster("local").setAppName("com.traveldata.leimly.getFeatures.DataProcess")
    val sc = new SparkContext(conf)
    judgeLocalPathExist(sc, args(0))
    judgeLocalPathExist(sc, args(3))
    judgeLocalPathExist(sc, args(4))
    produceTestDataLibSVM(sc, args(2), args(3))
    val trainFeatures = getTrainData(sc, args(1))
    //    trainFeatures.saveAsTextFile(args(4))
    MLUtils.saveAsLibSVMFile(trainFeatures, args(4))
    val testFeatures = getTestData(sc, args(3))


    val xgbTestFeatures = getXGBModelTestData(sc, args(3))
    val xgbTrainFeatures = getXGBModelTrainData(sc, args(4))

    getPredictedResult(trainFeatures, testFeatures, args(0), xgbTrainFeatures, xgbTestFeatures)
  }

  def produceTestDataLibSVM(sc: SparkContext, rawFilePath: String, outputTestFilePath: String): Unit = {
    val rawData = sc.textFile(rawFilePath)
    val fe = new TestFeaturesExtract(rawData)
    val feData = fe.constructFeatures()
    val testFeatures = feData.sortBy(_.id).map {
      temp =>
        LabeledPoint(temp.id, Vectors.dense(temp.features))
    }
    //    testFeatures.repartition(1).saveAsTextFile(outputTestFilePath)
    MLUtils.saveAsLibSVMFile(testFeatures.repartition(1), outputTestFilePath)
  }


  def judgeLocalPathExist(sc: SparkContext, pathStr: String): Unit = {
    val sPath: ScalaPath = ScalaPath(pathStr)
    if (sPath.exists) {
      sPath.deleteRecursively()
    }
    /*try {
      val fileSystem = FileSystem.get(new Configuration())
      val hPath : Path = new Path(pathStr)
      if (fileSystem.exists(hPath)) {
        fileSystem.delete(hPath, true)
      }
    }catch{
      case e: Exception => println(e.getMessage)
    }*/
  }

  def getXGBModelTrainData(sc: SparkContext, rawTrainFeaturePath: String): RDD[MLLabeledPoint] = {
    val trainFeatures = MLUtils.loadLibSVMFile(sc, rawTrainFeaturePath).map {
      lp =>
        MLLabeledPoint(lp.label, new MLDenseVector(lp.features.toArray))
    }
    return trainFeatures
  }


  def getXGBModelTestData(sc: SparkContext, rawTestFeaturePath: String): Array[MLDenseVector] = {
    val testFeatures = MLUtils.loadLibSVMFile(sc, rawTestFeaturePath).collect().map {
      lp => new MLDenseVector(lp.features.toArray)
    }
    return testFeatures
  }


  def getTrainData(sc: SparkContext, rawTrainDataPath: String): RDD[LabeledPoint] = {
    val rawTrainData = sc.textFile(rawTrainDataPath)
    val fe = new FeaturesExtract(rawTrainData)

    val trainFeatures = fe.constructFeatures().sortBy(_.id)
    val negativeData = trainFeatures.filter(_.label == 0)
    //    val sampleProportion = negativeData.count() / positiveData.count()
    //    val sampleNegativeData = negativeData.sample(withReplacement = false, sampleProportion, 11L)

    /*    var tempFeatures = trainFeatures
        for (i <- 1 to 2){
          tempFeatures = tempFeatures.union(negativeData)
        }
    //    tempFeatures = tempFeatures.union(negativeData.filter(x => (x.id > 2600) &&(x.id <= 2700)))
        val finalTrainFeatures = tempFeatures.map{
            temp => LabeledPoint(temp.label.toDouble, Vectors.dense(temp.features))
          }*/

    val returnFeatures = trainFeatures.map {
      temp => LabeledPoint(temp.label.toDouble, Vectors.dense(temp.features))
    }

    //    val features = vectorSelect(returnFeatures)
    //    val scaler = new StandardScaler(withMean = true, withStd = true).fit(returnFeatures.map(x => x.features))
    //    val features = returnFeatures.map(x => LabeledPoint((x.label), scaler.transform(x.features)))
    return returnFeatures
  }

  def getTestData(sc: SparkContext, testFeaturesPath: String): RDD[TestFeatures] = {
    val rawTestFeaturesData = MLUtils.loadLibSVMFile(sc, testFeaturesPath)
    val testFeatures = rawTestFeaturesData.map {
      line =>
        TestFeatures(line.label.toInt, line.features)
    }
    //    val scaler = new StandardScaler(withMean = true, withStd = true).fit(testFeatures.map(x => x.features))
    //    val features = testFeatures.map(x => TestFeatures((x.id), scaler.transform(x.features)))
    return testFeatures
  }

  def getPredictedResult(trainFeatures: RDD[LabeledPoint],
                         testFeatures: RDD[TestFeatures],
                         outputPath: String,
                         xgbTrainFeatures: RDD[MLLabeledPoint],
                         xgbTestFeatures: Array[MLDenseVector]): Unit = {
    val result = model.model(trainFeatures, testFeatures, xgbTrainFeatures, xgbTestFeatures)
    val printResult = result.repartition(1).sortBy(_.id).map(_.id)
    println(printResult.count())
    printResult.saveAsTextFile(outputPath)
  }


  def vectorSelect(trainFeatures: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    val selector = new ChiSqSelector(19)
    // Create ChiSqSelector model (selecting features)
    val transformer = selector.fit(trainFeatures)
    // Filter the top 50 features from each feature vector
    val filteredData = trainFeatures.map { lp =>
      LabeledPoint(lp.label, transformer.transform(lp.features))
    }

    return filteredData
  }
}
