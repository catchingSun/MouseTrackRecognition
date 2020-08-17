package com.traveldata.leimly.predictmodel

import com.traveldata.leimly.predictmodel.CaseClass.{Result, TestFeatures}
//import ml.dmlc.xgboost4j.scala.spark.XGBoost
//import ml.dmlc.xgboost4j.scala.DMatrix
//import ml.dmlc.xgboost4j.scala.spark.{XGBoost, XGBoostModel}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{GradientBoostedTrees, RandomForest}
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.{GradientBoostedTreesModel, RandomForestModel}
import org.apache.spark.ml.feature.{LabeledPoint => MLLabeledPoint}
//import ml.dmlc.xgboost4j.scala.{Booster, DMatrix}
import org.apache.spark.ml.linalg.{DenseVector => MLDenseVector}
import org.apache.spark.rdd.RDD


/**
  * Created by lizm on 17-7-28.
  **/
class Model {
  def model(trainFeatures: RDD[LabeledPoint],
            testFeatures: RDD[TestFeatures],
            xgbTrainFeatures: RDD[MLLabeledPoint],
            xgbTestFeatures: Array[MLDenseVector]): RDD[Result] = {
    //    val model = SVMWithSGD.train(trainFeatures, 10000)

    val split = trainFeatures.randomSplit(Array(0.8, 0.2), seed = 11L)
    val parsedTrain = split(0)
    //split(0)
    //split(0)
    val parsedTest = split(1) // split(1)

    //        val result = lrModel(parsedTrain, parsedTest, testFeatures)
    val result = svmModel(parsedTrain, parsedTest, testFeatures)
    //        val result = rfdTreeModel(parsedTrain, parsedTest, testFeatures)
    //        val result = gbtModel(parsedTrain, parsedTest, testFeatures)
    //        val result = xgboostModel(xgbTrainFeatures, xgbTestFeatures)
    //    val result = SVMWithSGD.train(trainFeatures, 1000)
    return result
  }

  def svmModel(parsedTrain: RDD[LabeledPoint], parsedTest: RDD[LabeledPoint], testFeatures: RDD[TestFeatures]): RDD[Result] = {
    //    val model = SVMWithSGD.train(parsedTrain, 10000)
    val svmAlg = new SVMWithSGD()
    svmAlg.optimizer
      .setNumIterations(10000)
      .setRegParam(0.01) //08.0099
      .setStepSize(3)
    //      .setMiniBatchFraction(0.9)
    // 0.9
    val model = svmAlg.run(parsedTrain)
    val preAndLabels = parsedTest.map {
      case LabeledPoint(label, features) =>
        val pre = model.predict(features)
        (pre, label)
    }
    println(verifyTestAccurancy(preAndLabels))
    val result = testFeatures.map {
      temp =>
        val pre = model.predict(temp.features)
        Result(temp.id, pre.toInt)
    }.filter(_.label == 0)
    return result
  }

  def lrModel(parsedTrain: RDD[LabeledPoint], parsedTest: RDD[LabeledPoint], testFeatures: RDD[TestFeatures]): RDD[Result] = {
    val lrModel = LogisticRegressionWithSGD.train(parsedTrain, 10000)
    val preAndLabels = parsedTest.map {
      case LabeledPoint(label, features) =>
        val pre = lrModel.predict(features)
        (pre, label)
    }
    println(verifyTestAccurancy(preAndLabels))
    val result = testFeatures.map {
      temp =>
        val pre = lrModel.predict(temp.features)
        Result(temp.id, pre.toInt)
    }.filter(_.label == 0)
    return result
  }

  def rfdTreeModel(parsedTrain: RDD[LabeledPoint], parsedTest: RDD[LabeledPoint], testFeatures: RDD[TestFeatures]): RDD[Result] = {
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()

    val featureSubsetStrategy = "auto"
    val impurity = "entropy"
    // 501 3 8
    var numTree: Int = 301
    var maxDepth: Int = 3
    var maxBins: Int = 10
    //    var
    var maxAccurancy = 0.0

    var printMaxDepth = 0.0
    var printNumTree = 0.0
    var printMaxBIns = 0.0

    var model: RandomForestModel = null
    //    for (maxDepth <- 1 to 3; numTree <- 1 to 2201 if (numTree / 50) == 1; maxBins <- 3 to 25) {
    val rfdModel = RandomForest
      .trainClassifier(
        parsedTrain, numClasses,
        categoricalFeaturesInfo, numTree,
        featureSubsetStrategy, impurity,
        maxDepth, maxBins)

    val preAndLabels = parsedTest.map {
      case LabeledPoint(label, features) =>
        val pre = rfdModel.predict(features)
        (pre, label)
    }

    val precision = verifyTestAccurancy(preAndLabels)
    if (precision > maxAccurancy) {
      maxAccurancy = precision
      printMaxDepth = maxDepth
      printNumTree = numTree
      printMaxBIns = maxBins
      model = rfdModel
      //      }
    }
    println("maxDepth = " + printMaxDepth + "numTree = " + printNumTree + "maxBins = " + printMaxBIns)
    println("Precision = " + maxAccurancy)

    val result = testFeatures.map {
      temp =>
        val pre = model.predict(temp.features)
        Result(temp.id, pre.toInt)
    }.filter(_.label == 0)
    return result
  }

  def bayesModel(parsedTrain: RDD[LabeledPoint]): NaiveBayesModel = {
    val bayesModel = NaiveBayes.train(parsedTrain, 1.0)
    return bayesModel
  }

  def gbtModel(parsedTrain: RDD[LabeledPoint], parsedTest: RDD[LabeledPoint], testFeatures: RDD[TestFeatures]): RDD[Result] = {
    var boostStrategy = BoostingStrategy.defaultParams("Classification")
    boostStrategy.numIterations = 101
    boostStrategy.treeStrategy.numClasses = 2
    boostStrategy.treeStrategy.maxDepth = 2
    boostStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    val model = GradientBoostedTrees.train(parsedTrain, boostStrategy)
    val preAndLabels = parsedTest.map {
      case LabeledPoint(label, features) =>
        val pre = model.predict(features)
        (pre, label)
    }

    val precision = verifyTestAccurancy(preAndLabels)
    println(precision)
    val result = testFeatures.map {
      temp =>
        val pre = model.predict(temp.features)
        Result(temp.id, pre.toInt)
    }.filter(_.label == 0)
    return result

  }

  /*  def xgboostModel(xgbTrainFeatures: RDD[MLLabeledPoint],
                     xgbTestFeatures: Array[MLDenseVector]): RDD[Array[Float]] = {
      val split = xgbTrainFeatures.randomSplit(Array(0.8, 0.2), seed = 11L)
      val parsedTrain = split(0)
      //split(0)
      //split(0)
      val parsedTest = split(1)


      val paramMap = List(
        "eta" -> 0.1f,
        "max_depth" -> 2,
        "objective" -> "binary:logistic").toMap
      println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

      val model = XGBoost.trainWithRDD(parsedTrain, paramMap, nWorkers = 2, round= 2)//5, 5, useExternalMemory = true
            val preAndLabels = parsedTest.map {
              case MLLabeledPoint(label, features) =>
                val pre = model.booster.predict(new DMatrix(features))
                (pre, label)
            }
            val metrics = new MulticlassMetrics(parsedTest)
      println("-------------------------")
      val result = model.predict(parsedTest.map(_.features))
      println("-----*****************--------")
      println(result)
      return result
    }*/

  def verifyTestAccurancy(preAndLabels: RDD[(Double, Double)]): Double = {

    val metrics = new MulticlassMetrics(preAndLabels)
    val precision = metrics.recall(0)
    return precision
  }
}

