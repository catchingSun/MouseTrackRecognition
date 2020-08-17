package com.traveldata.leimly.getFeatures

import com.traveldata.leimly.getFeatures.CaseClass.{RawDataRecord, ResultFeatures, TrackPoint}
import breeze.numerics._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by lizm on 17-7-28.
  */

class FeaturesExtract(rawData: RDD[String]) extends Serializable {
  val dp = new DataProcess() with Serializable


  def constructFeatures(): RDD[ResultFeatures] = {

    val raw_data = dp.processRawData(rawData)
    val reFeature = raw_data.map {
      trackPath => getFeaturesVector(trackPath)
    }
    return reFeature
  }

  def getFeaturesVector(trackPath: RawDataRecord): ResultFeatures = {

    val tt = trackPath.track
    var temp: List[Double] = List()
    val len = tt.length
    val startDir = calculateAngle(tt.head, tt(1))
    temp = startDir :: temp
    val endDir = calculateAngle(tt(len - 2), tt.last)
    temp = endDir :: temp

    temp = calculateCornerNum(tt) :: temp

    temp = calculateVFragmentX(tt) ::: temp

    temp = calculatePointIntensive(tt) ::: temp

    temp = calculateVFragmentY(tt) ::: temp

    temp = calculateLinearDir(tt) ::: temp

    //    temp = calculateInalCornerDir(tt) ::: temp

    //    val etdDir = calculateEtdDir(trackPath)
    //    temp = etdDir :: temp

    temp = calculateVStats(tt) ::: temp


    //    temp = calculateVStatsX(tt) ::: temp

    //    temp = calculateVStatsY(tt) ::: temp

    temp = calculateDisStatsX(tt) ::: temp

    //    temp = calculateDisStatsY(tt) ::: temp


    /*    val totalDisX = trackPath.track.last.x - trackPath.track.head.x
        temp = totalDisX :: temp
        val totalDisY = trackPath.track.last.y - trackPath.track.head.y
        temp = totalDisY :: temp
        val totalT = trackPath.track.last.t - trackPath.track.head.t
        temp = totalT :: temp
        val totalVX = totalDisX / totalT
        temp = totalVX :: temp
        val totalVY = totalDisY / totalT
        temp = totalVY :: temp

        val etdX = trackPath.track.last.x - trackPath.des(0)
        temp = etdX :: temp
        val etdY = trackPath.track.last.y - trackPath.des(1)
        temp = etdY :: temp*/


    temp = temp.map(x => judegNanInf(x))

    return ResultFeatures(trackPath.id, temp.toArray)

  }

  def judegNanInf(x: Double): Double = {
    if (x.isInfinity || x.isNaN) {
      return 0
    }
    else return x
  }

  def calculateRegressionXT(rawData: RDD[RawDataRecord]): Unit = {
    val lineRTrainData = rawData.map(track => track.track.map(point => LabeledPoint(point.x, Vectors.dense(point.t))))

    //    val model = LinearRegressionWithSGD.train(lineRTrainData, 10, 0.1)
  }

  def calculateEtdDir(trackPath: RawDataRecord): Double = {

    val dis = sqrt(pow(calculateOdis(trackPath.track.last.x, trackPath.des(0)), 2) + pow(calculateOdis(trackPath.track.last.y, trackPath.des(1)), 2))
    val angle = asin(calculateOdis(trackPath.track.last.y, trackPath.des(1)) / dis)
    return angle
  }

  def calculateVFragmentY(track: List[TrackPoint]): List[Double] = {
    var tempFeatures: List[Double] = List()
    val temp = track.last.t / 4
    var tempS = 0.0
    var tempE = temp
    var sum = 0.0
    var loopCount = 1
    for (i <- 0 to 3) {
      tempS = i * temp
      tempE = (i + 1) * temp
      sum = 0
      while (loopCount < track.length && track(loopCount).t + 1 <= tempE) {
        if ((tempS <= track(loopCount).t) && (track(loopCount).t <= tempE)) {
          sum += calculateOdis(track(loopCount - 1).y, track(loopCount).y)
        }
        loopCount += 1
      }
      tempFeatures = (sum / temp) :: tempFeatures
    }

    return tempFeatures

  }

  def calculateVFragmentX(track: List[TrackPoint]): List[Double] = {
    var tempFeatures: List[Double] = List()
    val temp = track.last.t / 4
    var tempS = 0.0
    var tempE = temp
    var sum = 0.0
    var loopCount = 1
    for (i <- 0 to 3) {
      tempS = i * temp
      tempE = (i + 1) * temp
      sum = 0
      while (loopCount < track.length && track(loopCount).t + 1 <= tempE) {
        if ((tempS <= track(loopCount).t) && (track(loopCount).t <= tempE)) {
          sum += calculateOdis(track(loopCount - 1).x, track(loopCount).x)
        }
        loopCount += 1
      }
      tempFeatures = (sum / temp) :: tempFeatures
    }

    return tempFeatures

  }

  def calculatePointIntensive(track: List[TrackPoint]): List[Double] = {

    var tempFeatures: List[Double] = List()
    val temp = track.last.t / 4
    var tempS = 0.0
    var tempE = temp
    var tempPointNum = 0
    var loopCount = 1
    for (i <- 0 to 3) {
      tempS = i * temp
      tempE = (i + 1) * temp
      tempPointNum = 0
      var dis = 0.0
      while (loopCount < track.length && track(loopCount).t + 1 <= tempE) {
        if ((tempS <= track(loopCount).t) && (track(loopCount).t <= tempE)) {
          tempPointNum += 1
          dis += track(loopCount).x
        }
        loopCount += 1
      }
      tempFeatures = (tempPointNum) :: tempFeatures
    }
    return tempFeatures

  }

  def calculateLinearDir(track: List[TrackPoint]): List[Double] = {
    var aDir: Array[Double] = new Array[Double](4)
    /*
        for (i <- 0 to track.length - 3) {

          val j = i + 1
          val k = i + 2
          if (track(i).y != track(j).y || (track(k).y != track(j).y)) {
            val tmp_angle = calculateAngle(track(i), track(j))
            calculateDirNum(tmp_angle, aDir)
          }
        }*/

    for (i <- 0 to track.length - 2) {

      val j = i + 1

      val tmp_angle = calculateAngle(track(i), track(j))
      calculateDirNum(tmp_angle, aDir)
    }
    /*
        val end_angle = calculateAngle(track(track.length - 2), track(track.length - 1))
        calculateDirNum(end_angle, aDir)*/
    return aDir.toList
  }

  def calculateCornerNum(track: List[TrackPoint]): Double = {
    var iftp_num = 0
    for (i <- 1 to track.length - 2) {

      val j = i + 1
      val k = i - 1
      if (track(i).y != track(j).y || (track(i).y != track(j).y)) {
        iftp_num += 1
      }
    }
    return iftp_num

  }

  def calculateVStats(track: List[TrackPoint]): List[Double] = {

    var temp: List[Double] = List()
    var tempV: List[Double] = List()
    for (i <- 0 to track.length - 2) {

      val j = i + 1
      tempV = calculateDis(track(i), track(j)) / (track(j).t - track(i).t) :: tempV

    }
    val tempVCount = tempV.length
    //        val internalVMean = tempV.sum / tempVCount
    //        temp = internalVMean :: temp
    //        val internalVStdev = tempV.map(v => pow((v - internalVMean), 2)).sum / tempVCount
    //        temp = internalVStdev :: temp

    tempV = tempV.sorted
    var internalMedianV = 0.0
    if (tempVCount % 2 == 0) {
      internalMedianV = (tempV(tempVCount / 2 - 1) + tempV(tempVCount / 2)) / 2
    } else {
      internalMedianV = tempV(tempVCount / 2)
    }
    temp = internalMedianV :: temp
    //    val extremumV = tempV.max - tempV.min
    //    temp = extremumV :: temp

    return temp
  }

  def calculateVStatsX(track: List[TrackPoint]): List[Double] = {
    var tempFeatures: List[Double] = List()
    var eachParagraphV: List[Double] = List()
    var eachParagraphA: List[Double] = List()
    for (i <- 0 to track.length - 2) {

      val j = i + 1
      val tempV = (track(j).x - track(i).x) / (track(j).t - track(i).t)
      eachParagraphV = tempV :: eachParagraphV
      eachParagraphA = 2 * (2 * tempV - (track(j).x - track(i).x)) / pow((track(j).t - track(i).t), 2) :: eachParagraphA

    }
    tempFeatures = calculateStats(eachParagraphV) ::: tempFeatures
    tempFeatures = calculateStats(eachParagraphA) ::: tempFeatures

    return tempFeatures
  }


  def calculateVStatsY(track: List[TrackPoint]): List[Double] = {
    var tempFeatures: List[Double] = List()
    var eachParagraphV: List[Double] = List()
    var eachParagraphA: List[Double] = List()
    for (i <- 0 to track.length - 2) {

      val j = i + 1
      val tempV = (track(j).y - track(i).y) / (track(j).t - track(i).t)
      eachParagraphV = tempV :: eachParagraphV
      eachParagraphA = 2 * (2 * tempV - (track(j).y - track(i).y)) / pow((track(j).t - track(i).t), 2) :: eachParagraphA

    }
    tempFeatures = calculateStats(eachParagraphV) ::: tempFeatures
    tempFeatures = calculateStats(eachParagraphA) ::: tempFeatures

    return tempFeatures
  }

  def calculateDisStatsX(track: List[TrackPoint]): List[Double] = {
    var tempFeatures: List[Double] = List()
    var eachParagraphDis: List[Double] = List()
    for (i <- 0 to track.length - 2) {
      val j = i + 1
      val temp = calculateOdis(track(i).x, track(j).x)
      eachParagraphDis = temp :: eachParagraphDis
    }
    tempFeatures = calculateStats(eachParagraphDis) ::: tempFeatures
    return tempFeatures
  }

  def calculateDisStatsY(track: List[TrackPoint]): List[Double] = {
    var tempFeatures: List[Double] = List()
    var eachParagraphDis: List[Double] = List()
    for (i <- 0 to track.length - 2) {
      val j = i + 1
      val temp = calculateOdis(track(i).y, track(j).y)
      eachParagraphDis = temp :: eachParagraphDis
    }
    tempFeatures = calculateStats(eachParagraphDis) ::: tempFeatures
    return tempFeatures
  }

  def calculateStats(sequence: List[Double]): List[Double] = {
    var tempFeatures: List[Double] = List()
    val sequenceCount = sequence.length
    val sequenceMean = sequence.sum / sequenceCount
    //      tempFeatures = sequenceMean :: tempFeatures
    val sequenceStdev = sequence.map(y => pow((y - sequenceMean), 2)).sum / sequenceCount
    tempFeatures = sequenceStdev :: tempFeatures

    val max = sequence.max
    //        tempFeatures = max :: tempFeatures
    val min = sequence.min
    //        tempFeatures = min :: tempFeatures
    val extremum = max - min
    tempFeatures = extremum :: tempFeatures
    val sequenceSorted = sequence.sorted

    var sequenceMedian = 0.0
    if (sequenceCount % 2 == 0) {
      sequenceMedian = (sequenceSorted(sequenceCount / 2 - 1) + sequenceSorted(sequenceCount / 2)) / 2
    } else {
      sequenceMedian = sequenceSorted(sequenceCount / 2)
    }
    tempFeatures = sequenceMedian :: tempFeatures
    return tempFeatures
  }

  def calculateInalCornerDir(track: List[TrackPoint]): List[Double] = {
    var iDir: Array[Double] = new Array[Double](4)
    for (i <- 1 to track.length - 2) {
      val j = i - 1
      val k = i + 1

      if (track(i).y != track(j).y || (track(i).y != track(k).y)) {

        val tmpIAngle = (math.Pi / 2 - calculateInalAngle(track(j), track(i))) + (math.Pi / 2 - calculateInalAngle(track(i), track(k)))
        //      println(tmpIAngle)
        if ((tmpIAngle >= 0) && (tmpIAngle < math.Pi / 4)) {
          iDir(0) += 1
        } else if ((tmpIAngle >= math.Pi / 4) && (tmpIAngle < math.Pi / 2)) {
          iDir(1) += 1
        }
        else if ((tmpIAngle >= math.Pi / 2) && (tmpIAngle < 3 * math.Pi / 4)) {
          iDir(2) += 1
        }
        else if ((tmpIAngle >= 3 * math.Pi / 4) && (tmpIAngle < math.Pi)) {
          iDir(3) += 1
        }
      }
    }
    return iDir.toList
  }

  def calculateDirNum(tmp_angle: Double, aDir: Array[Double]): Array[Double] = {
    if ((tmp_angle >= 0) && (tmp_angle < math.Pi / 4)) {
      aDir(0) += 1
    } else if ((tmp_angle >= math.Pi / 4) && (tmp_angle < math.Pi / 2)) {
      aDir(1) += 1
    }
    else if ((tmp_angle >= math.Pi / 2) && (tmp_angle < 3 * math.Pi / 4)) {
      aDir(2) += 1
    }
    else if ((tmp_angle >= 3 * math.Pi / 4) && (tmp_angle < math.Pi)) {
      aDir(3) += 1
    }
    else if ((tmp_angle >= math.Pi) && (tmp_angle < 5 * math.Pi / 4)) {
      aDir(4) += 1
    }
    else if ((tmp_angle >= 5 * math.Pi / 4) && (tmp_angle < 3 * math.Pi / 2)) {
      aDir(5) += 1
    }
    else if ((tmp_angle >= 3 * math.Pi / 2) && (tmp_angle < 7 * math.Pi / 4)) {
      aDir(6) += 1
    }
    else if ((tmp_angle >= 7 * math.Pi / 4) && (tmp_angle < 2 * math.Pi)) {
      aDir(7) += 1
    }
    return aDir
  }


  def calculateInalAngle(point1: TrackPoint, point2: TrackPoint): Double = {
    return asin(abs(calculateOdis(point1.y, point2.y)) / calculateDis(point1, point2))
  }


  def calculateAngle(point1: TrackPoint, point2: TrackPoint): Double = {
    val dis = calculateDis(point1, point2)
    val angle = asin(calculateOdis(point1.y, point2.y) / dis)
    return angle
  }

  def calculateDis(point1: TrackPoint, point2: TrackPoint): Double = {
    val dis = sqrt(pow(calculateOdis(point1.x, point2.x), 2) + pow(calculateOdis(point1.y, point2.y), 2))
    return dis
  }

  def calculateOdis(c1: Double, c2: Double): Double = {
    return c2 - c1
  }

  /*
    def calculateInternalVX(point1 : TrackPoint, point2 : TrackPoint): Double ={
      val vx = (point2.x - point1.x) / (point2.t - point1.t)
      return vx
    }

    def calculateInternalVY(point1 : TrackPoint, point2 : TrackPoint): Double ={
      val vy = (point2.y - point1.y) / (point2.t - point1.t)
      return vy
    }

    def startDirection(point1: TrackPoint, point2: TrackPoint): Double ={
      return calculateDis(point1, point2)
    }

    def endDirection(point1: TrackPoint, point2: TrackPoint): Double ={
      return calculateDis(point1, point2)
    }

    def totalDis(track: Array[TrackPoint]): List[Double] ={
      val tempT : Double = track.map(_.t).sum
      var tempD : List[Double] = List()
      var temp : List[Double] = List()
      for(i <- 0 to track.length - 2){
        val j = i + 1
        tempD = calculateDis(track(i), track(j)) :: tempD
      }
      temp = tempD.sum :: temp
      temp = tempT :: temp
      return temp
    }

    def totalDisX(track: Array[TrackPoint]): Double ={
      var temp : List[Double] = List()
      for(i <- 0 to track.length - 2){
        val j = i + 1
        temp = calculateOdis(track(i).x, track(j).x) :: temp
      }
      return temp.sum
    }

    def totalDisY(track: Array[TrackPoint]): Double ={
      var temp : List[Double] = List()
      for(i <- 0 to track.length - 2){
        val j = i + 1
        temp = calculateOdis(track(i).y, track(j).y) :: temp
      }
      return temp.sum
    }

    def totalSpeed(dis: Double, time: Double): Double ={
      return dis / time
    }

    def totalSpeedX(disX: Double, time: Double): Double ={

      return disX / time
    }

    def totalSpeedY(disY: Double, time: Double): Double ={

      return disY / time
    }*/


}
