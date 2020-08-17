package com.traveldata.leimly.predictmodel

import org.apache.spark.mllib.linalg.Vector

/**
  * Created by lizm on 17-7-29.
  */
object CaseClass {
  case class RawDataRecord(id: Int, track: List[TrackPoint], des: Array[Double], label: Int)
  case class ResultFeatures(id: Int, label: Int, features: Array[Double])

  case class TrackPoint(x: Double, y: Double, t: Double)
  case class TestFeatures(id: Int, features: Vector)
  case class Result(id: Int, label: Int)
}
