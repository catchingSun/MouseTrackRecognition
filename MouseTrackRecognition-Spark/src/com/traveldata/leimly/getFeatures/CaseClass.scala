package com.traveldata.leimly.getFeatures

import org.apache.spark.mllib.linalg.Vector

/**
  * Created by lizm on 17-7-29.
  */
object CaseClass {
  case class TrackPoint(x: Double, y: Double, t: Double)
  case class RawDataRecord(id: Int, track: List[TrackPoint], des: Array[Double])
  case class ResultFeatures(id: Int, features: Array[Double])
  case class TestFeatures(id: Int, features: Vector)
  case class Result(id: Int, label: Int)
}
