package com.traveldata.leimly.predictmodel

import com.traveldata.leimly.predictmodel.CaseClass.{RawDataRecord, TrackPoint}
import org.apache.spark.rdd.RDD

/**
  * Created by lizm on 17-7-28.
  */
class DataProcess {


  def processRawData(rawData: RDD[String]): RDD[RawDataRecord]= {

    val result = rawData.map{
      line =>
        val data = line.split(" ")
        RawDataRecord(
          data(0).trim.toInt,
          data(1).split(";").map{
            track => val temp = track.split(",")
              TrackPoint(temp(0).trim.toDouble, temp(1).trim.toDouble, temp(2).trim.toDouble)}
            .sortBy(_.t).toList,
          data(2).split(",").map{_.toDouble}
          ,data(3).trim.toInt
        )
    }.filter(_.track.length >= 2)
    return result
  }


}
