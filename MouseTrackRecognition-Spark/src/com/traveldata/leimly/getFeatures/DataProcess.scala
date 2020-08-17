package com.traveldata.leimly.getFeatures

import com.traveldata.leimly.getFeatures.CaseClass.{RawDataRecord, TrackPoint}
import org.apache.spark.rdd.RDD

/**
  * Created by lizm on 17-7-28.
  */
class DataProcess extends Serializable {


  def processRawData(rawData: RDD[String]): RDD[RawDataRecord]= {

    val result = rawData.map{
      line =>
        val data = line.split(" ")
        RawDataRecord(
          data(0).toInt,
          data(1).split(";").map{
            track => val temp = track.split(",")
              TrackPoint(temp(0).toDouble, temp(1).toDouble, temp(2).toDouble)}
            .sortBy(_.t).toList,
          data(2).split(",").map{_.toDouble}
        )
    }.filter(_.track.length >= 2)
    return result
  }


}
