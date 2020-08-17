package com.traveldata.leimly.predictmodel

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.{SparkConf, SparkContext}

import scala.reflect.io.{Path => ScalaPath}

/**
  * Created by lizm on 17-8-13.
  */
object DeleteMain {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("com.traveldata.leimly.getFeatures.DataProcess")
    //        val conf = new SparkConf().setMaster("local").setAppName("com.traveldata.leimly.getFeatures.DataProcess")
    val sc = new SparkContext(conf)
    judgeLocalPathExist(sc, args(0))
    judgeLocalPathExist(sc, args(3))
    judgeLocalPathExist(sc, args(4))
  }

  def judgeLocalPathExist(sc: SparkContext, pathStr: String): Unit = {

    val sPath: ScalaPath = ScalaPath(pathStr)
    if (sPath.exists) {
      sPath.deleteRecursively()
    }
    try {
      val conf = new Configuration()
      println(conf.getRaw("fs.default.name"))
      conf.set("fs.default.name", "hdfs://")
      println(conf.getRaw("fs.default.name"))
      val fileSystem = FileSystem.get(conf)

      val hPath: Path = new Path(pathStr)
      if (fileSystem.exists(hPath)) {
        fileSystem.delete(hPath, true)
      }
    } catch {
      case e: Exception => println(e.getMessage)
    }
  }

}

