package main

import etl.DataProcessing
import org.apache.log4j.Logger

class spark_main extends Serializable {
  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)
}

object spark_main extends Serializable{

  def main(args: Array[String]): Unit = {

    println("testing main")
    val dummy = args(0)
    dummy match {
      case "GrabData" => DataProcessing.getParquet(args(1))
      case _ => throw new ClassNotFoundException(s"$dummy class does not exist !")
    }
  }
}


