package etl

import org.apache.spark.sql.{DataFrame, SparkSession}


object DataProcessing {
    println("larkin")

    def getParquet(parquetPath: String): DataFrame = {
        val spark = SparkSession.builder().getOrCreate()
        println("read parquet!")
        spark.read.parquet(parquetPath)
    }

    def writeToCSV(df: DataFrame, fileName: DataFrame): Unit = {
        val spark = SparkSession.builder().getOrCreate()
        println("writing dataframe to csv!")
        df.write.format("com.databricks.spark.csv").option("header", "true").save("mydata.csv")
    }

}