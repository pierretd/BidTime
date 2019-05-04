// By using Historical CTR features, linear regression achieved a AUC of 0.7589

// COMMAND ----------

val toInt = udf[Int, String](_.toInt)

// COMMAND ----------

 val sampleData = spark.read.option("header", "true").csv(
            "/FileStore/tables/avazu_sample_1mil.csv").withColumn(
            "hour_int", toInt($"hour"))


// COMMAND ----------

sampleData.count

// COMMAND ----------

sampleData.printSchema

// COMMAND ----------

 sampleData.select("hour_int").describe().show


// COMMAND ----------

val pastData = sampleData.filter($"hour_int" < 14102100 + 300)
val currentData =  sampleData.filter($"hour_int" >= 14102100 + 300)

// COMMAND ----------

import org.apache.spark.sql.functions._

val pastDataSiteCategory = pastData.select("id", "click", "site_category").withColumn("click", toInt($"click")) 

// Get number of clicks per banner_pos
        val clickCount = pastDataSiteCategory.groupBy($"site_category".as("site_category_feature")
            ).agg(sum("click").alias("clicks"), 
            count("id").alias("impressions")
            ).withColumn("site_category_ctr", $"clicks"/$"impressions")
        
        val currentDataWithCTRFeatures = pastData.join(
            clickCount, 
            pastData("site_category") === clickCount("site_category_feature"), 
            "left_outer")

currentDataWithCTRFeatures.show(1)
        

// COMMAND ----------


val pastDataSubBanner = pastData.select("id", "click", "banner_pos").withColumn("click", toInt($"click")) 
  // Get number of clicks per banner_pos
        val clickCount1 = pastDataSubBanner.groupBy($"banner_pos".as("banner_pos_feature")
            ).agg(sum("click").alias("clicks"), 
            count("id").alias("impressions")
            ).withColumn("banner_ctr", $"clicks"/$"impressions")
        val currentDataWithCTRFeatures1 = currentDataWithCTRFeatures.join(
            clickCount1, 
            currentDataWithCTRFeatures("banner_pos") === clickCount1("banner_pos_feature"), 
            "left_outer")

currentDataWithCTRFeatures1.show(1)        
      

// COMMAND ----------

// Todo-More features and data cleaning to improve score

// COMMAND ----------

val ctrPrediction = currentDataWithCTRFeatures1

// COMMAND ----------

val stringFeatures = Array("site_domain","site_category","app_domain","app_category","device_model")
val catCol = Array("C1","banner_pos","device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21", "site_category_ctr","banner_ctr" )
val stringCatFeatures = stringFeatures ++ catCol
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
val catFeaturesIndexor = stringCatFeatures.map(
	cname => new StringIndexer()
		.setInputCol(cname)
		.setOutputCol(s"${cname}_index")
)

// COMMAND ----------

import org.apache.spark.ml.Pipeline
val indexPipeline = new Pipeline().setStages(catFeaturesIndexor)
val model = indexPipeline.fit(ctrPrediction)
val indexedDf = model.transform(ctrPrediction)

// COMMAND ----------

// One Hot Encoding for categorical features
val indexedCols = indexedDf.columns.filter(x => x contains "index")


// COMMAND ----------

val indexedFeatureEncoder = indexedCols.map(
	indexed_cname => new OneHotEncoder()
		.setInputCol(indexed_cname)
		.setOutputCol(s"${indexed_cname}_vec"))

// COMMAND ----------

val encodedPipeline = indexPipeline.setStages(indexedFeatureEncoder)

// COMMAND ----------

val encodeModel = encodedPipeline.fit(indexedDf)

// COMMAND ----------

val encodedDf = encodeModel.transform(indexedDf)

// COMMAND ----------

// Remove features that hurt score and keep only vectorized featueres
val nonFeatureCol = Array("id", "click", "site_id", "app_id", "device_id", "device_ip") // not using them

val featureCol = encodedDf.columns.filter(x => x contains "_vec")

// COMMAND ----------

// Declare assembler
import org.apache.spark.ml.feature.VectorAssembler
val assembler = new VectorAssembler().setInputCols(featureCol).setOutputCol("features")
val encodedTrainingSet = assembler.transform(encodedDf)

// COMMAND ----------

val finalTrainSet = encodedTrainingSet.selectExpr("*", "double(click) as click_output")


// COMMAND ----------

// Test Train Split
val Array(training, test) = finalTrainSet.randomSplit(Array(0.7, 0.3))


// COMMAND ----------

// training.cache()
// test.cache()

// COMMAND ----------

// Train the model
import org.apache.spark.ml.classification.LogisticRegression
val logisticRegModel = new LogisticRegression().setLabelCol("click_output").setFeaturesCol("features")
val lrFitted = logisticRegModel.fit(training)

// COMMAND ----------

// Test the Model
val holdout = lrFitted.transform(test)
val holdoutResult = holdout.selectExpr("id", "prediction", "click_output")
holdoutResult.cache()
val ranked = holdoutResult.filter(holdoutResult("prediction").between(0.1, 0.9))

// COMMAND ----------

// Re-assign for reproduction
val lrModel = lrFitted

// COMMAND ----------

val trainingSummary = lrModel.binarySummary

// COMMAND ----------

val objectiveHistory = trainingSummary.objectiveHistory
println("objectiveHistory:")
objectiveHistory.foreach(loss => println(loss))

// COMMAND ----------

val roc = trainingSummary.roc
roc.show()
println(s"areaUnderROC: ${trainingSummary.areaUnderROC}")

// COMMAND ----------

// Save model for future use
lrModel.save("target/tmp/LogRegScala")

// COMMAND ----------

// Set the model threshold to maximize F-Measure-todo
val fMeasure = trainingSummary.fMeasureByThreshold
val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure)
  .select("threshold").head().getDouble(0)
lrModel.setThreshold(bestThreshold)
