package edu.gatech.cse6250.main

import java.text.SimpleDateFormat

import edu.gatech.cse6250.clustering.Metrics
import edu.gatech.cse6250.features.FeatureConstruction
import edu.gatech.cse6250.helper.{ CSVHelper, SparkHelper }
import edu.gatech.cse6250.model.{ Diagnostic, LabResult, Medication }
import edu.gatech.cse6250.phenotyping.T2dmPhenotype
import org.apache.spark.mllib.clustering.{ GaussianMixture, KMeans, StreamingKMeans }
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{ DenseMatrix, Matrices, Vector, Vectors }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.io.Source

/**
 * @author Hang Su <hangsu@gatech.edu>,
 * @author Yu Jing <yjing43@gatech.edu>,
 * @author Ming Liu <mliu302@gatech.edu>
 */
object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.{ Level, Logger }

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkHelper.spark
    val sc = spark.sparkContext
    //  val sqlContext = spark.sqlContext

    /** initialize loading of data */
    val (medication, labResult, diagnostic) = loadRddRawData(spark)
    val (candidateMedication, candidateLab, candidateDiagnostic) = loadLocalRawData

    /** conduct phenotyping */
    val phenotypeLabel = T2dmPhenotype.transform(medication, labResult, diagnostic)

    /** feature construction with all features */
    val featureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult),
      FeatureConstruction.constructMedicationFeatureTuple(medication))

    // =========== USED FOR AUTO GRADING CLUSTERING GRADING =============
    // phenotypeLabel.map{ case(a,b) => s"$a\t$b" }.saveAsTextFile("data/phenotypeLabel")
    // featureTuples.map{ case((a,b),c) => s"$a\t$b\t$c" }.saveAsTextFile("data/featureTuples")
    // return
    // ==================================================================

    val rawFeatures = FeatureConstruction.construct(sc, featureTuples)

    val (kMeansPurity, gaussianMixturePurity, streamingPurity) = testClustering(phenotypeLabel, rawFeatures)
    println(f"[All feature] purity of kMeans is: $kMeansPurity%.5f")
    println(f"[All feature] purity of GMM is: $gaussianMixturePurity%.5f")
    println(f"[All feature] purity of StreamingKmeans is: $streamingPurity%.5f")

    /** feature construction with filtered features */
    val filteredFeatureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic, candidateDiagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult, candidateLab),
      FeatureConstruction.constructMedicationFeatureTuple(medication, candidateMedication))

    val filteredRawFeatures = FeatureConstruction.construct(sc, filteredFeatureTuples)

    val (kMeansPurity2, gaussianMixturePurity2, streamingPurity2) = testClustering(phenotypeLabel, filteredRawFeatures)
    println(f"[Filtered feature] purity of kMeans is: $kMeansPurity2%.5f")
    println(f"[Filtered feature] purity of GMM is: $gaussianMixturePurity2%.5f")
    println(f"[Filtered feature] purity of StreamingKmeans is: $streamingPurity2%.5f")
  }

  def testClustering(phenotypeLabel: RDD[(String, Int)], rawFeatures: RDD[(String, Vector)]): (Double, Double, Double) = {
    import org.apache.spark.mllib.linalg.Matrix
    import org.apache.spark.mllib.linalg.distributed.RowMatrix

    println("phenotypeLabel: " + phenotypeLabel.count)
    /** scale features */
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(rawFeatures.map(_._2))
    val features = rawFeatures.map({ case (patientID, featureVector) => (patientID, scaler.transform(Vectors.dense(featureVector.toArray))) })
    println("features: " + features.count)
    val rawFeatureVectors = features.map(_._2).cache()
    println("rawFeatureVectors: " + rawFeatureVectors.count)

    /** reduce dimension */
    val mat: RowMatrix = new RowMatrix(rawFeatureVectors)
    val pc: Matrix = mat.computePrincipalComponents(10) // Principal components are stored in a local dense matrix.
    val featureVectors = mat.multiply(pc).rows

    val densePc = Matrices.dense(pc.numRows, pc.numCols, pc.toArray).asInstanceOf[DenseMatrix]

    def transform(feature: Vector): Vector = {
      val scaled = scaler.transform(Vectors.dense(feature.toArray))
      Vectors.dense(Matrices.dense(1, scaled.size, scaled.toArray).multiply(densePc).toArray)
    }

    /**
     * TODO: K Means Clustering using spark mllib
     * Train a k means model using the variabe featureVectors as input
     * Set maxIterations =20 and seed as 6250L
     * Assign each feature vector to a cluster(predicted Class)
     * Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
     * Find Purity using that RDD as an input to Metrics.purity
     * Remove the placeholder below after your implementation
     */
    featureVectors.cache()
    val KMeans_Model = new KMeans().setK(3).setMaxIterations(20).setSeed(6250L).run(featureVectors).predict(featureVectors)
    val KMeans_Predict = features.map(_._1).zip(KMeans_Model).join(phenotypeLabel).map(_._2)
    val kMeansPurity = Metrics.purity(KMeans_Predict)

//    val KMeans_cluster_1 = KMeans_Predict.filter(x => (x._1 == 0 && x._2 == 3)).count()
//    val KMeans_cluster_2 = KMeans_Predict.filter(x => (x._1 == 1 && x._2 == 3)).count()
//    val KMeans_cluster_3 = KMeans_Predict.filter(x => (x._1 == 2 && x._2 == 3)).count()



    /**val kMeansPurity = 0.0 */

    /**
     * TODO: GMMM Clustering using spark mllib
     * Train a Gaussian Mixture model using the variabe featureVectors as input
     * Set maxIterations =20 and seed as 6250L
     * Assign each feature vector to a cluster(predicted Class)
     * Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
     * Find Purity using that RDD as an input to Metrics.purity
     * Remove the placeholder below after your implementation
     */

    val GMM_Model = new GaussianMixture().setK(3).setMaxIterations(20).setSeed(6250L).run(featureVectors).predict(featureVectors)
    val GMM_Predict = features.map(_._1).zip(GMM_Model).join(phenotypeLabel).map(_._2)
    val gaussianMixturePurity = Metrics.purity(GMM_Predict)
//
//    val GMM_cluster_1 = GMM_Predict.filter(x => (x._1 == 0 && x._2 == 3)).count()
//    val GMM_cluster_2 = GMM_Predict.filter(x => (x._1 == 1 && x._2 == 3)).count()
//    val GMM_cluster_3 = GMM_Predict.filter(x => (x._1 == 2 && x._2 == 3)).count()


    /** val gaussianMixturePurity = 0.0 */

    /**
     * TODO: StreamingKMeans Clustering using spark mllib
     * Train a StreamingKMeans model using the variabe featureVectors as input
     * Set the number of cluster K = 3, DecayFactor = 1.0, number of dimensions = 10, weight for each center = 0.5, seed as 6250L
     * In order to feed RDD[Vector] please use latestModel, see more info: https://spark.apache.org/docs/2.2.0/api/scala/index.html#org.apache.spark.mllib.clustering.StreamingKMeans
     * To run your model, set time unit as 'points'
     * Assign each feature vector to a cluster(predicted Class)
     * Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
     * Find Purity using that RDD as an input to Metrics.purity
     * Remove the placeholder below after your implementation
     */
    /** val streamKmeansPurity = 0.0  */

    /**
     * val Stream_Model = new StreamingKMeans().setK(3).setDecayFactor(1.0).setRandomCenters(10, 0.5, seed = 6250L).timeUnit("points".toInt)
     * .trainOn(featureVectors).predictOnValues(featureVectors)
     */

    val Stream_Model = new StreamingKMeans().setK(3).setDecayFactor(1.0).setRandomCenters(10, 0.5, seed = 6250L)
      .latestModel().update(featureVectors, 1.0, "points").predict(featureVectors)

    val Stream_Predict = features.map(_._1).zip(Stream_Model).join(phenotypeLabel).map(_._2)
    val streamKmeansPurity = Metrics.purity(Stream_Predict)

//    val Stream_cluster_1 = Stream_Predict.filter(x => (x._1 == 0 && x._2 == 3)).count()
//    val Stream_cluster_2 = Stream_Predict.filter(x => (x._1 == 1 && x._2 == 3)).count()
//    val Stream_cluster_3 = Stream_Predict.filter(x => (x._1 == 2 && x._2 == 3)).count()


    (kMeansPurity, gaussianMixturePurity, streamKmeansPurity)
  }

  /**
   * load the sets of string for filtering of medication
   * lab result and diagnostics
   *
   * @return
   */
  def loadLocalRawData: (Set[String], Set[String], Set[String]) = {
    val candidateMedication = Source.fromFile("data/med_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateLab = Source.fromFile("data/lab_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateDiagnostic = Source.fromFile("data/icd9_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    (candidateMedication, candidateLab, candidateDiagnostic)
  }

  def sqlDateParser(input: String, pattern: String = "yyyy-MM-dd'T'HH:mm:ssX"): java.sql.Date = {
    val dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssX")
    new java.sql.Date(dateFormat.parse(input).getTime)
  }

  def loadRddRawData(spark: SparkSession): (RDD[Medication], RDD[LabResult], RDD[Diagnostic]) = {
    /* the sql queries in spark required to import sparkSession.implicits._ */
    import spark.implicits._
    import java.text.SimpleDateFormat
    import edu.gatech.cse6250.helper.{ CSVHelper, SparkHelper }
    val sqlContext = spark.sqlContext

    /* a helper function sqlDateParser may useful here */

    /**
     * load data using Spark SQL into three RDDs and return them
     * Hint:
     * You can utilize edu.gatech.cse6250.helper.CSVHelper
     * through your sparkSession.
     *
     * This guide may helps: https://bit.ly/2xnrVnA
     *
     * Notes:Refer to model/models.scala for the shape of Medication, LabResult, Diagnostic data type.
     * Be careful when you deal with String and numbers in String type.
     * Ignore lab results with missing (empty or NaN) values when these are read in.
     * For dates, use Date_Resulted for labResults and Order_Date for medication.
     *
     */

    /**
     * TODO: implement your own code here and remove
     * existing placeholder code below
     */

    val db_med = CSVHelper.loadCSVAsTable(spark, path = "data/medication_orders_INPUT.csv")
    val db_phenotype = CSVHelper.loadCSVAsTable(spark, path = "data/phenotypeLabels.csv")
    val db_labResults = CSVHelper.loadCSVAsTable(spark, path = "data/lab_results_INPUT.csv")
    val db_encouterDiag = CSVHelper.loadCSVAsTable(spark, path = "data/encounter_dx_INPUT.csv")
    val db_encounter = CSVHelper.loadCSVAsTable(spark, path = "data/encounter_INPUT.csv")

    db_med.createOrReplaceTempView("tab_med")
    db_phenotype.createOrReplaceTempView("tab_phenotype")
    db_labResults.createOrReplaceTempView("tab_labResults")
    db_encouterDiag.createOrReplaceTempView("tab_encounterDiag")
    db_encounter.createOrReplaceTempView("tab_encounter")

    val encounterDiag_sc = spark.sql("select code as code, Encounter_ID as EncounterID from tab_encounterDiag")
    val encounter_sc = spark.sql("select Encounter_ID as EncounterID, Encounter_DateTime as date, Member_ID as patientID from tab_encounter")
    val newDiag_sc = encounterDiag_sc.join(encounter_sc, Seq("EncounterID"))
    newDiag_sc.createOrReplaceTempView("tab_newDiag")

    val medication: RDD[Medication] = spark.sql("select Member_ID as patientID, Order_Date as date, Drug_Name as medicine from tab_med").na.drop().map(row => Medication(row.getString(0), sqlDateParser(row.getString(1)), row.getString(2))).rdd
    val labResult: RDD[LabResult] = spark.sql("select Member_ID as patientID, Date_Resulted as date, Result_Name as testName, Numeric_Result as value from tab_labResults").na.drop().map(row => LabResult(row.getString(0), sqlDateParser(row.getString(1)), row.getString(2), row.getString(3).replace(",", "").toDouble)).rdd
    val diagnostic: RDD[Diagnostic] = spark.sql("select patientID as patientID, date as date, code as code from tab_newDiag").na.drop().map(row => Diagnostic(row.getString(0), sqlDateParser(row.getString(1)), row.getString(2))).rdd

    (medication, labResult, diagnostic)
  }

}
