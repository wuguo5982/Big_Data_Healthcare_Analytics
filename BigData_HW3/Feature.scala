import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors }
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._

object FeatureConstruction {

  /**
   * ((patient-id, feature-name), feature-value)
   */
  type FeatureTuple = ((String, String), Double)

  /**
   * Aggregate feature tuples from diagnostic with COUNT aggregation,
   *
   * @param diagnostic RDD of diagnostic
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic]): RDD[FeatureTuple] = {
    diagnostic.map(x => ((x.patientID, x.code), 1.0)).reduceByKey(_ + _)


  def constructMedicationFeatureTuple(medication: RDD[Medication]): RDD[FeatureTuple] = {
    medication.map(x => ((x.patientID, x.medicine), 1.0)).reduceByKey(_ + _)
  }

  /**
   * Aggregate feature tuples from lab result, using AVERAGE aggregation
   *
   * @param labResult RDD of lab result
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult]): RDD[FeatureTuple] = {

    labResult.map(x => ((x.patientID, x.testName), x.value, 1)).keyBy(_._1).reduceByKey((a, b) => (a._1, a._2 + b._2, a._3 + b._3))
      .map(x => (x._1, x._2._2 / x._2._3))
  }

  /**  labResult.sparkContext.parallelize(List((("patient", "lab"), 1.0))) */

  /**
   * Aggregate feature tuple from diagnostics with COUNT aggregation, but use code that is
   * available in the given set only and drop all others.
   *
   * @param diagnostic   RDD of diagnostics
   * @param candiateCode set of candidate code, filter diagnostics based on this set
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic], candiateCode: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    diagnostic.filter(x => candiateCode.contains(x.code)).map(x => ((x.patientID, x.code), 1.0)).reduceByKey(_ + _)

  }


  def constructMedicationFeatureTuple(medication: RDD[Medication], candidateMedication: Set[String]): RDD[FeatureTuple] = {
    medication.filter(x => candidateMedication.contains(x.medicine)).map(x => ((x.patientID, x.medicine), 1.0)).reduceByKey(_ + _)
  }


  def constructLabFeatureTuple(labResult: RDD[LabResult], candidateLab: Set[String]): RDD[FeatureTuple] = {
    labResult.filter(x => candidateLab.contains(x.testName.toLowerCase)).map(x => ((x.patientID, x.testName), x.value, 1))
      .keyBy(_._1).reduceByKey((a, b) => (a._1, a._2 + b._2, a._3 + b._3)).map(x => (x._1, x._2._2 / x._2._3))

  }

  /**
   * Given a feature tuples RDD, construct features in vector
   * format for each patient. feature name should be mapped
   * to some index and convert to sparse feature format.
   *
   * @param sc      SparkContext to run
   * @param feature RDD of input feature tuples
   * @return
   */
  def construct(sc: SparkContext, feature: RDD[FeatureTuple]): RDD[(String, Vector)] = {

    /** save for later usage */
    feature.cache()

    val feature_Map = feature.map(_._1._2).distinct.collect.zipWithIndex.toMap
    val scFeature_Map = sc.broadcast(feature_Map)
    val num_feature = scFeature_Map.value.size

    /** transform input feature */

    val result = feature.map(x => (x._1._1, scFeature_Map.value(x._1._2), x._2)).groupBy(_._1).map(x => {
      val feature_Vector = Vectors.sparse(num_feature, x._2.toList.map(x => (x._2, x._3)))
      val vectors = (x._1, feature_Vector)
      vectors
    })

    result

  }
}

