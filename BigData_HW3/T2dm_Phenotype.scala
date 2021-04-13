import model.{ Diagnostic, LabResult, Medication }
import org.apache.spark.rdd.RDD
import helper.SparkHelper.spark
import helper.CSVHelper
import spark.implicits._
import helper.SparkHelper.sc

object T2dmPhenotype {

  val T1DM_DX = Set("250.01", "250.03", "250.11", "250.13", "250.21", "250.23", "250.31", "250.33", "250.41", "250.43",
    "250.51", "250.53", "250.61", "250.63", "250.71", "250.73", "250.81", "250.83", "250.91", "250.93")

  val T2DM_DX = Set("250.3", "250.32", "250.2", "250.22", "250.9", "250.92", "250.8", "250.82", "250.7", "250.72", "250.6",
    "250.62", "250.5", "250.52", "250.4", "250.42", "250.00", "250.02")

  val T1DM_MED = Set("lantus", "insulin glargine", "insulin aspart", "insulin detemir", "insulin lente", "insulin nph", "insulin reg", "insulin,ultralente")

  val T2DM_MED = Set("chlorpropamide", "diabinese", "diabanase", "diabinase", "glipizide", "glucotrol", "glucotrol xl",
    "glucatrol ", "glyburide", "micronase", "glynase", "diabetamide", "diabeta", "glimepiride", "amaryl",
    "repaglinide", "prandin", "nateglinide", "metformin", "rosiglitazone", "pioglitazone", "acarbose",
    "miglitol", "sitagliptin", "exenatide", "tolazamide", "acetohexamide", "troglitazone", "tolbutamide",
    "avandia", "actos", "actos", "glipizide")

  val DM_RELATED_DX = Set("790.21", "790.22", "790.2", "790.29", "648.81", "648.82", "648.83", "648.84", "648", "648", "648.01", "648.02", "648.03", "648.04", "791.5", "277.7", "V77.1", "256.4")

  /**
   * Transform given data set to a RDD of patients and corresponding phenotype
   *
   * @param medication medication RDD
   * @param labResult  lab result RDD
   * @param diagnostic diagnostic code RDD
   * @return tuple in the format of (patient-ID, label). label = 1 if the patient is case, label = 2 if control, 3 otherwise
   */
  def transform(medication: RDD[Medication], labResult: RDD[LabResult], diagnostic: RDD[Diagnostic]): RDD[(String, Int)] = {
    /**
     * Remove the place holder and implement your code here.
     * Hard code the medication, lab, icd code etc. for phenotypes like example code below.
     * When testing your code, we expect your function to have no side effect,
     */

    val sc = medication.sparkContext

    /** Hard code the criteria */
    // val type1_dm_dx = Set("code1", "250.03")
    // val type1_dm_med = Set("med1", "insulin nph")
    // use the given criteria above like T1DM_DX, T2DM_DX, T1DM_MED, T2DM_MED and hard code DM_RELATED_DX criteria as well

    /** Find CASE Patients */

    val total_Patient = medication.map(x => x.patientID).union(labResult.map(x => x.patientID).union(diagnostic.map(x => x.patientID))).distinct()
    /** 3688 */
    val Type1_DM = diagnostic.filter(x => T1DM_DX.contains(x.code)).map(x => x.patientID).distinct()
    /**  686 */
    val Diag_AfterType1DM = total_Patient.subtract(Type1_DM)
    /**  after 3002 */
    val Type2_DM = diagnostic.filter(x => T2DM_DX.contains(x.code)).map(x => x.patientID).distinct()
    /**  1265 */
    val Diag_Afterboth = Diag_AfterType1DM.subtract(Type2_DM)
    /**  after 1737 !*/
    val Type1_Med = medication.filter(x => T1DM_MED.contains(x.medicine.toLowerCase)).map(x => x.patientID).distinct()
    /** 1328 */
    val Med_AfterType1Med = total_Patient.subtract(Type1_Med)
    /** 2360 */
    val Type2_Med = medication.filter(x => T2DM_MED.contains(x.medicine.toLowerCase)).map(x => x.patientID).distinct()
    /** 907 */
    val Med_AfterType2Med = total_Patient.subtract(Type2_Med)
    /** 2781 */
    val condition1 = Type2_DM.intersection(Med_AfterType1Med)
    /** 427 */
    val condition2 = Type2_DM.intersection(Type1_Med).intersection(Med_AfterType2Med)
    /** 255 */
    val condition3 = Type2_DM.intersection(Type1_Med).intersection(Type2_Med)
    /** 583 */

    val new_condition3 = medication.map(x => (x.patientID, x)).join(condition3.map(x => (x, 0))).map(x => Medication(x._2._1.patientID, x._2._1.date, x._2._1.medicine))
    val date1 = new_condition3.filter(row => T1DM_MED.contains(row.medicine.toLowerCase)).map(x => (x.patientID, x.date.getTime())).distinct()
    val date2 = new_condition3.filter(row => T2DM_MED.contains(row.medicine.toLowerCase)).map(x => (x.patientID, x.date.getTime())).distinct()
    val newCase = date1.reduceByKey(math.min).join(date2.reduceByKey(math.min)).filter(x => x._2._1 > x._2._2).map(_._1)
    /** 294 */

    val newcasePatients = sc.union(condition1, condition2, newCase)
    val casePatients = newcasePatients.map(x => (x, 1))
    /** 976  */

    /** Find CONTROL Patients */

    val glucose_lab = labResult.filter(x => x.testName.toLowerCase.contains("glucose")).map(x => x.patientID).distinct()
    /**1823 */

    val abnormal = labResult.filter(x => (x.testName.toLowerCase.contains("hba1c") && x.value >= 6.0)).map(x => x.patientID).distinct()
      .union(labResult.filter(x => (x.testName.toLowerCase.contains("hemoglobin a1c") && x.value >= 6.0)).map(x => x.patientID).distinct())
      .union(labResult.filter(x => (x.testName.toLowerCase.contains("fasting glucose") && x.value >= 110)).map(x => x.patientID).distinct())
      .union(labResult.filter(x => (x.testName.toLowerCase.contains("fasting blood glucose") && x.value >= 110)).map(x => x.patientID).distinct())
      .union(labResult.filter(x => (x.testName.toLowerCase.contains("fasting plasma glucose") && x.value >= 110)).map(x => x.patientID).distinct())
      .union(labResult.filter(x => (x.testName.toLowerCase.contains("glucose") && x.value >= 110)).map(x => x.patientID).distinct())
      .union(labResult.filter(x => (x.testName.toLowerCase.contains("glucose, serum") && x.value >= 110)).map(x => x.patientID).distinct()).distinct()

    /**585 */
    val noAbnormal = glucose_lab.subtract(abnormal)
    /**1238 */
    val DM_related = diagnostic.filter(x => DM_RELATED_DX.contains(x.code) | x.code.contains("250.")).map(x => x.patientID)
    val controlPatients = noAbnormal.subtract(DM_related).distinct().map(x => (x, 2))
    /**948 */
    /** Find OTHER Patients */

    val other_Patients = total_Patient.subtract(newcasePatients).subtract(controlPatients.map(_._1))
    val others = other_Patients.map(x => (x, 3))
    /** 1764 */

    /** Once you find patients for each group, make them as a single RDD[(String, Int)] */
    val phenotypeLabel = sc.union(casePatients, controlPatients, others)
    /**3688*/
    /** Return */

    phenotypeLabel
  }
}