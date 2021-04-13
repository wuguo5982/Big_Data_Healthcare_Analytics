/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse6250.graphconstruct

import edu.gatech.cse6250.model._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object GraphLoader {
  /**
   * Generate Bipartite Graph using RDDs
   *
   * @input: RDDs for Patient, LabResult, Medication, and Diagnostic
   * @return: Constructed Graph
   *
   */
  def load(patients: RDD[PatientProperty], labResults: RDD[LabResult],
           medications: RDD[Medication], diagnostics: RDD[Diagnostic]): Graph[VertexProperty, EdgeProperty] = {

    /** HINT: See Example of Making Patient Vertices Below */
    val sc = patients.sparkContext
    val patientVertex: RDD[(VertexId, VertexProperty)] = patients.map(x => (x.patientID.toLong, x.asInstanceOf[VertexProperty])).cache()
    val patientCount = patientVertex.count() + 1

    val labVertexRDD = labResults.map(_.labName).distinct().zipWithIndex().map{case(lab, zeroBasedIndex) => (lab, zeroBasedIndex + patientCount)}.cache()
    val labVertex: RDD[(VertexId, VertexProperty)] = labVertexRDD.map{case(lab, number) => (number, LabResultProperty(lab))}
    val lab2VertexID = labVertexRDD.collect.toMap
    val labCount = labVertex.count() + patientCount + 1

    val medVertexRDD = medications.map(_.medicine).distinct().zipWithIndex().map{case(med, zeroBasedIndex) => (med, zeroBasedIndex + labCount)}.cache()
    val medVertex: RDD[(VertexId, VertexProperty)] = medVertexRDD.map{case(med, number) => (number, MedicationProperty(med))}
    val med2VertexID  = medVertexRDD.collect.toMap
    val medCount = medVertex.count() + patientCount + labCount + 1

    val diagVertexRDD = diagnostics.map(_.icd9code).distinct().zipWithIndex().map{case(icd9code, zeroBasedIndex) => (icd9code, zeroBasedIndex + medCount)}.cache()
    val diagVertex: RDD[(VertexId, VertexProperty)] = diagVertexRDD.map{case(icd9code, number) => (number, DiagnosticProperty(icd9code)) }
    val diag2VertexID = diagVertexRDD.collect.toMap

    val LabBroadcast = sc.broadcast(lab2VertexID)
    val MedBroadcast = sc.broadcast(med2VertexID)
    val DiagBroadcast = sc.broadcast(diag2VertexID)

    val labEvents = labResults.map(x => ((x.patientID, x.labName), x)).reduceByKey((a, b) => if (a.date > b.date) a else b).map(_._2)
    val Edges_patientLab = labEvents.map(p => Edge(p.patientID.toLong, LabBroadcast.value(p.labName), PatientLabEdgeProperty(p).asInstanceOf[EdgeProperty]))
    val Edges_labPatient = labEvents.map(p => Edge(LabBroadcast.value(p.labName), p.patientID.toLong, PatientLabEdgeProperty(p).asInstanceOf[EdgeProperty]))

    val medEvents = medications.map(x => ((x.patientID, x.medicine), x)).reduceByKey((a, b) => if (a.date > b.date) a else b).map(_._2)
    val Edges_patientMedication = medEvents.map(p => Edge(p.patientID.toLong, MedBroadcast.value(p.medicine), PatientMedicationEdgeProperty(p).asInstanceOf[EdgeProperty]))
    val Edges_medicationPatient = medEvents.map(p => Edge(MedBroadcast.value(p.medicine), p.patientID.toLong, PatientMedicationEdgeProperty(p).asInstanceOf[EdgeProperty]))

    val diagEvents = diagnostics.map(x => ((x.patientID, x.icd9code), x)).reduceByKey((a, b) => if (a.date > b.date) a else b).map(_._2)
    val Edges_patientDiag = diagEvents.map(p=> Edge(p.patientID.toLong, DiagBroadcast.value(p.icd9code), PatientDiagnosticEdgeProperty(p).asInstanceOf[EdgeProperty]))
    val Edges_diagPatient = diagEvents.map(p=> Edge(DiagBroadcast.value(p.icd9code), p.patientID.toLong, PatientDiagnosticEdgeProperty(p).asInstanceOf[EdgeProperty]))

    val Edges_Patient_Lab = sc.union(Edges_patientLab, Edges_labPatient)
    val Edges_Patient_Medication = sc.union(Edges_patientMedication, Edges_medicationPatient)
    val Edges_Patient_Diagnostic = sc.union(Edges_patientDiag, Edges_diagPatient)

    val Vertices = sc.union(patientVertex, diagVertex, labVertex, medVertex)
    val Edges = sc.union(Edges_Patient_Diagnostic, Edges_Patient_Lab, Edges_Patient_Medication)
    val graph: Graph[VertexProperty, EdgeProperty] = Graph[VertexProperty, EdgeProperty](Vertices, Edges)

    graph
  }
}