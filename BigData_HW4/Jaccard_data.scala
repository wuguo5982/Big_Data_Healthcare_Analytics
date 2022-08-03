import model.{ EdgeProperty, VertexProperty }
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object Jaccard {
  def jaccardSimilarityOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long): List[Long] = {
    /**
     * Given a patient ID, compute the Jaccard similarity w.r.t. to all other patients.
     * Return a List of top 10 patient IDs ordered by the highest to the lowest similarity.
     * For ties, random order is okay. The given patientID should be excluded from the result.
     */

    val direction = graph.collectNeighborIds(EdgeDirection.Out).lookup(patientID).head
    val patientVerts = graph.vertices.filter(x => x._2.isInstanceOf[PatientProperty]).map(_._1)
    val newpatientVerts = patientVerts.filter(_!= patientID).collect().toSet
    val Jaccards = graph.collectNeighborIds(EdgeDirection.Out).filter(x => newpatientVerts.contains(x._1)).map(x => (x._1, jaccard(direction.toSet, x._2.toSet)))
    val newJaccards = Jaccards.sortBy(_._2.toString, false).map(_._1.toLong).take(10).toList
    newJaccards

    /** Remove this placeholder and implement your code */
    /** List(1, 2, 3, 4, 5) */
  }

  def jaccardSimilarityAllPatients(graph: Graph[VertexProperty, EdgeProperty]): RDD[(Long, Long, Double)] = {

    val sc = graph.edges.sparkContext
    val subGraph = graph.subgraph(vpred = {case(id, attr) => attr.isInstanceOf[PatientProperty]})
    val newGraph = subGraph.collectNeighborIds(EdgeDirection.Out).map(_._1).collect().toSet
    val patientIds = graph.collectNeighborIds(EdgeDirection.Out).filter(x => newGraph.contains(x._1))
    val graphPatients = patientIds.cartesian(patientIds).filter(x => x._1._1 < x._2._1)
    val result = graphPatients.map(x => (x._1._1, x._2._1, jaccard(x._1._2.toSet, x._2._2.toSet)))
    result

    /**sc.parallelize(Seq((1L, 2L, 0.5d), (1L, 3L, 0.4d))) */
  }

  def jaccard[A](a: Set[A], b: Set[A]): Double = {
    /**
     * Helper function
     *
     * Given two sets, compute its Jaccard similarity and return its result.
     * If the union part is zero, then return 0.
     */

    if (a.isEmpty | b.isEmpty) {
      return 0.0
    }
    val intersection = a.intersect(b).size.toDouble
    val totalUnion = a.union(b).size.toDouble
    val Similarity = intersection / totalUnion
    return Similarity


    /** Remove this placeholder and implement your code */
    /**0.0 */
  }
}
