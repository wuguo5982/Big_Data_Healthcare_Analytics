
import model.{ PatientProperty, EdgeProperty, VertexProperty }
import org.apache.spark.graphx._

object RandomWalk {

  def randomWalkOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long, iterations: Int = 100, alpha: Double = 0.15): List[Long] = {
    /**
     * Given a patient ID, compute the random walk probability w.r.t. to all other patients.
     * Return a List of patient IDs ordered by the highest to the lowest similarity.
     * For ties, random order is okay
     */

    val vertexSource: VertexId = patientID
    val graphVectrices = graph.outerJoinVertices(graph.outDegrees) {(e, f, g) => g.getOrElse(0) }.mapTriplets(m => 1.0 / m.srcAttr, TripletFields.Src)
    var graphRandomWalk: Graph[Double, Double] = graphVectrices.mapVertices {(x, y) => if (x != vertexSource) 0.0 else 1.0 }
    def distance(A: VertexId, B: VertexId): Double = {if (A != B) 0.0 else 1.0 }
    var n = 0
    var graphRandomInit: Graph[Double, Double] = null
    if (n < iterations) {
      graphRandomWalk.cache()
      val rankOne = graphRandomWalk.aggregateMessages[Double](x => x.sendToDst(x.srcAttr * x.attr), _ + _, TripletFields.Src)
      graphRandomInit = graphRandomWalk
      val rankTwo = {(vertexSource: VertexId, vertexDest: VertexId) => alpha * distance(vertexSource, vertexDest) }
      graphRandomWalk = graphRandomWalk.outerJoinVertices(rankOne) {(a, b, c) => rankTwo(vertexSource, a) + (1.0 - alpha) * c.getOrElse(0.0)}.cache()
      graphRandomWalk.edges.foreachPartition(_ => {})
      n += 1
    }

    val RandomsubGraph = graph.subgraph(vpred = {case(x, y) => y.isInstanceOf[PatientProperty]})
    val RandomNewGraph = RandomsubGraph.collectNeighborIds(EdgeDirection.Out).map(_._1).collect().toSet
    val result_rank10 = graphRandomWalk.vertices.filter(n => RandomNewGraph.contains(n._1)).sortBy(_._2.toString, false).map(_._1.toLong).take(11)
    val final_result = result_rank10.slice(1, 11).toList
    final_result


    /** Remove this placeholder and implement your code */
    //List(1, 2, 3, 4, 5)
  }
}
