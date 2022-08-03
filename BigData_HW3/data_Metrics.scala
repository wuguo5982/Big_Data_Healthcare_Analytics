import org.apache.spark.rdd.RDD

object Metrics {
  /**
   * Given input RDD with tuples of assigned cluster id by clustering,
   * and corresponding real class. Calculate the purity of clustering.
   * Purity is defined as
   * \fract{1}{N}\sum_K max_j |w_k \cap c_j|
   * where N is the number of samples, K is number of clusters and j
   * is index of class. w_k denotes the set of samples in k-th cluster
   * and c_j denotes set of samples of class j.
   *
   * @param clusterAssignmentAndLabel RDD in the tuple format
   *                                  (assigned_cluster_id, class)
   * @return
   */
  def purity(clusterAssignmentAndLabel: RDD[(Int, Int)]): Double = {
    /**
     * TODO: Remove the placeholder and implement your code here
     */

    val total = clusterAssignmentAndLabel.map((_, 1)).keyBy(_._1)
      .reduceByKey((x, y) => (x._1, x._2 + y._2)).map(z => (z._2._1._1, z._2._2))
    val nums = clusterAssignmentAndLabel.count()
    val summation = total.keyBy(_._1).reduceByKey((x, y) => (1, math.max(x._2, y._2)))
    val purity = summation.map(_._2._2).reduce((x, y) => x + y) / nums.toDouble
    purity
  }
}
