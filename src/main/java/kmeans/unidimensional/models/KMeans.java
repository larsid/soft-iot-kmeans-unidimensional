package kmeans.unidimensional.models;

import java.util.List;
import java.util.Random;
import kmeans.unidimensional.services.IKMeans;

public class KMeans implements IKMeans {

  private int numClusters;
  private int numIterations;
  private double[] data;
  private double[] centroids;
  private int[] labels;

  /**
   * Executa o Algoritmo K-Means.
   *
   * @param dataInList List<Float> - Lista com os dados que serão agrupados.
   */
  @Override
  public void execute(List<Float> dataInList) {
    this.data = new double[dataInList.size()];

    for (int i = 0; i < dataInList.size(); i++) {
      this.data[i] = dataInList.get(i);
    }

    this.init();
    this.calculate();
  }

  /**
   * Realiza a inicialização.
   */
  private void init() {
    this.centroids = new double[this.numClusters];
    this.labels = new int[this.data.length];

    Random random = new Random();
    for (int i = 0; i < this.numClusters; i++) {
      this.centroids[i] = this.data[random.nextInt(this.data.length)];
    }
  }

  /**
   * Realiza o agrupamento dos dados.
   */
  private void calculate() {
    boolean finish = false;
    int iteration = 0;

    while (!finish && iteration < this.numIterations) {
      this.assignClusters();
      double[] newCentroids = this.calculateNewCentroids();

      double distance = 0;
      for (int i = 0; i < numClusters; i++) {
        distance += Math.abs(this.centroids[i] - newCentroids[i]);
      }

      this.centroids = newCentroids;
      iteration++;

      if (distance == 0) {
        finish = true;
      }
    }
  }

  /**
   * Atribui cada ponto de dados ao cluster mais próximo.
   */
  private void assignClusters() {
    for (int i = 0; i < this.data.length; i++) {
      double minDistance = Double.MAX_VALUE;
      int clusterIndex = 0;

      for (int j = 0; j < this.numClusters; j++) {
        double distance = Math.abs(this.data[i] - this.centroids[j]);
        if (distance < minDistance) {
          minDistance = distance;
          clusterIndex = j;
        }
      }

      this.labels[i] = clusterIndex;
    }
  }

  /**
   * Calcula novos centróides com base na média dos pontos atribuídos a cada
   * cluster.
   *
   * @return Um array contendo os novos centróides.
   */
  private double[] calculateNewCentroids() {
    double[] newCentroids = new double[this.numClusters];
    int[] counts = new int[this.numClusters];

    for (int i = 0; i < this.data.length; i++) {
      newCentroids[this.labels[i]] += this.data[i];
      counts[this.labels[i]]++;
    }

    for (int i = 0; i < this.numClusters; i++) {
      if (counts[i] != 0) {
        newCentroids[i] /= counts[i];
      }
    }

    return newCentroids;
  }

  public void setNumClusters(int numClusters) {
    this.numClusters = numClusters;
  }

  public void setNumIterations(int numIterations) {
    this.numIterations = numIterations;
  }
}
