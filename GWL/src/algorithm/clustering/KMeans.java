package algorithm.clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import comparison.distance.Distance;
import comparison.distance.SquaredEuclideanDistance;

public class KMeans {
	
	Random rng = new Random();

	private Distance<double[]> distance;

	public KMeans()
	{
		this.distance = new SquaredEuclideanDistance();
	}
	
	public KMeans(Distance<double[]> distance)
	{
		this.distance = distance;
	}
	public class KMeansCluster {
		public ArrayList<double[]> points;
		public double[] mean;

		public KMeansCluster(KMeansCluster c) {
			points = new ArrayList<double[]>(c.points);
			if (c.mean != null) mean = c.mean.clone();
		}

		public KMeansCluster() {
			points = new ArrayList<>();
		}
		
		public KMeansCluster(double[] mean) {
			this();
			this.mean = mean;
		}
		
		public String toString() {
			StringBuilder sb = new StringBuilder();

			for (double[] point : points) {
				sb.append(Arrays.toString(point));
				sb.append(" ");
			}
			sb.append("m "+Arrays.toString(mean));
			return sb.toString();
		}
		
		public ArrayList<double[]> getPoints() {
			return points;
		}
		
		public int size() {
			return points.size();
		}
		
		public void updateMean() {
			int dim = points.get(0).length;
			mean = new double[dim];
			for (double[] p : points) {
				for (int i=0; i<dim; i++) {
					mean[i] += p[i];
				}
			}
			for (int i=0; i<dim; i++) {
				mean[i] /= points.size();
			}
		}
		
		public double compactness() {
			double d = 0;
			for (double[] p : points) {
				d += distance(p, mean);
			}
			return d;
		}
	}
	
	public static double compactness(ArrayList<KMeansCluster> clusters) {
		double d = 0;
		for (KMeansCluster c: clusters) {
			d += c.compactness();
		}
		return d;
	}
	
	public ArrayList<KMeansCluster> run(ArrayList<double[]> elements, int k) {
		ArrayList<KMeansCluster> clusters = initialize(elements, k);
		
		while (reassign(clusters)) {
			update(clusters, elements);
		}
		
		return clusters;
		
	}
	
	/**
	 * Forgy method
	 */
	private ArrayList<KMeansCluster> initialize(ArrayList<double[]> elements, int k) {
		k = Math.min(k, elements.size());
		ArrayList<KMeansCluster> clusters = new ArrayList<>();
		ArrayList<double[]> elementsCopy = new ArrayList<>(elements);
		for (int i=0; i<k; i++) {
			int randomIndex = rng.nextInt(elementsCopy.size());
			double[] randomPoint = elementsCopy.get(randomIndex);
			clusters.add(new KMeansCluster(randomPoint));
			// swap with last element and remove
			int lastIndex = elementsCopy.size()-1;
			elementsCopy.set(randomIndex, elementsCopy.get(lastIndex));
			elementsCopy.remove(lastIndex);
		}
		clusters.get(0).points.addAll(elements);
		return clusters;
	}
	
	/**
	 * 
	 * @param clusters
	 * @return true if changed
	 */
	private boolean reassign(ArrayList<KMeansCluster> clusters) {
		ArrayList<ArrayList<double[]>> newClusters = new ArrayList<>(clusters.size());
		for (int i=0; i<clusters.size(); i++) {
			newClusters.add(new ArrayList<>());
		}

		boolean changed = false;
		for (int i=0; i<clusters.size(); i++) {
			KMeansCluster cluster = clusters.get(i);
			for (double[] e : cluster.points) {
				double minDist = Double.POSITIVE_INFINITY;
				int nearestClusterIndex = -1;
				for (int j=0; j<clusters.size(); j++) {
					KMeansCluster c = clusters.get(j);
					double dist = distance(e, c.mean);
					if (dist<minDist) {
						minDist = dist;
						nearestClusterIndex = j;
					}
				}
				newClusters.get(nearestClusterIndex).add(e);
				if (i != nearestClusterIndex) {
					changed = true;
				}
			}
		}
		
		for (int i=0; i<clusters.size(); i++) {
			clusters.get(i).points = newClusters.get(i);
		}

		return changed;
	}
	
	public double distance(double[] e1, double[] e2) {
		return this.distance.compute(e1,e2);
	}
	
	private void update(ArrayList<KMeansCluster> clusters, ArrayList<double[]> elements) {
		for (KMeansCluster c : clusters) {
			if (c.points.isEmpty()) {
				// select random point
				int randomIndex = rng.nextInt(elements.size());
				double[] randomPoint = elements.get(randomIndex);
				c.mean = Arrays.copyOf(randomPoint, randomPoint.length);
			} else {
				c.updateMean();
			}
		}	
	}
	
	public double getOverallDissimilarity(ArrayList<KMeansCluster> clusters) {
		double ssd = 0;
		for (KMeansCluster cluster : clusters) {
			for (double[] p : cluster.points) {
				double dist = this.distance(p, cluster.mean);
				ssd += dist * dist;
			}
		}
		return ssd;

	}
	
	public static double[] randomPoint(int dim, Random rng, double offset) {
		double[] d = new double[dim];
		for (int i=0; i<dim; i++) {
			d[i] = rng.nextInt(10);
		}
		d[0] += offset;
		return d;
	}

}
