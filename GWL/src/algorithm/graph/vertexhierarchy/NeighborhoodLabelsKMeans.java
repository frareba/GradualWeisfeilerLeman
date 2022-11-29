package algorithm.graph.vertexhierarchy;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;

import algorithm.clustering.KMeans;
import algorithm.clustering.KMeans.KMeansCluster;
import algorithm.graph.isomorphism.labelrefinement.VertexLabelConverter;
import comparison.distance.Distance;
import comparison.distance.SquaredEuclideanDistance;
import graph.LGraph;
import graph.Graph.Vertex;

/**
 * @author kriege/bause
 *
 * Vertex refinement that clusters the resulting new colors into (max) k clusters.
 *
 * @param <E>
 */
public class NeighborhoodLabelsKMeans<E> implements GraphSetVertexRefinement<E> {
	
	double branching;
	private Distance<double[]> distance;

	/**
	 * 
	 * @param branching if > 0, this gives the number of clusters to be found; 
	 *                  if < 0, distinctPoints/-branching are searched
	 * @param distance Distance function to be used
	 */
	public NeighborhoodLabelsKMeans(double branching,Distance<double[]> distance ) {
		this.branching = branching;
		this.distance = distance;
	}
	
	/**
	 * Default (squared euclidean distance is used
	 * @param branching if > 0, this gives the number of clusters to be found; 
	 *                  if < 0, distinctPoints/-branching are searched
	 */
	public NeighborhoodLabelsKMeans(double branching) {
		this.branching = branching;
		this.distance = new SquaredEuclideanDistance();
	}

	@Override
	public ArrayList<ArrayList<Vertex>> refine(LGraph<Integer, E> LG, Iterable<? extends Vertex> V) {
		VertexLabelConverter<E> vlc = new VertexLabelConverter<E>();
		LGraph<Integer, E> LG2 = vlc.transform(LG);
		
		//map (neighbor label + edge label) to entry 
		HashMap<Integer,HashMap<E,Integer>> mapToEntry = new HashMap<Integer,HashMap<E,Integer>>();
		int k = 0;
		for(Vertex v:V)
		{
			for(Vertex w: v.neighbors())
			{
				Integer label = LG2.getVertexLabel().get(w);
				if(!mapToEntry.containsKey(label))
				{
					mapToEntry.put(label, new HashMap<E,Integer>());
				}
				E edgelabel = LG2.getEdgeLabel().get(LG2.getGraph().getEdge(v, w));
				if(!mapToEntry.get(label).containsKey(edgelabel))
				{
					HashMap<E,Integer> mapi = mapToEntry.get(label);
					mapi.put(edgelabel, k);
					k++;
				}
			}
		}
		//
		
		HashMap<Object,Vertex> map = new HashMap<>();
		HashSet<ArrayList<Double>> distinctPoints = new HashSet<>();
		ArrayList<double[]> neighborVectors = new ArrayList<>();
		for (Vertex v : V) {
			double[] neighborVector = new double[k];
			for (Vertex w : v.neighbors()) {
				neighborVector[mapToEntry.get(LG2.getVertexLabel().get(w)).get(LG2.getEdgeLabel().get(LG2.getGraph().getEdge(v, w)))] += 1;
			}
			neighborVectors.add(neighborVector);
			map.put(neighborVector, v);
			ArrayList<Double> neighborVectorAL = new ArrayList<>();
			for (double d : neighborVector) neighborVectorAL.add(d);
			distinctPoints.add(neighborVectorAL);
		}
		
		
		
		
		int clusterCount = branching > 0 ? (int)branching : (int)Math.floor((double)distinctPoints.size()/(-branching));
		
//		System.out.println(neighborVectors.size()+" "+distinctPoints.size()+"  "+branching+"\t--> " + clusterCount);
		
		if (clusterCount == distinctPoints.size()) clusterCount--;		
		if (clusterCount == 0) clusterCount = 1;
		
		
		
		int iter = 10;
		KMeans km = new KMeans(this.distance);
		ArrayList<KMeansCluster> bestResult = null;
		double bestOverallDissimilarity = Double.POSITIVE_INFINITY;
		for(int i=0; i<iter; i++) {
			ArrayList<KMeansCluster> result = km.run(neighborVectors, clusterCount);
			double overallDissimilarity = km.getOverallDissimilarity(result);
			if (overallDissimilarity<bestOverallDissimilarity) {
				bestOverallDissimilarity = overallDissimilarity;
				bestResult = result;
			}
		}
		
		ArrayList<ArrayList<Vertex>> r = new ArrayList<>();
		for (KMeansCluster c : bestResult) {
			if (c.getPoints().isEmpty()) continue;
			ArrayList<Vertex> cv = new ArrayList<>();
			for (double[] d : c.getPoints()) {
				cv.add(map.get(d));
			}
			r.add(cv);
		}
		return r;
	}
	
	@Override
	public String toString() {
		return "BIWL"+branching;
	}

	@Override
	public ArrayList<ArrayList<Vertex>> refine(LGraph<Integer, E> LG) {
		return refine(LG, LG.getGraph().vertices());
	}

	@Override
	public String getID() {
		return "kMeans"+this.distance.getClass().getSimpleName()+"_"+String.format(Locale.ENGLISH, "%1.0f", this.branching);
	}

	@Override
	public void fitToDataset(ArrayList<LGraph<Integer, E>> lgs) {
		// TODO Auto-generated method stub
		
	}
}
