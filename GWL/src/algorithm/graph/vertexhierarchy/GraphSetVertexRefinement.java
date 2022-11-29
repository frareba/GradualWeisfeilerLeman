package algorithm.graph.vertexhierarchy;

import java.util.ArrayList;

import graph.Graph.Vertex;
import graph.LGraph;

/**
 * @author kriege/bause
 * Interface for Refinements
 *
 * @param <E>
 */
public interface GraphSetVertexRefinement<E> {

	
	/**
	 * Splits the given vertices into clusters.
	 */
	public ArrayList<ArrayList<Vertex>> refine(LGraph<Integer, E> LG, Iterable<? extends Vertex> vertices);
	
	/**
	 * Splits all vertices into clusters
	 */
	public ArrayList<ArrayList<Vertex>> refine(LGraph<Integer, E> LG);

	public String getID();

	public void fitToDataset(ArrayList<LGraph<Integer, E>> lgs);
	
	default public void setTrainingsetIndex(boolean[] index)
	{
		System.err.print("method not implemented");
	}

}
