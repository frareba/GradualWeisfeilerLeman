package algorithm.graph.vertexhierarchy;

import java.util.ArrayList;

import graph.LGraph;
import graph.AdjListRootedTree.AdjListRTreeVertex;
import graph.Graph.Vertex;
import graph.properties.VertexArray;

/**
 * @author bause
 * Interface for refinements
 * @param <E>
 */
public interface RefinementStep<E> {
	/**
	 * Do the next refinementStep and return the resulting clustering
	 * @param rtCluster the map cluster -> vertex
	 * @param leaves the current clustering
	 * @param LG graph with vertices, that are to be refined
	 * @return the resulting clustering
	 */
	public ArrayList<ArrayList<ArrayList<Vertex>>> getNextStep(VertexArray<ArrayList<Vertex>> rtCluster, ArrayList<AdjListRTreeVertex> leaves, LGraph<Integer, E> LG);

	public String getID();

	public void fitToDataset(ArrayList<LGraph<Integer, E>> lgs);
	
	default public void setTrainingsetIndex(boolean[] index)
	{
		System.err.print("method not implemented");
	}
	
}
