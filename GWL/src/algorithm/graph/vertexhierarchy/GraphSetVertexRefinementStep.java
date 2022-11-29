package algorithm.graph.vertexhierarchy;

import java.util.ArrayList;

import graph.AdjListRootedTree.AdjListRTreeVertex;
import graph.Graph.Vertex;
import graph.properties.VertexArray;
import graph.LGraph;

/**
 * @author bause
 *
 * Class that does a whole refinement step using specified GraphSetVertexRefinement
 *
 * @param <E>
 */
public class GraphSetVertexRefinementStep<E> implements RefinementStep<E>{

	private GraphSetVertexRefinement<E> ref;
	public GraphSetVertexRefinementStep(GraphSetVertexRefinement<E> ref)
	{
		this.ref = ref;
	}
	@Override
	public ArrayList<ArrayList<ArrayList<Vertex>>> getNextStep(VertexArray<ArrayList<Vertex>> rtCluster, ArrayList<AdjListRTreeVertex> leaves, LGraph<Integer, E> LG) {
		ArrayList<ArrayList<ArrayList<Vertex>>> clusterings = new ArrayList<ArrayList<ArrayList<Vertex>>>();
		for (AdjListRTreeVertex leaf : leaves) {				
			
			ArrayList<Vertex> vertices = rtCluster.get(leaf);
			
			if (vertices.size() < 2) {
				// adopt current vertex set
				ArrayList<ArrayList<Vertex>> clustering = new ArrayList<ArrayList<Vertex>>();
				clustering.add(vertices);
				clusterings.add(clustering); 
				continue;
			}
			// cluster the vertices
			ArrayList<ArrayList<Vertex>> clustering = ref.refine(LG, rtCluster.get(leaf));
			clusterings.add(clustering); 
		}
		return clusterings;
	}

	@Override
	public String getID() {
		return this.ref.getID();
	}
	@Override
	public void fitToDataset(ArrayList<LGraph<Integer, E>> lgs) {
		this.ref.fitToDataset(lgs);
	}
	
	@Override
	public void setTrainingsetIndex(boolean[] index)
	{
		this.ref.setTrainingsetIndex(index);
	}

}
