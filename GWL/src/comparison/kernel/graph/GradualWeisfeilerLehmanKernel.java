package comparison.kernel.graph;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import algorithm.graph.isomorphism.labelrefinement.VertexLabelConverter;
import algorithm.graph.vertexhierarchy.RefinementStep;
import comparison.kernel.ExplicitMappingKernel;
import comparison.kernel.basic.DotProductKernel;
import concepts.TransformationTools;
import datastructure.FeatureVector;
import datastructure.SparseFeatureVector;
import graph.AdjListEditableGraph;
import graph.AdjListRootedTree;
import graph.Graph;
import graph.LGraph;
import graph.LGraphTools;
import graph.AdjListRootedTree.AdjListRTreeVertex;
import graph.Graph.Vertex;
import graph.properties.EdgeArray;
import graph.properties.VertexArray;

/**
 * @author kriege/bause
 *
 * Weisfeiler Lehman Kernel, that uses a step by step refinement (gradual weisfeiler lehman refinement)
 *
 * @param <V>
 * @param <E>
 */
public class GradualWeisfeilerLehmanKernel<V,E> implements ExplicitMappingKernel<LGraph<V, E>, Integer> {
	HashMap<Vertex, AdjListRTreeVertex> mapToLeaf;
	AdjListRootedTree rt;
	public VertexArray<ArrayList<Vertex>> rtCluster;
	LGraph<Integer, E> LG;
	private boolean initializedWithGraphs;
	
	RefinementStep<E> refStep;
	int refinementSteps;
	private boolean sparse;
	
	public GradualWeisfeilerLehmanKernel(RefinementStep<E> refStep, int refinementSteps) {
		this(refStep, refinementSteps, false);
	}
	
	public GradualWeisfeilerLehmanKernel(ArrayList<? extends LGraph<V,E>> graphs, RefinementStep<E> refStep, int refinementSteps) {
		this(graphs, refStep, refinementSteps, false);
	}
	
	public GradualWeisfeilerLehmanKernel(ArrayList<? extends LGraph<V,E>> graphs, RefinementStep<E> refStep, int refinementSteps, boolean sparse) {
		this.refStep = refStep;
		this.refinementSteps = refinementSteps;
		generateHierarchy(graphs);
		this.initializedWithGraphs = true;
		this.sparse = sparse;
	}
	
	public GradualWeisfeilerLehmanKernel(RefinementStep<E> refStep, int refinementSteps, boolean sparse) {
		this.refStep = refStep;
		this.refinementSteps = refinementSteps;
		this.initializedWithGraphs = false;
		this.sparse = sparse;
	}
	
	private void generateHierarchy(List<? extends LGraph<V, E>> list) {
		
		// assign integer label from {1 ,... , # labels}
		// the labels correspond to the ids of the tree nodes; 0 is reserved for the root
		VertexLabelConverter<E> vlc = new VertexLabelConverter<E>(1);
		ArrayList<LGraph<Integer, E>> lgs = TransformationTools.transformAll(vlc, list);
		vlc.clearLabelMap(); 
		
		this.refStep.fitToDataset(lgs); //important for some refinements
		// create a single graph containing the data set
		// count vertices and edges
		int nV = 0, nE = 0;
		for (LGraph<Integer, E> lg : lgs) {
			nV += lg.getGraph().getVertexCount();
			nE += lg.getGraph().getEdgeCount();
		}
		AdjListEditableGraph G = new AdjListEditableGraph(nV, nE);
		VertexArray<Vertex> VVA = new VertexArray<>(G, nV, false);
		VertexArray<Integer> VA = new VertexArray<>(G, nV, false);
		EdgeArray<E> EA = new EdgeArray<>(G, nE, false);
		LG = new LGraph<>(G, VA, EA);
		for (LGraph<Integer, E> lg : lgs) {
			HashMap<Vertex, Vertex> map = LGraphTools.copyLGraph(lg, LG, false);
			for (Map.Entry<Vertex, Vertex>  e : map.entrySet()) {
				VVA.set(e.getValue(), e.getKey());
			}
		}
		
		// create the tree and initial partition
		rt = new AdjListRootedTree();
		rt.createRoot();
		rtCluster = new VertexArray<>(rt, true);
		// assign all vertices to the root
		ArrayList<Vertex> rootCluster = new ArrayList<>();
		rtCluster.set(rt.getRoot(), rootCluster);
		for (Vertex v : G.vertices()) {
			rootCluster.add(v);
		}

		// iteration 0, level at depth 1
		refineFromLabel(rt.getRoot());
		
		for (int i=0; i<refinementSteps; i++) {
			ArrayList<AdjListRTreeVertex> leaves = rt.findLeaves();
			ArrayList<ArrayList<ArrayList<Vertex>>> clusterings = this.refStep.getNextStep(rtCluster, leaves, LG);

			// apply clusterings to obtain refined labels and the next tree level
			for (int j=0; j<leaves.size(); j++) {
				refineFromClustering(leaves.get(j), clusterings.get(j));
			}
		}
		
		// create leafmap
		mapToLeaf = new HashMap<>();
		ArrayList<AdjListRTreeVertex> leaves = rt.findLeaves();
		for (AdjListRTreeVertex l : leaves) {
			ArrayList<Vertex> cluster = rtCluster.get(l);
			for (Vertex v : cluster) {
				mapToLeaf.put(VVA.get(v), l);
			}
		}
		
	}
	


	
	/**
	 * Creates the first level of the tree from the initial labels
	 * 
	 * @param root the root node
	 */
	private void refineFromLabel(AdjListRTreeVertex root) {
		for (Vertex v : rtCluster.get(root)) {
			Integer color = LG.getVertexLabel().get(v);
			while (rt.getVertexCount() <= color) {
				Vertex tv = rt.createChild(root);
				rtCluster.set(tv, new ArrayList<>());
			}
			AdjListRTreeVertex tv = rt.getVertex(color);
			rtCluster.get(tv).add(v);			
		}
	}
	
	/**
	 * Extends the tree according to the clusterings and updates the labels of the graph accordingly.
	 * 
	 * @param p a leaf node of the tree 
	 * @param clusterings a clustering of the vertices associated with the leaf
	 */
	private void refineFromClustering(AdjListRTreeVertex p, ArrayList<ArrayList<Vertex>> clusterings) {
		for (ArrayList<Vertex> clustering : clusterings) {
			AdjListRTreeVertex tv = rt.createChild(p);
			rtCluster.set(tv, clustering);
			for (Vertex v : clustering) {
				LG.getVertexLabel().set(v, tv.getIndex());
			}
		}
	}
	
	public AdjListRootedTree getHierarchy() {
		return rt;
	}


	@Override
	public String getID() {
		String s = "";
		if(this.sparse)
		{
			s = "sparse";
		}
		if(this.initializedWithGraphs)
		{
			return "LazyRefine"+refStep.getID()+"_"+refinementSteps+s;
		}
		return "GWL"+refStep.getID()+"_"+refinementSteps+s;
	}
	
	@Override
	public ArrayList<FeatureVector<Integer>> getFeatureVectors(List<? extends LGraph<V, E>> list) throws IllegalStateException {
		generateHierarchy(list);
		ArrayList<FeatureVector<Integer>> r = new ArrayList<FeatureVector<Integer>>(list.size());
		for (LGraph<V, E> t : list) {
			r.add(getFeatureVector(t));
		}
		return r;		
	}

	@Override
	public FeatureVector<Integer> getFeatureVector(LGraph<V, E> lg) throws IllegalStateException {
		if(this.rt==null)
		{
			throw new IllegalStateException("This should not be used in this way. The whole dataset has to be fed to the algorithm first.");
		}
		
		SparseFeatureVector<Integer> fv = new SparseFeatureVector<>();

		Graph g = lg.getGraph();
		for (Vertex v : g.vertices()) {
			AdjListRTreeVertex rv = mapToLeaf.get(v);
			while (rt.getRoot() != rv) {
				if(!this.sparse || rv.getParent().children().size()>1)
				{
					fv.increaseByOne(rv.getIndex());
				}
				rv = rv.getParent();
			}
		}
		return fv;
	}
	
	@Override
	public double[][] computeAll(List<? extends LGraph<V, E>> set) {
		ArrayList<FeatureVector<Integer>> fvs = getFeatureVectors(set);
		DotProductKernel<Integer> k = new DotProductKernel<Integer>();
		return k.computeAll(fvs);
	}


	@Override
	public double compute(LGraph<V, E> g1, LGraph<V, E> g2) {
		if(this.rt==null)
		{
			throw new IllegalStateException("This should not be used in this way. The whole dataset has to be fed to the algorithm first.");
		}
		DotProductKernel<Integer> k = new DotProductKernel<Integer>();
		return k.compute(getFeatureVector(g1), getFeatureVector(g2));
	}

}
