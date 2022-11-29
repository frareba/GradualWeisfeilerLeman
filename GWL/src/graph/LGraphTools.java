package graph;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

import datastructure.Triple;
import graph.Graph.Edge;
import graph.Graph.Vertex;
import graph.properties.EdgeArray;
import graph.properties.VertexArray;

public class LGraphTools {
	
	private static final long seed = 88686765;
	private static final Random rand = new Random(seed);
	
	public static <V,E> HashMap<Vertex, Vertex> copyLGraph(LGraph<V,E> source, LGraph<V, E> target, boolean shuffle) {
		Graph sg = source.getGraph();
		VertexArray<V> sva = source.getVertexLabel();
		EdgeArray<E> sea = source.getEdgeLabel();
		ExtendibleGraph tg = (ExtendibleGraph)target.getGraph();
		VertexArray<V> tva = target.getVertexLabel();
		EdgeArray<E> tea = target.getEdgeLabel();
		
		HashMap<Vertex, Vertex> sourceToTargetMap = new HashMap<Vertex, Vertex>();
		Iterable<? extends Vertex> sV = sg.vertices();
		if (shuffle) {
			ArrayList<Vertex> copy = new ArrayList<>(sg.getVertexCount());
			for (Vertex v : sV) copy.add(v);
			Collections.shuffle(copy);
			sV = copy;
		}
		for (Vertex sv : sV) {
			Vertex tv = tg.createVertex();
			sourceToTargetMap.put(sv, tv);
			tva.set(tv, sva.get(sv));
		}
		for (Edge se : sg.edges()) {
			Vertex su = se.getFirstVertex();
			Vertex sv = se.getSecondVertex();
			Vertex tu = sourceToTargetMap.get(su);
			Vertex tv = sourceToTargetMap.get(sv);
			Edge te = tg.createEdge(tu, tv);
			tea.set(te, sea.get(se));
		}
		for (String property : source.graph.getProperties())
		{
			target.graph.setProperty(property, source.graph.getProperty(property));
		}
		
		return sourceToTargetMap;
	}
	
	public static <V,E> LGraph<V, E> copyLGraph(LGraph<V,E> source) {
		Graph g = new AdjListGraph(source.getGraph().getVertexCount(), source.getGraph().getEdgeCount());
		VertexArray<V> va = new VertexArray<V>(g, true);
		EdgeArray<E> ea = new EdgeArray<E>(g, true);
		LGraph<V, E> target = new LGraph<V, E>(g, va, ea);
		copyLGraph(source, target, false);
		return target;
	}

	/**
	 * Returns a copy of the given labeled graph, where the nodes are shuffled.
	 * The affects their id and iteration oder.
	 * 
	 * @param source the original labeled graph.
	 * @return a shuffled copy
	 */
	public static <V,E> LGraph<V, E> shuffleLGraph(LGraph<V,E> source) {
		Graph g = new AdjListGraph(source.getGraph().getVertexCount(), source.getGraph().getEdgeCount());
		VertexArray<V> va = new VertexArray<V>(g, true);
		EdgeArray<E> ea = new EdgeArray<E>(g, true);
		LGraph<V, E> target = new LGraph<V, E>(g, va, ea);
		copyLGraph(source, target, true);
		return target;
		
	}
	
	public static LGraph<String, String> createInducedLabeledSubgraph(LGraph<?, ?> lg, Collection<Vertex> vertices) {
		Graph g = lg.getGraph();
		VertexArray<?> va = lg.getVertexLabel();
		EdgeArray<?> ea = lg.getEdgeLabel();
		
		AdjListGraph g2 = new AdjListGraph();
		VertexArray<String> va2 = new VertexArray<String>(g2,true);
		EdgeArray<String> ea2 = new EdgeArray<String>(g2,true);
		
		HashMap<Vertex, Vertex> sourceToTargetMap = new HashMap<Vertex, Vertex>();
		for (Vertex v : vertices) {
			Vertex v2 = g2.createVertex();
			va2.set(v2, va.get(v).toString());
			sourceToTargetMap.put(v, v2);
		}
		for (Edge e : g.edges()) {
			Vertex u = e.getFirstVertex();
			Vertex v = e.getSecondVertex();
			// TODO use hashset?
			if (vertices.contains(u) && vertices.contains(v)) {
				Vertex u2 = sourceToTargetMap.get(u);
				Vertex v2 = sourceToTargetMap.get(v);
				Edge e2 = g2.createEdge(u2, v2);
				ea2.set(e2, ea.get(e).toString());
			}
		}
		
		//TODO: what is with the properties? I want the subgraph to have at least the same class as the original one
		for(String p : g.getProperties())
		{
			g2.setProperty(p, g.getProperty(p));
		}
		return new LGraph<String, String>(g2, va2, ea2);
	}
	
	public static <V extends Comparable<V>,E> LGraph<Triple<V, E, V>, V> createLineGraph(LGraph<V, E> lg) {
		Graph g = lg.getGraph();
		VertexArray<V> va = lg.getVertexLabel();
		EdgeArray<E> ea = lg.getEdgeLabel();
		
		// the correct edge size should be 1/2 * \sum_v deg(v)*(deg(v)-1)
		ExtendibleGraph rg = new AdjListGraph(g.getEdgeCount(), 0);
		VertexArray<Triple<V, E, V>> rva = new VertexArray<Triple<V, E, V>>(rg,g.getEdgeCount());
		EdgeArray<V> rea = new EdgeArray<V>(rg,true);
		
		for (Edge e : g.edges()) {
			Vertex v = rg.createVertex();
			assert (e.getIndex() == v.getIndex());
			V lu = va.get(e.getFirstVertex());
			V lv = va.get(e.getSecondVertex());
			Triple<V, E, V> label;
			if (lu.compareTo(lv) < 0) {
				label = new Triple<V, E, V>(lu, ea.get(e), lv);
			} else {
				label = new Triple<V, E, V>(lv, ea.get(e), lu);
			}
			rva.set(v, label);
		}
		// TODO use indices and start most inner loop from i
		for (Vertex v : g.vertices()) {
			for (Edge e : v.edges()) {
				for (Edge f : v.edges()) {
					if (e.getOppositeVertex(v).getIndex() < f.getOppositeVertex(v).getIndex()) {
						Edge rge = rg.createEdge(rg.getVertex(e.getIndex()), rg.getVertex(f.getIndex()));
						rea.set(rge, va.get(v));
					}
				}				
			}
		}
		return new LGraph<Triple<V, E, V>, V>(rg, rva, rea);
	}

	
	/**
	 * Generates vertex and edge labels which are uniformly distributed in
	 * {0, ..., edgeLabelCount-1} and {0, ..., vertexLabelCount-1}.
	 * @param g the graph to generate labels for
	 * @param edgeLabelCount the number of different edge labels
	 * @param vertexLabelCount the number of different vertex labels
	 * @return a labeled graph containing g together with the generated labels
	 */
	public static LGraph<Integer, Integer> createRandomLabeling(Graph g, int edgeLabelCount, int vertexLabelCount) {
		VertexArray<Integer> va = new VertexArray<Integer>(g);
		EdgeArray<Integer> ea = new EdgeArray<Integer>(g);

		for (Vertex v : g.vertices()) {
			va.set(v, rand.nextInt(vertexLabelCount));
		}
		for (Edge e : g.edges()) {
			ea.set(e, rand.nextInt(edgeLabelCount));
		}

		return new LGraph<Integer, Integer>(g, va, ea);
	}
	
	public static LGraph<Integer, Integer> createRandomLabeling(Graph g, int edgeLabelCount, int vertexLabelCount, double pV, double pE) {
		VertexArray<Integer> va = new VertexArray<Integer>(g);
		EdgeArray<Integer> ea = new EdgeArray<Integer>(g);

		for (Vertex v : g.vertices()) {
			if (rand.nextDouble() < pV) {
				va.set(v, rand.nextInt(vertexLabelCount)+1);
			} else {
				va.set(v, 0);
			}
		}
		for (Edge e : g.edges()) {
			if (rand.nextDouble() < pE) {
				ea.set(e, rand.nextInt(edgeLabelCount)+1);
			} else {
				ea.set(e, 0);
			}
		}

		return new LGraph<Integer, Integer>(g, va, ea);
	}
	
	/**
	 * @see GraphTools#createRandomLabeling(Graph, int, int)
	 */
	public static LGraph<String, String> createRandomStringLabeling(Graph g, int edgeLabelCount, int vertexLabelCount) {
		VertexArray<String> va = new VertexArray<String>(g);
		EdgeArray<String> ea = new EdgeArray<String>(g);

		for (Vertex v : g.vertices()) {
			va.set(v, String.valueOf(rand.nextInt(vertexLabelCount)));
		}
		for (Edge e : g.edges()) {
			ea.set(e, String.valueOf(rand.nextInt(edgeLabelCount)));
		}

		return new LGraph<String, String>(g, va, ea);
	}
	
	public static <V,E> LGraph<V,E> createUniformLabeling(Graph g, V vertexLabel, E edgeLabel) {
		VertexArray<V> va = new VertexArray<V>(g);
		EdgeArray<E> ea = new EdgeArray<E>(g);

		for (Vertex v : g.vertices()) {
			va.set(v, vertexLabel);
		}
		for (Edge e : g.edges()) {
			ea.set(e, edgeLabel);
		}

		return new LGraph<V, E>(g, va, ea);
	}
	
	public static LGraph<Integer, Integer> getIndexLabeledGraph(Graph g) {
		VertexArray<Integer> va = new VertexArray<Integer>(g);
		EdgeArray<Integer> ea = new EdgeArray<Integer>(g);
		for (Vertex v : g.vertices()) {
			va.set(v, v.getIndex());
		}
		for (Edge e : g.edges()) {
			ea.set(e, e.getIndex());
		}
		return new LGraph<Integer, Integer>(g, va, ea);
	}
	
	public static <V,E> LGraph<V,E> graphUnion(Collection<LGraph<V,E>> lgs) {
		AdjListEditableGraph g = new AdjListEditableGraph();
		VertexArray<V> va = new VertexArray<V>(g, true);
		EdgeArray<E> ea = new EdgeArray<E>(g, true);
		LGraph<V, E> result = new LGraph<V, E>(g, va, ea);
		
		for (LGraph<V,E> lg : lgs) {
			LGraphTools.copyLGraph(lg, result, false);
		}
		return result;
	}
	
	public static <V,E> void printGraph(LGraph<V,E> graph)
	{
		for(String property : graph.getGraph().getProperties())
		{
			System.out.println(property +": "+graph.getGraph().getProperty(property).toString());
		}
		System.out.println();
		for(Vertex v: graph.getGraph().vertices())
		{
			System.out.println(v.getIndex() +": "+graph.getVertexLabel().get(v).toString());
			System.out.print(v.getDegree() +" [");
			for(Vertex u: v.neighbors())
			{
				System.out.print(u.getIndex()+ " ");
			}
			System.out.println("]");
		}
		System.out.println();
	}
}
