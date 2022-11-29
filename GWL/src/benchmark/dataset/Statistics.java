package benchmark.dataset;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;

import algorithm.shortestpath.SPTools;
import graph.ConnectivityTools;
import graph.Graph;
import graph.Graph.Edge;
import graph.Graph.Vertex;
import graph.GraphTools;
import graph.LGraph;
import graph.properties.EdgeArray;
import graph.properties.VertexArray;

public class Statistics {
	
	
	public static void printClassStatistics(String[] classes) {
		
		HashMap<String,Integer> classLabels = new HashMap<String,Integer>();

		for (String c : classes) {
			Integer counter = classLabels.get(c);
			classLabels.put(c, (counter == null ? 1 : counter + 1));
		}
		System.out.println("Class labels:      "+classLabels.size()+"  "+classLabels);
	}
	
	public static <IV,IE> void printLabeledGraphStatistics(Collection<? extends LGraph<?,?>> graphs) {
		
		printClassStatistics(LGDataset.getClassesArray(graphs));
		printGraphStatistics(LGraph.toGraphCollection(graphs));
		
		HashMap<Object,Integer> vertexLabel = new HashMap<Object,Integer>();
		HashMap<Object,Integer> edgeLabel = new HashMap<Object,Integer>();
		for (LGraph<?,?> lg : graphs) {
			Graph g = lg.getGraph();
			VertexArray<?> va = lg.getVertexLabel();
			for (Vertex v : g.vertices()) {
				Integer counter = vertexLabel.get(va.get(v));
				vertexLabel.put(va.get(v), (counter == null ? 1 : counter + 1));
			}
			EdgeArray<?> ea = lg.getEdgeLabel();
			for (Edge e : g.edges()) {
				Integer counter = edgeLabel.get(ea.get(e));
				edgeLabel.put(ea.get(e), (counter == null ? 1 : counter + 1));
			}
		}
		
		System.out.println("Vertex labels:     "+vertexLabel.size()+"  "+vertexLabel);
		System.out.println("Edge labels:       "+edgeLabel.size()+"  "+edgeLabel);
	}
	
	public static <IV,IE> void printLabeledGraphStatistics(AttrDataset graphs) {

		printLabeledGraphStatistics(graphs.getSDataset());
		
		// get attribute counts, first graph, first vertex/edge
		int vAttrCount = graphs.get(0).getVertexLabel().get(graphs.get(0).getGraph().getVertex(0)).getRealValuedAttributeCount();
		int eAttrCount = graphs.get(0).getEdgeLabel().get(graphs.get(0).getGraph().getEdge(0)).getRealValuedAttributeCount();

		System.out.println("Vertex attributes: "+vAttrCount);
		System.out.println("Edge attributes:   "+eAttrCount);
	}

	
	public static void printGraphStatistics(Collection<? extends Graph> graphs) {
		int n = graphs.size();
		long totalV = 0;
		long totalE = 0;
		long totalDegree = 0;
		int maxV = Integer.MIN_VALUE;
		int minV = Integer.MAX_VALUE;
		int maxE = Integer.MIN_VALUE;
		int minE = Integer.MAX_VALUE;
		int maxDegree = Integer.MIN_VALUE;
		int minDegree = Integer.MAX_VALUE;
		int disconnectedGraphs = 0;
		HashMap<Integer,Integer> degreeDistribution = new HashMap<>();
		for (Graph g : graphs) {
			totalV += g.getVertexCount();
			totalE += g.getEdgeCount();
			maxV = Math.max(maxV, g.getVertexCount());
			minV = Math.min(minV, g.getVertexCount());
			maxE = Math.max(maxE, g.getEdgeCount());
			minE = Math.min(minE, g.getEdgeCount());
			for (Vertex v : g.vertices()) {
				totalDegree += v.getDegree();
				maxDegree = Math.max(maxDegree, v.getDegree());
				minDegree = Math.min(minDegree, v.getDegree());
				Integer counter = degreeDistribution.get(v.getDegree());
				degreeDistribution.put(v.getDegree(), (counter == null ? 1 : counter + 1));
			}
			if (!ConnectivityTools.isConnected(g)) disconnectedGraphs++;
		}

		System.out.println("Number of graphs:    "+n);
		System.out.println("Disconnected graphs: "+disconnectedGraphs);
		System.out.println("Total |V|:           "+totalV);
		System.out.println("Total |E|:           "+totalE);
		System.out.println("Max. |V|:            "+maxV);
		System.out.println("Min. |V|:            "+minV);
		System.out.println("Avg. |V|:            "+(double)totalV/n);
		System.out.println("Max. |E|:            "+maxE);
		System.out.println("Min. |E|:            "+minE);
		System.out.println("Avg. |E|:            "+(double)totalE/n);
		System.out.println("Max. deg.:           "+maxDegree);
		System.out.println("Min. deg.:           "+minDegree);
		System.out.println("Avg. deg.:           "+(double)totalDegree/totalV);
		System.out.println("Deg. frequencies:    "+degreeDistribution);
		
	}
	
	/**
	 * Density and diameter
	 * @param graphs
	 */
	public static void printGraphStatistics2(Collection<? extends Graph> graphs) {
		double[] diameters = new double[graphs.size()];
		double[] densities = new double[graphs.size()];
		
		int i=0;
		for (Graph g : graphs) {
			diameters[i]=SPTools.getDiameter(g);
			densities[i]=GraphTools.getDensity(g);
			i++;
		}
		
//		System.out.println(Arrays.toString(densities));
		Arrays.sort(diameters);
		Arrays.sort(densities);
		
		System.out.println("Min. Diameter:  "+diameters[0]);
		System.out.println("Max. Diameter:  "+diameters[diameters.length-1]);
		System.out.println("Avg. Diameter:  "+average(diameters));
		System.out.println("Med. Diameter:  "+median(diameters));
		System.out.println("Min. Density:  "+densities[0]);
		System.out.println("Max. Density:  "+densities[densities.length-1]);
		System.out.println("Avg. Density:  "+average(densities));
		System.out.println("Med. Density:  "+median(densities));
		
//		System.out.println(Arrays.toString(densities));
	}
	
	public static double median(double[] numArray) {
		Arrays.sort(numArray);
		if (numArray.length % 2 == 0)
		    return ((double)numArray[numArray.length/2] + (double)numArray[numArray.length/2 - 1])/2;
		else
		    return (double) numArray[numArray.length/2];
	}
	
	public static double average(double[] numArray) {
		double avg = 0;
		for (double d : numArray) {
			avg += d;
		}
		avg /= numArray.length;
		return avg;
	}


	public static void printGraphStatistics(Graph g) {
		long totalDegree = 0;
		int maxDegree = Integer.MIN_VALUE;
		for (Vertex v : g.vertices()) {
			totalDegree += v.getDegree();
			maxDegree = Math.max(maxDegree, v.getDegree());
		}
		int n = g.getVertexCount();
		int m = g.getEdgeCount();
		
		System.out.println("|V|: "+n);
		System.out.println("|E|: "+m);
		System.out.println("Density: "+((double)m)/(n*(n-1)));
		System.out.println("Max. deg.: "+maxDegree);
		System.out.println("Avg. deg.: "+(double)totalDegree/g.getVertexCount());
		
	}
	
	public static void printLabeledGraphStatistics(LGraph<?,?> lg) {
		Graph g = lg.getGraph();
		
		printGraphStatistics(g);
		
		HashMap<Object,Integer> vertexLabel = new HashMap<Object,Integer>();
		HashMap<Object,Integer> edgeLabel = new HashMap<Object,Integer>();
		VertexArray<?> va = lg.getVertexLabel();
		for (Vertex v : g.vertices()) {
			Integer counter = vertexLabel.get(va.get(v));
			vertexLabel.put(va.get(v), (counter == null ? 1 : counter + 1));
		}
		EdgeArray<?> ea = lg.getEdgeLabel();
		for (Edge e : g.edges()) {
			Integer counter = edgeLabel.get(ea.get(e));
			edgeLabel.put(ea.get(e), (counter == null ? 1 : counter + 1));
		}
		
		System.out.println("Vertex labels:       "+vertexLabel.size()+"  "+vertexLabel);
		System.out.println("Edge labels:         "+edgeLabel.size()+"  "+edgeLabel);
	}
	
}
