package graph;

import java.util.ArrayList;
import java.util.Iterator;

import util.SkippingIterator;

public class AdjListEditableGraph extends AdjListGraph implements EditableGraph {
	
	int vertexCount = 0;
	int edgeCount = 0;
	
	public AdjListEditableGraph() {
		super();
	}
	
	public AdjListEditableGraph(int n, int m) {
		super(n,m);
	}

	// TODO use more efficient implementation without skipping
	public Iterable<AdjListEdge> edges() {
		return new Iterable<AdjListEdge>() {
			public Iterator<AdjListEdge> iterator() {
				return new SkippingIterator<AdjListEdge>(edges.iterator());
			}
		};
	}

	// TODO use more efficient implementation without skipping
	public Iterable<AdjListVertex> vertices() {
		return new Iterable<AdjListVertex>() {
			public Iterator<AdjListVertex> iterator() {
				return new SkippingIterator<AdjListVertex>(vertices.iterator());
			}
		};
	}

	public int getVertexCount() {
		return vertexCount;
	}
	
	public int getEdgeCount() {
		return edgeCount;
	}
	
	public int getNextVertexIndex() {
		return vertices.size();
	}

	public int getNextEdgeIndex() {
		return edges.size();
	}
	
	public AdjListVertex createVertex() {
		vertexCount++;
		return super.createVertex();
	}
	
	public AdjListEdge createEdge(Vertex u, Vertex v) {
		edgeCount++;
		return super.createEdge(u, v);
	}
	
	public void deleteVertex(Vertex v) {
		vertices.set(v.getIndex(), null);
		ArrayList<AdjListEdge> es = ((AdjListVertex)v).edges();
		while (!es.isEmpty()) {
			deleteEdge(es.get(0));
		}
		vertexCount--;
		
		notifyVertexDeleted(v);
	}
	
	public void deleteEdge(Edge e) {
		edges.set(e.getIndex(), null);
		((AdjListVertex)e.getFirstVertex()).removeEdge((AdjListEdge)e);
		((AdjListVertex)e.getSecondVertex()).removeEdge((AdjListEdge)e);
		edgeCount--;
		
		notifyEdgeDeleted(e);
	}
}