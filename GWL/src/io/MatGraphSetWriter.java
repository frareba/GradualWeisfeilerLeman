package io;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLSparse;
import com.jmatio.types.MLStructure;

import benchmark.dataset.AttrDataset;
import graph.Graph.Edge;
import graph.Graph.Vertex;
import graph.attributes.AttributedGraph;
import graph.attributes.Attributes;

public class MatGraphSetWriter {

	
	public void write(AttrDataset ags, File matFile) throws IOException {

		MLStructure ds = new MLStructure(ags.getID(), new int[]{1, ags.size()});
		int i=-1;
		for (AttributedGraph ag : ags) {
			i++;

			// adjacency matrix am
//			double[][] A = GraphTools.getAdjacencyMatrix(ag.getGraph());
//			MLArray mlA = new MLDouble("mat", A);
			int n = ag.getGraph().getVertexCount();
			int m = ag.getGraph().getEdgeCount();
			MLSparse am = new MLSparse("am", new int[]{n,n}, 0, 2*m);
			for (Edge e : ag.getGraph().edges()) {
				int u = e.getFirstVertex().getIndex();
				int v = e.getSecondVertex().getIndex();
				am.set(1d, u, v);
				am.set(1d, v, u);
			}
			ds.setField("am", am, 0, i);
		
			if (ag.hasNominalNodeLabel()) {
				// node label nl
				MLStructure nl = new MLStructure("nl", new int[]{1,1});
				MLDouble nlValues = new MLDouble("values", new int[]{n,1});
				for (Vertex v : ag.getGraph().vertices()) {
					Attributes aV = ag.getVertexLabel().get(v);
					if (aV.getNominalAttributeCount() != 1) {
						throw new IllegalArgumentException("Multiple node labels.");
					}
					nlValues.set(Double.valueOf(aV.getNominalAttribute(0).toString()), v.getIndex());
				}
				nl.setField("values", nlValues);
				ds.setField("nl", nl,  0, i);
			}
			
			if (ag.hasNominalEdgeLabel()) {
				// edge labels
				MLStructure el = new MLStructure("el", new int[]{1,1});
				MLDouble elValues = new MLDouble("values", new int[]{2*m,3});
				int iE = 0;
				for (Vertex v : ag.getGraph().vertices()) {
					for (Edge e : v.edges()) {
						Attributes aE = ag.getEdgeLabel().get(e);
						if (aE.getNominalAttributeCount() != 1) {
							throw new IllegalArgumentException("Multiple edge labels.");
						}
						elValues.set(Double.valueOf(v.getIndex()+1), iE, 0);
						elValues.set(Double.valueOf(e.getOppositeVertex(v).getIndex()+1), iE, 1);
						elValues.set(Double.valueOf(aE.getNominalAttribute(0).toString()), iE, 2);
						iE++;
					}
				}
				el.setField("values", elValues);
				ds.setField("el", el,  0, i);
			}
			
			// adjacency list
			MLCell al = new MLCell("al", new int[]{n,1});
			for (Vertex u : ag.getGraph().vertices()) {
				MLDouble alU = new MLDouble("alU", new int[]{1,u.getDegree()});
				int nCount = 0;
				for (Vertex v : u.neighbors()) {
					alU.set(Double.valueOf(v.getIndex()+1), 0, nCount++);
				}
				al.set(alU, u.getIndex(),0);
			}
			ds.setField("al", al,  0, i);


		}
		
		// convert class labels
		MLDouble classLabelData = new MLDouble("l"+ags.getID(), new int[] {ags.size(), 1});
		String[] classLabels = ags.getClassLabels();
//		System.out.println(classLabels.length);
//		System.out.println(ags.size());
		for (int k=0; k<classLabels.length; k++) {
//			System.out.println(k);
			classLabelData.set(Double.valueOf(classLabels[k]), k);
		}
		
		ArrayList<MLArray> data = new ArrayList<MLArray>(1);
		data.add(ds);
		data.add(classLabelData);

		
		MatFileWriter mfr = new MatFileWriter(matFile, data);
	}
	
	
}
