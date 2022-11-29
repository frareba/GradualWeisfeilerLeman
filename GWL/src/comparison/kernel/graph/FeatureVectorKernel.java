package comparison.kernel.graph;

import java.util.ArrayList;
import java.util.List;

import comparison.kernel.ExplicitMappingKernel;
import comparison.kernel.Kernel;
import datastructure.FeatureVector;
import graph.LGraph;

/**
 * Applies a kernel to the feature vectors obtained from an explicit
 * mapping kernel. 
 * 
 * @author Nils Kriege
 *
 * @param <V>
 * @param <E>
 * @param <F>
 */
public class FeatureVectorKernel<V, E, F> implements Kernel<LGraph<V, E>> {

	ArrayList<LGraph<V, E>> graphs = new ArrayList<>(2);

	ExplicitMappingKernel<LGraph<V, E>, F> explicitKernel;
	Kernel<FeatureVector<F>> fvKernel;
	
	public FeatureVectorKernel(Kernel<FeatureVector<F>> fvKernel, ExplicitMappingKernel<LGraph<V, E>, F> explicitKernel) {
		this.explicitKernel = explicitKernel;
		this.fvKernel = fvKernel;
	}
	
	@Override
	public double compute(LGraph<V, E> g1, LGraph<V, E> g2) {
		graphs.set(0, g1);
		graphs.add(1, g2);
		ArrayList<FeatureVector<F>> maps = explicitKernel.getFeatureVectors(graphs);
		return fvKernel.compute(maps.get(0), maps.get(1));
	}
	
	@Override
	public double[][] computeAll(List<? extends LGraph<V, E>> set) {
		ArrayList<FeatureVector<F>> maps = explicitKernel.getFeatureVectors(set);
		return fvKernel.computeAll(maps);
	}
	
	@Override
	public double[][] computeAll(List<? extends LGraph<V, E>> setA, List<? extends LGraph<V, E>> setB) {
		ArrayList<LGraph<V, E>> all = new ArrayList<>();
		all.addAll(setA);
		all.addAll(setB);
		ArrayList<FeatureVector<F>> maps = explicitKernel.getFeatureVectors(all);
		List<FeatureVector<F>> mapsA = maps.subList(0, setA.size());
		List<FeatureVector<F>> mapsB = maps.subList(setA.size(), maps.size());
		
		return fvKernel.computeAll(mapsA, mapsB);

	}

	@Override
	public String getID() {
		return "FV-"+fvKernel.getID()+"-"+explicitKernel.getID();
	}

}
