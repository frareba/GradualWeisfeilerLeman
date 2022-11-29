package comparison.kernel.graph;

import algorithm.graph.vertexhierarchy.GraphSetVertexRefinementStep;
import algorithm.graph.vertexhierarchy.NeighborhoodLabelsKMeans;
import comparison.kernel.ExplicitMappingKernel;
import comparison.kernel.basic.DiracKernel;
import comparison.kernel.basic.HistogramIntersectionKernel;
import datastructure.Triple;
import graph.LGraph;

/**
 * Computes optimal assignment kernels by histogram intersection.
 * 
 * @author Nils Kriege
 *
 * @param <V>
 * @param <E>
 * @param <F>
 */
public class OptimalAssignmentGraphKernel<V, E, F> extends FeatureVectorKernel<V, E, F> {
	
	
	public OptimalAssignmentGraphKernel(ExplicitMappingKernel<LGraph<V, E>, F> explicitKernel) {
		super(new HistogramIntersectionKernel<>(), explicitKernel);
	}
	
	@Override
	public String getID() {
		return "OA-"+explicitKernel.getID();
	}

	public static class WeisfeilerLehmanSubtree<V,E> extends OptimalAssignmentGraphKernel<V, E, Integer> {
		public WeisfeilerLehmanSubtree(int height) {
			super(new WeisfeilerLehmanSubtreeKernel<>(height));
		}

		@SuppressWarnings("rawtypes")
		@Override
		public String getID() {
			return "WLOA_"+((WeisfeilerLehmanSubtreeKernel)explicitKernel).getHeight();
		}
	}
	
	public static class GradualWeisfeilerLehmankMeansSubtreeKernel<V,E> extends OptimalAssignmentGraphKernel<V, E, Integer> {
		public GradualWeisfeilerLehmankMeansSubtreeKernel(int height, int k) {
			super(new GradualWeisfeilerLehmanKernel<>(new GraphSetVertexRefinementStep(new NeighborhoodLabelsKMeans<>(k)),height));
		}

		@Override
		public String getID() {
			return "GWLOA_"+(explicitKernel.getID());
		}
	}
}
