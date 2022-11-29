package comparison.kernel.basic;

import comparison.kernel.ExplicitMappingKernel;
import datastructure.FeatureVector;
import datastructure.SparseFeatureVector;

public class DotProductKernel<T> implements ExplicitMappingKernel<FeatureVector<T>,T> {

	/*
	public static final boolean USE_EIGEN = true;
	*/
	
	public double compute(FeatureVector<T> g1, FeatureVector<T> g2) {
		return g1.dotProduct(g2);
	}

	/*
	@Override
	public double[][] compute(List<? extends FeatureVector<T>> set) {
		if (USE_EIGEN) {
			return FeatureVectors.computeGram(FeatureVectors.toIntegerIndexed(set));
		} else {
			return ExplicitMappingKernel.super.compute(set);
		}
	}
	*/
	
	public String getID() {
		return "IP";
	}

	@Override
	public FeatureVector<T> getFeatureVector(FeatureVector<T> t) throws IllegalStateException {
		return new SparseFeatureVector<T>(t);
	}
	
}
