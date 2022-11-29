package util.kernel;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_print_interface;
import libsvm.svm_problem;


public class AccuracyTestLibSVMNative extends AccuracyTest {

	public AccuracyTestLibSVMNative(String folder) {
		super(folder);
		init();
	}

	public AccuracyTestLibSVMNative(File folder, List<String> datasets, List<String> kernels, File accLogFile,
			File selLogFile) {
		super(folder, datasets, kernels, accLogFile, selLogFile);
		init();
	}
	
	private void init() {
		// silence svm
		
		svm.svm_set_print_string_function(new svm_print_interface() {
			@Override
			public void print(String arg0) {
				// TODO Auto-generated method stub
			}
		});
	}


	@Override
	public double predict(ArrayList<Integer> testSet, ArrayList<Integer> trainSet, Parameter param) throws IOException, InterruptedException {

		// train 
		svm_problem prob = new svm_problem();
		prob.l = trainSet.size();
		prob.x = new svm_node[prob.l][];
		prob.y = new double[prob.l];
		for(int i=0;i<prob.l;i++) {
			svm_node[] x = new svm_node[param.gram.getDimension()+1];
			prob.x[i] = x;
			svm_node node = new svm_node();
			node.index = 0;
			int iIndex = trainSet.get(i);
			node.value = iIndex+1;
			prob.x[i][0] = node;
			for (int j=0; j<param.gram.getDimension(); j++) {
				node = new svm_node();
				node.index = j+1;
				node.value = param.gram.gram[iIndex][j];
				prob.x[i][j+1] = node;
			}
			prob.y[i] = Double.valueOf(param.gram.classes[iIndex]);
		}
		
		svm_parameter svm_param = getDefaultParameters();
		svm_param.C = param.C;

		svm_model model = svm.svm_train(prob,svm_param);
		
		// predict
		int d = testSet.size();
		int total_correct = 0;
		for(int i=0;i<d;i++) {
			svm_node[] x = new svm_node[param.gram.getDimension()+1];
			svm_node node = new svm_node();
			node.index = 0;
			int iIndex = testSet.get(i);
			node.value = iIndex+1;
			x[0] = node;
			for (int j=0; j<param.gram.getDimension(); j++) {
				node = new svm_node();
				node.index = j+1;
				node.value = param.gram.gram[iIndex][j];
				x[j+1] = node;
			}
					
			double y = Double.valueOf(param.gram.classes[iIndex]);
			double v = svm.svm_predict(model,x);
			if (y == v) total_correct++;
		}

		return 100.0*total_correct/d;
	}
	
	/**
	 * Finds the best parameters based on the test set
	 * @param names available kernels
	 * @return name of the best kernel and C parameter
	 */
	public Parameter searchBestParameter(ArrayList<GramMatrix> matrices, ArrayList<Integer> trainSet) {

		ExecutorService es = Executors.newFixedThreadPool(PARALLEL_THREADS);
		ArrayList<Future<Parameter>> futures = new ArrayList<>();

		for (GramMatrix matrix : matrices) {
			
			// create problem
			svm_problem prob = new svm_problem();
			prob.l = trainSet.size();
			prob.x = new svm_node[prob.l][];
			prob.y = new double[prob.l];
			for(int i=0;i<prob.l;i++) {
				svm_node[] x = new svm_node[matrix.getDimension()+1];
				prob.x[i] = x;
				int iIndex = trainSet.get(i);
				// index node
				svm_node node = new svm_node();
				node.index = 0;
				node.value = iIndex+1;
				prob.x[i][0] = node;
				// kernel values
				for (int j=0; j<matrix.getDimension(); j++) {
					node = new svm_node();
					node.index = j+1;
					node.value = matrix.gram[iIndex][j];
					prob.x[i][j+1] = node;
				}
				prob.y[i] = Double.valueOf(matrix.classes[iIndex]);
			}
			
			for (double c : C_RANGE) {
				
				svm_parameter param = getDefaultParameters();
				param.C = c;

				futures.add(es.submit(new Callable<Parameter>() {
					@Override
					public Parameter call() throws Exception {
						final Parameter result = new Parameter(null, 0, Double.MIN_VALUE);
						Thread d = new Thread(new Runnable() {
							public void run() {
								int total_correct = 0;
								double[] target = new double[prob.l];
								svm.svm_cross_validation(prob,param,FOLDS_OPT,target);			
								for(int i=0;i<prob.l;i++) {
									if (target[i] == prob.y[i]) total_correct++;
								}
								double acc = 100.0*total_correct/prob.l;
								result.accuracy = acc;
								result.gram = matrix;
								result.C = c;
							}
						});
						d.start();
						d.join(TIMEOUT);
						d.stop();
						if (result.gram == null) {
							System.out.println("Process not finished for name="+matrix.name+" C="+c+" due to timeout.");
						}
						return result;
					}
				}));
				
			}
		}
		
		es.shutdown();
		try {
			es.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
		} catch (InterruptedException e) {
			// this will not happen
		}
		
		Parameter best = new Parameter(matrices.get(0), 1, Double.MIN_VALUE);
		boolean foundBest = false;
		for (Future<Parameter> f : futures) {
			Parameter p = null;
			try {
				p = f.get();
			} catch (InterruptedException e) {
			} catch (ExecutionException e) { }
			if (p != null && p.accuracy > best.accuracy) {
				best = p;
				foundBest = true;
			}
		}
		if (!foundBest) {
			System.out.println("No parameters found due to timeouts; selecting first choice.");
		}
		return best;
	}

	
	protected svm_parameter getDefaultParameters() {
		// create parameters
		svm_parameter param = new svm_parameter();
		param.svm_type = svm_parameter.C_SVC;
		param.kernel_type = svm_parameter.PRECOMPUTED;
		param.degree = 3;
		param.gamma = 0;	// 1/num_features
		param.coef0 = 0;
		param.nu = 0.5;
		param.cache_size = 100;
		param.C = 1;
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 1;
		param.probability = 0;
		param.nr_weight = 0;
		param.weight_label = new int[0];
		param.weight = new double[0];
		return param;

	}
	
	
}
