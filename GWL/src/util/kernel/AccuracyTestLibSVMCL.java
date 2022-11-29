package util.kernel;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import datastructure.FeatureVector;
import datastructure.Pair;

/**
 * Performs tests using the libSVM command line tools.
 * 
 * @author Nils Kriege
 *
 */
public class AccuracyTestLibSVMCL extends AccuracyTest {
	
	public AccuracyTestLibSVMCL(File folder, List<String> datasets, List<String> kernels, File accLogFile, File selLogFile) {
		super(folder, datasets, kernels, accLogFile, selLogFile);
	}

	public AccuracyTestLibSVMCL(String iD) {
		super(iD);
	}

	public AccuracyTestLibSVMCL(String folder, String... datasets) {
		super(folder, datasets);
	}

	/**
	 * Used to write temporary files; heavy read/write access in this folder! 
	 */
	public String TMP_PATH = "/tmp/";
	

	/**
	 * Allows to specify the path to libSVM, i.e., the folder containing svm-train and 
	 * svm-predict; leave blank if both are available in the current path.
	 */
	public String LIBSVM_PATH = "";
	
	
	public void run() throws IOException, InterruptedException {
		
		// create tmp directory for this instance
		int i = 0;
		File tmp;
		do {
			tmp = new File(TMP_PATH+"/svmlib_"+i++);
		} while (tmp.exists());
		tmp.mkdir();
		TMP_PATH = tmp.getAbsolutePath()+"/";
		
		
		super.run();
		
		
		tmp.delete();
	}
	
	
	/**
	 * Computes the accuracy for a given fold assignment. Each fold serves as test 
	 * fold once; the resulting accuracies are averaged. The C parameter and 
	 * kernel parameters (different grams) are selected on the training fold.
	 * @param gramFiles the gram files (kernel parameterizations)
	 * @param selections used to store the selected parameters (gram and C)
	 * @param foldNo the number of folds
	 * @return average accuracy for the given fold assignment
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public double crossValidate(ArrayList<File> gramFiles, FeatureVector<String> selections, int foldNo) throws IOException, InterruptedException {
		if (DEBUG) {
			System.out.println("Cross-Validation");
		}

		// prepare train and test files
		System.out.println("\tPreparing train and test files according to random folds.");
		
		// load first gram matrix
		Iterator<File> itGramFiles = gramFiles.iterator(); 
		GramMatrix gram0 = GramUtil.readLibSVMFile(itGramFiles.next()); // load first gram file
		
		// compute random fold indices
		ArrayList<ArrayList<Integer>> folds = partition(gram0.getDimension(), foldNo);

		ArrayList<ArrayList<String>> allNames = new ArrayList<ArrayList<String>>(folds.size());
		for (int i=0; i<folds.size(); i++) {
			allNames.add(new ArrayList<>());
		}
		// write train and test files for each gram file and fold
		do {
			for (int iFold=0; iFold<folds.size(); iFold++) {
				// iFold indexes the independent evaluation (test) set
				ArrayList<Integer> test = folds.get(iFold);
				ArrayList<Integer> train = new ArrayList<>();
				for (int iFold2=0; iFold2<folds.size(); iFold2++) {
					if (iFold2 == iFold) continue; // skip test fault
					train.addAll(folds.get(iFold2));
				}
				
				String name = gram0.name+"_fold"+(iFold+1);
				allNames.get(iFold).add(name);

				String trainFile = name+".train";
				GramUtil.writeLibSVMFile(gram0, TMP_PATH+trainFile, train, train);				

				String testFile = name+".test";
				GramUtil.writeLibSVMFile(gram0, TMP_PATH+testFile, test, train);
			}
			
			// load next gram
			if (itGramFiles.hasNext()) {
				gram0 = GramUtil.readLibSVMFile(itGramFiles.next());
			} else {
				gram0 = null;
			}
		} while (gram0 != null);
		
		// compute accuracies
		double avgAccuracy=0;
		for (int iFold=0; iFold<folds.size(); iFold++) {
			
			System.out.println("\tFold  "+(iFold+1)+"  of  "+folds.size());
			
			ArrayList<String> names = allNames.get(iFold);
			
			// parameter optimization
			Pair<String,Double> par = searchBestParameter(names);
			String selectedGram = par.getFirst();
			String[] selectedGramSplit = selectedGram.split(".gram_fold");
			if (Integer.parseInt(selectedGramSplit[1])!= iFold+1) {
				throw new IllegalStateException("Selected wrong gram file.");
			} else {
				selectedGram = selectedGramSplit[0];
			}			
			selections.increaseByOne(selectedGram);
			
			// build model
			String modelFile = buildModel(par.getFirst()+".train", par.getSecond());
			
			// predict
			double accuracy = predict(modelFile, par.getFirst()+".test");
			System.out.println("\tReached Accuracy: "+accuracy);
			avgAccuracy += accuracy/folds.size();
			
			// delete files
			for (String name : names) {
				new File(TMP_PATH+name+".train").delete();
				new File(TMP_PATH+name+".test").delete();
			}
			new File(TMP_PATH+par.getFirst()+".train.model").delete();
			new File(TMP_PATH+par.getFirst()+".train.model.out").delete();
		}
		
//		System.out.println("Avg. Accuracy "+avgAccuracy);
		return avgAccuracy;
	}
	
	/**
	 * Finds the best parameters based on the test set
	 * @param names available kernels
	 * @return name of the best kernel and C parameter
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public Pair<String,Double> searchBestParameter(ArrayList<String> names) throws IOException, InterruptedException {

		ArrayList<Pair<String,Double>> taskKey = new ArrayList<Pair<String,Double>>();
		ArrayList<Future<Double>> taskFutureValue = new ArrayList<Future<Double>>();
		ExecutorService es = Executors.newFixedThreadPool(PARALLEL_THREADS);
		
		for (String name : names) {
			for (double c : C_RANGE) {
				
				taskKey.add(new Pair<String, Double>(name, c));
				taskFutureValue.add(es.submit(new Callable<Double>() {
					@Override
					public Double call() throws Exception {
						String cmd = LIBSVM_PATH+"svm-train -s 0 -t 4 -c "+c+" -v "+FOLDS_OPT+" "+TMP_PATH+name+".train";
						if (DEBUG) System.out.println(cmd);
						return executeSVM(cmd, TIMEOUT);
					}
				}));
			}
		}
		
		es.shutdown();
		es.awaitTermination(0, TimeUnit.NANOSECONDS);

		Pair<String, Double> bestParam = null;
		double bestAccuracy = Double.MIN_VALUE;

		for (int i=0; i<taskKey.size(); i++) {
			Pair<String, Double> p = taskKey.get(i); 
			double r;
			try {
				r = taskFutureValue.get(i).get();
			} catch (ExecutionException e) {
				throw new IllegalStateException("Execution error libSVM!");
			}
			if (r == -2) { 
				// this usually happens because c gets to high for the test set
				// we return the current best result
				System.out.println("Process not finished for name="+p.getFirst()+" C="+p.getSecond()+" due to timeout.");
			} else { 
				if (r < 0) throw new IllegalStateException("Unexpected libSVM result!\nInfo name="+p.getFirst()+" C="+p.getSecond()+"");
			}
			
			if (r>bestAccuracy) {
				bestAccuracy=r;
				bestParam = p;
			}
		}
		
		if (bestParam == null) {
			// none found, everything timed out -- this indicates that the
			// configuration should be changed!
			bestParam = taskKey.get(0);
			System.out.println("\t\tWARNING: Cross-validation did not finish! Using first setting.");
		}
		System.out.println("\t\tSelected Gram: "+bestParam.getFirst());
		System.out.println("\t\tSelected C: "+bestParam.getSecond());
		System.out.println("\t\tAnticipated Accuracy: "+bestAccuracy);
		
		return bestParam;
	}
	
	
	/**
	 * Learns a model based on the given training file with the given C 
	 * parameter.
	 * @param trainFile the training file name
	 * @param c the regularization parameter
	 * @return model file name the file name containing the model information
	 * @throws IOException
	 */
	public String buildModel(String trainFile, double c) throws IOException, InterruptedException {
		if (DEBUG) {
			System.out.println("buildModel");
		}

		String modelFile = trainFile+".model";
		String cmd = LIBSVM_PATH+"svm-train -s 0 -t 4 -c "+c+" "+TMP_PATH+trainFile+" "+TMP_PATH+modelFile;
		if (DEBUG) System.out.println(cmd);
		
		double r = executeSVM(cmd, 0);
		if (r != -1) throw new IllegalStateException("Unexpected libSVM result!+\nCommand: "+cmd);
		
		
		return modelFile;
	}
	
	/**
	 * Predict the given test set based on the given model.
	 * @param modelFile model file name
	 * @param testFile test file name
	 * @return the obtained accuracy
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public double predict(String modelFile, String testFile) throws IOException, InterruptedException {
		if (DEBUG) {
			System.out.println("Predict");
		}

		String outFile = modelFile+".out";
		String cmd = LIBSVM_PATH+"svm-predict "+TMP_PATH+testFile+" "+TMP_PATH+modelFile+" "+TMP_PATH+outFile;
		if (DEBUG) System.out.println(cmd);
		
		double r = executeSVM(cmd, 0);
		if (r < 0) throw new IllegalStateException("Unexpected libSVM result!");
		
		return r;
	}
	

	/**
	 * Run the given command, returns the accuracy provided by libSVM.
	 * @param commandLine
	 * @param timeout the process is killed if not finished within the specified time
	 * @return accuracy, or -1 if call does output accuracy or -2 because of timeout, -3 if call failed
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public double executeSVM(final String commandLine, final long timeout) throws IOException, InterruptedException {
		Runtime runtime = Runtime.getRuntime();
		Process process = runtime.exec(commandLine);

		Worker worker = new Worker(process);
		worker.start();
		try {
			worker.join(timeout);
			if (worker.exit != null && worker.exit == 0) {
				return worker.accuracy;
			} else {
				if (worker.exit == null) {
					return -2; // Process not finished: Timeout
				} else {
					return -3; // Process not finished: libSVM call failed!
				}
			}
		} catch(InterruptedException ex) {
			worker.interrupt();
			Thread.currentThread().interrupt();
			throw ex;
		} finally {
			process.destroy();
		}
	}
		
	/**
	 * Worker to handle process output in a different thread.
	 * @author Nils Kriege
	 */
	final class Worker extends Thread {
		final Process process;
		Integer exit;
		double accuracy = -1;
		
		private Worker(Process process) {
			this.process = process;
		}
		
		public void run() {
			try {
				BufferedReader buff = new BufferedReader(new InputStreamReader(process.getInputStream()));
				String str;
				while ((str = buff.readLine()) != null) {
					if (DEBUG) System.out.println(str);
					if (str.contains("Accuracy = ")) {
						String accuracyString = str.split("%")[0].split("Accuracy = ")[1];
						accuracy = Double.valueOf(accuracyString);
					}
				}
				exit = process.waitFor();
			} catch (InterruptedException ignore) {
				return;
			} catch (IOException io) {
				return;
			}
		}
	}

	@Override
	public Parameter searchBestParameter(ArrayList<GramMatrix> matrices, ArrayList<Integer> trainSet)
			throws IOException, InterruptedException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double predict(ArrayList<Integer> test, ArrayList<Integer> train, Parameter param)
			throws IOException, InterruptedException {
		// TODO Auto-generated method stub
		return 0;
	}

	
}
