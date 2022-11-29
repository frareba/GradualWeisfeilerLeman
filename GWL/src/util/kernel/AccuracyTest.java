package util.kernel;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map.Entry;
import java.util.Random;

import datastructure.FeatureVector;
import datastructure.SparseFeatureVector;


// TODO: support option for computing stddev from each accuracy; 
// over all individual folds!

// TODO: add option for stratified cross validation, i.e., all folds
// should have approximately the same distribution of class labels

// TODO: improve parallelization to achieve a higher workload

/**
 * Computes prediction accuracies using cross-validation
 * based on the gram files in a given <code>FOLDER</code>, which all 
 * must be in libSVM format. Each file must be named as follows:
 * <code>DATASET__KERNEL_PARAMETERS.gram</code>
 * 
 * For each dataset and each kernel the accuracy is computed, the 
 * parameters used for prediction as well as the regularization 
 * constant C are selected by cross-validation for each training 
 * fold separately.  
 * 
 * The accuracies are stored in the file <code>rAccFOLDER.txt</code>, 
 * the file <code>rAccSelFOLDER.txt</code> stores information on the 
 * parameters selected. The number of repetitions determines the 
 * number of random fold assignments. The accuracies are averaged 
 * over these repetitions and standard deviation is reported w.r.t.
 * to the results obtained for different fold assignments.
 * 
 * The behavior of the class is configured by several static variables
 * which should be checked carefully.
 * 
 * 
 * @author kriege
 *
 */
public abstract class AccuracyTest {

	/** 
	 * The regularization constant of the C-SVM is selected by cross-validation 
	 * from these values.
	 */
	public Double[] C_RANGE = {
		Math.pow(10, -3),
		Math.pow(10, -2),
		Math.pow(10, -1),
		Math.pow(10, 0),
		Math.pow(10, 1),
		Math.pow(10, 2),
		Math.pow(10, 3)
	};

	/**
	 * The path containing the considered FOLDER, result files will be places here.
	 */
	public static String BASE_PATH = "/scratch/kriege/";
	
	/**
	 * The number of parallel threads used.
	 */
	public int PARALLEL_THREADS = 7;
	
	/** The number of folds used for accuracy calculation */
	public int FOLDS = 10;
	
	/** Number of folds for learning parameters on the training set */
	public int FOLDS_OPT = FOLDS;
	
	/** 
	 * Number of repetitions, each repetitions has a different fold assignment. The 
	 * reported standard deviation refers to these repetitions. 
	 */
	public int REPETITIONS = 10;
	
	/** Print additional information */
	public boolean DEBUG = false;
	
	/**
	 * Gram matrices with a file name containing the stop word are ignored; set to
	 * <code>null</code> to deactivate.
	 */
	public String BLACKLIST_STOP_WORD = null;
	
	/** 
	 * The maximum number of milliseconds to wait before svm processes are killed; set to 
	 * 0 to deactivate timeout.
	 * This was introduced for the following reason:
	 * libSVM often converges slowly for inappropriate choices of the constant C, in this 
	 * case a timeout can be useful, since these choices of C typically lead to bad results.
	 * However, for large data sets libSVM may also require a long time. Setting this variable 
	 * to a high value or deactivating timeouts is safe!
	 */ 
	public long TIMEOUT = 50000; // ms
	
	
	//------------- no configuration below this line is required -------------
	public static String LOG_FILE_PREFIX = "rAcc";
	public Random rng;
	
	protected File pFolder;
	protected List<String> pDatasets;
	protected List<String> pKernels;
	protected File pAccLogFile;
	protected File pSelLogFile;
	
	/**
	 * Creates an object for accuracy testing.
	 * @param folder the folder containing the gram files.
	 * @param folder
	 */
	public AccuracyTest(String folder) {
		this(folder, (List<String>)null);
	}
	
	public AccuracyTest(String folder, String... datasets) {
		this(folder, Arrays.asList(datasets));
	}
	
	/**
	 * Creates an object for accuracy testing.
	 * @param folder the folder containing the gram files.
	 * @param datasets the data sets used for testing; null to test all data sets.
	 */
	public AccuracyTest(String folder, List<String> datasets) {
		this(new File(BASE_PATH+folder),
			datasets,
			null,
			new File(BASE_PATH+LOG_FILE_PREFIX+folder+".txt"),
			new File(BASE_PATH+LOG_FILE_PREFIX+"Sel"+folder+".txt")
		);
	}
	
	/**
	 * Creates an object for accuracy testing.
	 * @param folder the folder containing the gram files.
	 * @param datasets the data sets used for testing; null to test all data sets.
	 * @param kernels used for testing
	 * @param accLogFile accuracy log file
	 * @param selLogFile selection log file
	 */
	public AccuracyTest(File folder, List<String> datasets, List<String> kernels, File accLogFile, File selLogFile) {
		this.pFolder = folder;
		this.pDatasets = datasets;
		this.pKernels = kernels;
		this.pAccLogFile = accLogFile;
		this.pSelLogFile = selLogFile;
	}
	
	/**
	 * Starts accuracy tests.
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public void run() throws IOException, InterruptedException {
		
		// prepare folders if necessary
		
		// retrieve list of data sets
		ArrayList<String> datasets = getDatsets();
		if (pDatasets != null) {
			datasets.retainAll(pDatasets);
		}
		
		// print what will be tested
		for (String ds : datasets) {
			System.out.println(ds);
			
			// retrieve list of kernels
			ArrayList<String> kernels = getKernels(ds, pKernels);
			
			for (String bk : kernels) {
				System.out.println("\t"+bk);
				ArrayList<File> grams = getGramsFiles(ds, bk);
				for (File g : grams) {
					System.out.println("\t\t"+g.getName());
				}
			}
		}
		
		System.out.println("---------------------");

		// write header to log files
		StringBuilder header = new StringBuilder();
		Date now = new Date();
		header.append("--------------------------------------\n");
		header.append("Time:    "+now.toString()+"\n");
		header.append("Folder:  "+pFolder.toString()+"\n");
		header.append("--------------------------------------\n");

		BufferedWriter bw = new BufferedWriter(new FileWriter(pAccLogFile, true));
		bw.append(header.toString());
		bw.close();
		bw = new BufferedWriter(new FileWriter(pSelLogFile, true));
		bw.append(header.toString());
		bw.close();
		
		// test data sets
		for (String ds : datasets) {
			System.out.println(ds);
			
			// retrieve list of kernels
			ArrayList<String> kernels = getKernels(ds, pKernels);
			
			for (String bk : kernels) {
				System.out.println("\t"+bk);
				ArrayList<File> gramFiles = getGramsFiles(ds, bk);
				for (File f : gramFiles) {
					System.out.println("\t\t"+f.getName());
				}
				AccuracyResult r = computeAccurcay(gramFiles, REPETITIONS, FOLDS);
				addResult(ds, bk, r.accuracy, r.stdev);
				addSelectionResult(r.parameterSelection);
			}
		}
		
		// clean up if necessary

	}
		
	protected void addResult(String dataset, String kernel, double accuracy, double stdev) throws IOException {
		FileWriter fw = new FileWriter(pAccLogFile, true);
		BufferedWriter bw = new BufferedWriter(fw);
		bw.append(dataset + "\t" + kernel + "\t" + String.format(Locale.ENGLISH, "%1.2f",accuracy) + "\t" + String.format(Locale.ENGLISH, "%1.2f",stdev) + "\n");
		bw.close();
	}
	
	protected void addSelectionResult(FeatureVector<String> sel) throws IOException {
		FileWriter fw = new FileWriter(pSelLogFile, true);
		BufferedWriter bw = new BufferedWriter(fw);
		for (Entry<String, Double> e : sel.nonZeroEntries()) {
			bw.append(e.getKey() + "\t" + e.getValue() + "\n");
		}
		bw.close();
	}

	/**
	 * The data sets in the considered folder.
	 * @return list of available data sets in the folder
	 */
	private ArrayList<String> getDatsets() {
		HashSet<String> ds = new HashSet<String>();
		System.out.println(pFolder);
		for (File f : pFolder.listFiles()) {
			if (!f.getName().endsWith(".gram")) continue;
			ds.add(f.getName().split("__")[0]);
		}
		ArrayList<String> r = new ArrayList<>(ds);
		Collections.sort(r);	
		return r;
	}
	
	/**
	 * The kernels for the given data set. Multiple gram files
	 * may be contained in the folder for each kernel and data set 
	 * having different parameters. If kernel is not null, then only
	 * kernels specified in the list are returned. The strings may
	 * contain specific parameters.
	 * @param dataset the data set
	 * @param kenrnels restrict to these kernels
	 * @return list of available kernels for the given data set
	 */
	public ArrayList<String> getKernels(String dataset, List<String> kernels) {
		HashSet<String> bk = new HashSet<String>();
		for (File f : pFolder.listFiles()) {
			if (!f.getName().endsWith(".gram")) continue;
			String name = f.getName().substring(0, f.getName().length()-5); // remove '.gram'
			String[] dsKernel = name.split("__");
			if (!dsKernel[0].equals(dataset)) continue;
			name = dsKernel[1];
			if (kernels != null) {
				for (String k : kernels) {
					if (name.startsWith(k+"_") || name.endsWith(k)) {
						bk.add(k);
					}
				}
			} else {
				bk.add(name.split("_")[0]); 
			}
		}
		ArrayList<String> r = new ArrayList<>(bk);
		Collections.sort(r);	
		return r;
	}
	
	/**
	 * List of gram files for the given data set and kernel.
	 * @param dataset the data set name
	 * @param kernel the kernel name
	 * @return gram matrices, obtained for each parameterization
	 * @throws IOException
	 */
	private ArrayList<File> getGramsFiles(String dataset, String kernel) throws IOException {
		ArrayList<File> grams = new ArrayList<File>();
		for (File f : pFolder.listFiles()) {
			if (!f.getName().endsWith(".gram")) continue;
			if (f.getName().startsWith(dataset+"__"+kernel+"_") || 
				f.getName().equals(dataset+"__"+kernel+".gram") )
				if (BLACKLIST_STOP_WORD == null || !f.getName().contains(BLACKLIST_STOP_WORD))
					grams.add(f);
		}
		Collections.sort(grams);
		return grams;
	}
	
	
	/**
	 * Creates a random partition for of the set {0, ..., n-1} into the given number
	 * of cells.
	 * @param n number of elements
	 * @param cells number of cells of the desired partition
	 * @return the partition
	 */
	public ArrayList<ArrayList<Integer>> partition(int n, int cells) {
		ArrayList<ArrayList<Integer>> r = new ArrayList<ArrayList<Integer>>(cells);
		for (int i=0; i<cells; i++) {
			r.add(new ArrayList<Integer>());
		}
		
		ArrayList<Integer> pool = new ArrayList<Integer>(n);
		for (int i=0; i<n; i++) pool.add(i);
		
		int i=0;
		while (!pool.isEmpty()) {
			Integer e = pool.remove(rng.nextInt(pool.size()));
			r.get(i).add(e);			
			i = ++i%cells;
		}
		
		return r;
	}
	
	/**
	 * Computes the accuracy by cross validation; all grams are split according
	 * to randomly chosen folds. For each training fold the gram matrix performing
	 * best is selected and used for prediction on the test fold. 
	 * @param gramFiles the gram files (typically for different parameterizations of a kernel)
	 * @param repetitions number of repetitions, i.e., different fold assignments
	 * @param foldNo the number of folds
	 * @return the accuracy result
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public AccuracyResult computeAccurcay(ArrayList<File> gramFiles, int repetitions, int foldNo) throws IOException, InterruptedException {
		
		double[] accuracy = new double[repetitions];

		SparseFeatureVector<String> selections = new SparseFeatureVector<String>();
		
		rng = new Random(42); // same partitions for each call
		for (int i=0; i<repetitions; i++) {
			System.out.println();
			System.out.println("REPETITION  "+(i+1)+"  of  "+repetitions);
			double acc = crossValidate(gramFiles, selections, foldNo);
			accuracy[i] = acc;
			System.out.println("Avg. Accuracy: "+acc);
		}
		System.out.println("========================");
		double avg = 0;
		for (double d : accuracy) avg += d;
		avg /= repetitions;
		
		// compute stdev
		double stdev = 0;
		for (double d : accuracy) stdev += (d-avg)*(d-avg);
		stdev = Math.sqrt(stdev/repetitions);
		
		System.out.println("Accuracy: "+String.format(Locale.ENGLISH, "%1.2f",avg));
		System.out.println("Standard deviation: "+String.format(Locale.ENGLISH, "%1.2f",stdev));

		return new AccuracyResult(avg, stdev, selections);
	}
	
	/**
	 * Container for the results of a classification experiment.
	 * 
	 * @author Nils Kriege
	 */
	public static class AccuracyResult {
		public double accuracy;
		public double stdev;
		public FeatureVector<String> parameterSelection;
		
		public AccuracyResult(double accuracy, double stdev, FeatureVector<String> parameterSelection) {
			this.accuracy = accuracy;
			this.stdev = stdev;
			this.parameterSelection = parameterSelection;
		}
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
	
		// load all gram matrices
		ArrayList<GramMatrix> gramMatrices = new ArrayList<>();
		for (File f : gramFiles) {
			gramMatrices.add(GramUtil.readLibSVMFile(f));
		}
		
		// compute random fold indices
		ArrayList<ArrayList<Integer>> folds = partition(gramMatrices.get(0).getDimension(), foldNo);

		// compute accuracies
		double avgAccuracy=0;
		for (int iFold=0; iFold<folds.size(); iFold++) {
			
			System.out.println("\tFold  "+(iFold+1)+"  of  "+folds.size());
			
			// iFold indexes the independent evaluation (test) set
			ArrayList<Integer> test = folds.get(iFold);
			ArrayList<Integer> train = new ArrayList<>();
			for (int iFold2=0; iFold2<folds.size(); iFold2++) {
				if (iFold2 == iFold) continue; // skip test fault
				train.addAll(folds.get(iFold2));
			}
			
			Parameter param = searchBestParameter(gramMatrices, train);
			selections.increaseByOne(param.gram.name);
			
			System.out.println("\t\tSelected Gram: "+param.gram.name);
			System.out.println("\t\tSelected C: "+param.C);
			System.out.println("\t\tAnticipated Accuracy: "+param.accuracy);

			double accuracy = predict(test, train, param);
			System.out.println("\tReached Accuracy: "+accuracy);
			avgAccuracy += accuracy/folds.size();
			
		}
		
		return avgAccuracy;
	}
	
	public static class Parameter {
		public GramMatrix gram;
		public double C;
		public Double accuracy;

		public Parameter(GramMatrix gram, double C, Double accuracy) {
			this.gram = gram;
			this.C = C;
			this.accuracy = accuracy;
		}
	}
	
	/**
	 * Finds the best parameters based on the test set
	 * @param names available kernels
	 * @return name of the best kernel and C parameter
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public abstract Parameter searchBestParameter(ArrayList<GramMatrix> matrices, ArrayList<Integer> trainSet) throws IOException, InterruptedException;

	
	public abstract double predict (ArrayList<Integer> test, ArrayList<Integer> train, Parameter param) throws IOException, InterruptedException;	
	
	
}
