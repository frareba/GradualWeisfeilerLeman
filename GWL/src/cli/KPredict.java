package cli;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.converters.FileConverter;

import util.kernel.PredictionMatrixGenerator;

public class KPredict extends KCommon.CommandMain {
	
	@Parameter(names = { "-d", "--datasets"}, description = "Restrict to these datasets, e.g., MUTAG")
	private List<String> datasets;
	
	@Parameter(names = { "-k", "--kernels"}, description = "Restrict to these kernels, e.g., wlsk or wlsk_5; in the former "
			+ "case all available parameters will be consider for parameter selection; in the latter only those with h=5")
	private List<String> kernels;

	@Parameter(names = { "-G", "--gramdir" }, description = "Directory for gram files", converter = FileConverter.class)
	File gramDir = new File("gram");
	
	@Parameter(names = { "-c" }, description = "The regularization constant of the C-SVM is selected by cross-validation from these values.")
	List<Double> cRange = new ArrayList<Double>();
	{
		cRange.add(Math.pow(10, -3));
		cRange.add(Math.pow(10, -2));
		cRange.add(Math.pow(10, -1));
		cRange.add(Math.pow(10, 0));
		cRange.add(Math.pow(10, 1));
		cRange.add(Math.pow(10, 2));
		cRange.add(Math.pow(10, 3));
	}
	
	
	@Parameter(names = { "-p" }, description = "The number of parallel threads used. This corresponds to the numberof libSVM processes running simultaneously.")
	int parallelThreads = 4;
	
	@Parameter(names = { "-x", "--timeout"}, description = "The maximum number of milliseconds to wait before "
			+ "svm processes are killed; set to 0 to deactivate timeout. "
			+ "This was introduced for the following reason: "
			+ "libSVM often converges slowly for inappropriate choices of the regularizartion parameter c, in this "
			+ "case a timeout can be useful, since these choices of c typically lead to bad results. "
			+ "However, for large data sets libSVM may also require a long time. Setting this variable "
			+ "to a high value or deactivating timeouts is safe!")
	long timeout = 50000;
	
	@Parameter(names = { "-f", "--folds" }, description = "The number of folds used for accuracy calculation")
	int folds = 10;
	
	@Parameter(names = { "-f2", "--folds2" }, description = "Number of folds for learning C (libSVM Parameter -v n-fold cross-validation mode)")
	int folds2 = folds;
	
	@Parameter(names = { "-l", "--log" }, description = "Name of the log file for accuracy values.", converter = FileConverter.class)
	public File accLogFile = new File("log_pred_accuracy.txt");
	
	@Parameter(names = { "-l2", "--log2" }, description = "Name of the log file for parameter selection.", converter = FileConverter.class)
	public File selLogFile = new File("log_pred_selection.txt");
	
	
	public static void main(String[] args) throws IOException, InterruptedException {
		
		KPredict cm = new KPredict();
		JCommander jc = new JCommander(cm);
		jc.setProgramName("kpred");
		jc.parse(args);

		if (cm.help) {
			jc.usage();
			System.exit(0);
		}
		
		if (!cm.gramDir.exists()) {
	        throw new IOException("Gram directory does not exist.");
		}

		
		PredictionMatrixGenerator pmg  = new PredictionMatrixGenerator(cm.gramDir, cm.datasets, cm.kernels, cm.accLogFile, cm.selLogFile);
		pmg.DEBUG = cm.verbose;
		pmg.FOLDS = cm.folds;
		pmg.PARALLEL_THREADS = cm.parallelThreads;
		pmg.TIMEOUT = cm.timeout;
		pmg.FOLDS_OPT = cm.folds2;
		pmg.C_RANGE = cm.cRange.toArray(new Double[cm.cRange.size()]);

		pmg.run();
		
	}
	

}
