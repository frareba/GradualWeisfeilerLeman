package cli;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.converters.FileConverter;

import util.kernel.AccuracyTest;
import util.kernel.AccuracyTestLibSVMCL;
import util.kernel.AccuracyTestLibSVMNative;

public class KAccuracy extends KCommon.CommandMain {
	
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
	
	@Parameter(names = { "-T", "--tmpdir" }, description = "Used to write temporary files when the parameter -t (--clt) is specified; heavy read/write access in this folder!", converter = FileConverter.class)
	File tmpDir = new File("/tmp");
	
	@Parameter(names = { "-p" }, description = "The number of parallel threads used. This corresponds to the numberof libSVM processes running simultaneously.")
	int parallelThreads = 32;
	
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
	
	@Parameter(names = { "-r", "--repeat" }, description = "Number of repetitions, each repetitions has a different fold assignment. The reported standard deviation refers to these repetitions.")
	int repetitions = 10;

	@Parameter(names = { "-t", "--clt" }, description = "Use libSVM command line tools instead of the build-in native Java version.")
	boolean libSVMCL = false;

	@Parameter(names = { "-s", "--svm" }, description = "Allows to specify the path of the folder containing svm-train and svm-predict when the parameter -t (--clt) is specified; leave blank if both are available in the current path.")
	String libSVMPath = "";
	
	@Parameter(names = { "-l", "--log" }, description = "Name of the log file for accuracy values.", converter = FileConverter.class)
	public File accLogFile = new File("log_accuracy.txt");
	
	@Parameter(names = { "-l2", "--log2" }, description = "Name of the log file for parameter selection.", converter = FileConverter.class)
	public File selLogFile = new File("log_selection.txt");
	
	
	
	public static void main(String[] args) throws IOException, InterruptedException {
		
		KAccuracy cm = new KAccuracy();
		JCommander jc = new JCommander(cm);
		jc.setProgramName("kacc");
		jc.parse(args);

		if (cm.help) {
			jc.usage();
			System.exit(0);
		}
		
		if (!cm.gramDir.exists()) {
	        throw new IOException("Gram directory does not exist.");
		}
		
		// set up accuracy test using command line tools
		AccuracyTest at;
		if (cm.libSVMCL) { 
			AccuracyTestLibSVMCL atCLI = new AccuracyTestLibSVMCL(cm.gramDir, cm.datasets,cm.kernels, cm.accLogFile, cm.selLogFile);
			atCLI.TMP_PATH = cm.tmpDir.getAbsolutePath()+"/";
			atCLI.LIBSVM_PATH = cm.libSVMPath;
			at = atCLI;
		} else {
			at = new AccuracyTestLibSVMNative(cm.gramDir, cm.datasets,cm.kernels, cm.accLogFile, cm.selLogFile);
		}
		
		at.DEBUG = cm.verbose;
		at.FOLDS = cm.folds;
		at.PARALLEL_THREADS = cm.parallelThreads;
		at.REPETITIONS = cm.repetitions;
		at.TIMEOUT = cm.timeout;
		at.FOLDS_OPT = cm.folds2;
		at.C_RANGE = cm.cRange.toArray(new Double[cm.cRange.size()]);

		
		at.run();
		
	}
	

}
