package cli;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.beust.jcommander.converters.FileConverter;

import util.kernel.GramMatrix;
import util.kernel.GramUtil;



public class KGram {
	
	static class CommandMain extends KCommon.CommandMain {

		@Parameter(names = { "-G", "--gramdir" }, description = "Directory for gram files", converter = FileConverter.class)
		File gramDir = new File("gram");
	}
	
	static class CommandBase {

		@Parameter(description = "gram files")
		List<File> files;
		
		@Parameter(names = { "-a", "--all" }, description = "Apply to all gram files")
		boolean all;
		
	}
	
	@Parameters(commandDescription = "Applies cosine normalization")
	static class CommandNorm extends CommandBase { }
	
	@Parameters(commandDescription = "Applies the Gaussian kernel using the kernel metric")
	static class CommandGauss extends CommandBase {
		
		@Parameter(names = { "-s", "--sigma" }, description = "Sigma values of Gaussian kernel")
		private List<Double> sigmas = new ArrayList<>();
		{
			sigmas.add(Math.pow(10, -2));
			sigmas.add(Math.pow(10, -1));
			sigmas.add(Math.pow(10, 0));
			sigmas.add(Math.pow(10, 1));
			sigmas.add(Math.pow(10, 2));
		}
		
	}

	@Parameters(commandDescription = "Shows statistics for gram matrices")
	static class CommandStats extends CommandBase { }
	
	@Parameters(commandDescription = "Plots gram matrices (Requires gnuplot)")
	static class CommandPlot extends CommandBase { }
	
	static CommandMain cm = new CommandMain();
	static CommandNorm norm = new CommandNorm();
	static CommandGauss gauss = new CommandGauss();
	static CommandStats stats = new CommandStats();
	static CommandPlot plot = new CommandPlot();

	public static void main(String[] argsString) throws IOException, InterruptedException {
		
		JCommander jc = new JCommander(cm);
		jc.addCommand("norm", norm);
		jc.addCommand("gauss", gauss);
		jc.addCommand("stats", stats);
		jc.addCommand("plot", plot);
		jc.setProgramName("kgram");

		jc.parse(argsString);
		
		if (cm.help || jc.getParsedCommand() == null) {
			jc.usage();
			System.exit(0);
		}
		
		switch (jc.getParsedCommand()) {
			case "norm" : {
				if (norm.all) {
					norm.files = getGramFiles(cm.gramDir, "__norm", "__gauss"); 
				}
				for (File f : norm.files) {
					normalize(f);
				}
				break;
			}
			case "gauss" : {
				if (gauss.all) {
					gauss.files = getGramFiles(cm.gramDir, "__norm", "__gauss");
				}
				for (File f : gauss.files) {
					for (Double sigma : gauss.sigmas)
						gaussify(f, sigma);
				}
				break;
			}
			case "stats" : {
				if (stats.all) {
					stats.files = getGramFiles(cm.gramDir);
				}
				for (File f : stats.files) {
					statistics(f);
				}
				break;
			}
			case "plot" : {
				if (plot.all) {
					plot.files = getGramFiles(cm.gramDir);
				}
				for (File f : plot.files) {
					plot(f);
				}
				break;
			}
		}
	}
	
	public static List<File> getGramFiles(File gramDir, String... ignore) {
		List<File> bk = new LinkedList<>();
		for (File f : gramDir.listFiles()) {
			if (!f.getName().endsWith(".gram")) continue;
			boolean add = true;
			for (int i=0; i<ignore.length; i++) {
				if (f.getName().contains(ignore[i])) {
					add = false;
					break;
				}
			}
			if (add) bk.add(f);
		}
		return bk;
	}
	
	static void normalize(File gramFile) throws FileNotFoundException, IOException {
		GramMatrix g = GramUtil.readLibSVMFile(gramFile);
		g.gram = GramUtil.normalize(g.gram);
		String fileName = gramFile.getAbsolutePath();
		GramUtil.writeLibSVMFile(g, addSuffix(fileName, "__norm"));
	}
	
	static void gaussify(File gramFile, double sigma) throws FileNotFoundException, IOException {
		GramMatrix g = GramUtil.readLibSVMFile(gramFile);
		g.gram = GramUtil.gaussify(g.gram, sigma);
		String fileName = gramFile.getAbsolutePath();
		GramUtil.writeLibSVMFile(g, addSuffix(fileName, "__gauss_"+sigma));
	}
	
	static void statistics(File gramFile) throws FileNotFoundException, IOException {
		System.out.println("Statistics of file: "+gramFile.getName());
		GramMatrix g = GramUtil.readLibSVMFile(gramFile);
		System.out.println("Positive semidefinite:               "+ GramUtil.isPSD(g.gram));
		System.out.println("Complete:                            "+ GramUtil.isComplete(g.gram));
		System.out.println("Non-distiguished pairs:              "+ GramUtil.nonDistinguishedPairs(g.gram));
		System.out.println("Completeness ratio:                  "+ GramUtil.completenessRatioPairs(g.gram));
		System.out.println("Complete for class labels:           "+ GramUtil.isComplete(g.gram, g.classes));
		System.out.println("Completeness ratio for class labels: "+ GramUtil.completenessRatioPairs(g.gram, g.classes));
		System.out.println("Most similar molecules in different classes: ");
		GramUtil.findMostSimilar(g.gram, g.classes);
		System.out.println();
	}
	
	static void plot(File gramFile) throws FileNotFoundException, IOException, InterruptedException {
		GramMatrix g = GramUtil.readLibSVMFile(gramFile);
		GramUtil.plot(gramFile.getAbsolutePath(), g.gram);
	}

	private static String addSuffix(String fileName, String suffix) {
		if (fileName.endsWith(".gram")) {
			fileName = fileName.substring(0, fileName.length()-5);
		}
		fileName+=suffix+".gram";
		return fileName;
	}
	
}
