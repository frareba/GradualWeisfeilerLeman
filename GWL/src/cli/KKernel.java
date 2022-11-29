package cli;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import com.beust.jcommander.Parameters;
import com.beust.jcommander.converters.FileConverter;

import algorithm.graph.vertexhierarchy.GraphSetVertexRefinementStep;
import algorithm.graph.vertexhierarchy.NeighborhoodLabelsKMeans;
import benchmark.dataset.AttrDataset;
import benchmark.dataset.SDataset;
import cli.KCommon.RangeConverter;
import cli.KCommon.RangeSplitter;
import comparison.kernel.ExplicitMappingKernel;
import comparison.kernel.Kernel;
import comparison.kernel.graph.GradualWeisfeilerLehmanKernel;
import comparison.kernel.graph.OptimalAssignmentGraphKernel;
import comparison.kernel.graph.WeisfeilerLehmanSubtreeKernel;
import graph.LGraph;
import util.kernel.GramUtil;


public class KKernel {
	
	static class CommandMain extends KCommon.CommandMain {

		@Parameter(names = { "-D", "--datadir" }, description = "Directory containing the data files", converter = FileConverter.class)
		File dataDir = new File("data");
		
		@Parameter(names = { "-G", "--gramdir" }, description = "Output directory where the gram files are stored", converter = FileConverter.class)
		File gramDir = new File("gram");

		@Parameter(names = { "-e", "--explicit" }, description = "Force computation by explicit feature maps")
		boolean explicit = false;

		@Parameter(names = { "-i", "--implicit" }, description = "Force computation by implicit feature maps")
		boolean implicit = false;
		
		@Parameter(names = { "-l", "--log" }, description = "File for saving running times", converter = FileConverter.class)
		File logFile = new File("log_runtime.txt");
		
		@Parameter(names = {"-d", "--datasets"}, description = "List of data sets")
		List<String> datasets;
		
		@Parameter(names = { "-a", "--all" }, description = "Compute kernel for all data sets in the data directory")
		private boolean all;

	}
	
	public static abstract class KernelConfig {
		
		abstract SDataset preprocessDataset(AttrDataset ds);
		
		abstract ArrayList<Kernel<LGraph<String, String>>> getKernels();
	}
	
	public static abstract class SimpleGraphKernelConfig extends KernelConfig {
		
		SDataset preprocessDataset(AttrDataset ds) {
			return ds.getSDataset();
		}
	}
	
	public static abstract class CommandWL extends SimpleGraphKernelConfig {

		@Parameter(names = { "-h", "--height" }, 
				description = "height, i.e., the number of refinement steps", 
				variableArity = true,
				splitter = RangeSplitter.class,
				converter = RangeConverter.class)
		List<Integer> iterations = Arrays.asList(0,1,2,3,4,5);

		ArrayList<Kernel<LGraph<String, String>>> getKernels() {
			ArrayList<Kernel<LGraph<String, String>>> kernels = new ArrayList<>();
			for (Number h : iterations) {
				kernels.add(getKernel(h.intValue()));
			}
			return kernels;
		}

		abstract Kernel<LGraph<String, String>> getKernel(int height);
	}
	
	
	////////////////////////////////////////////////////////////////////////////////////////////
	@Parameters(commandDescription = "Compute the gradual Weisfeiler-Lehman kernel using k-means clustering.")
	public static class CommandGWLK extends CommandWL {
		@Parameter(names = { "-k", "--k" }, 
				description = "k of kmeans, i.e., the number of clusters" )
		int k=4;
		Kernel<LGraph<String, String>> getKernel(int height) {
			return new GradualWeisfeilerLehmanKernel<>(new GraphSetVertexRefinementStep(new NeighborhoodLabelsKMeans<>(k)),height);
		}
	}
		
	@Parameters(commandDescription = "Compute the gradual Weisfeiler-Lehman optimal assignment kernel.")
	public static class CommandGWLOA extends CommandWL {
		@Parameter(names = { "-k", "--k" }, 
				description = "k of kmeans, i.e., the number of clusters" )
		int k=4;
		@Override
		Kernel<LGraph<String, String>> getKernel(int height) {
			return new OptimalAssignmentGraphKernel.GradualWeisfeilerLehmankMeansSubtreeKernel<>(height,k);
		}
	}
	////////////////////////////////////////////////////////////////////////////////////////////
	
	@Parameters(commandDescription = "Compute the Weisfeiler-Lehman subtree kernel.")
	public static class CommandWLS extends CommandWL {
		Kernel<LGraph<String, String>> getKernel(int height) {
			return new WeisfeilerLehmanSubtreeKernel<>(height);
		}
	}

	

	@Parameters(commandDescription = "Compute the Weisfeiler-Lehman optimal assignment kernel.")
	public static class CommandWLOA extends CommandWL {

		@Override
		Kernel<LGraph<String, String>> getKernel(int height) {
			return new OptimalAssignmentGraphKernel.WeisfeilerLehmanSubtree<>(height);
		}
	}
	
	
	static CommandMain cm = new CommandMain();
	static CommandWLS wls = new CommandWLS();
	static CommandWLOA wloa = new CommandWLOA();
	static CommandGWLOA gwloa = new CommandGWLOA();

	static CommandGWLK gwlk = new CommandGWLK();
	@SuppressWarnings({ "unchecked", "rawtypes" })
	public static void main(String[] argsString) throws IOException, InterruptedException {
		
		JCommander jc = new JCommander(cm);
		jc.addCommand("wls", wls);
		jc.addCommand("wloa", wloa);
		jc.addCommand("gwlk", gwlk);
		jc.addCommand("gwloa", gwloa);
		
		jc.getMainParameter();
		jc.setProgramName("kkernel");

		jc.parse(argsString);

		if (cm.help || jc.getParsedCommand() == null) {
			jc.usage();
			System.exit(0);
		}
		
	    if (cm.explicit && cm.implicit) {
	        throw new ParameterException("Choose either -i or -e.");
	    }
	    
	    if (!cm.all && cm.datasets == null) {
	        throw new ParameterException("No datasets specified.");
	    }
		
		KernelConfig kc = null;

		switch (jc.getParsedCommand()) {
			case "wls"   : kc = wls;    break;
			case "wloa"  : kc = wloa;   break;	
			case "gwlk"	 : kc = gwlk;   break;	
			case "gwloa" : kc = gwloa;  break;
		}

		if (!cm.gramDir.exists()) cm.gramDir.mkdir();
		
		if (cm.all) {
			cm.datasets = KCommon.getDatasets(cm.dataDir);
		}
		
		for (String dName : cm.datasets) {
			AttrDataset ds = KCommon.load(dName, cm.dataDir);
			SDataset ds2 = kc.preprocessDataset(ds);
			for (Kernel<LGraph<String, String>> k : kc.getKernels()) {

				System.out.println("Kernel:   "+k.getID());
				System.out.println("Dataset:  "+ds2.getID()+"  converted from: "+ds.getID());
				double[][] gram;
				boolean explicit = false;
				long startTime = System.nanoTime();
				if (cm.explicit) {
					gram = ((ExplicitMappingKernel)k).computeExplicit(ds2);
					explicit = true;
				} else if (cm.implicit || !(k instanceof ExplicitMappingKernel)) {
					gram = k.computeAll(ds2);
					explicit = false;
				} else {
					ExplicitMappingKernel emk = (ExplicitMappingKernel)k;
					try {
						gram = emk.computeExplicit(ds2);
						explicit = true;
					} catch (IllegalStateException e) {
						System.out.println("Non-explicit computation due to kernel choice!");
						gram = k.computeAll(ds2);
						explicit = false;
					}
				}
				long runtime = System.nanoTime() - startTime;
				
				// write running time
				FileWriter fw = new FileWriter(cm.logFile, true);
				BufferedWriter bw = new BufferedWriter(fw);
				bw.append(ds2.getID()+"\t"+k.getID()+"\t"+(explicit?"exp":"imp")+"\t"+(double)runtime/1000d/1000+" ms \n");
				bw.close();

				// write gram file 
				String fileName = cm.gramDir.getAbsolutePath()+"/"+ds2.getID()+"__"+k.getID()+".gram";
				GramUtil.writeLibSVMFile(gram, ds2.getClassLabels(), fileName);
				
				System.out.println();
			}
		}

	}
		
}
