package cli;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import com.beust.jcommander.converters.FileConverter;

import benchmark.dataset.AttrDataset;
import benchmark.dataset.SDataset;
import comparison.kernel.ExplicitMappingKernel;
import comparison.kernel.Kernel;
import datastructure.FeatureVector;
import datastructure.FeatureVectors;
import datastructure.SparseFeatureVector;
import graph.LGraph;


public class KVec {
	
	static class CommandMain extends KCommon.CommandMain {

		@Parameter(names = { "-D", "--datadir" }, description = "Directory for data files", converter = FileConverter.class)
		File dataDir = new File("data");
		
		@Parameter(names = { "-F", "--fvecdir" }, description = "Directory for feature vector files", converter = FileConverter.class)
		File fvecDir = new File("fvec");

		@Parameter(names = { "-l", "--log" }, description = "File for saving running times", converter = FileConverter.class)
		File logFile = new File("log_runtime.txt");
		
		@Parameter(names = {"-d", "--datasets"}, description = "datasets")
		List<String> datasets;
		
		@Parameter(names = { "-a", "--all" }, description = "Compute for all data sets")
		private boolean all;

	}

	
	static CommandMain cm = new CommandMain();
	static KKernel.CommandWLS wls = new KKernel.CommandWLS();
	static KKernel.CommandGWLK gwlk = new KKernel.CommandGWLK();

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public static void main(String[] argsString) throws IOException, InterruptedException {
		
		JCommander jc = new JCommander(cm);
		jc.addCommand("wls", wls);
		jc.addCommand("gwlk", gwlk);
		
		jc.getMainParameter();
		jc.setProgramName("kvec");

		jc.parse(argsString);

		if (cm.help || jc.getParsedCommand() == null) {
			jc.usage();
			System.exit(0);
		}
		
	    if (!cm.all && cm.datasets == null) {
	        throw new ParameterException("No datasets specified.");
	    }
		
	    KKernel.KernelConfig kc = null;

		switch (jc.getParsedCommand()) {
			case "wls"  : kc = wls;   break;
			case "gwlk" : kc = gwlk;   break;	
		}

		if (!cm.fvecDir.exists()) cm.fvecDir.mkdir();
		
		if (cm.all) {
			cm.datasets = KCommon.getDatasets(cm.dataDir);
		}
		
		for (String dName : cm.datasets) {
			AttrDataset ds = KCommon.load(dName, cm.dataDir);
			SDataset ds2 = kc.preprocessDataset(ds);
			for (Kernel<LGraph<String, String>> k : kc.getKernels()) {

				System.out.println("Kernel:   "+k.getID());
				System.out.println("Dataset:  "+ds2.getID()+"  converted from: "+ds.getID());
				long startTime = System.nanoTime();
				ArrayList<FeatureVector<?>> fvs = ((ExplicitMappingKernel)k).getFeatureVectors(ds2);
				long runtime = System.nanoTime() - startTime;
				
				// write running time
				FileWriter fw = new FileWriter(cm.logFile, true);
				BufferedWriter bw = new BufferedWriter(fw);
				bw.append(ds2.getID()+"\t"+k.getID()+"\t"+"fv"+"\t"+(double)runtime/1000d/1000+"\n");
				bw.close();

				// write feature vector file 
				String fileName = cm.fvecDir.getAbsolutePath()+"/"+ds2.getID()+"__"+k.getID()+".fvec";
				ArrayList<SparseFeatureVector<Integer>> fvsInt =  FeatureVectors.toIntegerIndexed(fvs);
				FeatureVectors.writeLibSVMFile(fvsInt, ds2.getClassLabels(), fileName);
				
				System.out.println();
			}
		}

	}
		
}
