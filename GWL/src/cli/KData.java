package cli;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import com.beust.jcommander.Parameters;
import com.beust.jcommander.converters.FileConverter;

import benchmark.dataset.AttrDataset;
import benchmark.dataset.Statistics;
import io.MatGraphSetWriter;


public class KData {
	
	public static String REPO_BASE_URL = "http://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/";

	static class CommandMain extends KCommon.CommandMain {

		@Parameter(names = { "-D", "--datadir" }, description = "Directory for data files", converter = FileConverter.class)
		File dataDir = new File("data"); 
		
	}
	
	static class CommandBase {
		
		@Parameter(description = "datasets")
		List<String> datasets;

		@Parameter(names = { "-a", "--all" }, description = "All available data sets")
		boolean all;

	}
	
	@Parameters(commandDescription = "Downloads data sets (Requires: wget, unzip)")
	static class CommandGet extends CommandBase {
		
		@Parameter(names = { "-l", "--list" }, description = "List available datasets")
		private boolean list;

	}
	
	@Parameters(commandDescription = "Converts to Matlab file format")
	public static class CommandConvert extends CommandBase {}
	
	@Parameters(commandDescription = "Shows statistics for data sets.")
	public static class CommandStats extends CommandBase {}
	
	static CommandMain cm = new CommandMain();
	static CommandGet get = new CommandGet();
	static CommandConvert convert = new CommandConvert();
	static CommandStats stats = new CommandStats();

	public static void main(String[] argsString) throws IOException, InterruptedException {
		
		JCommander jc = new JCommander(cm);
		jc.addCommand("get", get);
		jc.addCommand("convert", convert);
		jc.addCommand("stats", stats);
		jc.setProgramName("kdata");

		jc.parse(argsString);
		
		if (cm.help || jc.getParsedCommand() == null) {
			jc.usage();
			System.exit(0);
		}
		
		switch (jc.getParsedCommand()) {
			case "get" : {
				if (get.list) {
					list();
					break;
				}
				if (get.all) {
					get.datasets = getList(); 
				}
				if (get.datasets == null) {
					throw new ParameterException("No datasets specified!");
				}
				for (String ds : get.datasets) {
					download(ds);
				}
				break;
			}
			case "convert" : {
				if (convert.all) {
					convert.datasets = KCommon.getDatasets(cm.dataDir);
				}
				for (String ds : convert.datasets) {
					convert(ds);
				}
				break;
			}
			case "stats" : {
				if (stats.all) {
					stats.datasets = KCommon.getDatasets(cm.dataDir);
				}
				for (String ds : stats.datasets) {
					statistics(ds);
				}
				break;
			}
		}
			
		
	}
	
	public static void convert(String name) throws IOException {
		System.out.println("Converting datatset "+name);
		AttrDataset ds = KCommon.load(name, cm.dataDir);		
		MatGraphSetWriter mw = new MatGraphSetWriter();
		mw.write(ds, new File(cm.dataDir.getAbsolutePath()+"/"+name+".mat"));
		System.out.println("Success!");
	}
	
	public static void statistics(String name) throws IOException {
		System.out.println("Loading datatset "+name);
		AttrDataset ds = KCommon.load(name, cm.dataDir);
		Statistics.printLabeledGraphStatistics(ds);
		System.out.println();
	}
	
	public static void download(String name) throws IOException, InterruptedException {
		System.out.println("Downloading datatset "+name);
		
		if (!cm.dataDir.exists()) cm.dataDir.mkdir();
		
		KCommon.runCommand("wget "+REPO_BASE_URL+name+".zip --output-document="+cm.dataDir+"/"+name+".zip", cm.verbose);
		System.out.println("Unzipping "+name+".zip");
		KCommon.runCommand("unzip -o "+cm.dataDir+"/"+name+".zip -d "+cm.dataDir, cm.verbose);
	}
	
	public static void list() throws IOException {
		
		ArrayList<String> list = getList();
		for (String ds : list) {
			System.out.println("\t"+ds);
		}
	}
	
	private static ArrayList<String> getList() throws IOException {
		ArrayList<String> list = new ArrayList<String>();
		
        URL repo = new URL(REPO_BASE_URL);
        BufferedReader in = new BufferedReader(
        new InputStreamReader(repo.openStream()));

        Pattern p = Pattern.compile("href=\"(.*?).zip\"");
        String inputLine;
        while ((inputLine = in.readLine()) != null) {
            Matcher m = p.matcher(inputLine);
            if (m.find()) {
            	String link = m.group(1);
            	if (!"DS_all".equals(link)) {
            		list.add(link);
            	}
            }
        }
        in.close();
        return list;
	}

}
