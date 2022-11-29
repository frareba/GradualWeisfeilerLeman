package util.kernel;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.Writer;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

import org.ejml.ops.MatrixFeatures;
import org.ejml.simple.SimpleEVD;
import org.ejml.simple.SimpleMatrix;

import comparison.kernel.Kernel;
import datastructure.Pair;

public class GramUtil {
	
	public static double[][] gaussify(double[][] gram, double sigma) {
		int n = gram.length;
		
		double[][] dist = distance(gram);
		double[][] gauss = new double[n][n];

		for (int i=0; i<n; i++) {
			for (int j=0; j<n; j++) {
				double d = Math.pow(dist[i][j], 2);
				d /= 2 * Math.pow(sigma, 2);
				gauss[i][j] = Math.exp(-d);
			}
		}
		
		return gauss;
	}
	
	public static double[][] distance(double[][] gram) {
		int n = gram.length;
		double[][] dist = new double[n][n];

		for (int i=0; i<n; i++) {
			for (int j=0; j<n; j++) {
				dist[i][j]= Math.sqrt((gram[i][i]+gram[j][j]-2*gram[i][j]));
			}
		}
		
		return dist;
	}

	
	
	
	public static double[][] normalize(double[][] gram) {
		int n = gram.length;
		double[][] gramNorm = new double[n][n];

		for (int i=0; i<n; i++) {
			for (int j=0; j<n; j++) {
				gramNorm[i][j]= ((double)gram[i][j]/Math.sqrt((gram[i][i]*gram[j][j])));
			}
		}
		
		return gramNorm;
	}

	public static double[][] tanimoto(double[][] gram) {
		int n = gram.length;

		double[][] gramTani = new double[n][n];

		for (int i=0; i<n; i++) {
			for (int j=0; j<n; j++) {
				gramTani[i][j]= ((double)gram[i][j]/((gram[i][i] + gram[j][j] - gram[i][j])));
			}
		}
		return gramTani;
	}

	public static void writeLibSVMFile(double[][] m, String[] classes, String fileName) throws IOException {
		FileWriter fw = new FileWriter(fileName, false);
		BufferedWriter bw = new BufferedWriter(fw);
		GramUtil.writeLibSVMString(m, classes, bw);
		bw.close();
	}
	
	public static void writeLibSVMFile(GramMatrix gram, String fileName) throws IOException {
		GramUtil.writeLibSVMFile(gram.gram, gram.classes, fileName);
	}
	
	public static void writeLibSVMFile(GramMatrix gram, String fileName, ArrayList<Integer> rows, ArrayList<Integer> columns) throws IOException {
		FileWriter fw = new FileWriter(fileName, false);
		BufferedWriter bw = new BufferedWriter(fw);
		writeLibSVMString(gram.gram, gram.classes, rows, columns, bw);
		bw.close();
	}
	
	/**
	 * Writes data in LibSVM format.
	 * @param m the matrix of quadratic size
	 * @param classes the class labels
	 * @param w the output writer 
	 * @throws IOException
	 */
	private static void writeLibSVMString(double[][] m, String[] classes, Writer w) throws IOException {
		ArrayList<Integer> rc = new ArrayList<>();
		for (int i=0;i<m.length; i++) {
			rc.add(i);
		}
		writeLibSVMString(m, classes, rc, rc, w);
	}

	/**
	 * Writes data in LibSVM format.
	 * @param m the matrix
	 * @param classes the class labels
	 * @param rows restrict to these rows 
	 * @param columns restrict to these columns
	 * @param w the output writer 
	 * @throws IOException
	 */
	private static void writeLibSVMString(double[][] m, String[] classes, ArrayList<Integer> rows, ArrayList<Integer> columns, Writer w) throws IOException {
		int idRow=1;
		for (int i : rows) {
			w.append(classes[i]);
			w.append(" 0:"+idRow++);
			int idColumn=1;
			for (int j : columns) {
				double val = m[i][j];
				// such small values result in an error with strtod()
				// used by libsvm when parsing the input file.
				// it should be safe to set the value to zero here!
				if (val > 0 && val <= 1E-300) {
					System.out.println("Warning: "+val+" was set to 0");
					val = 0d;
				}
				w.append(" "+(idColumn++)+":"+val);
			}
			w.append("\n");
		}
	}
	
	public static void writeSPGGMKLDir(GramMatrixComposed c, String dirName,  ArrayList<Integer> rows, ArrayList<Integer> columns, String suffix) throws IOException {
		File dir = new File(dirName);
		dir.mkdirs();

		FileWriter fw = new FileWriter(dir.getPath()+"/main"+suffix, false);
		BufferedWriter bw = new BufferedWriter(fw);
		for (int i=0; i<c.grams.size(); i++) {
			double[][] gram = c.grams.get(i);
//			if (skipZeroGrams) {
//				boolean isZeroGram = true;
//				for (int x : rows) {
//					for (int y : columns) {
//						if (gram[x][y] != 0d) isZeroGram = false;
//					}
//				}
//				if (isZeroGram) continue;
//			}
			
			String fileName = dir.getPath()+"/"+i+".gram"+suffix;
			FileWriter fw2 = new FileWriter(fileName, false);
			BufferedWriter bw2 = new BufferedWriter(fw2);
			bw2.write(rows.size()+" "+columns.size()+"\n");
			for (int x : rows) {
				for (int y : columns) {
					bw2.append(String.valueOf(gram[x][y]));
					bw2.append(' ');
				}
				bw2.append('\n');
			}
			bw2.close();
			
			bw.append("-t 4 -f "+fileName+"\n");
		}
		bw.close();
		
		fw = new FileWriter(dir.getPath()+"/y"+suffix, false);
		bw = new BufferedWriter(fw);
		for (int x : rows) {
			bw.append(c.classes[x]);
			bw.append('\n');
		}
		bw.close();
	}
	
	
	public static void writeSPGGMKLDir(GramMatrixComposed c, String dirName) throws IOException {
		ArrayList<Integer> rc = new ArrayList<>();
		for (int i=0;i<c.getDimension(); i++) {
			rc.add(i);
		}
		writeSPGGMKLDir(c, dirName, rc, rc,"");
	}
	
	
	public static GramMatrixComposed readSPGGMKLDir(File dir) throws FileNotFoundException, IOException {
		
		GramMatrixComposed r = new GramMatrixComposed();
		r.name = dir.getName();
		int lineCount = 0;
		
		// load classes
		ArrayList<String> classes = new ArrayList<>();
		try (BufferedReader br = new BufferedReader(new FileReader(dir.getPath()+"/y"))) {
		    String line;
		    while ((line = br.readLine()) != null) {
		    	lineCount++;
		    	classes.add(line);
		    }
		}
		r.classes = classes.toArray(new String[lineCount]);
		
		// load grams
		File[] directoryListing = dir.listFiles();
		Arrays.sort(directoryListing);
		for (File f : directoryListing) {
			if (!f.getName().endsWith(".gram")) continue;
					
			// load this gram matrix
			ArrayList<double[]> gram = new ArrayList<>();

			try (BufferedReader br = new BufferedReader(new FileReader(f))) {
				String line = br.readLine();
				int rows = Integer.valueOf(line.split(" ")[0]);
				if (rows != lineCount) {
					throw new IllegalArgumentException("Number of lines doens not match "+line.split(" ")[0]+" "+lineCount);
				}
				while ((line = br.readLine()) != null) {
					String[] ts = line.split(" ");
					double[] values = new double[ts.length];
					for (int i=0; i<ts.length; i++) {
						double val = Double.valueOf(ts[i]);
						values[i] = val;
					}
					gram.add(values);
				}
			}
			double[][] gramD = gram.toArray(new double[lineCount][lineCount]);
			r.grams.add(gramD);

			
		}
		return r;
	}
	
	public static GramMatrix readLibSVMFile(File file) throws FileNotFoundException, IOException {
		ArrayList<String> classes = new ArrayList<>();
		ArrayList<double[]> gram = new ArrayList<>();
		int lineCount = 0;
		try (BufferedReader br = new BufferedReader(new FileReader(file))) {
			String line;
			while ((line = br.readLine()) != null) {
				lineCount++;
				String[] ts = line.split(" ");
				classes.add(ts[0]);
				if (Integer.valueOf(ts[1].split(":")[1]) != lineCount) {
					System.out.println("Error line count: "+lineCount);
				}
				double[] values = new double[ts.length-2];
				for (int i=2; i<ts.length; i++) {
					String[] p = ts[i].split(":");
					int idx = Integer.valueOf(p[0]); 
					double val = Double.valueOf(p[1]);
					if (idx != i-1) System.out.println("Error: "+i+" "+idx);
					values[i-2] = val;
				}
				gram.add(values);

			}
		}
		return new GramMatrix(
						gram.toArray(new double[lineCount][lineCount]),
						classes.toArray(new String[lineCount]),
						file.getName());
	}
	
	public static String matrixToString(double[][] m, char separator) {
		StringBuffer sb = new StringBuffer();
		int n1 = m.length;
		int n2 = m[0].length;
		for (int i=0; i<n1; i++) {
			for (int j=0; j<n2; j++) {
				double val = m[i][j];
				if (j!=0) sb.append(separator);
				sb.append(val);
			}
			sb.append("\n");
		}

		return sb.toString();
	}
	
	public static String matrixToString(double[][] m) {
		return matrixToString(m, '\t');
	}
	
	
	
	public static String matrixToStringFormat(double[][] m) {
		DecimalFormat df = new DecimalFormat("0.00");
		
		StringBuffer sb = new StringBuffer();
		int n = m.length;
		for (int i=0; i<n; i++) {
			for (int j=0; j<n; j++) {
				sb.append("\t"+df.format(m[i][j]));
			}
			sb.append("\n");
		}

		return sb.toString();
	}
	
	public static String matrixToMathematicaString(double[][] m) {
		StringBuffer sb = new StringBuffer();
		int n = m.length;
		sb.append("{");
		for (int i=0; i<n; i++) {
			sb.append("{");
			for (int j=0; j<n; j++) {
				double val = m[i][j];
				sb.append(val);
				sb.append(",");
			}
			sb.append("},");
		}
		sb.append("}");
		return sb.toString();
	}
	
	public static boolean isPSD(double[][] gram) {
		SimpleMatrix m = new SimpleMatrix(gram);

		boolean isPSD = MatrixFeatures.isPositiveSemidefinite(m.getMatrix());
		if (!isPSD) {
			SimpleEVD<?> evd = m.eig();
			double minEV = Integer.MAX_VALUE;
			for (int i=0; i<evd.getNumberOfEigenvalues(); i++) {
				minEV = Math.min(minEV, evd.getEigenvalue(i).getReal());
			}
			// other negative eigenvalues are likely to be due to numerical instability! 
			if (minEV > -0.0000000001) isPSD = true;
		}
		
		return isPSD; 

	}
	
	/**
	 * 
	 * @param gram
	 * @return true iff smallest eigenvalue > 100
	 */
	public static boolean checkPSD(double[][] gram) {
		SimpleMatrix m = new SimpleMatrix(gram);
		int psd = -1;
		try {
			// this sometimes causes a null pointer exception
			// ejml bug?!
			psd = MatrixFeatures.isPositiveSemidefinite(m.getMatrix()) ? 1 : 0;
		} catch (NullPointerException e) {}
		if (psd == 0) {
			SimpleEVD<?> evd = m.eig();
			double minEV = Integer.MAX_VALUE;
			for (int i=0; i<evd.getNumberOfEigenvalues(); i++) {
				minEV = Math.min(minEV, evd.getEigenvalue(i).getReal());
			}
			// other negative eigenvalues are likely to be due to numerical instability! 
			if (minEV < -0.0000001) {
				System.out.print("!!WARNING!! ");
			}
			System.out.println("Smallest eigenvalue is "+minEV);
			
			if (minEV < -100) {
				return false;
			} else {
				return true;
			}
		} else if (psd == 1) {
			System.out.println("FINE: Gram matrix is psd!");
		} else if (psd == -1) {
			System.out.println("Error computing eigenvalues! Matrix possibly is psd, returning false anyway.");
			return false;
		}
		return true;
	}
	
	public static void findMostSimilar(final double[][] m, String[] classes) {
		int n = m.length;
		
		ArrayList<Pair<Integer,Integer>> issues = new ArrayList<Pair<Integer,Integer>>();
		
		for (int i=0; i<n; i++) {
			String iActivity = classes[i];
			for (int j=i; j<n; j++) {
				String jActivity = classes[j];
				if (!iActivity.equals(jActivity)) {
					issues.add(new Pair<Integer, Integer>(i,j));
				}
			}
		}
		
		Collections.sort(issues, new Comparator<Pair<Integer,Integer>>() {
			public int compare(Pair<Integer, Integer> o1, Pair<Integer, Integer> o2) {
				double m1 = m[o1.getFirst()][o1.getSecond()];
				double m2 = m[o2.getFirst()][o2.getSecond()];
				
				if (m1<m2) return 1;
				if (m1>m2) return -1;
				return 0;
			}
		});
		for (int x=0; x<Math.min(10, issues.size()); x++) {
			Pair<Integer,Integer> p = issues.get(x);
			System.out.println("\t"+m[p.getFirst()][p.getSecond()]+"\t"+p);
		}
		
	}
	
	public static boolean checkPosDiagonal(final double[][] m) {
		for (int i=0; i<m.length; i++) {
			if (m[i][i] < 0) {
				return false;
			}
		}
		return true;
	}
	
	public static int nonDistinguishedPairs(final double[][] m) {
		int count = 0;
		int n = m.length;
		for (int i=0; i<n; i++) {
			for (int j=i+1; j<n; j++) {
				if (m[i][i]+m[j][j]-2*m[i][j] == 0) count++;
			}
		}
		return count;
	}
	
	public static boolean isComplete(final double[][] m) {
		int n = m.length;
		for (int i=0; i<n; i++) {
			double[] row1 = m[i];
			for (int j=i+1; j<n; j++) {
				double[] row2 = m[j];
				// compare lines
				boolean valid = false;
				for (int k=0; k<n; k++) {
					if (row1[k] != row2[k]) valid = true;
				}
				if (!valid) return false;
			}
		}
		return true;
	}

	public static boolean isComplete(final double[][] m, String[] classes) {
		int n = m.length;
		for (int i=0; i<n; i++) {
			double[] row1 = m[i];
			for (int j=i+1; j<n; j++) {
				if (classes[i].equals(classes[j])) continue;
				double[] row2 = m[j];
				// compare lines
				boolean valid = false;
				for (int k=0; k<n; k++) {
					if (row1[k] != row2[k]) valid = true;
				}
				if (!valid) return false;
			}
		}
		return true;
	}
	
	public static double completenessRatioPairs(final double[][] m) {
		int distinguishablePairs = 0;
		int testedPairs = 0;
		
		int n = m.length;
		for (int i=0; i<n; i++) {
			for (int j=i+1; j<n; j++) {
				testedPairs++;
				// distance in Hilbert space
				double dist = Math.sqrt(m[i][i]+m[j][j]-2*m[i][j]);
				if (dist != 0) distinguishablePairs++;
			}
		}
		return (double)distinguishablePairs/testedPairs;
	}
	
	public static double completenessRatio(final double[][] m) {
		return completenessRatio(m, 0);
	}
	
	public static double completenessRatio(final double[][] m, double epsilon) {
		int distinguishable = 0;

		int n = m.length;
		for (int i=0; i<n; i++) {
			boolean distinct = true;
			for (int j=0; j<n; j++) {
				// distance in Hilbert space
				double sqDist = m[i][i]+m[j][j]-2*m[i][j];
				if (sqDist <= epsilon  && j != i) distinct = false;
			}
			if (distinct) distinguishable++; 
		}
		return (double)distinguishable/n;
	}
	
	public static double completenessRatio(final double[][] m, String[] classes) {
		return completenessRatio(m, classes, 0);
	}
	
	public static double completenessRatio(final double[][] m, String[] classes, double epsilon) {
		int distinguishable = 0;

		int n = m.length;
		for (int i=0; i<n; i++) {
			boolean distinct = true;
			for (int j=0; j<n; j++) {
				if (classes[i].equals(classes[j])) continue;
				// distance in Hilbert space
				double sqDist = m[i][i]+m[j][j]-2*m[i][j];
				if (sqDist <= epsilon  && j != i) distinct = false;
			}
			if (distinct) distinguishable++; 
		}
		return (double)distinguishable/n;
	}
	
	public static double completenessRatioPairs(final double[][] m, String[] classes) {
		int distinguishable = 0;
		int testedPairs = 0;
		
		int n = m.length;
		for (int i=0; i<n; i++) {
			for (int j=i+1; j<n; j++) {
				if (classes[i].equals(classes[j])) continue;
				testedPairs++;
				// distance in Hilbert space
				double dist = Math.sqrt(m[i][i]+m[j][j]-2*m[i][j]);
				if (dist != 0) distinguishable++;
			}
		}
		return (double)distinguishable/testedPairs;
	}
	
	public static double[][] distanceMatrix(double[][] gram) {
		int n = gram.length;
		double[][] dist = new double[n][n];
		for (int i=0; i<n; i++) {
			for (int j=i; j<n; j++) {
				dist[i][j] = dist[j][i] = gram[i][i] + gram[j][j] - 2*gram[i][j];
			}
		}
		return dist;
	}
	
	public static double[][] angleMatrix(double[][] gram) {
		int n = gram.length;
		double[][] ang = new double[n][n];
		for (int i=0; i<n; i++) {
			for (int j=i; j<n; j++) {
				double normI = Math.sqrt(gram[i][i]);
				double normJ = Math.sqrt(gram[j][j]);
				double angle = Math.acos(gram[i][j]/(normI*normJ));
				ang[i][j] = ang[j][i] = angle;
			}
		}
		return ang;
	}
	
	
	public static boolean isEqual(double[][] gram1, double[][] gram2) {
		int n = gram1.length;
		for (int i=0; i<n; i++) {
			double e[] = gram1[i];
			double a[] = gram2[i];
			for (int j=0; j<n; j++) {
				if (e[j]!=a[j]) return false;
			}
		}
		return true;
	}
	
	public static double elementSum(double[][] gram) {
		int n = gram.length;
		double k=0;
		for (int i=0; i<n; i++) {
			for (int j=0; j<n; j++) {
				k += gram[i][j];
			}
		}
		return k;
	}
	
	public static void plot(String name, double[][] data, int steps) throws IOException, InterruptedException {
		
		// write data to file
		try(PrintWriter out = new PrintWriter(name+".data" )  ){
		    out.println( GramUtil.matrixToString(data) );
		}

		
		// write gnuplot script
		try(PrintWriter out = new PrintWriter(name+".gp" )  ){
			out.println("set xrange [-0.5:"+(steps-.5)+"]");
			out.println("set yrange [-0.5:"+(steps-.5)+"]");
//			out.println("set cbrange [0:1]");
			out.println("set tics nomirror out scale 0.75");
			out.println("unset ytics");
			out.println("unset xtics");
			out.println("set cbtics 0,1");
		    out.println("unset key");
			out.println("set terminal pdf size 5,5 font \"Verdana,24\" linewidth 4");
		    out.println("set output '"+name+".pdf'");
		    out.println("set view map");
		    out.println("splot '"+name+".data' matrix with image");

		}
		
		
		// run gnuplot
		final Process p = Runtime.getRuntime().exec("gnuplot "+name+".gp",new String[0],new File("./plot/"));
		new Thread(new Runnable() {
		    public void run() {
		     BufferedReader input = new BufferedReader(new InputStreamReader(p.getInputStream()));
		     String line = null; 

		     try {
		        while ((line = input.readLine()) != null)
		            System.out.println(line);
		     } catch (IOException e) {
		            e.printStackTrace();
		     }
		    }
		}).start();
	    p.waitFor();
	}
	
	public static void plot(String name, double[][] data) throws IOException, InterruptedException {
		
		// write data to file
		try(PrintWriter out = new PrintWriter(name+".data" )  ){
		    out.println( GramUtil.matrixToString(data) );
		}

		
		// write gnuplot script
		try(PrintWriter out = new PrintWriter(name+".gp" )  ){
			out.println("set autoscale xfix");
			out.println("set autoscale yfix");
			out.println("set autoscale cbfix");
			out.println("set terminal pdf");
		    out.println("set output '"+name+".pdf'");
		    out.println("set size square");
		    out.println("plot '"+name+".data' matrix with image notitle");

		}
		
		
		// run gnuplot
		final Process p = Runtime.getRuntime().exec("gnuplot "+name+".gp",new String[0],new File("."));
		new Thread(new Runnable() {
		    public void run() {
		     BufferedReader input = new BufferedReader(new InputStreamReader(p.getInputStream()));
		     String line = null; 

		     try {
		        while ((line = input.readLine()) != null)
		            System.out.println(line);
		     } catch (IOException e) {
		            e.printStackTrace();
		     }
		    }
		}).start();
	    p.waitFor();
	}
	
	public static void plot(Kernel<double[]> k, double xMin, double xMax, double yMin, double yMax, int steps) throws IOException, InterruptedException {

		double xStepWidth = (xMax-xMin)/(steps-1);
		double yStepWidth = (yMax-yMin)/(steps-1);

		double[][] result = new double[steps+1][steps+1];
		for (int x=0; x<steps; x++) {
			for (int y=0; y<steps; y++) {
				result[x][y] = k.compute(new double[]{xMin+x*xStepWidth}, new double[]{yMin+y*yStepWidth});
			}
		}
		plot(k.getID(), result, steps);
		
	}

}
