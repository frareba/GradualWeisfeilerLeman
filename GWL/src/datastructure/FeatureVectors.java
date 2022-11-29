package datastructure;

import static jeigen.Shortcuts.spzeros;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;

import jeigen.SparseMatrixLil;

public class FeatureVectors {
	
	public ArrayList<? extends FeatureVector<Integer>> fvs;
	public String[] classes;
	
	public FeatureVectors(ArrayList<? extends FeatureVector<Integer>> fvs, String[] classes) {
		this.fvs=fvs;
		this.classes=classes;
	}
	
	@Override
	public boolean equals(Object obj) {
		if (!(obj instanceof FeatureVectors)) return false;
		FeatureVectors other = (FeatureVectors)obj;
		
		if (!this.fvs.equals(other.fvs)) return false;
		if (!Arrays.equals(this.classes, other.classes)) return false;
		
		return true;
	}
	
	
	// --------------------
	// static methods
	// -------------------
	
	public static ArrayList<SparseFeatureVector<Integer>> toIntegerIndexed(Collection<? extends FeatureVector<?>> fvs) {
		ArrayList<SparseFeatureVector<Integer>> r = new ArrayList<SparseFeatureVector<Integer>>(fvs.size());
		
		int nextIndex = 0;
		HashMap<Object,Integer> map = new HashMap<>();
		for (FeatureVector<?> fv : fvs) {
			SparseFeatureVector<Integer> rfv = new SparseFeatureVector<>();
			for (Entry<?, Double> e : fv.nonZeroEntries()) {
				Integer index = map.get(e.getKey());
				if (index == null) {
					index = nextIndex++;
					map.put(e.getKey(), index);
				}
				rfv.setValue(index, e.getValue());
			}
			r.add(rfv);
		}
		return r;
	}
	
	/**
	 * Uses C++ library Eigen for efficient computation!
	 * @param fvs the feature vectors
	 * @return the gram matrix
	 */
	public static double[][] computeGram(List<? extends FeatureVector<Integer>> fvs) {
		int max = Integer.MIN_VALUE;
		for (FeatureVector<Integer> fv : fvs) {
			for (Entry<Integer, Double> e : fv.nonZeroEntries()) {
				max = Math.max(max, e.getKey());
			}
		}
		int dim = max+1;
		int dsSize = fvs.size();
		
		SparseMatrixLil mfv1=spzeros(dim,dsSize);
		for (int i=0; i<dsSize; i++) {
			FeatureVector<Integer> fv = fvs.get(i);
			for (Entry<Integer, Double> e : fv.nonZeroEntries()) {
//				System.out.println(e.getKey()+" "+i+" "+e.getValue());;
				mfv1.append(e.getKey(), i, e.getValue());
			}
		}
		SparseMatrixLil mfv2=mfv1.t();
		SparseMatrixLil r = mfv2.mmul(mfv1);
		
		double[][] gram = new double[dsSize][dsSize];
		for (int i=0; i< r.getSize(); i++) {
			gram[r.getRowIdx(i)][r.getColIdx(i)] = r.getValue(i);
		}
		return gram;
	}
	
	/**
	 * Write feature vector in LibSVM format.
	 * @see #toIntegerIndexed(List) to transform arbitrary features to integers
	 * @throws IOException
	 */
	public static void writeLibSVMFile(FeatureVectors d, ArrayList<Integer> ids, String fileName) throws IOException {
		FileWriter fw = new FileWriter(fileName, false);
		BufferedWriter w = new BufferedWriter(fw);
		
		for (Integer id : ids) {
			FeatureVector<Integer> fv = d.fvs.get(id);
			// sort the key value pairs
			ArrayList<Entry<Integer, Double>> es = new ArrayList<>();
			for (Entry<Integer, Double> e : fv.nonZeroEntries()) {
				es.add(e);
			}
			Collections.sort(es, Entry.comparingByKey());
			
			w.append(d.classes[id]);
			for (Entry<Integer, Double> e : es) {
				w.append(' ');
				w.append(String.valueOf(e.getKey()+1)); // 0 is reserved, start with 1 
				w.append(':');
				w.append(e.getValue().toString());
			}
			w.append("\n");
		}
		
		w.close();
	}
	
	public static void writeSPGGMKLKernelFile(FeatureVectors d, String fileName) throws IOException {
		HashSet<Integer> features = new HashSet<>();
		for (FeatureVector<Integer> fv : d.fvs) {
			for (Integer i : fv.nonZeroFeatures()) {
				features.add(i);
			}
		}
		ArrayList<Integer> features2 = new ArrayList<>(features);
		Collections.sort(features2);
		FileWriter fw = new FileWriter(fileName, false);
		BufferedWriter bw = new BufferedWriter(fw);
		for (int i : features) {
			bw.append("-t 6 -c "+(i+1)+"\n");
		}
//		bw.append("-t 5\n");
		bw.close();
	}
	
	public static void writeLibSVMFile(FeatureVectors d, String fileName) throws IOException {
		ArrayList<Integer> rc = new ArrayList<>();
		for (int i=0;i<d.fvs.size(); i++) {
			rc.add(i);
		}
		writeLibSVMFile(d, rc, fileName);
	}

	public static void writeLibSVMFile(ArrayList<SparseFeatureVector<Integer>> fvs, String[] classLabels, String fileName) throws IOException {
		writeLibSVMFile(new FeatureVectors(fvs, classLabels), fileName);
	}
	
	public static FeatureVectors readLibSVMFile(String fileName) throws IOException {
		try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
			return readLibSVMFile(new File(fileName));
		}
	}
	
	public static FeatureVectors readLibSVMFile(File file) throws IOException {
		ArrayList<String> classes = new ArrayList<>();
		ArrayList<SparseFeatureVector<Integer>> fvs = new ArrayList<>();
		int lineCount = 0;
		try (BufferedReader br = new BufferedReader(new FileReader(file))) {
			String line;
			while ((line = br.readLine()) != null) {
				lineCount++;
				String[] ts = line.split(" ");
				classes.add(ts[0]);
				SparseFeatureVector<Integer> fv = new SparseFeatureVector<>();
				for (int i=1; i<ts.length; i++) {
					String[] p = ts[i].split(":");
					int idx = Integer.valueOf(p[0]); 
					double val = Double.valueOf(p[1]);
					fv.setValue(idx-1, val);
				}
				fvs.add(fv);
			}
		}
		return new FeatureVectors(fvs, classes.toArray(new String[lineCount]));
	}

	/**
	 * Note: The keys of the feature vectors must be identifiable via the String value
	 * obtained by String.valueOf()!
	 * @param A
	 * @param B
	 * @return
	 */
	public static SparseFeatureVector<String> kronecker(FeatureVector<?> A, FeatureVector<?> B) {
		SparseFeatureVector<String> r = new SparseFeatureVector<>();
		for (Entry<?, Double> eA : A.nonZeroEntries()) {
			for (Entry<?, Double> eB : B.nonZeroEntries()) {
				String featureKey = String.valueOf(eA.getKey())+String.valueOf(eB.getKey());
				r.setValue(featureKey, eA.getValue()*eB.getValue());
			}
		}
		return r;
	}
	

}
