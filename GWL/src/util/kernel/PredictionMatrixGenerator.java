package util.kernel;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import datastructure.FeatureVector;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

public class PredictionMatrixGenerator extends AccuracyTestLibSVMNative {
	
	HashMap<Integer,Integer> prediction = null;


	public PredictionMatrixGenerator(File folder, List<String> datasets, List<String> kernels,  File accLogFile,
			File selLogFile) {
		super(folder, datasets, kernels, accLogFile, selLogFile);
		REPETITIONS = 1;
	}


	public ArrayList<String> getKernels(String dataset, List<String> kernels) {
		HashSet<String> bk = new HashSet<String>();
		File[] files = pFolder.listFiles();
		for (File f : files) {
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
//				bk.add(name.split("_")[0]); 
				bk.add(name);
			}
		}
		ArrayList<String> r = new ArrayList<>(bk);
		Collections.sort(r);
		return r;
	}
	
	@Override
	public double crossValidate(ArrayList<File> gramFiles, FeatureVector<String> selections, int foldNo)
			throws IOException, InterruptedException {
		
		if (DEBUG) {
			System.out.println("Cross-Validation");
		}
	
		// load all gram matrices
		ArrayList<GramMatrix> gramMatrices = new ArrayList<>();
		for (File f : gramFiles) {
			gramMatrices.add(GramUtil.readLibSVMFile(f));
		}
		
		// create matrix log file
		prediction= new HashMap<>();
		File matrixFile = new File(gramMatrices.get(0).name.split("__")[0]+"_matrix.txt");
		BufferedWriter bwMatrix = null;
		if (!matrixFile.exists()) {
			bwMatrix = new BufferedWriter(new FileWriter(matrixFile, false));
			bwMatrix.write("class");
			for (int i = 0; i<gramMatrices.get(0).getDimension(); i++) {
				bwMatrix.write("\t"+gramMatrices.get(0).classes[i]);
			}
			bwMatrix.write("\n");
		} else {
			bwMatrix = new BufferedWriter(new FileWriter(matrixFile, true));
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
		
		// write matrix
		String kernel = gramMatrices.get(0).name;
		kernel = kernel.split("__")[1];
		kernel = kernel.substring(0, kernel.length()-5);
		bwMatrix.write(kernel);
		for (int i=0; i<gramMatrices.get(0).getDimension(); i++) {
			bwMatrix.write("\t"+prediction.get(i));
		}
		bwMatrix.write("\n");
		
		bwMatrix.close();
		
		return avgAccuracy;
		
	}
	
	
	@Override
	public double predict(ArrayList<Integer> testSet, ArrayList<Integer> trainSet, Parameter param) throws IOException, InterruptedException {

		// train 
		svm_problem prob = new svm_problem();
		prob.l = trainSet.size();
		prob.x = new svm_node[prob.l][];
		prob.y = new double[prob.l];
		for(int i=0;i<prob.l;i++) {
			svm_node[] x = new svm_node[param.gram.getDimension()+1];
			prob.x[i] = x;
			svm_node node = new svm_node();
			node.index = 0;
			int iIndex = trainSet.get(i);
			node.value = iIndex+1;
			prob.x[i][0] = node;
			for (int j=0; j<param.gram.getDimension(); j++) {
				node = new svm_node();
				node.index = j+1;
				node.value = param.gram.gram[iIndex][j];
				prob.x[i][j+1] = node;
			}
			prob.y[i] = Double.valueOf(param.gram.classes[iIndex]);
		}
		
		svm_parameter svm_param = getDefaultParameters();
		svm_param.C = param.C;

		svm_model model = svm.svm_train(prob,svm_param);
		
		// predict
		int d = testSet.size();
		int total_correct = 0;
		for(int i=0;i<d;i++) {
			svm_node[] x = new svm_node[param.gram.getDimension()+1];
			svm_node node = new svm_node();
			node.index = 0;
			int iIndex = testSet.get(i);
			node.value = iIndex+1;
			x[0] = node;
			for (int j=0; j<param.gram.getDimension(); j++) {
				node = new svm_node();
				node.index = j+1;
				node.value = param.gram.gram[iIndex][j];
				x[j+1] = node;
			}
					
			double y = Double.valueOf(param.gram.classes[iIndex]);
			double v = svm.svm_predict(model,x);
			prediction.put(iIndex, (int)v);
			if (y == v) total_correct++;
		}
		
		return 100.0*total_correct/d;
	}
		
}
