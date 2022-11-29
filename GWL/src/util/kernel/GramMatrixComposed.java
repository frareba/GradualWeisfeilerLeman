package util.kernel;

import java.util.ArrayList;

/**
 * Container for multiple gram matrices.
 * 
 * @author Nils Kriege
 */
public class GramMatrixComposed {
	
	public ArrayList<double[][]> grams;
	public String[] classes;
	public String name;
	
	public GramMatrixComposed() {
		grams = new ArrayList<>();
	}

	public int getComponentCount() {
		return grams.size();
	}
	
	public int getDimension() {
		return grams.get(0).length;
	}
	
}
