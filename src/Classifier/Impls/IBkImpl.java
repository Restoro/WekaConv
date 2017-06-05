package Classifier.Impls;

import java.io.File;

import Classifier.AbsClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class IBkImpl extends AbsClassifier{

	int k;
	public IBkImpl(File arffTrain, File testFile, String pathToData) {
		super(arffTrain, testFile, pathToData);
		this.k = 0;
	}
	
	public IBkImpl(File arffTrain, File testFile, String pathToData, int k) {
		super(arffTrain, testFile, pathToData);
		this.k = k;
	}
	
	@Override
	public Evaluation executeClassifier(boolean output, boolean useSave) throws Exception {
		return evaluateClassifier(this.generateClassifier(false), output);
	}
	
	@Override
	public Float executeSegmentClassifier(boolean output, boolean useSave) throws Exception {
		return evaluateClassifierSegment(this.generateClassifier(false));
		
	}

	@Override
	public boolean setParams(int[] param) {
		if(param.length > 0) {
			k = param[0];
			return true;
		}
		return false;
	}



	@Override
	public Classifier generateClassifier(boolean saveModel) throws Exception {
		IBk classi = new IBk();
		System.out.println("Start " + classi.getClass().getName());
		
		//Instances trainData = getInstance(arffTrain);
		ArffLoader loader = getLoader(arffTrain);
		Instances structure = getStructure(loader);
		
		if(this.k >= 1 ) {
			classi.setKNN(k);
		}
		
		Instance current;
		classi.buildClassifier(structure);
		
		while ((current = loader.getNextInstance(structure)) != null)
			   classi.updateClassifier(current);
		
		System.out.println("Model Done!");
		return classi;
	}

	@Override
	public String getClassifierName() {
		return IBk.class.getSimpleName();
	}
}
