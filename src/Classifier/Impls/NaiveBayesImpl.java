package Classifier.Impls;

import java.io.File;

import Classifier.AbsClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class NaiveBayesImpl extends AbsClassifier{
	
	public NaiveBayesImpl(File arffTrain, File testFile, String pathToData) {
		super(arffTrain, testFile, pathToData);
	}

	@Override
	public Evaluation executeClassifier(boolean output , boolean useSave) throws Exception {
		return evaluateClassifier(this.generateClassifier(false), output);
	}
	

	@Override
	public boolean setParams(int[] param) {
		return true; //No params to set
	}

	@Override
	public Classifier generateClassifier(boolean saveModel) throws Exception {
		NaiveBayes classi = new NaiveBayes();
		System.out.println("Start " + classi.getClass().getName());
		
		ArffLoader loader = getLoader(arffTrain);
		Instances structure = getStructure(loader);
		Instance current;
		classi.buildClassifier(structure);
		
		while ((current = loader.getNextInstance(structure)) != null)
			   classi.updateClassifier(current);

		System.out.println("Model Done!");
		if (saveModel)
			saveClassifier(classi);
		
		return classi;
	}

	@Override
	public Float executeSegmentClassifier(boolean output, boolean useSave) throws Exception {
		return this.loadModelExecuteSegment(useSave);
	}

	@Override
	public String getClassifierName() {
		return NaiveBayes.class.getSimpleName();
	}
	
}
