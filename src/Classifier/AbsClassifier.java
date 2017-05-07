package Classifier;

import java.io.File;
import java.io.IOException;

import weka.attributeSelection.PrincipalComponents;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

abstract public class AbsClassifier {
	protected File arffTrain;
	protected File arffTest;
	
	public AbsClassifier(File arffTrain, File arffTest) {
		this.arffTrain = arffTrain;
		this.arffTest = arffTest;
	}
	
	abstract public Evaluation executeClassifier(boolean output) throws Exception;
	
	abstract public boolean setParams(int[] param);
	
	protected Evaluation evaluateClassifier(weka.classifiers.Classifier classi, boolean output) throws Exception {
		System.out.println("Start Evaluation");
		Instances test = getInstance(arffTest);
		Evaluation eval = new Evaluation(test);
		eval.evaluateModel(classi, test);
		
		if(output)System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		return eval;
	}
	
	protected ArffLoader getLoader(File arffFile) throws IOException {
		ArffLoader loader = new ArffLoader();
		loader.setFile(arffFile);
		return loader;
	}
	
	protected Instances getInstance(File arffFile) throws IOException {
		ArffLoader loader = getLoader(arffFile);
		Instances returnInstance = new Instances(loader.getDataSet());
		returnInstance.setClassIndex(returnInstance.numAttributes() - 1);
		return returnInstance;
	}
	
	protected Instances getStructure(File arffFile) throws IOException {
		ArffLoader loader = getLoader(arffFile);
		return getStructure(loader);
	}
	
	protected Instances getStructure(ArffLoader loader) throws IOException{
		Instances structure = loader.getStructure();
		structure.setClassIndex(structure.numAttributes() -1 );
		return structure;
	}
}
