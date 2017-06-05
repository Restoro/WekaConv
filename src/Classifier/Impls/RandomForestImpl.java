package Classifier.Impls;

import java.io.File;

import Classifier.AbsClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class RandomForestImpl extends AbsClassifier {

	int iterations;
	int numOfThreads;

	public RandomForestImpl(File arffTrain, File testFile, String pathToData) {
		super(arffTrain, testFile, pathToData);
		this.iterations = 120;
		this.numOfThreads = 8;
	}

	@Override
	public Evaluation executeClassifier(boolean output, boolean useSave) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Float executeSegmentClassifier(boolean output, boolean useSave) throws Exception {
		return this.loadModelExecuteSegment(RandomForest.class.getSimpleName(), useSave);
	}

	@Override
	public boolean setParams(int[] param) {
		if(param.length == 1) {
			iterations = param[0];
		}
		if (param.length == 2) {
			iterations = param[0];
			numOfThreads = param[1];
			return true;
		}
		return false;
	}

	@Override
	public Classifier generateClassifier(boolean saveModel) throws Exception {
		RandomForest classi = new RandomForest();
		System.out.println("Start " + classi.getClass().getName());
		classi.setNumIterations(this.iterations);
		classi.setNumExecutionSlots(this.numOfThreads);

		Instances data = getInstance(arffTrain);
		classi.buildClassifier(data);
		System.out.println("Model Done!");
		if (saveModel)
			saveClassifier(classi);

		return classi;
	}

	@Override
	public String getClassifierName() {
		return RandomForest.class.getSimpleName();
	}

}
