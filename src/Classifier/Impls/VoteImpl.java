package Classifier.Impls;

import java.io.File;

import Classifier.AbsClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class VoteImpl extends AbsClassifier{

	public VoteImpl(File arffTrain, File testFile, String pathToData) {
		super(arffTrain, testFile, pathToData);
	}

	@Override
	public Classifier generateClassifier(boolean saveModel) throws Exception {
		Vote classifier = new Vote();
		NaiveBayesImpl bayesImpl = new NaiveBayesImpl(arffTrain, testFile, pathToData);
		RandomForestImpl forestImpl = new RandomForestImpl(arffTrain, testFile, pathToData);
		weka.classifiers.Classifier bayesClassifier = bayesImpl.loadOrGenerateClassifier(saveModel);
		weka.classifiers.Classifier forestClassifier = forestImpl.loadOrGenerateClassifier(saveModel);
		classifier.addPreBuiltClassifier(bayesClassifier);
		classifier.addPreBuiltClassifier(forestClassifier);
		
		Instances data = getInstance(arffTrain);
		classifier.buildClassifier(data);
		System.out.println("Model Done!");
		if (saveModel)
			saveClassifier(classifier);
		return classifier;
	}


	@Override
	public Float executeSegmentClassifier(boolean output, boolean useSave) throws Exception {
		return this.loadModelExecuteSegment(useSave);
	}

	@Override
	public boolean setParams(int[] param) {
		return true;
	}

	@Override
	public String getClassifierName() {
		return Vote.class.getSimpleName();
	}

	@Override
	public Evaluation executeClassifier(boolean output, boolean useSave) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

}
