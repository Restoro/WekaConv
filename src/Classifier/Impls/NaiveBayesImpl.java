package Classifier.Impls;

import java.io.File;

import Classifier.AbsClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class NaiveBayesImpl extends AbsClassifier{

	public NaiveBayesImpl(File arffTrain, File arffTest) {
		super(arffTrain, arffTest);
	}

	@Override
	public Evaluation executeClassifier(boolean output) throws Exception {
		NaiveBayes classi = new NaiveBayes();
		System.out.println("Start " + classi.getClass().getName());
		
		Instances trainData = getInstance(arffTrain);
		classi.buildClassifier(trainData);
		
		if(output)System.out.println(classi);
		else System.out.println("Model Done!");
		
		return evaluateClassifier(classi, output);
	}

	@Override
	public boolean setParams(int[] param) {
		return true; //No params to set
	}
	
}
