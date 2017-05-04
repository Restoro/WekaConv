package Classifier.Impls;

import java.io.File;

import Classifier.AbsClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

public class IBkImpl extends AbsClassifier{

	int k;
	public IBkImpl(File arffTrain, File arffTest) {
		super(arffTrain, arffTest);
		this.k = 0;
	}
	
	public IBkImpl(File arffTrain, File arffTest, int k) {
		super(arffTrain, arffTest);
		this.k = k;
	}
	
	@Override
	public Evaluation executeClassifier(boolean output) throws Exception {
		IBk classi = new IBk();
		System.out.println("Start " + classi.getClass().getName());
		
		Instances trainData = getInstance(arffTrain);
		classi.buildClassifier(trainData);
		
		if(this.k >= 1 ) {
			classi.setKNN(k);
		}
		
		if(output)System.out.println(classi);
		else System.out.println("Model Done!");
		
		return evaluateClassifier(classi, output);
	}

	@Override
	public boolean setParams(int[] param) {
		if(param.length > 0) {
			k = param[0];
			return true;
		}
		return false;
	}
}
