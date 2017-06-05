package Classifier.Impls;

import java.io.File;

import Classifier.AbsClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;

public class SVMImpl extends AbsClassifier{

	int c;
	double gamma;
	public SVMImpl(File arffTrain, File testFile, String pathToData) {
		super(arffTrain, testFile, pathToData);
		this.c = 1024;
		this.gamma = 1d/512d;
	}

	@Override
	public String getClassifierName() {
		return SMO.class.getSimpleName();
	}

	@Override
	public Classifier generateClassifier(boolean saveModel) throws Exception {
		SMO classi = new SMO();
		System.out.println("Start " + classi.getClass().getName());
		classi.setC(this.c);
		RBFKernel kernel = new RBFKernel();
		kernel.setGamma(gamma);
		classi.setKernel(kernel);

		Instances data = getInstance(arffTrain);
		classi.buildClassifier(data);
		System.out.println("Model Done!");
		if (saveModel)
			saveClassifier(classi);

		return classi;
	}

	@Override
	public Evaluation executeClassifier(boolean output, boolean useSave) throws Exception {
		return this.evaluateClassifier(this.generateClassifier(useSave), output);
	}

	@Override
	public Float executeSegmentClassifier(boolean output, boolean useSave) throws Exception {
		return this.loadModelExecuteSegment(useSave);
	}

	@Override
	public boolean setParams(int[] param) {
		if(param != null) {
			if(param.length == 1) {
				this.c = param[0];
				return true;
			} else if(param.length == 2) {
				this.c = param[0];
				this.gamma = 1d/param[1];
				return true;
			}
		}
		return false;
	}

}
