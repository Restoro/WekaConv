import java.io.File;
import java.io.IOException;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class Classifier {
	File arffTrain;
	File arffTest;
	
	public Classifier(File arffTrain, File arffTest) {
		this.arffTrain = arffTrain;
		this.arffTest = arffTest;
	}
	
	
	public double ibk(boolean output) {
		try {
			System.out.println("Start IBk");
			return executeClassifier(new IBk(), output);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return 0;
	}

	public double naiveBayes(boolean output) {
		try {
			System.out.println("Start Bayes");
			return executeClassifier(new NaiveBayes(), output);
		} catch (Exception e) {
			e.printStackTrace();
			return 0d;
		}
	}
	
	private double executeClassifier(weka.classifiers.Classifier classi, boolean output) throws Exception {
		System.out.println("Start " + classi.getClass().getName());
		
		Instances trainData = getInstance(arffTrain);
		Instances test = getInstance(arffTest);
		classi.buildClassifier(trainData);
		if(output)System.out.println(classi);
		else System.out.println("Model Done!");
		
		Evaluation eval = new Evaluation(test);
		eval.evaluateModel(classi, test);
		
		if(output)System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		return eval.pctCorrect();
	}
	
	private ArffLoader getLoader(File arffFile) throws IOException {
		ArffLoader loader = new ArffLoader();
		loader.setFile(arffFile);
		return loader;
	}
	
	private Instances getInstance(File arffFile) throws IOException {
		ArffLoader loader = getLoader(arffFile);
		Instances returnInstance = new Instances(loader.getDataSet());
		returnInstance.setClassIndex(returnInstance.numAttributes() - 1);
		return returnInstance;
	}
}
