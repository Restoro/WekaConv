package Classifier;
import java.io.File;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import Classifier.Impls.IBkImpl;
import Classifier.Impls.NaiveBayesImpl;
import weka.classifiers.Evaluation;

public class Main {

	static final int dataLineCount = 1500;
	static final int randomNumberCount = 20;
	static final int numberOfFolds = 4;
	static final boolean convertFiles = true;
	static final boolean useRandomData = true;

	public static void main(String[] args) {
		try {
			if(args.length == 2) { 
				String pathToTrain = System.getProperty("user.dir") + "/resources/";
				String pathToData = args[0];
				String pathToOutput = args[1];
				if(convertFiles) convertFiles(useRandomData, pathToTrain, pathToData, pathToOutput);
				classifyFold(Classifiers.IBk, pathToOutput, new int[] {1});
				//classifyBayesFold(pathToOutput);
				//for(int i=1;i<=100;i=i==1?5:i + 5)
				//	classifyFold(Classifiers.IBk, pathToOutput, new int[] {i});
			} else {
				System.out.println("Not every parameter set (Length must be 2)");
			}
		} catch (Exception e) {
			System.out.println(e.toString());
		}
	}
	
	private static void convertFiles(boolean selectRandom, String pathToTrain, String pathToData, String pathToOutput) {
		ExecutorService executor = Executors.newFixedThreadPool(4);
		Converter conv = new Converter(pathToTrain, pathToData, pathToOutput, randomNumberCount, dataLineCount);
		for(int i=1; i <= numberOfFolds; i++) {
			conv.executeInThread(executor, pathToTrain + "fold" + i + "_train.txt", pathToData, pathToOutput + "combinedFold" + i + ".arff",
					selectRandom);
			conv.executeInThread(executor, pathToTrain + "fold" +i + "_test.txt", pathToData,
					pathToOutput + "combinedFold" + i + "Test.arff", selectRandom);
		}
		executor.shutdown();
        while (!executor.isTerminated()) {
        	//Waiting to finish
        }
	}
	
	private static void classifyFold(Classifiers e, String pathToOutput, int[] param) throws InterruptedException, ExecutionException {
		ExecutorService executor = Executors.newFixedThreadPool(4);
		ArrayList<Future<Evaluation>> taskList = new ArrayList<>();
		for(int i=1; i <= numberOfFolds; i++) {
			AbsClassifier classifier = factory(e, pathToOutput, i);
			classifier.setParams(param);
			MyCallable callable = new MyCallable(classifier, false);
			taskList.add(executor.submit(callable));
		}
		double crossValiAcc = 0;
		for(int i=0; i < taskList.size(); i++) {
			Future<Evaluation> task = taskList.get(i);
			Evaluation result = task.get();
			System.out.println("Fold:" + (i+1) + " Correct:" + result.pctCorrect());
			crossValiAcc += result.pctCorrect();
		}
		System.out.println("CrossValidation Correct:" + (crossValiAcc/taskList.size()));
		executor.shutdown();
		while (!executor.isTerminated()) {
        	//Waiting to finish
        }
	}

	private static Evaluation classify(Classifiers e, boolean debugOutput, String pathToOutput, int number, int[] param) throws Exception {
        AbsClassifier classifier = factory(e, pathToOutput, number);
        classifier.setParams(param);
        return classifier.executeClassifier(debugOutput);
	}
	
	
	private static AbsClassifier factory(Classifiers c, String pathToOutput, int number) {
		File arffTrain = new File(pathToOutput + "combinedFold" + number + ".arff");
		File arffTest = new File(pathToOutput + "combinedFold" + number + "Test.arff");
		switch(c) {
		case IBk:
			return new IBkImpl(arffTrain, arffTest);
		case NaiveBayes:
			return new NaiveBayesImpl(arffTrain, arffTest);
		default:
			return null; //Should not happen!
		}
	}
	
	

}
