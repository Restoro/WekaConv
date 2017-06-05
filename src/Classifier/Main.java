package Classifier;
import java.io.File;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import Classifier.Impls.IBkImpl;
import Classifier.Impls.NaiveBayesImpl;
import Classifier.Impls.RandomForestImpl;
import Enums.Classifiers;
import weka.classifiers.Evaluation;

public class Main {

	static final int dataLineCount = 1500;
	static final int randomNumberCount = 250;
	static final int numberOfFolds = 4;
	static final boolean convertFiles = false;
	static final boolean useRandomData = true;

	public static void main(String[] args) {
		try {
			if(args.length == 2) { 
				String pathToTrain = System.getProperty("user.dir") + "/resources/";
				String pathToData = args[0];
				String pathToOutput = args[1];
				if(convertFiles) convertFiles(useRandomData, pathToTrain, pathToData, pathToOutput);
				//classifySegmentFold(Classifiers.NaiveBayes, pathToData, pathToOutput, null, false, true);
				classifySegmentFoldNoThread(Classifiers.RandomForest, pathToData, pathToOutput, new int[] {120, 8}, false, true);
			} else {
				System.out.println("Not every parameter set (Length must be 2)");
			}
		} catch (Exception e) {
			System.out.println(e.toString());
		}
	}
	
	private static void convertFullFile(boolean selectRandom, String pathToTrain, String pathToData, String pathToOutput) {
		Converter conv = new Converter(pathToTrain, pathToData, pathToOutput, randomNumberCount, dataLineCount);
		ExecutorService executor = Executors.newFixedThreadPool(1);
		conv.executeInThreadFull(executor, pathToTrain, pathToData, pathToOutput + "complete.arff", selectRandom);
		executor.shutdown();
        while (!executor.isTerminated()) {
        	//Waiting to finish
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
	
	private static void classifyFold(Classifiers e, String pathToData, String pathToOutput, int[] param) throws InterruptedException, ExecutionException {
		ExecutorService executor = Executors.newFixedThreadPool(4);
		ArrayList<Future<Evaluation>> taskList = new ArrayList<>();
		for(int i=1; i <= numberOfFolds; i++) {
			AbsClassifier classifier = factory(e, pathToData, pathToOutput, i, false);
			classifier.setParams(param);
			MyCallable callable = new MyCallable(classifier, false);
			taskList.add(executor.submit(callable));
		}
		double crossValiAcc = 0;
		for(int i=0; i < taskList.size(); i++) {
			Future<Evaluation> task = taskList.get(i);
			Evaluation result = task.get();
			System.out.println("Fold:" + (i+1) + " Correct:" + result.pctCorrect() + " Error Rate:" + result.errorRate());
			
			crossValiAcc += result.pctCorrect();
		}
		System.out.println("CrossValidation Correct:" + (crossValiAcc/taskList.size()));
		executor.shutdown();
		while (!executor.isTerminated()) {
        	//Waiting to finish
        }
	}

	private static Evaluation classify(Classifiers e, boolean debugOutput, String pathToData, String pathToOutput, int number, int[] param) throws Exception {
        AbsClassifier classifier = factory(e, pathToData, pathToOutput, number, false);
        classifier.setParams(param);
        return classifier.executeClassifier(debugOutput, false);
	}
	
	private static void classifySegmentFold(Classifiers e, String pathToData, String pathToOutput, int[] param, boolean debugOutput, boolean useSave) throws Exception {
		ExecutorService executor = Executors.newFixedThreadPool(4);
		
		ArrayList<Future<Float>> taskList = new ArrayList<>();
		for(int i=1; i <= numberOfFolds; i++) {
			AbsClassifier classifier = factory(e, pathToData, pathToOutput, i, true);
			classifier.setParams(param);
			MyCallableSegment callable = new MyCallableSegment(classifier, debugOutput, useSave);
			taskList.add(executor.submit(callable));
		}
		float crossValiAcc = 0;
		for(int i=0; i < taskList.size(); i++) {
			Future<Float> task = taskList.get(i);
			Float result = task.get();
			System.out.println("Fold:" + (i+1) + " Correct:" + result);
			crossValiAcc += result;
		}
		System.out.println("CrossValidation Correct:" + (crossValiAcc/taskList.size()));
		executor.shutdown();
		while (!executor.isTerminated()) {
        	//Waiting to finish
        }
	}
	
	private static void classifySegmentFoldNoThread(Classifiers e, String pathToData, String pathToOutput, int[] param, boolean debugOutput, boolean useSave) throws Exception {
		float crossValiAcc = 0;
		int folds = numberOfFolds;
		for(int i=1; i <= folds; i++) {
			AbsClassifier classifier = factory(e, pathToData, pathToOutput, i, true);
			classifier.setParams(param);
			float result = classifier.executeSegmentClassifier(debugOutput, useSave);
			System.out.println("Fold:" + (i) + " Correct:" + result);
			crossValiAcc += result;
		}
		System.out.println("CrossValidation Correct:" + (crossValiAcc/folds));
	}
	
	
	private static AbsClassifier factory(Classifiers c, String pathToData, String pathToOutput, int number, boolean segment) {
		File arffTrain = new File(pathToOutput + "combinedFold" + number + ".arff");
		File testFile;
		if(!segment) {
			testFile = new File(pathToOutput + "combinedFold" + number + "Test.arff");
		} else {
			String pathToTrain = System.getProperty("user.dir") + "/resources/";
			testFile = new File(pathToTrain + "fold" + number + "_evaluate.txt");
		}
		switch(c) {
		case IBk:
			return new IBkImpl(arffTrain, testFile, pathToData);
		case NaiveBayes:
			return new NaiveBayesImpl(arffTrain, testFile, pathToData);
		case RandomForest:
			return new RandomForestImpl(arffTrain, testFile, pathToData);
		default:
			return null; //Should not happen!
		}
	}
	
	

}
