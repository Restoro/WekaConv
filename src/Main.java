import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Main {

	static final int dataLineCount = 1500;
	static final int randomNumberCount = 20;
	static final int numberOfFolds = 4;
	static final boolean convertFiles = true;

	public static void main(String[] args) {
		try {
			if(args.length == 2) { 
				String pathToTrain = System.getProperty("user.dir") + "/resources/";
				String pathToData = args[0];
				String pathToOutput = args[1];
				if(convertFiles) convertFiles(true, pathToTrain, pathToData, pathToOutput);
				classifyBayesFold(pathToOutput);
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
	
	private static void classifyIBkFold(String pathToOutput) {
		ExecutorService executor = Executors.newFixedThreadPool(4);
		for(int i=1; i <= numberOfFolds; i++) {
			final int zw = i;
			executor.execute(new Thread(() -> System.out.println("Fold: " + zw + " Correct:" + classifyIBk(zw, false, pathToOutput))));
		}
		executor.shutdown();
        while (!executor.isTerminated()) {
        	//Waiting to finish
        }
	}
	
	private static void classifyBayesFold(String pathToOutput) {
		ExecutorService executor = Executors.newFixedThreadPool(4);
		for(int i=1; i <= numberOfFolds; i++) {
			final int zw = i;
			executor.execute(new Thread(() -> System.out.println("Fold: " + zw + " Correct:" + classifyBayes(zw, false, pathToOutput))));
		}
		executor.shutdown();
        while (!executor.isTerminated()) {
        	//Waiting to finish
        }
	}
	
	private static double classifyBayes(int number, boolean debugOutput, String pathToOutput) {
        Classifier classifier = new Classifier(new File(pathToOutput + "combinedFold" + number + ".arff"), new File(pathToOutput + "combinedFold" + number + "Test.arff"));
        return classifier.naiveBayes(debugOutput);
	}
	
	private static double classifyIBk(int number, boolean debugOutput, String pathToOutput) {
        Classifier classifier = new Classifier(new File(pathToOutput + "combinedFold" + number + ".arff"), new File(pathToOutput + "combinedFold" + number + "Test.arff"));
        return classifier.ibk(debugOutput);
	}

}
