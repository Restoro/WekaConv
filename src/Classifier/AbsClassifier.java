package Classifier;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map.Entry;

import Enums.Classes;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

abstract public class AbsClassifier {
	protected File arffTrain;
	protected File testFile;
	protected String pathToData;
	
	public AbsClassifier(File arffTrain, File testFile, String pathToData) {
		this.arffTrain = arffTrain;
		this.testFile = testFile;
		this.pathToData = pathToData;
	}
	
	abstract public weka.classifiers.Classifier generateClassifier(boolean saveModel) throws Exception;
	
	abstract public Evaluation executeClassifier(boolean output, boolean useSave) throws Exception;
	
	abstract public Float executeSegmentClassifier(boolean output, boolean useSave) throws Exception;
	
	abstract public boolean setParams(int[] param);
	
	protected Float loadModelExecuteSegment(String className, boolean useSave) throws Exception {
		if (useSave) {
			weka.classifiers.Classifier classi = (weka.classifiers.Classifier) loadClassifier(className);
			if (classi == null) {
				System.out.println("Found no Model for " + className);
				classi = (weka.classifiers.Classifier) this.generateClassifier(true);
			}
			return evaluateClassifierSegment(classi);
		} else {
			return evaluateClassifierSegment(this.generateClassifier(false));
		}
	}
	
	protected Evaluation evaluateClassifier(weka.classifiers.Classifier classi, boolean output) throws Exception {
		System.out.println("Start Evaluation");
		Instances test = getInstance(testFile);
		Evaluation eval = new Evaluation(test);
		eval.evaluateModel(classi, test);
		
		if(output)System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		return eval;
	}
	
	protected String voteOnEvaluation(weka.classifiers.Classifier classi, File selectedArff) throws Exception {
		Instances testFile = getInstance(selectedArff);
		int maximumIndex = 0;
		int[] classes = new int[15];
		for(int i=0; i < testFile.numInstances(); i++) {
			int val = (int) classi.classifyInstance(testFile.instance(i));
			classes[val] += 1;
			if(classes[val] > classes[maximumIndex]) {
				maximumIndex = val;
			}
		}
		String predictedClass = Classes.fromOrdinal(maximumIndex).name();
		//System.out.println(generateVoteTable(classes, predictedClass));
		return predictedClass;
	}
	
	private String generateVoteTable(int[] classes, String predicted) {
		StringBuilder sb = new StringBuilder();
		sb.append("Classes --- Predicted: ");
		sb.append(predicted);
		sb.append("\n");
		for(Classes clazz : Classes.values()) {
			int count = classes[clazz.ordinal()];
			sb.append(clazz + " - " + count + "\n");
		}
		sb.append("END -------------\n");
		return sb.toString();
	}
	
	//Testfile = textDatei
	protected float evaluateClassifierSegment(weka.classifiers.Classifier classi) throws Exception {
		System.out.println("Start Voting for classifier " + classi.getClass().getName());
		BufferedReader reader = new BufferedReader(new FileReader(testFile));
		String line;
		int count = 0;
		int correct = 0;
		while ((line = reader.readLine()) != null) {
			String fileName = line.substring(6, line.indexOf('.'));
			String[] split = line.split("	");
			String correctClass = split[1].replaceAll("/", "_");
			File selectedArff = new File(pathToData + fileName + ".arff");
			
			String className = voteOnEvaluation(classi, selectedArff);
			//System.out.println("Predicted for Segment:" + className + " Correct:" + correctClass);
			if(className.equals(correctClass)) {
				correct++;
			}
			count++;
		}
		float percentage = ((float) correct) / count;
		reader.close();
		return percentage;
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
	
	protected void saveClassifier(weka.classifiers.Classifier classi) throws Exception {
		System.out.println("Save model for classifier " + classi.getClass().getSimpleName() + " with train file " + this.arffTrain.getName());
		String path = System.getProperty("user.dir") + "/models/" + classi.getClass().getSimpleName() + "_" + this.arffTrain.getName() + ".model";
		weka.core.SerializationHelper.write(path, classi);
	}
	
	protected weka.classifiers.Classifier loadClassifier(String classiName) throws Exception
	{
		String fileName = System.getProperty("user.dir") + "/models/" + classiName + "_" + this.arffTrain.getName() + ".model";
		if(new File(fileName).isFile()) {
			System.out.println("Load model for classifier " + classiName + " with train file " + this.arffTrain.getName());
			return (weka.classifiers.Classifier) weka.core.SerializationHelper.read(fileName);
		} else {
			return null;
		}
	}
}
