package Classifier.Impls;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;

import Classifier.AbsClassifier;
import Enums.Classes;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;

public class StackClassifierImpl extends AbsClassifier{

	File trainFile;
	public StackClassifierImpl(File arffTrain, File trainFile, File testFile, String pathToData) {
		super(arffTrain, testFile, pathToData);
		this.trainFile = trainFile;
		// TODO Auto-generated constructor stub
	}

	@Override
	public String getClassifierName() {
		return "StackClassifier";
	}

	@Override
	public Classifier generateClassifier(boolean saveModel) throws Exception {
		return null;
	}

	@Override
	public Evaluation executeClassifier(boolean output, boolean useSave) throws Exception {
		
		
		// TODO Auto-generated method stub
		return null;
	}
	
	private Instances generateInstances(weka.classifiers.Classifier classifier, File toTestFile) throws Exception {
		
		ArrayList<Attribute> atts = new ArrayList<Attribute>(1501);
		for(int i=0; i < 1500; i++) {
			atts.add(new Attribute("frameLabel"+i));
		}
        ArrayList<String> classVal = new ArrayList<String>();
        for(Classes clazz : Classes.values()) {
        	classVal.add(clazz.name());
        }
		atts.add(new Attribute("class",classVal));
		Instances predictedLabels = new Instances("PredictedLabels",atts,0);

		System.out.println("Start creating Instances for File " + toTestFile.getName());

		String line;
		BufferedReader reader = new BufferedReader(new FileReader(toTestFile));
		while ((line = reader.readLine()) != null) {
			String fileName = line.substring(6, line.indexOf('.'));
			String[] split = line.split("	");
			String correctClass = split[1].replaceAll("/", "_");
			File selectedArff = new File(pathToData + fileName + ".arff");
			Instances selectedTestFile = getInstance(selectedArff);
			
			double[] instanceValue = new double[predictedLabels.numAttributes()];
			for(int i=0; i < selectedTestFile.numInstances(); i++) {
				double val = classifier.classifyInstance(selectedTestFile.instance(i));
				instanceValue[i] = val;
			}
			instanceValue[1500] = (double) Classes.valueOf(correctClass).ordinal();
			predictedLabels.add(new DenseInstance(1.0, instanceValue));
			
			//System.out.println("Predicted for Segment:" + className + " Correct:" + correctClass);
		}
		reader.close();
		return predictedLabels;
	}

	@Override
	public Float executeSegmentClassifier(boolean output, boolean useSave) throws Exception {
		RandomForestImpl forestImpl = new RandomForestImpl(arffTrain, trainFile, pathToData);
		weka.classifiers.Classifier forestClassifier = forestImpl.loadOrGenerateClassifier(useSave);
		
		System.out.println("Start generating predicted labels for classifier " + this.getClassifierName());
		
		
		Instances trainingLabels = generateInstances(forestClassifier, trainFile);
		Instances predictedLabels = generateInstances(forestClassifier, testFile);
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(trainingLabels);
		saver.setFile(new File("trainingLabels.arff"));
		saver.writeBatch();
		
		
		saver.setInstances(predictedLabels);
		saver.setFile(new File("predictedLabels.arff"));
		saver.writeBatch();
		

		RandomForestImpl forestStackImpl = new RandomForestImpl(new File("trainingLabels.arff"), new File("predictedLabels.arff"), pathToData);
		Evaluation eval = forestStackImpl.executeClassifier(output, false);
		return (float) eval.pctCorrect();
	}

	@Override
	public boolean setParams(int[] param) {
		// TODO Auto-generated method stub
		return false;
	}

}
