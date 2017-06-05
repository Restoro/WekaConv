package Classifier;
import java.util.concurrent.Callable;

import weka.classifiers.Evaluation;

public class MyCallable implements Callable<Evaluation> {

	private AbsClassifier classi;
	private boolean fullOutput;
	
	public MyCallable(AbsClassifier classi, boolean fullOutput){
		this.classi=classi;
		this.fullOutput = fullOutput;
	}

	@Override
	public Evaluation call() throws Exception {
		return classi.executeClassifier(fullOutput, false);
	}

}