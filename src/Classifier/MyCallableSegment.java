package Classifier;

import java.util.concurrent.Callable;

public class MyCallableSegment implements Callable<Float> {

	private AbsClassifier classi;
	private boolean output;
	private boolean useSave;
	
	public MyCallableSegment(AbsClassifier classi, boolean output, boolean useSave) {
		this.classi = classi;
		this.output = output;
		this.useSave = useSave;
	}
	
	
	@Override
	public Float call() throws Exception {
		return classi.executeSegmentClassifier(output,useSave);
	}

}
