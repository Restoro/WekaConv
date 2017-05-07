package Classifier;

import weka.attributeSelection.PrincipalComponents;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;


import java.io.File;
import java.io.IOException;

/**
 * Created by Stefan on 05.05.2017.
 */
public abstract class AbsPCAClassifier extends AbsClassifier {

    String[] options = null;
    PrincipalComponents pca = null;

    public AbsPCAClassifier(File arffTrain, File arffTest) {
        this(arffTrain, arffTest,"-R","1.0","-C");
    }

    public AbsPCAClassifier(File arffTrain, File arffTest, String... options) {
        super(arffTrain, arffTest);
        this.options = options;
    }


    @Override
    protected Instances getInstance(File arffFile) throws IOException {
        Instances instances = super.getInstance(arffFile);
        //return instances;
        return getTransformedData(instances);
    }

    protected Instances getTransformedData(Instances instances){
        try {
            if(pca == null) initPCA();
            return pca.transformedData(instances);
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
        return null;
    }

    protected Instance convertInstance(Instance instance) throws Exception {
        try {
            if(pca == null) initPCA();
            return pca.convertInstance(instance);
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
        return null;//pca.convertInstance(instance);
    }

    private void initPCA() throws Exception{
        pca = new PrincipalComponents();
        pca.setOptions(options);
        pca.buildEvaluator(super.getInstance(arffTrain));
    }


}
