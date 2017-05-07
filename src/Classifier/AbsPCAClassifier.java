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
    PrincipalComponents pca;

    public AbsPCAClassifier(File arffTrain, File arffTest) {
        this(arffTrain, arffTest,"-C");
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
            if(pca == null){
                pca = new PrincipalComponents();
                //pca.setInputFormat(super.getInstance(arffTrain));

                pca.buildEvaluator(super.getInstance(arffTrain));
            } return /*Filter.useFilter(instances,pca);*/pca.transformedData(instances);
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
        return null;
    }

    protected Instance convertInstance(Instance instance) throws Exception {
        try {
            if(pca == null){
                pca = new PrincipalComponents();
                //pca.setInputFormat(super.getInstance(arffTrain));
                //pca.buildEvaluator(super.getInstance(arffTrain));
            } return null;//pca.convertInstance(instance);
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
        return null;//pca.convertInstance(instance);
    }


}
