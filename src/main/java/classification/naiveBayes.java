package classification;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
//import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.ComplementNaiveBayes;
import weka.classifiers.misc.SerializedClassifier;
//import weka.classifiers.pmml.consumer.Regression;

import weka.classifiers.evaluation.Prediction;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.FastVector;

public class naiveBayes {
    public static Instances isTrainingSet  ;
    public static Instances isTestSet  ;

	public static void main(String[] args) {
		 try {
			 // define output file
			 // can be replaced by database as well
			 
			 System.setOut(new PrintStream(new BufferedOutputStream(new FileOutputStream("output.txt")), true) );
			 	
			 
			 // parse arguments (if any) to be passed to Create Data
			 
			 int firstArg;
			 int secondArg;
			 if (args.length > 0) {
			     try {
			         firstArg = Integer.parseInt(args[0]);
			         secondArg = Integer.parseInt(args[1]);
			         if (firstArg >= secondArg){
						 System.out.println("Running with sizes: Training=" + firstArg + " Test = " + secondArg);			        	 
			        	 createData(firstArg, secondArg);
			         } else { 
			        	 Exception dataError = new Exception("!!!!! Training Data size should be more or equal to test data size");
			        	 throw dataError;
			         }
			     }  catch (NumberFormatException e) {
			         System.err.println("Arguments" + args[0] + "::" + args[1] + " must be an integers.");
			         System.exit(1);
			     }  catch (Exception e){
			    	 System.err.println(e.getMessage());
			    	 System.exit(1);
			     }
			 } else {
				 System.out.println("Running with default 1000, 100 sizes");
				 createData(1000, 100); // default createData values
			 }
			 /* possible values could be
			  * AdaBoostM1, AdditiveRegression, AttributeSelectedClassifier, 
			  * Bagging, BayesNet, BayesNetGenerator, BIFReader, ClassificationViaRegression, CostSensitiveClassifier, 
			  * CVParameterSelection, DecisionStump, DecisionTable, EditableBayesNet, FilteredClassifier, 
			  * GaussianProcesses, GeneralRegression, HoeffdingTree, IBk, InputMappedClassifier, 
			  * IteratedSingleClassifierEnhancer, IterativeClassifierOptimizer, J48, JRip, KStar, 
			  * LinearRegression, LMT, LMTNode, Logistic, LogisticBase, LogitBoost, LWL, M5Base, M5P, M5Rules, 
			  * MultiClassClassifier, MultiClassClassifierUpdateable, MultilayerPerceptron, 
			  * MultipleClassifiersCombiner, MultiScheme, NaiveBayes, NaiveBayesMultinomial, 
			  * NaiveBayesMultinomialText, NaiveBayesMultinomialUpdateable, NaiveBayesUpdateable, NeuralNetwork, 
			  * OneR, ParallelIteratedSingleClassifierEnhancer, ParallelMultipleClassifiersCombiner, PART, 
			  * PMMLClassifier, PreConstructedLinearModel, RandomCommittee, RandomForest, RandomizableClassifier, 
			  * RandomizableFilteredClassifier, RandomizableIteratedSingleClassifierEnhancer, 
			  * RandomizableMultipleClassifiersCombiner, RandomizableParallelIteratedSingleClassifierEnhancer, 
			  * RandomizableParallelMultipleClassifiersCombiner, RandomizableSingleClassifierEnhancer, 
			  * RandomSubSpace, RandomTree, Regression, RegressionByDiscretization, REPTree, RuleNode, RuleSetModel, 
			  * SerializedClassifier, SGD, SGDText, SimpleLinearRegression, SimpleLogistic, SingleClassifierEnhancer, 
			  * SMO, SMOreg, Stacking, SupportVectorMachineModel, TreeModel, Vote, VotedPerceptron, ZeroR

			  */
	         Classifier cModel = (Classifier)new ComplementNaiveBayes() ; //NaiveBayes();   
	         cModel.buildClassifier(isTrainingSet);
	 
	         // Test the model
	         Evaluation eTest = new Evaluation(isTestSet);
	         eTest.evaluateModel(cModel, isTestSet);
	          
	         // Print the result à la Weka explorer:
	         String strSummary = eTest.toSummaryString();
	         System.out.println(strSummary);
	
	         // atul: specific stats
	         /*
	          * Weka results output:
	          * http://www.cs.usfca.edu/~pfrancislyon/courses/640fall2015/WekaDataAnalysis.pdf
				TP = true positives: number of examples predicted positive that are actually positive
				FP = false positives: number of examples predicted positive that are actually negative
				TN = true negatives: number of examples predicted negative that are actually negative
				FN = false negatives: number of examples predicted negative that are actually positive

	          */
	         System.out.println("http://www.cs.usfca.edu/~pfrancislyon/courses/640fall2015/WekaDataAnalysis.pdf");
	         System.out.println("TP = true positives: number of examples predicted positive that are actually positive");
	         System.out.println("FP = false positives: number of examples predicted positive that are actually negative");
	         System.out.println("TN = true negatives: number of examples predicted negative that are actually negative");
	         System.out.println("FN = false negatives: number of examples predicted negative that are actually positive");

	         System.out.println("False negatives 0 :" + eTest.numFalseNegatives(0));//0  **1
	         System.out.println("False negatives 1 :" + eTest.numFalseNegatives(1));//25
	         //System.out.println("False negatives 2 :" + eTest.numFalseNegatives(2));

	         System.out.println("True negatives 0 :" + eTest.numTrueNegatives(0));//0  **2
	         System.out.println("True negatives 1 :" + eTest.numTrueNegatives(1));//25
	         //System.out.println("True negatives 2 :" + eTest.numTrueNegatives(2));

	         System.out.println("False Positives 0 :" + eTest.numFalsePositives(0));//25 *2
	         System.out.println("False Positives 1 :" + eTest.numFalsePositives(1));//0
	         //System.out.println("False Positives 2 :" + eTest.numFalsePositives(2));

	         System.out.println("True Positives 0 :" + eTest.numTruePositives(0));//25 *1
	         System.out.println("True Positives 1 :" + eTest.numTruePositives(1));//0
	         //System.out.println("True Positives 2 :" + eTest.numTruePositives(2));
	         
	         // atul:get the specific details of result
	         
	         System.out.println("########## here are the predictions: ########");
	         FastVector bayPredictions =  eTest.predictions();
	         Object [] arrPredictions = bayPredictions.toArray();
	         for (Object bPrediction : arrPredictions){
	        	 Prediction p = (Prediction) bPrediction ;
	        	 System.out.println("predicted::actual::weight -->> " + p.actual() + " :: " + p.predicted() + " :: " + p.weight());
	         }
	         // Get the confusion matrix
	         double[][] cmMatrix = eTest.confusionMatrix();
	         for(int row_i=0; row_i<cmMatrix.length; row_i++){
	             for(int col_i=0; col_i<cmMatrix.length; col_i++){
	                 System.out.print(cmMatrix[row_i][col_i]);
	                 System.out.print("|");
	             }
	             System.out.println();
	 			 
	         }
				 System.out.print("!!!!!!!!!!!!!!!!!!!!! RUN OVER !!!!!!!!!!!!!!!!!!!!!");
	             System.out.println();
				 System.out.print("!!!!!!!!!!!!!!!!!!!!! RUN OVER !!!!!!!!!!!!!!!!!!!!!");
		 } catch (Exception e) {
			 System.out.print("Something went wrong .... " + e.getMessage());
		 }
	} // End of Main()

	/*
	 * Create data on the run
	 * TrainingSet to build the model
	 * TestSet to test the model
	 */
	public static void createData(int trnSize, int tstSize){
        // Declare two numeric attributes
        Attribute Attribute1 = new Attribute("firstNumeric");
        Attribute Attribute2 = new Attribute("secondNumeric");
         
        // Declare a nominal attribute along with its values
        FastVector fvNominalVal = new FastVector(3);
        fvNominalVal.addElement("blue");
        fvNominalVal.addElement("gray");
        fvNominalVal.addElement("black");
        Attribute Attribute3 = new Attribute("aNominal", fvNominalVal);
         
        // Declare the class attribute along with its values
        FastVector fvClassVal = new FastVector(2);
        fvClassVal.addElement("positive");
        fvClassVal.addElement("negative");
        //fvClassVal.addElement("neutral");		// valid only for three possible output case 
        Attribute ClassAttribute = new Attribute("theClass", fvClassVal);
         
        // Declare the feature vector
        FastVector fvWekaAttributes = new FastVector(4);
        fvWekaAttributes.addElement(Attribute1);    
        fvWekaAttributes.addElement(Attribute2);    
        fvWekaAttributes.addElement(Attribute3);    
        fvWekaAttributes.addElement(ClassAttribute);
         
        //////// Create training set
        isTrainingSet = new Instances("Rel", fvWekaAttributes, trnSize);  //PARAMETER training size 
         
        // Set class index
        isTrainingSet.setClassIndex(3);
         
        // Create the instances to be added.
		 for (int tSize = 0; tSize < trnSize; tSize++){ // PARAMETER training size
			 Instance iExample = new Instance(4);
			 iExample.setValue((Attribute)fvWekaAttributes.elementAt(0), tSize + 1.0);      
			 iExample.setValue((Attribute)fvWekaAttributes.elementAt(1), tSize + 0.5);  
			 String attVal3 = ((tSize % 3) == 0)? "blue" : (((tSize % 3) == 1)? "gray" : "black"   ) ;
			 System.out.print(">>>> TRAINING ****** third parameter = " + attVal3 + "  FOR iteration = " + tSize);
			 System.out.println();
			 iExample.setValue((Attribute)fvWekaAttributes.elementAt(2), "gray");

			 //String conclusion = ((tSize % 3) == 0)? "positive" : (((tSize % 3) == 1)? "negative" : "neutral"   ) ;
			 String conclusion = (((tSize % 2) == 0)? "positive" :  "negative" ) ;

			 iExample.setValue((Attribute)fvWekaAttributes.elementAt(3), conclusion);
			 System.out.print(">>>> TRAINING ****** INDEPENDENT PARAM = " + conclusion + "  FOR iteration = " + tSize);
			 System.out.println();
			  
			 // add the instance
			 isTrainingSet.add(iExample);
		 } 
		 //////// create Test set
        isTestSet = new Instances("Rel", fvWekaAttributes, tstSize);   // PARAMETER testing size    
         
        // Set class index
        isTestSet.setClassIndex(3);
         
        // Create the instances to be added.
		 for (int tSize = 0; tSize < tstSize; tSize++){    // PARAMETER testing size
			 Instance iExampleTest = new Instance(4);
			 iExampleTest.setValue((Attribute)fvWekaAttributes.elementAt(0), tSize + 1.0);      
			 iExampleTest.setValue((Attribute)fvWekaAttributes.elementAt(1), tSize + 0.5);  
			 String attVal33 = ((tSize % 3) == 0)? "blue" : (((tSize % 3) == 1)? "gray" : "black"   ) ;
			 System.out.print(">>>> TEST ###### third parameter = " + attVal33 + "  FOR iteration = " + tSize);
			 System.out.println();
			 iExampleTest.setValue((Attribute)fvWekaAttributes.elementAt(2), attVal33);
			 
			 //String conclusion2 = ((tSize % 3) == 0)? "positive" : (((tSize % 3) == 1)? "negative" : "neutral"   ) ;
			 String conclusion2 =  ( ((tSize % 2) == 0)? "positive" : "negative" ) ;
			 iExampleTest.setValue((Attribute)fvWekaAttributes.elementAt(3), conclusion2);
			 System.out.print(">>>> TEST ###### INDEPENDENT PARAM = " + conclusion2 + "  FOR iteration = " + tSize);
			 System.out.println();

			  
			 // add the instance
			 isTestSet.add(iExampleTest);			
		 }	
	} // End of CreateData()
	

}
