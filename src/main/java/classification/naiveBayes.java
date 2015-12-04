package classification;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.PrintStream;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class naiveBayes {
    public static Instances isTrainingSet  ;
    public static Instances isTestSet  ;

	public static void main(String[] args) {
		 try {
			 System.setOut(new PrintStream(new BufferedOutputStream(new FileOutputStream("output.txt")), true) );
			 createData();
	         Classifier cModel = (Classifier)new NaiveBayes();   
	         cModel.buildClassifier(isTrainingSet);
	 
	         // Test the model
	         Evaluation eTest = new Evaluation(isTestSet);
	         eTest.evaluateModel(cModel, isTestSet);
	          
	         // Print the result à la Weka explorer:
	         String strSummary = eTest.toSummaryString();
	         System.out.println(strSummary);
	          
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
	public static void createData(){
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
        FastVector fvClassVal = new FastVector(3);
        fvClassVal.addElement("positive");
        fvClassVal.addElement("negative");
        fvClassVal.addElement("neutral");		 
        Attribute ClassAttribute = new Attribute("theClass", fvClassVal);
         
        // Declare the feature vector
        FastVector fvWekaAttributes = new FastVector(4);
        fvWekaAttributes.addElement(Attribute1);    
        fvWekaAttributes.addElement(Attribute2);    
        fvWekaAttributes.addElement(Attribute3);    
        fvWekaAttributes.addElement(ClassAttribute);
         
        //////// Create training set
        isTrainingSet = new Instances("Rel", fvWekaAttributes, 10);       
         
        // Set class index
        isTrainingSet.setClassIndex(3);
         
        // Create the instances to be added.
		 for (int tSize = 0; tSize <10; tSize++){
			 Instance iExample = new Instance(4);
			 iExample.setValue((Attribute)fvWekaAttributes.elementAt(0), tSize + 1.0);      
			 iExample.setValue((Attribute)fvWekaAttributes.elementAt(1), tSize + 0.5);  
			 String attVal3 = ((tSize % 3) == 0)? "blue" : (((tSize % 3) == 1)? "gray" : "black"   ) ;
			 System.out.print(">>>> TRAINING ****** third parameter = " + attVal3 + "  FOR iteration = " + tSize);
			 System.out.println();
			 iExample.setValue((Attribute)fvWekaAttributes.elementAt(2), "gray");

			 String conclusion = ((tSize % 3) == 0)? "positive" : (((tSize % 3) == 1)? "negative" : "neutral"   ) ;
			 iExample.setValue((Attribute)fvWekaAttributes.elementAt(3), conclusion);
			 System.out.print(">>>> TRAINING ****** INDEPENDENT PARAM = " + conclusion + "  FOR iteration = " + tSize);
			 System.out.println();
			  
			 // add the instance
			 isTrainingSet.add(iExample);
		 } 
		 //////// create Test set
        isTestSet = new Instances("Rel", fvWekaAttributes, 3);       
         
        // Set class index
        isTestSet.setClassIndex(3);
         
        // Create the instances to be added.
		 for (int tSize = 0; tSize <3; tSize++){
			 Instance iExampleTest = new Instance(4);
			 iExampleTest.setValue((Attribute)fvWekaAttributes.elementAt(0), tSize + 1.0);      
			 iExampleTest.setValue((Attribute)fvWekaAttributes.elementAt(1), tSize + 0.5);  
			 String attVal33 = ((tSize % 3) == 0)? "blue" : (((tSize % 3) == 1)? "gray" : "black"   ) ;
			 System.out.print(">>>> TEST ###### third parameter = " + attVal33 + "  FOR iteration = " + tSize);
			 System.out.println();
			 iExampleTest.setValue((Attribute)fvWekaAttributes.elementAt(2), attVal33);
			 
			 String conclusion2 = ((tSize % 3) == 0)? "positive" : (((tSize % 3) == 1)? "negative" : "neutral"   ) ;
			 iExampleTest.setValue((Attribute)fvWekaAttributes.elementAt(3), conclusion2);
			 System.out.print(">>>> TEST ###### INDEPENDENT PARAM = " + conclusion2 + "  FOR iteration = " + tSize);
			 System.out.println();

			  
			 // add the instance
			 isTestSet.add(iExampleTest);			
		 }	
	} // End of CreateData()
	

}
