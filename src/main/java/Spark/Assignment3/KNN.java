package Spark.Assignment3;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.stream.IntStream;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.LineNumberReader;
import java.io.IOException;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.DoubleFlatMapFunction;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.api.java.function.VoidFunction2;

import scala.Tuple2;

public class KNN {
	
	public static int GetLineCount(String dir) throws Exception{
		FileReader fr = new FileReader(dir);
		BufferedReader br = new BufferedReader(fr);
		int row_cnt=0;
		String CurLine;
		while ((CurLine = br.readLine()) != null) {
			row_cnt++;
		}
		br.close();
		return row_cnt;
	}
	
	// read the value of each element from testing_set
	public static double[][] ReadDataSet(String dir, int linecnt) throws Exception{	
		double[][] vals = new double[linecnt][];
		
		// store the lines read from file
		String sCurrentLine;
		FileReader fr = new FileReader(dir);
		BufferedReader br = new BufferedReader(fr);
		
		// row index
		int row = 0;
		while ((sCurrentLine = br.readLine()) != null) {	
			// split current strings by ","
			String[] splitVals = sCurrentLine.split(",");
			// assign memory for current string line
			vals[row] = new double[splitVals.length];
			// read the value and assign to vals[][]
	        for(int i=0;  i < splitVals.length; i++) {
	        	vals[row][i] = Double.parseDouble(splitVals[i]);
	        }
	        row++;
		}
		br.close();
		return vals;
	}
	
	// display
	public static void Display(double[][] vals) throws Exception {
		for(int i=0; i<vals.length; i++){
			for( int j =0; j<vals[0].length; j++){
				System.out.format("%10.6f ", vals[i][j]);
			}
			System.out.format("\n");
		}
	}
	
	// get training set RDD data
	// maybe we will not call this function
	public static JavaRDD<double[]> Get_RDD_Double_Array(JavaRDD<String> rdd_str){
		 return rdd_str.map(new Function<String, double[]>(){
		    public double[] call(String s){
		    	String[] splitVals = s.split(",");
		        double[] vals = new double[splitVals.length];
		        for(int i=0; i < splitVals.length; i++) {
		            vals[i] = Double.parseDouble(splitVals[i]);
		        }		       
		        return vals;
		    }
		});
	}
	
	// get testing set RDD data
	// im my implementation, I call this function
	public static JavaRDD<double[]> Get_Testing_Set_RDD_Double_Array(JavaRDD<String> rdd_str){
		 return rdd_str.map(new Function<String, double[]>(){
			 public double[] call(String s){
			    String[] splitVals = s.split(",");
			    //System.out.println("The element is: " + splitVals.length);
			    // one extra element to indentify which objects it is
			    double[] vals = new double[splitVals.length];
			    for(int i=0; i < splitVals.length-1; i++) {
			    	vals[i] = Double.parseDouble(splitVals[i]);
			    }
			    // get the last element of testing set data 
			    // the last element means its object index
			    vals[splitVals.length-1] = Double.parseDouble(splitVals[splitVals.length-1]);
			    return vals;
			 }
		});
	}
	
	// add object index to testing set RDD data
	public static void AddObjIdx(double[][] rdd){
		int row = rdd.length;
		int col = rdd[0].length;
		for( int i=0; i<row; i++){
			rdd[i][col-1] = i;
		}
	}
	
	// display java resilient distributed dataset
	public static void RDD_Display_1D(JavaRDD<int[]> rdd) throws Exception {
		rdd.foreach(new VoidFunction<int[]>(){
			public void call(int[] d){
				System.out.format("%2d ---> %2d\n ", d[0], d[1]);
			}
		});
	}
	
	
	// display java resilient distributed dataset
	public static void RDD_Display_2D(JavaRDD<double[][]> rdd) throws Exception {
		rdd.foreach(new VoidFunction<double[][]>(){
			public void call(double[][] d){
				for( int i=0; i<d.length; i++){
					System.out.format("{%10.5f, %4.1f, %4.1f} \n", d[i][0], d[i][1], d[i][2]);
				}
				System.out.format("\n");
			}
		});
	}	
	
	public static void main(String[] args) throws IOException, Exception {
		// get testing set
		double[][] training_array = KNN.ReadDataSet(args[0], GetLineCount(args[0]));
		int TrainingSetLineCnt = GetLineCount(args[0]);
		//int TestingSetLineCnt = GetLineCount(args[1]);
		
		//double[][] testing_array = KNN.ReadDataSet(args[1], GetLineCount(args[1]));
		
		int k = Integer.parseInt(args[2]);
		
		SparkConf conf = new SparkConf().setAppName("KNN Spark");
		JavaSparkContext sc = new JavaSparkContext(conf);
		
		// read the training set
		JavaRDD<String> testing = sc.textFile(args[1]);
		JavaRDD<double[]> testing_rdd_array = KNN.Get_Testing_Set_RDD_Double_Array(testing);
		JavaRDD<int[]> KNN_Classification = testing_rdd_array.map(new Function<double[], int[]>(){
			public int[] call(double[] d){
				int[] KNN_Classification = new int[2];
				
				double distance = 0;
				
				// the width of training set data
				// the last element of each row stands for its label
				int w_training = training_array[0].length;
				
				// the width of testing set data
				// the last element of each row stands for its object index 
				int w_testing = d.length;
				
				KNN_Classification[0] = (int)d[w_testing-1];
				
				// three dimension variables
				// 1 dimension(0): is the distance
				// 2 dimension(1): label
				double[][] trj = new double[TrainingSetLineCnt][2];
				for(int i=0; i<training_array.length-1; i++){
					distance = 0;
					for( int j = 0; j<w_training-1; j++){
						distance += d[j]*training_array[i][j];
					}
					trj[i][0] = distance;
					trj[i][1] = training_array[i][w_training-1];
				}
						
				InsertSort(trj);
				KNN_Classification[1] = VotingSystem(trj, k);
				return KNN_Classification;	
			}
			
			// Insert sorting to get first k numbers
			public void InsertSort(double[][] a){
				for( int i=1; i<a.length; i++){
					double temp_distc = a[i][0];
					double temp_label = a[i][1];
					int j;
					for( j=i-1; j>=0 && temp_distc<a[j][0];j--){
						a[j+1][0] = a[j][0];
						a[j+1][1] = a[j][1];
					}
					a[j+1][0] = temp_distc;
					a[j+1][1] = temp_label;
				}
			}
			
			// voting system to get the final label
			public int VotingSystem(double[][] a, int k){		
				LinkedList<Double> l = new LinkedList<Double>();
				int[] label_cnt = new int[k];
				
				for( int i=0; i<k; i++){
					// if linkedlist is empty
					if(l.isEmpty()){
						// add current element in the linkedlist
						l.add(a[i][1]);
						// add 1 for corresponding element
						label_cnt[l.indexOf(a[i][1])]++;
					}
					// if current element is not in l and l is not empty
					else{
						// if the element is in the 
						if(l.contains(a[i][1])){
							label_cnt[l.indexOf(a[i][1])]++;
						} else {
							l.add(a[i][1]);
							label_cnt[l.indexOf(a[i][1])]++;
						}		
					}	
				}
				
				int max_index = -1;
				int index = 0;
				for(int i=0; i<label_cnt.length; i++){
					if(label_cnt[i]>max_index){
						max_index = label_cnt[i];
						index = i;
					}	
				}
				return (l.get(index)).intValue();
			}	
		});
				
		System.out.println("Obj #  label ");
		KNN.RDD_Display_1D(KNN_Classification);
		sc.close();	
	}
}
