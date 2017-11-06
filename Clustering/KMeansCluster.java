// package edu.buffalo.project2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KMeansCluster {

	public static Map<Integer, Integer> groundTruthMap = new HashMap<Integer, Integer>();
	public static Map<Integer, double[]> geneList = new HashMap<Integer, double[]>();
	public static int[][] carr;
	public static int[][] garr;
	public static Map<Integer, Integer> currentGeneCentroidMap = new HashMap<Integer, Integer>();
	public static Map<Integer, Integer> updatedGeneCentroidMap = new HashMap<Integer, Integer>();
	public static Map<Integer, double[]> centroidClusterMap = new HashMap<Integer, double[]>();
	static String filePath = null;
	static boolean isFileParsed = false;
	static String cvsSplitBy = "\t";
	static int k;
	boolean updated = true;
	static int totalGenes = 0;
	private static String initialCentroids;
	private static int vectorLength;
	public static void parseTxtFile(String TxtFile) {
		String line = "";
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(TxtFile));
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
		try {

			while ((null != br) && (line = br.readLine()) != null) {

				totalGenes++;
				String[] geneRowVector = line.split(cvsSplitBy);
				int geneId = Integer.parseInt(geneRowVector[0]);
				int groundTruthClusterId = Integer.parseInt(geneRowVector[1]);
				groundTruthMap.put(geneId, groundTruthClusterId);
				currentGeneCentroidMap.clear();
				updatedGeneCentroidMap.clear();
				currentGeneCentroidMap.put(geneId, 0);
				updatedGeneCentroidMap.put(geneId, 0);
				vectorLength = geneRowVector.length - 2;
				double[] geneVector = new double[vectorLength];
				for (int i = 0; i < geneRowVector.length - 2; i++) {
					geneVector[i] = Double.parseDouble(geneRowVector[i + 2]);
				}				
				geneList.put(geneId, geneVector);
			}

		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	private static void initiateClustering() {
		int c = 1;
		
		if(null==initialCentroids || initialCentroids.equals(""))
		{
			for (int i = 1; i <= totalGenes && c <= k; i = i + (totalGenes / k)) {
			updatedGeneCentroidMap.put(i, c);
			centroidClusterMap.put(c, geneList.get(i));
			c++;
		}
		}
		else
		{
			initialCentroids = initialCentroids.replaceAll("\"", "");
			String iniCentroids[] = initialCentroids.split(",");
			for(int i = 1 ;i<=k;i++)
			{
				updatedGeneCentroidMap.put(Integer.parseInt(iniCentroids[i-1]),i );
				centroidClusterMap.put(i, geneList.get(Integer.parseInt(iniCentroids[i-1])));
				
			}
		}

	}

	public static class TokenizerMapper extends Mapper<Object, Text, IntWritable, IntWritable> {

		double minDis = Double.MAX_VALUE;
		double dis =0.0;

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			if (!isFileParsed) {
				parseTxtFile(filePath);
				initiateClustering();
				isFileParsed = true;
			}
			String row = value.toString();
			row = row.replaceAll("\"", "");
			String[] vectors = row.split(cvsSplitBy);
			int geneId = Integer.parseInt(vectors[0]);
			double[] geneVector = new double[vectorLength];
			for (int i = 0; i < vectorLength; i++) {
				geneVector[i] = Double.parseDouble(vectors[i + 2]);
			}
			int index = 0;
			minDis = Double.MAX_VALUE;
			for (Entry<Integer, double[]> entry : centroidClusterMap.entrySet()) {
				dis = computeDistance(entry.getValue(), geneVector);
				if (dis < minDis) {
					minDis = dis;
					index = entry.getKey();
				}
			}
			updatedGeneCentroidMap.put(geneId, index);
			context.write(new IntWritable(index), new IntWritable(geneId));

		}

	}

	private static double computeDistance(double[] geneVectorList, double[] v) {
		double sum = 0;
		int length = v.length;
		/*if (geneVectorList.length != length)
			return -1;*/
		for (int i = 0; i < length; i++) {
			sum += Math.pow((geneVectorList[i] - (Double) v[i]), 2);
		}

		return Math.sqrt(sum);

	}

	public static class IntSumReducer extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {

		public void reduce(IntWritable key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			int num = 0;
			double[] indVectorVal = new double[vectorLength];
			 
			for (IntWritable vx : values) {
			
				double[] geneVector = geneList.get(vx.get());
				for (int i = 0; i < vectorLength; i++) {
					
					indVectorVal[i] = indVectorVal[i] + geneVector[i];

				}
				num++;
			}
				for (int i = 0; i < vectorLength; i++)
				indVectorVal[i] = indVectorVal[i] / num;
				centroidClusterMap.put(key.get(), indVectorVal);

				context.write(key,key );
			
		}
	}

	public static void main(String[] args) throws Exception {
		
		boolean x = true;

		filePath = args[2];
		k = Integer.parseInt(args[3]);
		initialCentroids = args[4];
		
		while (x || isMapUpdated()) {
			x = false;
			Configuration conf = new Configuration();

			Job job = new Job(conf, "KMeans");
			job.setJarByClass(KMeansCluster.class);
			job.setMapperClass(TokenizerMapper.class);
			// job.setCombinerClass(IntSumReducer.class);
			job.setReducerClass(IntSumReducer.class);
			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(IntWritable.class);
			FileInputFormat.addInputPath(job, new Path(args[0]));
			FileSystem fileSystem = FileSystem.get(conf);

			Path path = new Path(args[1]);

			if (!fileSystem.exists(path)) {

				FileOutputFormat.setOutputPath(job, new Path(args[1]));

			}

			else {

				// Delete file

				fileSystem.delete(new Path(args[1]), true);

				fileSystem.close();

				FileOutputFormat.setOutputPath(job, new Path(args[1]));

			}
			job.waitForCompletion(true);
		}
		double[] res = findJaccard();
		System.out.println("Jaccard Co-efficient = "+res[0]);
		System.out.println("Rand index = "+res[1]);

	}
	private static double[] findJaccard() {
		double jacc =0.0;
		double rand =0.0;
		int clusLength = updatedGeneCentroidMap.size();
		int groundTruthLength = groundTruthMap.size();
		/*if(clusLength!=groundTruthLength)
			return jacc;*/
		int i,j;
		carr = new int[clusLength+1][clusLength+1];
		for(i = 1 ;i<=clusLength;i++)
		{
			for(j = 1 ;j<=clusLength;j++)
			{
				if(updatedGeneCentroidMap.get(i)== updatedGeneCentroidMap.get(j))
				{
					carr[i][j] = 1;
					
				}
				else
					carr[i][j] = 0;
			}
		}
		garr = new int[groundTruthLength+1][groundTruthLength+1];
		for(i = 1 ;i<=groundTruthLength;i++)
		{
			for(j = 1 ;j<=groundTruthLength;j++)
			{
				if(groundTruthMap.get(i)== groundTruthMap.get(j))
				{
					garr[i][j] = 1;
				}
				else
					garr[i][j] = 0;
			}
		}
		
		int m11 = 0;
		int	m01 = 0;
		int	m10 = 0;
		int m00 = 0;
		for (i = 1;i<=groundTruthLength;i++)
		{
			for (j = 1;j<=groundTruthLength;j++)
			{
				        if (garr[i][j] == 1 && carr[i][j] == 1)
				            m11 += 1;
				            else if( garr[i][j] == 0 && carr[i][j] == 1)
				            m01 += 1;
				            else if( garr[i][j] == 1 && carr[i][j] == 0)
				            m10 += 1;
				            else if( garr[i][j] == 0 && carr[i][j] == 0)
					            m00 += 1;
			}
		}
		
		jacc = (double) m11 / (m11 + m01 + m10);
		rand = (double) (m11+m00)/(m11+m00+m01+m10);
		double[] results = new double[]{jacc,rand};
		return results;
	}

	private static boolean isMapUpdated() {
		for (Integer entry : currentGeneCentroidMap.keySet()) {
			if (updatedGeneCentroidMap.get(entry) != currentGeneCentroidMap.get(entry))
				
			{
				currentGeneCentroidMap = new HashMap<Integer, Integer>(updatedGeneCentroidMap);
				updatedGeneCentroidMap.clear();
				return true;
			}
		}
		//write file
		writeGeneClustedMap();
		return false;
	}

	private static void writeGeneClustedMap() {
		//updatedGeneCentroidMap
		String csv = "/home/hadoop/outputCluster.csv";
		FileWriter fw = null;
		try {
			fw = new FileWriter(csv);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		   PrintWriter out = new PrintWriter(fw);
		   for(Map.Entry<Integer, Integer> entry : updatedGeneCentroidMap.entrySet())
		   {
			   out.print(entry.getKey()) ;
			   out.print(",");
			   out.println(entry.getValue()) ;
		   }
		   
		  
		   out.flush();
		   out.close();
		   try {
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}       }

}
