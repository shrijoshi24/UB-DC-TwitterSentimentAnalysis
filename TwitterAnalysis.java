package ub.mis.dc.cloud9;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;

import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.codehaus.jackson.JsonNode;
import org.codehaus.jackson.map.ObjectMapper;


public class TwitterAnalysis
{
	static HashSet<String> positiveWordsSet = new HashSet<String>();
	
	static HashSet<String> negativeWordsSet = new HashSet<String>();
	
	private static final boolean DEBUG = true;
	
	enum Sentiment
	{
		positive, negative, neutral;
	}
	
	public static class TMapper extends MapReduceBase implements Mapper<LongWritable, Text, Text, Text>
	{
		//NLP models' path.
		private static String modelsPath = "/home/ubuntu/Cluster/sentiment_data/models";
		
		@Override
		public void map(LongWritable _key, Text value,
				OutputCollector<Text, Text> output, Reporter reporter)
				throws IOException
		{
			ObjectMapper map = new ObjectMapper();
			Sentiment sent = Sentiment.neutral;
			try
			{
				String line = value.toString();
				JsonNode rootNode = map.readValue(line, JsonNode.class);
				JsonNode langNode = rootNode.path("lang");
				if(langNode != null)
				{
					String langValue = langNode.getTextValue();
					//If the Tweet is in English
					if("en".equalsIgnoreCase(langValue))
					{
						JsonNode userNode = rootNode.path("user");
						String userLocation = userNode.path("location").getTextValue();
						if(userLocation != null)
						{
							userLocation = userLocation.trim();
							//If the user location is not blank and doesn't contain special chars
							if(userLocation.length() != 0 && !userLocation.contains("&") && !userLocation.contains(":")/* && !userLocation.contains("-")*/)
							{
								sent = getTweetAndUserData(rootNode);
								userLocation = userLocation.replace(".", "");
								userLocation = userLocation.toLowerCase();
								Text sentStr = new Text(getSentimentAsString(sent));
								if(DEBUG)
									System.err.println("Mapped String: " + userLocation + "    " + sentStr);
								//Map the user location to its sentiment (positive, negative or neutral)
								output.collect(new Text(userLocation), sentStr);
							}
						}
					}
				}
			}
			catch (IOException e)
			{
				e.printStackTrace();
			}
		}
		
		private static Sentiment getTweetAndUserData(JsonNode rootNode) throws InvalidFormatException, IOException
		{
			JsonNode tweetContentNode = rootNode.path("text");
			String tweetString = tweetContentNode.getTextValue();
			Sentiment sent = doNLP(tweetString);
			return sent;
		}
		
		private static Sentiment doNLP(String tweetString)
		{
			if(DEBUG)
				System.err.println("Tweet: " + tweetString);
			String[] sentences = detectSentences(tweetString);
			if(DEBUG)
				System.err.println("Tokens: ");
			ArrayList<String> allTokensPerTweet = new ArrayList<String>();
			InputStream is = null;
			try
			{
				is = new FileInputStream(modelsPath + "/en-token.bin");
			}
			catch (FileNotFoundException e)
			{
				System.out.println("en-token.bin file not found in path: " + modelsPath);
				e.printStackTrace();
			}
			TokenizerModel model = null;
			try
			{
				model = new TokenizerModel(is);
			}
			catch (InvalidFormatException e)
			{
				e.printStackTrace();
			}
			catch (IOException e)
			{
				e.printStackTrace();
			}
			Tokenizer tokenizer = new TokenizerME(model);
			int sentencesLength = sentences.length;
			for(int i=0; i<sentencesLength; ++i)
			{
				String eachSentence = sentences[i];
				String tokens[] = tokenizer.tokenize(eachSentence);
				for (String a : tokens)
				{
					allTokensPerTweet.add(a);
					if(DEBUG)
						System.err.println(a);
				}
			}
			try
			{
				is.close();
			}
			catch (IOException e)
			{
				e.printStackTrace();
			}
			Sentiment sent = categorizeTokens(allTokensPerTweet);
			return sent;
		}
		
		private static String[] detectSentences(String tweetString)
		{
			InputStream is = null;
			try
			{
				is = new FileInputStream(modelsPath + "/en-sent.bin");
			}
			catch (FileNotFoundException e)
			{
				System.out.println("en-sent.bin file not found in path: " + modelsPath);
				e.printStackTrace();
			}
			
			SentenceModel model = null;
			try
			{
				model = new SentenceModel(is);
			}
			catch (InvalidFormatException e)
			{
				e.printStackTrace();
			}
			catch (IOException e)
			{
				e.printStackTrace();
			}
			SentenceDetectorME sdetector = new SentenceDetectorME(model);
			String sentences[] = sdetector.sentDetect(tweetString);
			try
			{
				is.close();
			}
			catch (IOException e)
			{
				e.printStackTrace();
			}
			return sentences;
		}
		
		private static Sentiment categorizeTokens(ArrayList<String> allTokensPerTweet)
		{
			int totalTokens = allTokensPerTweet.size();
			int positiveCount = 0, negativeCount = 0, neutralCount = 0;
			for(int i = 0; i<totalTokens; ++i)
			{
				String eachToken = allTokensPerTweet.get(i);
				if(positiveWordsSet.contains(eachToken))
				{
					++positiveCount;
					if(DEBUG)
						System.err.println("positive token: " + eachToken);
				}
				else if(negativeWordsSet.contains(eachToken))
				{
					++negativeCount;
					if(DEBUG)
						System.err.println("negative token: " + eachToken);
				}
				else
				{
					++neutralCount;
				}
			}
			if(DEBUG)
				System.err.println("positiveCount: " + positiveCount + " negativeCount: " 
			+ negativeCount + " neutralCount: " + neutralCount + " Total count: " + totalTokens);
			
			if(positiveCount > negativeCount)
				return Sentiment.positive;
			
			if(negativeCount > positiveCount)
				return Sentiment.negative;
			
			return Sentiment.neutral;			
		}
	}
	
	private static String getSentimentAsString(Sentiment sent)
	{
		switch (sent)
		{
			case positive:
				return "positive";
			case negative:
				return "negative";
			case neutral:
				return "neutral";
			default:
				return "neutral";
		}
	}
	
	public static class TReducer extends MapReduceBase implements Reducer<Text, Text, Text, Text>
	{
		@Override
		public void reduce(Text key, Iterator<Text> values,
				OutputCollector<Text, Text> output, Reporter reporter)
				throws IOException
		{
			int positiveCount = 0, negativeCount = 0, neutralCount = 0;
			while (values.hasNext())
			{
				// replace ValueType with the real type of your value
				Text value = (Text) values.next();
				String valueStr = value.toString();
				if("positive".equalsIgnoreCase(valueStr))
				{
					positiveCount++;
				}
				else if("negative".equalsIgnoreCase(valueStr))
				{
					negativeCount++;
				}
				else
				{
					neutralCount++;
				}
			}
			Text reduceText = new Text(positiveCount + "\t"+negativeCount + "\t"+neutralCount);
			if(DEBUG)
				System.err.println("Reduce String: " + key + "    " + reduceText.toString());
			//Reduce based on user location to the count of positive, negative and neutral tweets
			output.collect(key, reduceText);
		}
	}
	
	public static void main(String[] args) throws Exception
	{
		JobConf conf = new JobConf();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		if (otherArgs.length != 2)
		{
			System.err.println("JarName.jar <Input path> <Output path>");
			System.exit(2);
		}
		if(DEBUG)
		{
			for(int i=0; i<otherArgs.length; i++)
			{
				System.out.println("otherargs["+i+"] = "+otherArgs[i]);
			}
		}
		conf.setJobName("Twitter Sentiment Analysis");
		conf.setOutputKeyClass(Text.class);
		conf.setOutputValueClass(Text.class);
		conf.setJarByClass(TwitterAnalysis.class);
		
		conf.setMapperClass(TMapper.class);
		conf.setReducerClass(TReducer.class);
		
		conf.setInputFormat(TextInputFormat.class);
		conf.setOutputFormat(TextOutputFormat.class);
		
		FileInputFormat.setInputPaths(conf, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(conf, new Path(otherArgs[1]));
		
		readSentimentDictionary();
		JobClient.runJob(conf);
	}
	
	private static void readSentimentDictionary()
	{
		//Sentiment dictionary path
		String dictionaryPath = "/home/ubuntu/Cluster/sentiment_data/lexicon";
		String filePath = dictionaryPath + "/positive-words.txt";
		positiveWordsSet = readDictionary(filePath);
		filePath = dictionaryPath + "/negative-words.txt";
		negativeWordsSet = readDictionary(filePath);
	}
	
	private static HashSet<String> readDictionary(String filePath)
	{
		HashSet<String> wordsSet = new HashSet<String>();
		try
		{
			BufferedReader br = new BufferedReader(new FileReader(filePath));
			String line = br.readLine();
			while(line != null)
			{
				wordsSet.add(line);
				line = br.readLine();
			}
			br.close();
		}
		catch (FileNotFoundException e)
		{
			System.err.println("File path: " + filePath + " not found.");
			e.printStackTrace();
		}
		catch (IOException ex)
		{
			ex.printStackTrace();
		}
		return wordsSet;
	}
}
