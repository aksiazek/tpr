package me.soulmachine;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mrunit.mapreduce.MapDriver;
import org.apache.hadoop.mrunit.mapreduce.MapReduceDriver;
import org.apache.hadoop.mrunit.mapreduce.ReduceDriver;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;


public class WordCountTest {

    @Test
    public void testMapper() throws Exception {
        new MapDriver<LongWritable, Text, Text, IntWritable>()
                .withMapper(new WordCount.Map())
                .withInput(new LongWritable(0), new Text("foo bar foo bar foo"))
                .withOutput(new Text("foo"), new IntWritable(1))
                .withOutput(new Text("bar"), new IntWritable(1))
                .withOutput(new Text("foo"), new IntWritable(1))
                .withOutput(new Text("bar"), new IntWritable(1))
                .withOutput(new Text("foo"), new IntWritable(1))
                .runTest();
    }

    @Test
    public void testReducer() throws Exception {

        List<IntWritable> fooValues = new ArrayList<>();
        fooValues.add(new IntWritable(1));
        fooValues.add(new IntWritable(1));
        fooValues.add(new IntWritable(1));

        List<IntWritable> barValue = new ArrayList<>();
        barValue.add(new IntWritable(1));
        barValue.add(new IntWritable(1));

        new ReduceDriver<Text, IntWritable, Text, IntWritable>()
                .withReducer(new WordCount.Reduce())
                .withInput(new Text("foo"), fooValues)
                .withInput(new Text("bar"), barValue)
                .withOutput(new Text("foo"), new IntWritable(3))
                .withOutput(new Text("bar"), new IntWritable(2))
                .runTest();
    }


    @Test
    public void testMapReduce() throws Exception {

        new MapReduceDriver<LongWritable, Text, Text, IntWritable, Text, IntWritable>()
                .withMapper(new WordCount.Map())
                .withInput(new LongWritable(0), new Text("foo bar foo bar foo"))
                .withReducer(new WordCount.Reduce())
                .withOutput(new Text("bar"), new IntWritable(2))
                .withOutput(new Text("foo"), new IntWritable(3))
                .runTest();
    }
}