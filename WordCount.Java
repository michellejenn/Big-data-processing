import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;  // Import for using Stream API

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCount {

    // Mapper class
    public static class TokenizerMapper
            extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        // Map function processes each line of input
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] columns = value.toString().split("\t");
            if (columns.length >= 5) {
                word.set(columns[4]);
                context.write(word, one); // Emit (key, value) pair: (GO term, 1)
            }
        }
    }

    // Reducer class
    public static class IntSumReducer
            extends Reducer<Text, IntWritable, Text, IntWritable> {

        private IntWritable result = new IntWritable();

        // Reduce function sums up the counts for each GO term
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = StreamSupport.stream(values.spliterator(), false)  // Use Stream API to sum values
                                   .mapToInt(IntWritable::get)
                                   .sum();
            result.set(sum);
            context.write(key, result);  // Emit (key, value) pair: (GO term, total count)
        }
    }

    // Custom class for sorting GO terms by count
    public static class GOTermCount implements Comparable<GOTermCount> {
        private String term;
        private int count;

        public GOTermCount(String term, int count) {
            this.term = term;
            this.count = count;
        }

        public String getTerm() {
            return term;
        }

        public int getCount() {
            return count;
        }

        @Override
        public int compareTo(GOTermCount other) {
            return Integer.compare(other.count, this.count);
        }
    }

    // Reducer class to sort and limit the output to the top 10 GO terms
    public static class GOTermCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private Map<String, Integer> goTermCounts = new HashMap<>();

        // Reduce function calculates total counts for each GO term
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = StreamSupport.stream(values.spliterator(), false)  // Use Stream API to sum values
                                   .mapToInt(IntWritable::get)
                                   .sum();
            goTermCounts.put(key.toString(), sum);
        }

        // Cleanup function is called at the end of the Reducer
        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            // Sort GO terms by count, limit to the top 10, and emit
            List<GOTermCount> sortedCounts = goTermCounts.entrySet().stream()
                .map(entry -> new GOTermCount(entry.getKey(), entry.getValue()))
                .sorted(Comparator.comparingInt(GOTermCount::getCount).reversed())
                .limit(10)
                .collect(Collectors.toList());

            sortedCounts.forEach(count -> context.write(new Text(count.getTerm()), new IntWritable(count.getCount()));
        }
    }

    // Main method
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "GO term count");
        job.setJarByClass(WordCount.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setReducerClass(GOTermCountReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
