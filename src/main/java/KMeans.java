import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * KMeans class implements the K-means clustering algorithm using Apache Spark.
 */
public final class KMeans {
    /**
     * The pattern used to split input lines by commas.
     */
    private static final Pattern COMMA_PATTERN = Pattern.compile(",");

    /**
     * Main method to execute K-means clustering.
     *
     * @param args Command-line arguments: <input_path> <output_path> <k> [max_iterations] [epsilon]
     */
    public static void main(String[] args) {
        // Check if the required command-line arguments are provided
        if (args.length < 3) {
            System.out.println("Usage: KMeans <input_path> <output_path> <k> [max_iterations] [epsilon]");
            System.exit(1);
        }

        // Extract command-line arguments
        String inputPath = args[0];
        String outputPath = args[1];
        int k = Integer.parseInt(args[2]);
        int maxIterations = args.length > 3 ? Integer.parseInt(args[3]) : 1000;
        double epsilon = args.length > 4 ? Double.parseDouble(args[4]) : 0.001;

        // Run the K-means clustering algorithm
        runKMeans(inputPath, outputPath, k, maxIterations, epsilon);
    }

    /**
     * Method to run the K-means clustering algorithm.
     *
     * @param inputPath      Path to the input data file
     * @param outputPath     Path to save the output centroids
     * @param k              Number of clusters (centroids)
     * @param maxIterations  Maximum number of iterations for convergence
     * @param epsilon        Convergence threshold
     */
    public static void runKMeans(String inputPath, String outputPath, int k, int maxIterations, double epsilon) {
        try (JavaSparkContext context = new JavaSparkContext(new SparkConf().setMaster("local").setAppName("k-means"))) {
            JavaRDD<List<Double>> data = context.textFile(inputPath)
                    .map(KMeans::loadData)
                    .cache();

            JavaRDD<List<Double>> finalCentroids = kMeans(context, data, k, epsilon, maxIterations);

            finalCentroids.saveAsTextFile(outputPath);
        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        }
    }

    /**
     * Method to perform K-means clustering.
     *
     * @param context        Spark context
     * @param data           Input data as RDD of vectors
     * @param k              Number of clusters (centroids)
     * @param epsilon        Convergence threshold
     * @param maxIterations  Maximum number of iterations for convergence
     * @return               RDD containing the final centroids
     */
    public static JavaRDD<List<Double>> kMeans(JavaSparkContext context, JavaRDD<List<Double>> data, int k, double epsilon, long maxIterations) {
        // Initialize centroids
        List<List<Double>> centroids = initializeCentroids(data, k);
        long currentIteration = 0;
        double distance;

        long startTime = System.currentTimeMillis();

        // Perform iterations until convergence or maximum iterations reached
        do {
            JavaPairRDD<Integer, List<Double>> closestPairs = assignToClosestCentroid(data, centroids);
            JavaPairRDD<Integer, Iterable<List<Double>>> groupedData = closestPairs.groupByKey();
            Map<Integer, List<Double>> newCentroids = calculateNewCentroids(groupedData);
            distance = calculateDistance(centroids, newCentroids);

            // Update centroids with the new ones
            centroids = new ArrayList<>(newCentroids.values());

            currentIteration++;
        } while (distance > epsilon && currentIteration < maxIterations);

        long endTime = System.currentTimeMillis();
        long elapsedTime = endTime - startTime;

        // Print results
        System.out.println("Time taken: " + elapsedTime + " milliseconds.");
        System.out.println("Converged in " + currentIteration + " iterations.");
        System.out.println("Final centers:");
        centroids.forEach(System.out::println);

        // Convert centroids list to RDD and return
        return context.parallelize(centroids);
    }

    /**
     * Method to initialize centroids randomly from the data.
     *
     * @param data Input data as RDD of vectors
     * @param k    Number of clusters (centroids)
     * @return     List of initial centroids
     */
    private static List<List<Double>> initializeCentroids(JavaRDD<List<Double>> data, int k) {
        return data.takeSample(false, k);
    }

    /**
     * Method to assign each data point to its closest centroid.
     *
     * @param data      Input data as RDD of vectors
     * @param centroids List of centroids
     * @return          Pair RDD with data points and their closest centroid
     */
    private static JavaPairRDD<Integer, List<Double>> assignToClosestCentroid(JavaRDD<List<Double>> data, List<List<Double>> centroids) {
        return data.mapToPair(vector -> new Tuple2<>(closestPoint(vector, centroids), vector));
    }

    /**
     * Method to calculate new centroids based on assigned data points.
     *
     * @param groupedData Grouped data where each key is a centroid index
     *                    and values are the vectors assigned to that centroid
     * @return            Map of new centroids
     */
    private static Map<Integer, List<Double>> calculateNewCentroids(JavaPairRDD<Integer, Iterable<List<Double>>> groupedData) {
        return groupedData.mapValues(KMeans::average).collectAsMap();
    }

    /**
     * Method to calculate the total squared Euclidean distance between old and new centroids.
     *
     * @param centroids    List of old centroids
     * @param newCentroids Map of new centroids where keys are centroid indices
     * @return             Total squared Euclidean distance
     */
    private static double calculateDistance(List<List<Double>> centroids, Map<Integer, List<Double>> newCentroids) {
        return centroids.stream()
                .mapToDouble(centroid -> squaredDist(centroid, newCentroids.get(centroids.indexOf(centroid))))
                .sum();
    }

    /**
     * Method to parse a line of input data into a list of doubles.
     *
     * @param line Input line containing comma-separated values
     * @return     List of parsed double values
     */
    private static List<Double> loadData(String line) {
        return Arrays.stream(COMMA_PATTERN.split(line))
                .map(Double::parseDouble)
                .collect(Collectors.toList());
    }

    /**
     * Method to find the index of the closest centroid for a given vector.
     *
     * @param p       Input vector
     * @param centers List of centroids
     * @return        Index of the closest centroid
     */
    private static int closestPoint(List<Double> p, List<List<Double>> centers) {
        return IntStream.range(0, centers.size())
                .boxed()
                .min(Comparator.comparingDouble(i -> squaredDist(p, centers.get(i))))
                .orElse(0);
    }

    /**
     * Method to calculate the average vector from a collection of vectors.
     *
     * @param vectors Collection of vectors
     * @return        Average vector
     */
    private static List<Double> average(Iterable<List<Double>> vectors) {
        List<Double> sum = new ArrayList<>();
        int count = 0;

        for (List<Double> vector : vectors) {
            if (sum.isEmpty()) {
                sum.addAll(vector);
            } else {
                for (int i = 0; i < vector.size(); i++) {
                    sum.set(i, sum.get(i) + vector.get(i));
                }
            }
            count++;
        }

        for (int i = 0; i < sum.size(); i++) {
            sum.set(i, sum.get(i) / count);
        }

        return sum;
    }

    /**
     * Method to calculate the Euclidean distance between two vectors.
     *
     * @param p1 First vector
     * @param p2 Second vector
     * @return   Squared Euclidean distance
     */
    private static double squaredDist(List<Double> p1, List<Double> p2) {
        return IntStream.range(0, p1.size())
                .mapToDouble(i -> Math.pow(p1.get(i) - p2.get(i), 2))
                .sum();
    }
}
