import com.datadoghq.sketch.ddsketch.DDSketch;
import com.datadoghq.sketch.uddsketch.UniformDDSketch;
import com.github.stanfordfuturedata.momentsketch.MomentStruct;
import com.github.stanfordfuturedata.momentsketch.SimpleMomentSketch;
import org.apache.commons.math3.distribution.*;
import org.apache.commons.math3.util.FastMath;
import org.apache.datasketches.kll.KllFloatsSketch;
import org.apache.datasketches.req.ReqSketch;

import java.io.*;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.TimeUnit;

public class Quantiler {

    public static final boolean RUN_MERGE_SKETCHES = false;
    public static final int KLL_PARAM_K = 350;
    public static final int MOMEMNTS_PARAM_K = 18;
    public static final int UDDS_PARAM_MAX_NUM_BUCKETS = 1024;
    public static final int UDDS_PARAM_K = 12;
    public static final double DDS_PARAM_RELATIVE_ACCURACY = 0.01;
    public static final double UDDS_PARAM_RELATIVE_ACCURACY = 0.01;
    public static final int REQ_PARAM_K = 16;
    public static final boolean REQ_PARAM_HIGH_RANK_ACCURACY = false;
    public static final boolean REQ_PARAM_LT_EQ = true;
    public static final boolean CALC_POWER_STATS = false;
    public static final boolean CALC_NYT_STATS = false;
    public static final boolean RUN_K_TESTS = false;

    public static void main(String[] args) {
        int runMode = Integer.parseInt(args[0]);
        try {

            // Sketch Algorithms
            double[] percentiles = {0.01, 0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.98, 0.99};

            ZipfDistribution zipfDistribution = new ZipfDistribution(50, 1.5);
            int[] data = zipfDistribution.sample(1000000);

            /*
            double[] percentiles = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,  0.11, 0.12, 0.13,
                    0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,  0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27,
                    0.28, 0.29, 0.3,  0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4,  0.41,
                    0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,  0.51, 0.52, 0.53, 0.54, 0.55,
                    0.56, 0.57, 0.58, 0.59, 0.6,  0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
                    0.7,  0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,  0.81, 0.82, 0.83,
                    0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,  0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97,
                    0.98, 0.99};

            int[] percentiles2 = {1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
                    14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
                    28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
                    42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
                    56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
                    70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
                    84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
                    98,  99};
             */

            MomentStruct msketch = new MomentStruct(18);

            double alphaZero =
                Math.tanh(FastMath.atanh(UDDS_PARAM_RELATIVE_ACCURACY) / Math.pow(2.0, UDDS_PARAM_K - 1));
            DDSketch ddsketch = new DDSketch(DDS_PARAM_RELATIVE_ACCURACY);
            UniformDDSketch uddsketch = new UniformDDSketch(UDDS_PARAM_MAX_NUM_BUCKETS, alphaZero);
            KllFloatsSketch kllsketch = new KllFloatsSketch(KLL_PARAM_K);
            SimpleMomentSketch momentSketch = new SimpleMomentSketch(MOMEMNTS_PARAM_K);
            ReqSketch reqSketch =
                ReqSketch.builder().setK(REQ_PARAM_K).setHighRankAccuracy(REQ_PARAM_HIGH_RANK_ACCURACY)
                    .setLessThanOrEqual(REQ_PARAM_LT_EQ).build();
            // Synthetic Workloads

            if (RUN_K_TESTS) {
                runKTests();
            }

            //initTestSketches();
//            System.exit(0);

            int dataSizeMerge = 1_000_000;
            if (runMode == 3) {
                int numSketches = 10;
                runMergeTests(dataSizeMerge, numSketches);
            }


            // ADAPTABILITY TEST

            int dataSizeAdaptability = 500000;
            ArrayList<Double> all_data = new ArrayList<Double>();
            BinomialDistribution binD = new BinomialDistribution(30, 0.4);
            System.out.println("Binomial Distribution mean: " + binD.getNumericalMean());

            ZipfDistribution zipD = new ZipfDistribution(20, 0.6);
            System.out.println("Zipf mean: " + zipD.getNumericalMean());

            GammaDistribution gd = new GammaDistribution(2, 0.4);
            System.out.println("Gamma mean: " + gd.getNumericalMean());

            UniformRealDistribution unif_adapt_test = new UniformRealDistribution(40, 100);
            System.out.println("Uniform mean: " + unif_adapt_test.getNumericalMean());

            ParetoDistribution ptoD = new ParetoDistribution(1, 1);
            System.out.println("Pareto mean: " + ptoD.getNumericalMean());

            // *********************************
            // Adaptability tests
//            String sketchName = "UDDSketch";
//            long startInsert = System.nanoTime();
//
//            for (int i = 0; i < dataSizeAdaptability; i++) {
//                double sampled_value = binD.sample();
//                double sampled_value_2 = gd.sample();
//                all_data.add(sampled_value);
//                ddsketch.accept(sampled_value);
//                ddsketch.accept(sampled_value_2);
//                uddsketch.accept(sampled_value);
//                uddsketch.accept(sampled_value_2);
//                kllsketch.update((float) sampled_value);
//                reqSketch.update((float) sampled_value);
//                msketch.add(sampled_value);
//            }
//
//            long endInsert = System.nanoTime();
//
//            long elapsedTimeInsert = endInsert - startInsert;
//            System.out.println(sketchName);
//            System.out.println("Insert time - nanos (" + sketchName + "): " + elapsedTimeInsert);
//            System.out.println(
//                "Insert time - micros (" + sketchName + "): " + TimeUnit.NANOSECONDS.toMicros(elapsedTimeInsert));
//            System.out.println(
//                "Insert time - millis (" + sketchName + "): " + TimeUnit.NANOSECONDS.toMillis(elapsedTimeInsert));
//            if ("UDDSketch".equals(sketchName)) {
//                System.out.println(uddsketch);
//                System.out.println("Count:" + uddsketch.getCount());
//                System.out.println(Arrays.toString(uddsketch.getValuesAtQuantiles(percentiles)));
//            }
//
//            startInsert = System.nanoTime();
//            ArrayList<Double> dal = new ArrayList<Double>();
//
//            for (int i = 0; i < dataSizeAdaptability; i++) {
//                double sampled_value = binD.sample();
//                double sampled_value_2 = gd.sample();
//                dal.add(sampled_value);
//                //all_data.add(sampled_value);
//                //ddsketch.accept(sampled_value);
//                //kllsketch.update((float) sampled_value);
//                //kllsketch.update((float) sampled_value_2);
//                momentSketch.add(sampled_value);
//                momentSketch.add(sampled_value_2);
//            }
//
//            Kurtosis kurtosis = new Kurtosis();
//
//            double[] target = new double[dal.size()];
//            for (int i = 0; i < target.length; i++) {
//                target[i] = dal.get(i);                // java 1.5+ style (outboxing)
//            }
//
//            System.out.println("Kurtosis");
//            System.out.println(kurtosis.evaluate(target));
//
//            endInsert = System.nanoTime();
//
//            elapsedTimeInsert = endInsert - startInsert;
//
//            System.out.println("Moments Sketch");
//            System.out.println(elapsedTimeInsert);
//            System.out.println(TimeUnit.NANOSECONDS.toMicros(elapsedTimeInsert));
//            System.out.println(TimeUnit.NANOSECONDS.toMillis(elapsedTimeInsert));
//
//            startInsert = System.nanoTime();
//
//            for (int i = 0; i < dataSizeAdaptability; i++) {
//                double sampled_value = binD.sample();
//                double sampled_value_2 = gd.sample();
//                //all_data.add(sampled_value);
//                //ddsketch.accept(sampled_value);
//                kllsketch.update((float) sampled_value);
//                kllsketch.update((float) sampled_value_2);
//            }
//
//            endInsert = System.nanoTime();
//
//            elapsedTimeInsert = endInsert - startInsert;
//
//            System.out.println("KLL Sketch");
//            System.out.println(elapsedTimeInsert);
//            System.out.println(TimeUnit.NANOSECONDS.toMicros(elapsedTimeInsert));
//            System.out.println(TimeUnit.NANOSECONDS.toMillis(elapsedTimeInsert));
//
//            System.out.println(kllsketch.toString(true, false));
//
//            double[] percentilesAdaptability = {.05, .25, .50, .75, .90, .95, .98};

            //double[] resultsDDSAdapt = ddsketch.getValuesAtQuantiles(percentilesAdaptability);
            //float[] resultsKllAdapt = kllsketch.getQuantiles(percentilesAdaptability);

            //MomentSolver ms = new MomentSolver(msketch);
            //ms.setGridSize(1024);
            //ms.solve();

            //double[] resultsMomentsAdapt = ms.getQuantiles(percentilesAdaptability);

            //ArrayList<Double> real_percentiles = getPercentiles(all_data, percentilesAdaptability);

            //System.out.print(real_percentiles.get(0) + ", " + real_percentiles.get(1) + ", " +  real_percentiles.get(2) + ", " + real_percentiles.get(3) + ", " + real_percentiles.get(4) + ", " + real_percentiles.get(5) + ", " + real_percentiles.get(6) + "\n");
            //System.out.print(round(resultsMomentsAdapt[0], 4) +  "," + round(resultsMomentsAdapt[1], 4) + "," + round(resultsMomentsAdapt[2], 4) + ", " + round(resultsMomentsAdapt[3], 4) + ", " + round(resultsMomentsAdapt[4], 4) + ", " + round(resultsMomentsAdapt[5], 4) + ", " + round(resultsMomentsAdapt[6], 4) + "\n");
            //System.out.print(round(resultsDDSAdapt[0], 4) + ", " + round(resultsDDSAdapt[1], 4) + ", " + round(resultsDDSAdapt[2], 4) + ", " + round(resultsDDSAdapt[3], 4) + ", " + round(resultsDDSAdapt[4], 4) + ", " + round(resultsDDSAdapt[5], 4) + ", " + round(resultsDDSAdapt[6], 4) + "\n");
            //System.out.print(round(resultsKllAdapt[0], 4) + ", " + round(resultsKllAdapt[1], 4) + ", " + round(resultsKllAdapt[2], 4) + ", " + round(resultsKllAdapt[3], 4) + ", " + round(resultsKllAdapt[4], 4) + ", " + round(resultsKllAdapt[5], 4) + ", " + round(resultsKllAdapt[6], 4) + "\n");


            //FileWriter myWriter = new FileWriter("all_data.txt");
            //for (Double datad: all_data){
            //    myWriter.write(datad.toString() + "\n");
            //}
            // myWriter.close();

            // End adaptability tests
            // *********************************

            // Insertion tests

            if (runMode == 2) {
                System.out.println("=========== Starting insert time tests ==============");

                ArrayList<Integer> dataSizes = new ArrayList<>(4);
                dataSizes.add(10_000_000);
                dataSizes.add(10_000_000);
                dataSizes.add(10_000_000);
                dataSizes.add(10_000_000);
                dataSizes.add(10_000_000);
                dataSizes.add(100_000_000);

                FileWriter insertWriter = new FileWriter("insert_times.txt");

                for (Integer dataSize : dataSizes) {
                    long[] insertResults = new long[5];
                    ddsketch = new DDSketch(DDS_PARAM_RELATIVE_ACCURACY);
                    uddsketch = new UniformDDSketch(UDDS_PARAM_MAX_NUM_BUCKETS, alphaZero);
                    kllsketch = new KllFloatsSketch(KLL_PARAM_K);
                    momentSketch = new SimpleMomentSketch(MOMEMNTS_PARAM_K);
                    reqSketch =
                        ReqSketch.builder().setK(REQ_PARAM_K).setHighRankAccuracy(REQ_PARAM_HIGH_RANK_ACCURACY)
                            .setLessThanOrEqual(REQ_PARAM_LT_EQ).build();

                    long startInsert = System.nanoTime();

                    for (int i = 0; i < dataSize; i++) {
                        double sampled_value = ptoD.sample();
                        momentSketch.add(sampled_value);
                    }

                    long endInsert = System.nanoTime();

                    long elapsedTimeInsertNanos = endInsert - startInsert;
                    long elapsedTimeInsertMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeInsertNanos);
                    System.out.println(
                        "MomentSketch - Insert time [" + dataSize + "] (micros): " + elapsedTimeInsertMicros);
                    System.out.println(
                        "MomentSketch - Insert time [" + dataSize + "] (nanos): " + elapsedTimeInsertNanos);
                    insertResults[0] = elapsedTimeInsertMicros;

                    startInsert = System.nanoTime();

                    for (int i = 0; i < dataSize; i++) {
                        double sampled_value = ptoD.sample();
                        ddsketch.accept(sampled_value);
                    }

                    endInsert = System.nanoTime();

                    elapsedTimeInsertNanos = endInsert - startInsert;
                    elapsedTimeInsertMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeInsertNanos);
                    System.out.println(
                        "DDSketch - Insert time [" + dataSize + "] (micros): " + elapsedTimeInsertMicros);
                    System.out.println("DDSketch - Insert time [" + dataSize + "] (nanos): " + elapsedTimeInsertNanos);
                    insertResults[1] = elapsedTimeInsertMicros;

                    startInsert = System.nanoTime();

                    for (int i = 0; i < dataSize; i++) {
                        double sampled_value = ptoD.sample();
                        kllsketch.update((float) sampled_value);
                    }

                    endInsert = System.nanoTime();

                    elapsedTimeInsertNanos = endInsert - startInsert;
                    elapsedTimeInsertMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeInsertNanos);
                    System.out.println(
                        "KLLSketch - Insert time [" + dataSize + "] (micros): " + elapsedTimeInsertMicros);
                    System.out.println("KLLSketch - Insert time [" + dataSize + "] (nanos): " + elapsedTimeInsertNanos);
                    insertResults[2] = elapsedTimeInsertMicros;

                    // REQ
                    startInsert = System.nanoTime();

                    for (int i = 0; i < dataSize; i++) {
                        double sampled_value = ptoD.sample();
                        reqSketch.update((float) sampled_value);
                    }

                    endInsert = System.nanoTime();

                    elapsedTimeInsertNanos = endInsert - startInsert;
                    elapsedTimeInsertMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeInsertNanos);
                    System.out.println(
                        "REQSketch - Insert time [" + dataSize + "] (micros): " + elapsedTimeInsertMicros);
                    System.out.println("REQSketch - Insert time [" + dataSize + "] (nanos): " + elapsedTimeInsertNanos);
                    insertResults[3] = elapsedTimeInsertMicros;

                    // UDDS
                    startInsert = System.nanoTime();

                    for (int i = 0; i < dataSize; i++) {
                        double sampled_value = ptoD.sample();
                        uddsketch.accept(sampled_value);
                    }

                    endInsert = System.nanoTime();

                    elapsedTimeInsertNanos = endInsert - startInsert;
                    elapsedTimeInsertMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeInsertNanos);
                    System.out.println(
                        "UDDSketch - Insert time [" + dataSize + "] (micros): " + elapsedTimeInsertMicros);
                    System.out.println("UDDSketch - Insert time [" + dataSize + "] (nanos): " + elapsedTimeInsertNanos);
                    insertResults[4] = elapsedTimeInsertMicros;

                    insertWriter.write(dataSize + "," + Arrays.toString(insertResults) + "\n");
                }
                insertWriter.close();
            }
            // End of insertion tests

            // Real World Workloads
            if (CALC_NYT_STATS) {
                calcNYTDatasetStats();
            }
            if (CALC_POWER_STATS) {
                calcPowerDatasetStats();
            }

            if (runMode == 1) {
                ArrayList<Integer> dataSizes = new ArrayList<>(4);
                dataSizes.add(1_000_000);
                dataSizes.add(1_000_000);
                dataSizes.add(10_000_000);
                dataSizes.add(100_000_000);
                dataSizes.add(1000_000_000);

                FileWriter myWriter = new FileWriter("query_times.txt");

                for (Integer dataSize : dataSizes) {

                    ddsketch = new DDSketch(DDS_PARAM_RELATIVE_ACCURACY);
                    uddsketch = new UniformDDSketch(UDDS_PARAM_MAX_NUM_BUCKETS, alphaZero);
                    kllsketch = new KllFloatsSketch(KLL_PARAM_K);
                    momentSketch = new SimpleMomentSketch(MOMEMNTS_PARAM_K);
                    reqSketch =
                        ReqSketch.builder().setK(REQ_PARAM_K).setHighRankAccuracy(REQ_PARAM_HIGH_RANK_ACCURACY)
                            .setLessThanOrEqual(REQ_PARAM_LT_EQ).build();

                    /* PARETO GENERATOR
                     */
                    ArrayList<Double> ad3 = new ArrayList<Double>();
                    for (int i = 0; i < dataSize; i++) {
                        double sampled_value = ptoD.sample();
                        ddsketch.accept(sampled_value);
                        kllsketch.update((float) sampled_value);
                        momentSketch.add(sampled_value);
                        reqSketch.update((float) sampled_value);
                        uddsketch.accept(sampled_value);
                        ad3.add(sampled_value);
                    }
                    System.out.println("======= Pareto min max ===========");
                    System.out.println(Collections.min(ad3));
                    System.out.println(Collections.max(ad3));

/*
                // UNIFORM GENERATOR //
                ArrayList<DDSketch> ddSketches = new ArrayList<DDSketch>();
                UniformRealDistribution uD = new UniformRealDistribution(1, 5000);
                for (int i = 0; i < dataSize; i++) {
                    ddsketch.accept(uD.sample());
                    kllsketch.update((float) uD.sample());
                    msketch.add(uD.sample());
                }

                // NORMAL GENERATOR //
                NormalDistribution valnD = new NormalDistribution(100, 15);
                for (int i = 0; i < dataSize; i++) {
                    ddsketch.accept(valnD.sample());
                    // kllsketch.update(valnD.sample());
                    // msketch.add(valnD.sample());
                }
*/

                    // Query time test
                    System.out.println("=========== Starting query time tests ==============");
                    long[] queryResults = new long[5];

                    // Moments
                    long start = System.nanoTime();

                    double[] results_moments = momentSketch.getQuantiles(percentiles);

                    long end = System.nanoTime();
                    long elapsedTimeNanos = end - start;
                    long elapsedTimeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeNanos);

                    System.out.println(
                        "MomentSketch - Query time [" + dataSize + "] (micros): " + elapsedTimeMicros);
                    System.out.println("MomentSketch - Query time [" + dataSize + "] (nanos): " + elapsedTimeNanos);
                    queryResults[0] = elapsedTimeMicros;

                    // DDS
                    start = System.nanoTime();

                    double[] results_ddsketch = ddsketch.getValuesAtQuantiles(percentiles);

                    end = System.nanoTime();
                    elapsedTimeNanos = end - start;

                    elapsedTimeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeNanos);
                    System.out.println(
                        "DDSketch - Query time [" + dataSize + "] (micros): " + elapsedTimeMicros);
                    System.out.println("DDSketch - Query time [" + dataSize + "] (nanos): " + elapsedTimeNanos);
                    queryResults[1] = elapsedTimeMicros;

                    // KLL
                    start = System.nanoTime();

                    float[] results_kll = kllsketch.getQuantiles(percentiles);

                    end = System.nanoTime();
                    elapsedTimeNanos = end - start;

                    elapsedTimeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeNanos);
                    System.out.println(
                        "KLLSketch - Query time [" + dataSize + "] (micros): " + elapsedTimeMicros);
                    System.out.println("KLLSketch - Query time [" + dataSize + "] (nanos): " + elapsedTimeNanos);
                    queryResults[2] = elapsedTimeMicros;

                    // REQ
                    start = System.nanoTime();

                    float[] results_reqsketch = reqSketch.getQuantiles(percentiles);

                    end = System.nanoTime();
                    elapsedTimeNanos = end - start;

                    elapsedTimeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeNanos);
                    System.out.println(
                        "REQSketch - Query time [" + dataSize + "] (micros): " + elapsedTimeMicros);
                    System.out.println("REQSketch - Query time [" + dataSize + "] (nanos): " + elapsedTimeNanos);
                    queryResults[3] = elapsedTimeMicros;

                    // UDDS
                    start = System.nanoTime();

                    double[] results_uddsketch = uddsketch.getValuesAtQuantiles(percentiles);

                    end = System.nanoTime();
                    elapsedTimeNanos = end - start;

                    elapsedTimeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeNanos);
                    System.out.println(
                        "UDDSketch - Query time [" + dataSize + "] (micros): " + elapsedTimeMicros);
                    System.out.println("UDDSketch - Query time [" + dataSize + "] (nanos): " + elapsedTimeNanos);
                    queryResults[4] = elapsedTimeMicros;

                    // Requires a sorted set of values
                    //int[] percentiles = {1, 5, 25, 50, 75, 90, 95, 98, 99};
                    for (int i = 0; i < percentiles.length; i++) {
                        System.out.println("MomentsSketch [" + percentiles[i] + "]: " + results_moments[i]);
                        System.out.println("DDSketch [" + percentiles[i] + "]: " + results_ddsketch[i]);
                        System.out.println("KLLSketch [" + percentiles[i] + "]: " + results_kll[i]);
                        System.out.println("REQSketch [" + percentiles[i] + "]: " + results_reqsketch[i]);
                        System.out.println("UDDSketch [" + percentiles[i] + "]: " + results_uddsketch[i]);
                    }
                    myWriter.write(dataSize + "," + Arrays.toString(queryResults) + "\n");
                }
                myWriter.close();
            }
        } catch (FileNotFoundException e) {
            System.out.println("File not found exception occurred.");
            e.printStackTrace();
        } catch (IOException e) {
            System.out.println("IO exception occurred.");
            e.printStackTrace();
        }

    }

    private static void runKTests() {
        System.out.println("Get K");
        System.out.println(KllFloatsSketch.getKFromEpsilon(0.01, true));
        System.out.println(KllFloatsSketch.getKFromEpsilon(0.01, false));
        System.out.println("End K");
    }

    private static void calcPowerDatasetStats() throws IOException {
        String line;
        /* POWER ACTUAL_POWER GENERATOR */
        String delimiterPowerDataset = ";";
        BufferedReader br =
            new BufferedReader(new FileReader("/home/m34ferna/flink-benchmarks/household_power_consumption.txt"));
        ArrayList<Double> valuesRead = new ArrayList<Double>();
        while ((line = br.readLine()) != null) {
            String[] line_array = line.split(delimiterPowerDataset);    // use comma as separator
            double globalActivePower = Double.parseDouble(line_array[2]);
            valuesRead.add(globalActivePower);

            //ddsketch.accept(d);

            //kllsketch.update(d;

            //msketch.add(d);
        }

        System.out.println("power min max");
        System.out.println(Collections.min(valuesRead));
        System.out.println(Collections.max(valuesRead));
    }

    private static void calcNYTDatasetStats() throws IOException {
        /* NYT FARES GENERATOR */
        String line = "";
        String splitBy = ",";
        BufferedReader br = new BufferedReader(new FileReader("/home/m34ferna/flink-benchmarks/nyt-data.csv"));
        ArrayList<Double> valuesRead = new ArrayList<Double>();
        double min = 0;
        double max = 0;
        while ((line = br.readLine()) != null) {
            String[] line_array = line.split(splitBy);    // use comma as separator
            double totalAmount = Double.parseDouble(line_array[10]);
            //ddsketch.accept(d);
            //kllsketch.update(d;
            //msketch.add(d);
            valuesRead.add(totalAmount);
        }
        System.out.println("NYT min max");
        System.out.println(Collections.min(valuesRead));
        System.out.println(Collections.max(valuesRead));
    }

    private static void runMergeTests(int dataSize, int numSketches) {
        /*
        // MERGE TESTS
        ****************************************************
         */

        UniformRealDistribution uD2 = new UniformRealDistribution(1, 5000);
        NormalDistribution valnD2 = new NormalDistribution(100, 15);
        BinomialDistribution ptoD2 = new BinomialDistribution(100, 0.2);

        ArrayList<DDSketch> ddSketches = new ArrayList<DDSketch>();
        ArrayList<KllFloatsSketch> kllFloatsSketches = new ArrayList<KllFloatsSketch>();
        ArrayList<MomentStruct> momentSketches = new ArrayList<MomentStruct>();

        for (int i = 0; i < numSketches; i++) {
            DDSketch ddSketch = new DDSketch(0.01);
            ddSketches.add(ddSketch);
            KllFloatsSketch kllFloatsSketch = new KllFloatsSketch();
            kllFloatsSketches.add(kllFloatsSketch);
            MomentStruct momentStruct = new MomentStruct(15);
            momentSketches.add(momentStruct);
            kllFloatsSketches.add(kllFloatsSketch);

        }

        int k = 0;
        for (int i = 0; i < 100; i++) {
            DDSketch ds = ddSketches.get(i);
            KllFloatsSketch kll = kllFloatsSketches.get(i);
            MomentStruct momsketch = momentSketches.get(i);

            if ((k % 3) == 0) {
                for (int j = 0; j < dataSize; j++) {
                    double sampled_value = uD2.sample();
                    ds.accept(sampled_value);
                    kll.update((float) sampled_value);
                    momsketch.add(sampled_value);
                }
            }
            if ((k % 3) == 1) {
                for (int j = 0; j < dataSize; j++) {
                    double sampled_value = valnD2.sample();
                    ds.accept(sampled_value);
                    kll.update((float) sampled_value);
                    momsketch.add(sampled_value);
                }
            }
            if ((k % 3) == 2) {
                for (int j = 0; j < dataSize; j++) {
                    double sampled_value = ptoD2.sample();
                    ds.accept(sampled_value);
                    kll.update((float) sampled_value);
                    momsketch.add(sampled_value);
                }
            }
            k++;
        }

        DDSketch orig_dds = ddSketches.get(0);
        KllFloatsSketch orig_kll = kllFloatsSketches.get(0);
        MomentStruct orig_moments = momentSketches.get(0);
        long startMerge = System.nanoTime();

        for (int i = 1; i < 10; i++) {
            orig_dds.mergeWith(ddSketches.get(i));
        }

        long endMerge = System.nanoTime();

        long elapsedTimeMerge = endMerge - startMerge;
        System.out.println("DDSketch");
        System.out.println(TimeUnit.NANOSECONDS.toMicros(elapsedTimeMerge));
        System.out.println(elapsedTimeMerge);

        startMerge = System.nanoTime();

        for (int i = 1; i < 10; i++) {
            orig_kll.merge(kllFloatsSketches.get(i));
        }

        endMerge = System.nanoTime();
        elapsedTimeMerge = endMerge - startMerge;
        System.out.println("KLLSketch");
        System.out.println(TimeUnit.NANOSECONDS.toMicros(elapsedTimeMerge));
        System.out.println(elapsedTimeMerge);

        startMerge = System.nanoTime();

        for (int i = 1; i < 10; i++) {
            orig_moments.merge(momentSketches.get(i));
        }

        endMerge = System.nanoTime();
        elapsedTimeMerge = endMerge - startMerge;
        System.out.println("Moments Sketch");
        System.out.println(TimeUnit.NANOSECONDS.toMicros(elapsedTimeMerge));
        System.out.println(elapsedTimeMerge);
    }

    private static void initTestSketches() {
        System.out.println("======= Running initial testing of sketches ===========");
        SimpleMomentSketch ex_ms = new SimpleMomentSketch(18);
        DDSketch ex_dds = new DDSketch(0.01);
        UniformDDSketch ex_udds = new UniformDDSketch(1024, 0.001);

        double[] example_data = {1, 2, 3, 4, 5, 6, 7, 18, 30, 51};
        for (int i = 0; i < example_data.length; i++) {
            ex_ms.add(example_data[i]);
            ex_dds.accept(example_data[i]);
            ex_udds.accept(example_data[i]);
        }

        double[] ex_quantiles = {0.8, 0.9};

        System.out.println("Example data output");
        System.out.println("Moments Sketch 0.8: " + ex_ms.getQuantiles(ex_quantiles)[0]);
        System.out.println("Moments Sketch 0.9: " + ex_ms.getQuantiles(ex_quantiles)[1]);
        System.out.println("DDSketch: " + ex_dds.getValuesAtQuantiles(ex_quantiles)[0]);
        System.out.println("DDSketch: " + ex_dds.getValuesAtQuantiles(ex_quantiles)[1]);
        System.out.println("UDDSketch: " + ex_dds.getValuesAtQuantiles(ex_quantiles)[0]);
        System.out.println("UDDSketch: " + ex_dds.getValuesAtQuantiles(ex_quantiles)[1]);

        System.out.println(ex_udds);
        System.out.println("======= End initial testing of sketches ===========");
    }

    static private double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        BigDecimal bd = new BigDecimal(Double.toString(value));
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }

    public static Double percentile(ArrayList<Double> sort_values, double percentile) {
        assert (percentile > 0 && percentile < 1);
        int index = (int) Math.ceil(percentile * sort_values.size());
        return sort_values.get(index - 1);
    }

    public static ArrayList<Double> getPercentiles(ArrayList<Double> all_events, double[] percentiles) {
        ArrayList<Double> ret_list = new ArrayList<Double>();
        Collections.sort(all_events);
        for (double p : percentiles) {
            ret_list.add(round(percentile(all_events, p), 4));
        }
        return ret_list;
    }


}
