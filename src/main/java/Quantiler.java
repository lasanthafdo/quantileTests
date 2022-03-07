import com.datadoghq.sketch.ddsketch.DDSketch;
import com.datadoghq.sketch.ddsketch.mapping.LogarithmicMapping;
import com.datadoghq.sketch.ddsketch.store.CollapsingLowestDenseStore;
import com.datadoghq.sketch.uddsketch.UniformDDSketch;
import com.github.stanfordfuturedata.momentsketch.SimpleMomentSketch;
import org.apache.commons.math3.distribution.*;
import org.apache.commons.math3.stat.descriptive.moment.Kurtosis;
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

    public static final int KLL_PARAM_K = 350;
    public static final int MOMEMNTS_PARAM_K = 12;
    public static final int UDDS_PARAM_MAX_NUM_BUCKETS = 1024;
    public static final int UDDS_PARAM_K = 12;
    public static final double DDS_PARAM_RELATIVE_ACCURACY = 0.01;
    public static final double UDDS_PARAM_RELATIVE_ACCURACY = 0.01;
    public static final int REQ_PARAM_K = 30;
    public static final boolean REQ_PARAM_HIGH_RANK_ACCURACY = true;
    public static final boolean REQ_PARAM_LT_EQ = false;
    public static final boolean CALC_POWER_STATS = false;
    public static final boolean CALC_NYT_STATS = false;
    public static final boolean RUN_K_TESTS = false;
    public static final boolean RUN_INIT_TESTS = false;
    public static final boolean PRINT_QUERY_RESULTS = false;
    public static final boolean RUN_KURT_POWER = true;
    public static final boolean RUN_KURT_NYT = true;
    private static final boolean RUN_UDDS_TESTS = false;
    public static final boolean MOMENTS_PARAM_COMPRESSED = true;

    public static void main(String[] args) {
        int runMode = Integer.parseInt(args[0]);
        try {
            double[] percentiles = {0.01, 0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.98, 0.99};
            double alphaZero =
                Math.tanh(FastMath.atanh(UDDS_PARAM_RELATIVE_ACCURACY) / Math.pow(2.0, UDDS_PARAM_K - 1));

            // Sketch Algorithms
            DDSketch ddsketch = new DDSketch(DDS_PARAM_RELATIVE_ACCURACY);
            DDSketch ddSketchCollapsing = new DDSketch(new LogarithmicMapping(DDS_PARAM_RELATIVE_ACCURACY),
                () -> new CollapsingLowestDenseStore(UDDS_PARAM_MAX_NUM_BUCKETS));
            UniformDDSketch uddsketch = new UniformDDSketch(UDDS_PARAM_MAX_NUM_BUCKETS, alphaZero);
            KllFloatsSketch kllsketch = new KllFloatsSketch(KLL_PARAM_K);
            SimpleMomentSketch momentSketch = new SimpleMomentSketch(MOMEMNTS_PARAM_K);
            momentSketch.setCompressed(MOMENTS_PARAM_COMPRESSED);
            ReqSketch reqSketch =
                ReqSketch.builder().setK(REQ_PARAM_K).setHighRankAccuracy(REQ_PARAM_HIGH_RANK_ACCURACY)
                    .setLessThanOrEqual(REQ_PARAM_LT_EQ).build();

            if (RUN_K_TESTS) {
                runKTests();
            }

            if (RUN_INIT_TESTS) {
                initTestSketches();
            }

            if (RUN_UDDS_TESTS) {
                runUDDSTests();
            }
            int dataSizeMerge = 1_000_000;
            if (runMode == 3) {
                ArrayList<Integer> numSketchesList = new ArrayList<>();
                numSketchesList.add(10);
                numSketchesList.add(100);
                numSketchesList.add(1000);
                runMergeTests(dataSizeMerge, numSketchesList);
            }


            int dataSizeAdaptability = 1_000_000;
            int dataSizeKurtosis = 1_000_000;
            ArrayList<Double> all_data = new ArrayList<>();

            // Synthetic Workloads
            BinomialDistribution binD = new BinomialDistribution(30, 0.4);
            System.out.println("Binomial Distribution mean: " + binD.getNumericalMean());

            ZipfDistribution zipD = new ZipfDistribution(20, 0.6);
            System.out.println("Zipf mean: " + zipD.getNumericalMean());

            GammaDistribution gd = new GammaDistribution(2, 0.4);
            System.out.println("Gamma mean: " + gd.getNumericalMean());

            UniformRealDistribution uD = new UniformRealDistribution(40, 100);
            System.out.println("Uniform mean: " + uD.getNumericalMean());

            UniformRealDistribution unif_adapt_test = new UniformRealDistribution(30, 100);
            System.out.println("Uniform (Adapt) mean: " + unif_adapt_test.getNumericalMean());

            ParetoDistribution ptoD = new ParetoDistribution(1, 1);
            System.out.println("Pareto mean: " + ptoD.getNumericalMean());

            // *********************************
            // Adaptability tests
            if (runMode == 4) {
                System.out.println(
                    "======= Running adaptability tests with data size : " + dataSizeAdaptability + " x 2 =========");
                FileWriter adaptabilityWriter = new FileWriter("adaptability_data.txt");
                for (int iter = 0; iter < 6; iter++) {
                    ddsketch = new DDSketch(DDS_PARAM_RELATIVE_ACCURACY);
                    ddSketchCollapsing = new DDSketch(new LogarithmicMapping(DDS_PARAM_RELATIVE_ACCURACY),
                        () -> new CollapsingLowestDenseStore(UDDS_PARAM_MAX_NUM_BUCKETS));
                    uddsketch = new UniformDDSketch(UDDS_PARAM_MAX_NUM_BUCKETS, alphaZero);
                    kllsketch = new KllFloatsSketch(KLL_PARAM_K);
                    momentSketch = new SimpleMomentSketch(MOMEMNTS_PARAM_K);
                    momentSketch.setCompressed(MOMENTS_PARAM_COMPRESSED);
                    reqSketch =
                        ReqSketch.builder().setK(REQ_PARAM_K).setHighRankAccuracy(REQ_PARAM_HIGH_RANK_ACCURACY)
                            .setLessThanOrEqual(REQ_PARAM_LT_EQ).build();

                    long startInsert = System.nanoTime();

                    for (int i = 0; i < dataSizeAdaptability; i++) {
                        double sampled_value = binD.sample();
                        all_data.add(sampled_value);
                        momentSketch.add(sampled_value);
                        ddsketch.accept(sampled_value);
                        ddSketchCollapsing.accept(sampled_value);
                        kllsketch.update((float) sampled_value);
                        reqSketch.update((float) sampled_value);
                        uddsketch.accept(sampled_value);
                    }

                    for (int i = 0; i < dataSizeAdaptability; i++) {
                        double sampled_value_2 = unif_adapt_test.sample();
                        all_data.add(sampled_value_2);
                        momentSketch.add(sampled_value_2);
                        ddsketch.accept(sampled_value_2);
                        ddSketchCollapsing.accept(sampled_value_2);
                        kllsketch.update((float) sampled_value_2);
                        reqSketch.update((float) sampled_value_2);
                        uddsketch.accept(sampled_value_2);
                    }

                    long endInsert = System.nanoTime();

                    long elapsedTimeInsert = endInsert - startInsert;
                    System.out.println("Insert time - nanos : " + elapsedTimeInsert);
                    System.out.println(
                        "Insert time - micros : " + TimeUnit.NANOSECONDS.toMicros(elapsedTimeInsert));
                    System.out.println(
                        "Insert time - millis : " + TimeUnit.NANOSECONDS.toMillis(elapsedTimeInsert));

                    double[] percentilesAdaptability = {.01, .05, .25, .50, .75, .90, .95, .98, .99};

                    double[] resultsMomentsAdapt = momentSketch.getQuantiles(percentilesAdaptability);
                    double[] resultsDDSAdapt = ddsketch.getValuesAtQuantiles(percentilesAdaptability);
                    double[] resultsDDSCollapsingAdapt =
                        ddSketchCollapsing.getValuesAtQuantiles(percentilesAdaptability);
                    float[] resultsKllAdapt = kllsketch.getQuantiles(percentilesAdaptability);
                    float[] resultsReqAdapt = reqSketch.getQuantiles(percentilesAdaptability);
                    double[] resultsUDDSAdapt = uddsketch.getValuesAtQuantiles(percentilesAdaptability);

                    ArrayList<Double> realPercentiles = getPercentiles(all_data, percentilesAdaptability);

                    System.out.println("Results for percentiles: " + Arrays.toString(percentiles));
                    System.out.print(
                        "Real:" + realPercentiles.get(0) + ", " + realPercentiles.get(1) + ", " +
                            realPercentiles.get(2) +
                            ", " +
                            realPercentiles.get(3) + ", " + realPercentiles.get(4) + ", " + realPercentiles.get(5) +
                            ", " + realPercentiles.get(6) + "\n");
                    System.out.print(
                        "Moments:" + round(resultsMomentsAdapt[0], 4) + "," + round(resultsMomentsAdapt[1], 4) + "," +
                            round(resultsMomentsAdapt[2], 4) + ", " + round(resultsMomentsAdapt[3], 4) + ", " +
                            round(resultsMomentsAdapt[4], 4) + ", " + round(resultsMomentsAdapt[5], 4) + ", " +
                            round(resultsMomentsAdapt[6], 4) + "\n");
                    System.out.print(
                        "DDS:" + round(resultsDDSAdapt[0], 4) + ", " + round(resultsDDSAdapt[1], 4) + ", " +
                            round(resultsDDSAdapt[2], 4) + ", " + round(resultsDDSAdapt[3], 4) + ", " +
                            round(resultsDDSAdapt[4], 4) + ", " + round(resultsDDSAdapt[5], 4) + ", " +
                            round(resultsDDSAdapt[6], 4) + "\n");
                    System.out.print(
                        "DDSC:" + round(resultsDDSCollapsingAdapt[0], 4) + ", " +
                            round(resultsDDSCollapsingAdapt[1], 4) +
                            ", " +
                            round(resultsDDSCollapsingAdapt[2], 4) + ", " + round(resultsDDSCollapsingAdapt[3], 4) +
                            ", " +
                            round(resultsDDSCollapsingAdapt[4], 4) + ", " + round(resultsDDSCollapsingAdapt[5], 4) +
                            ", " +
                            round(resultsDDSCollapsingAdapt[6], 4) + "\n");
                    System.out.print(
                        "KLL:" + round(resultsKllAdapt[0], 4) + ", " + round(resultsKllAdapt[1], 4) + ", " +
                            round(resultsKllAdapt[2], 4) + ", " + round(resultsKllAdapt[3], 4) + ", " +
                            round(resultsKllAdapt[4], 4) + ", " + round(resultsKllAdapt[5], 4) + ", " +
                            round(resultsKllAdapt[6], 4) + "\n");
                    System.out.print(
                        "REQ:" + round(resultsReqAdapt[0], 4) + ", " + round(resultsReqAdapt[1], 4) + ", " +
                            round(resultsReqAdapt[2], 4) + ", " + round(resultsReqAdapt[3], 4) + ", " +
                            round(resultsReqAdapt[4], 4) + ", " + round(resultsReqAdapt[5], 4) + ", " +
                            round(resultsReqAdapt[6], 4) + "\n");
                    System.out.print(
                        "UDDS:" + round(resultsUDDSAdapt[0], 4) + ", " + round(resultsUDDSAdapt[1], 4) + ", " +
                            round(resultsUDDSAdapt[2], 4) + ", " + round(resultsUDDSAdapt[3], 4) + ", " +
                            round(resultsUDDSAdapt[4], 4) + ", " + round(resultsUDDSAdapt[5], 4) + ", " +
                            round(resultsUDDSAdapt[6], 4) + "\n");

                    for (int i = 0; i < realPercentiles.size(); i++) {
                        adaptabilityWriter.write(percentilesAdaptability[i] + "," +
                            round(realPercentiles.get(i), 4) + "," + round(resultsMomentsAdapt[i], 4) + "," +
                            round(resultsDDSAdapt[i], 4) + "," + round(resultsDDSCollapsingAdapt[i], 4) + "," +
                            round(resultsKllAdapt[i], 4) + "," + round(resultsReqAdapt[i], 4) + "," +
                            round(resultsUDDSAdapt[i], 4) + "\n");
                    }
                }
                adaptabilityWriter.close();
                System.out.println("======== End of adaptability tests ========");
            }
            // End adaptability tests
            // *********************************

            // Kurtosis test
            if (runMode == 5) {
                System.out.println(
                    "======= Running kurtosis tests with data size : " + dataSizeKurtosis + " =========");

                ArrayList<Double> actualKurtosisData = new ArrayList<>();
                double percentileOfInterest = 0.98;
                int numIters = 6;
                double realQ, momentsQ, ddsQ, ddscQ, kllQ, reqQ, uddsQ;
                String resultString;
                FileWriter kurtosisWriter = new FileWriter("kurtosis_data.txt");

                // Uniform
                for (int i = 0; i < dataSizeKurtosis; i++) {
                    double sampled_value = uD.sample();
                    actualKurtosisData.add(sampled_value);
                }

                Kurtosis kurtosis = new Kurtosis();

                double[] target = new double[actualKurtosisData.size()];
                for (int i = 0; i < target.length; i++) {
                    target[i] = actualKurtosisData.get(i);                // java 1.5+ style (outboxing)
                }

                double kurtosisVal = round(kurtosis.evaluate(target), 4);
                System.out.println("Kurtosis (Uniform) : " + kurtosisVal);

                for (int iter = 0; iter < numIters; iter++) {
                    ddsketch = new DDSketch(DDS_PARAM_RELATIVE_ACCURACY);
                    ddSketchCollapsing = new DDSketch(new LogarithmicMapping(DDS_PARAM_RELATIVE_ACCURACY),
                        () -> new CollapsingLowestDenseStore(UDDS_PARAM_MAX_NUM_BUCKETS));
                    uddsketch = new UniformDDSketch(UDDS_PARAM_MAX_NUM_BUCKETS, alphaZero);
                    kllsketch = new KllFloatsSketch(KLL_PARAM_K);
                    momentSketch = new SimpleMomentSketch(MOMEMNTS_PARAM_K);
                    momentSketch.setCompressed(MOMENTS_PARAM_COMPRESSED);
                    reqSketch =
                        ReqSketch.builder().setK(REQ_PARAM_K).setHighRankAccuracy(REQ_PARAM_HIGH_RANK_ACCURACY)
                            .setLessThanOrEqual(REQ_PARAM_LT_EQ).build();

                    for (int i = 0; i < dataSizeKurtosis; i++) {
                        double sampledValue = actualKurtosisData.get(i);
                        momentSketch.add(sampledValue);
                        ddsketch.accept(sampledValue);
                        ddSketchCollapsing.accept(sampledValue);
                        kllsketch.update((float) sampledValue);
                        reqSketch.update((float) sampledValue);
                        uddsketch.accept(sampledValue);
                    }

                    Collections.sort(actualKurtosisData);
                    realQ = round(percentile(actualKurtosisData, percentileOfInterest), 4);
                    momentsQ = round(momentSketch.getQuantiles(new double[]{percentileOfInterest})[0], 4);
                    ddsQ = round(ddsketch.getValueAtQuantile(percentileOfInterest), 4);
                    ddscQ = round(ddSketchCollapsing.getValueAtQuantile(percentileOfInterest), 4);
                    kllQ = round(kllsketch.getQuantile(percentileOfInterest), 4);
                    reqQ = round(reqSketch.getQuantile(percentileOfInterest), 4);
                    uddsQ = round(uddsketch.getValueAtQuantile(percentileOfInterest), 4);
                    resultString =
                        "Uniform," + kurtosisVal + "," + realQ + "," + momentsQ + "," + ddsQ + "," + ddscQ + "," +
                            kllQ +
                            "," + reqQ + "," + uddsQ;
                    System.out.println(resultString);
                    kurtosisWriter.write(resultString + "\n");
                }

                // Pareto
                actualKurtosisData.clear();

                for (int i = 0; i < dataSizeKurtosis; i++) {
                    double sampled_value = ptoD.sample();
                    actualKurtosisData.add(sampled_value);
                }

                kurtosis = new Kurtosis();

                target = new double[actualKurtosisData.size()];
                for (int i = 0; i < target.length; i++) {
                    target[i] = actualKurtosisData.get(i);                // java 1.5+ style (outboxing)
                }

                kurtosisVal = round(kurtosis.evaluate(target), 4);
                System.out.println("Kurtosis (Pareto) : " + kurtosisVal);

                for (int iter = 0; iter < numIters; iter++) {
                    ddsketch = new DDSketch(DDS_PARAM_RELATIVE_ACCURACY);
                    ddSketchCollapsing = new DDSketch(new LogarithmicMapping(DDS_PARAM_RELATIVE_ACCURACY),
                        () -> new CollapsingLowestDenseStore(UDDS_PARAM_MAX_NUM_BUCKETS));
                    uddsketch = new UniformDDSketch(UDDS_PARAM_MAX_NUM_BUCKETS, alphaZero);
                    kllsketch = new KllFloatsSketch(KLL_PARAM_K);
                    momentSketch = new SimpleMomentSketch(MOMEMNTS_PARAM_K);
                    momentSketch.setCompressed(MOMENTS_PARAM_COMPRESSED);
                    reqSketch =
                        ReqSketch.builder().setK(REQ_PARAM_K).setHighRankAccuracy(REQ_PARAM_HIGH_RANK_ACCURACY)
                            .setLessThanOrEqual(REQ_PARAM_LT_EQ).build();

                    for (int i = 0; i < dataSizeKurtosis; i++) {
                        double sampledValue = actualKurtosisData.get(i);
                        momentSketch.add(Math.log(sampledValue));
                        ddsketch.accept(sampledValue);
                        ddSketchCollapsing.accept(sampledValue);
                        kllsketch.update((float) sampledValue);
                        reqSketch.update((float) sampledValue);
                        uddsketch.accept(sampledValue);
                    }

                    Collections.sort(actualKurtosisData);
                    realQ = round(percentile(actualKurtosisData, percentileOfInterest), 4);
                    momentsQ = round(Math.exp(momentSketch.getQuantiles(new double[]{percentileOfInterest})[0]), 4);
                    ddsQ = round(ddsketch.getValueAtQuantile(percentileOfInterest), 4);
                    ddscQ = round(ddSketchCollapsing.getValueAtQuantile(percentileOfInterest), 4);
                    kllQ = round(kllsketch.getQuantile(percentileOfInterest), 4);
                    reqQ = round(reqSketch.getQuantile(percentileOfInterest), 4);
                    uddsQ = round(uddsketch.getValueAtQuantile(percentileOfInterest), 4);
                    resultString =
                        "Pareto," + kurtosisVal + "," + realQ + "," + momentsQ + "," + ddsQ + "," + ddscQ + "," + kllQ +
                            "," + reqQ + "," + uddsQ;
                    System.out.println(resultString);
                    kurtosisWriter.write(resultString + "\n");
                }

                // Power dataset
                String line;
                if (RUN_KURT_POWER) {
                    actualKurtosisData.clear();

                    String delimiterPowerDataset = ";";
                    BufferedReader brPower =
                        new BufferedReader(
                            new FileReader("/home/m34ferna/flink-benchmarks/household_power_consumption.txt"));
                    while ((line = brPower.readLine()) != null) {
                        String[] line_array = line.split(delimiterPowerDataset);    // use comma as separator
                        double globalActivePower = Double.parseDouble(line_array[2]);
                        actualKurtosisData.add(globalActivePower);
                    }

                    kurtosis = new Kurtosis();

                    target = new double[actualKurtosisData.size()];
                    for (int i = 0; i < target.length; i++) {
                        target[i] = actualKurtosisData.get(i);                // java 1.5+ style (outboxing)
                    }

                    kurtosisVal = round(kurtosis.evaluate(target), 4);
                    System.out.println("Kurtosis (Power) : " + kurtosisVal);

                    for (int iter = 0; iter < numIters; iter++) {
                        ddsketch = new DDSketch(DDS_PARAM_RELATIVE_ACCURACY);
                        ddSketchCollapsing = new DDSketch(new LogarithmicMapping(DDS_PARAM_RELATIVE_ACCURACY),
                            () -> new CollapsingLowestDenseStore(UDDS_PARAM_MAX_NUM_BUCKETS));
                        uddsketch = new UniformDDSketch(UDDS_PARAM_MAX_NUM_BUCKETS, alphaZero);
                        kllsketch = new KllFloatsSketch(KLL_PARAM_K);
                        momentSketch = new SimpleMomentSketch(MOMEMNTS_PARAM_K);
                        momentSketch.setCompressed(MOMENTS_PARAM_COMPRESSED);
                        reqSketch =
                            ReqSketch.builder().setK(REQ_PARAM_K).setHighRankAccuracy(REQ_PARAM_HIGH_RANK_ACCURACY)
                                .setLessThanOrEqual(REQ_PARAM_LT_EQ).build();

                        for (double globalActivePower : actualKurtosisData) {
                            momentSketch.add(Math.log(globalActivePower));
                            ddsketch.accept(globalActivePower);
                            ddSketchCollapsing.accept(globalActivePower);
                            kllsketch.update((float) globalActivePower);
                            reqSketch.update((float) globalActivePower);
                            uddsketch.accept(globalActivePower);
                        }

                        Collections.sort(actualKurtosisData);
                        realQ = round(percentile(actualKurtosisData, percentileOfInterest), 4);
                        momentsQ = round(Math.exp(momentSketch.getQuantiles(new double[]{percentileOfInterest})[0]), 4);
                        ddsQ = round(ddsketch.getValueAtQuantile(percentileOfInterest), 4);
                        ddscQ = round(ddSketchCollapsing.getValueAtQuantile(percentileOfInterest), 4);
                        kllQ = round(kllsketch.getQuantile(percentileOfInterest), 4);
                        reqQ = round(reqSketch.getQuantile(percentileOfInterest), 4);
                        uddsQ = round(uddsketch.getValueAtQuantile(percentileOfInterest), 4);
                        resultString =
                            "Power," + kurtosisVal + "," + realQ + "," + momentsQ + "," + ddsQ + "," + ddscQ + "," +
                                kllQ +
                                "," + reqQ + "," + uddsQ;
                        System.out.println(resultString);
                        kurtosisWriter.write(resultString + "\n");
                    }
                }

                if (RUN_KURT_NYT) {
                    actualKurtosisData.clear();

                    // NYT dataset
                    String splitBy = ",";
                    BufferedReader brNYT =
                        new BufferedReader(new FileReader("/home/m34ferna/flink-benchmarks/nyt-data.csv"));
                    while ((line = brNYT.readLine()) != null) {
                        String[] line_array = line.split(splitBy);    // use comma as separator
                        double totalAmount = Double.parseDouble(line_array[10]);
                        actualKurtosisData.add(totalAmount);
                    }

                    kurtosis = new Kurtosis();

                    target = new double[actualKurtosisData.size()];
                    for (int i = 0; i < target.length; i++) {
                        target[i] = actualKurtosisData.get(i);                // java 1.5+ style (outboxing)
                    }

                    kurtosisVal = round(kurtosis.evaluate(target), 4);
                    System.out.println("Kurtosis (NYT) : " + kurtosisVal);

                    for (int iter = 0; iter < numIters; iter++) {

                        ddsketch = new DDSketch(DDS_PARAM_RELATIVE_ACCURACY);
                        ddSketchCollapsing = new DDSketch(new LogarithmicMapping(DDS_PARAM_RELATIVE_ACCURACY),
                            () -> new CollapsingLowestDenseStore(UDDS_PARAM_MAX_NUM_BUCKETS));
                        uddsketch = new UniformDDSketch(UDDS_PARAM_MAX_NUM_BUCKETS, alphaZero);
                        kllsketch = new KllFloatsSketch(KLL_PARAM_K);
                        momentSketch = new SimpleMomentSketch(MOMEMNTS_PARAM_K);
                        momentSketch.setCompressed(MOMENTS_PARAM_COMPRESSED);
                        reqSketch =
                            ReqSketch.builder().setK(REQ_PARAM_K).setHighRankAccuracy(REQ_PARAM_HIGH_RANK_ACCURACY)
                                .setLessThanOrEqual(REQ_PARAM_LT_EQ).build();

                        for (double totalAmount : actualKurtosisData) {
                            momentSketch.add(totalAmount);
                            ddsketch.accept(totalAmount);
                            ddSketchCollapsing.accept(totalAmount);
                            kllsketch.update((float) totalAmount);
                            reqSketch.update((float) totalAmount);
                            uddsketch.accept(totalAmount);
                        }

                        Collections.sort(actualKurtosisData);
                        realQ = round(percentile(actualKurtosisData, percentileOfInterest), 4);
                        momentsQ = round(momentSketch.getQuantiles(new double[]{percentileOfInterest})[0], 4);
                        ddsQ = round(ddsketch.getValueAtQuantile(percentileOfInterest), 4);
                        ddscQ = round(ddSketchCollapsing.getValueAtQuantile(percentileOfInterest), 4);
                        kllQ = round(kllsketch.getQuantile(percentileOfInterest), 4);
                        reqQ = round(reqSketch.getQuantile(percentileOfInterest), 4);
                        uddsQ = round(uddsketch.getValueAtQuantile(percentileOfInterest), 4);
                        resultString =
                            "NYT," + kurtosisVal + "," + realQ + "," + momentsQ + "," + ddsQ + "," + ddscQ + "," +
                                kllQ +
                                "," + reqQ + "," + uddsQ;
                        System.out.println(resultString);
                        kurtosisWriter.write(resultString + "\n");
                    }
                }

                kurtosisWriter.close();
            }

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
                    momentSketch.setCompressed(MOMENTS_PARAM_COMPRESSED);
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

            // Query tests
            if (runMode == 1 || runMode == 6) {
                ArrayList<Integer> dataSizes = new ArrayList<>(4);

                int numOneMilIters = 1000;
                for (int i = 0; i < numOneMilIters; i++) {
                    dataSizes.add(1_000_000);
                }

                int numTenMilIters = 100;
                for (int i = 0; i < numTenMilIters; i++) {
                    dataSizes.add(10_000_000);
                }

                int numHundredMilIters = 20;
                for (int i = 0; i < numHundredMilIters; i++) {
                    dataSizes.add(100_000_000);
                }

                int numThousandMilIters = 5;
                for (int i = 0; i < numThousandMilIters; i++) {
                    dataSizes.add(1_000_000_000);
                }

                if (runMode == 6) {
                    Collections.reverse(dataSizes);
                }

                int itersForWarmUp = 3;
                for (int i = 0; i < itersForWarmUp; i++) {
                    dataSizes.add(0, 1_000_000_000);
                }

                long startPopulate = System.currentTimeMillis();
                int maxDataSize = Collections.max(dataSizes);
                System.out.println("Populating " + maxDataSize + " data points from Pareto distribution...");
                double[] sampledFullData = new double[maxDataSize];
                for (int i = 0; i < maxDataSize; i++) {
                    sampledFullData[i] = ptoD.sample();
                }
                long elapsedPopulate = System.currentTimeMillis() - startPopulate;
                System.out.println("Took " + elapsedPopulate + " milliseconds to populate data");

                FileWriter queryWriter = new FileWriter("query_times.txt");

                for (Integer dataSize : dataSizes) {

                    ddsketch = new DDSketch(DDS_PARAM_RELATIVE_ACCURACY);
                    uddsketch = new UniformDDSketch(UDDS_PARAM_MAX_NUM_BUCKETS, alphaZero);
                    kllsketch = new KllFloatsSketch(KLL_PARAM_K);
                    momentSketch = new SimpleMomentSketch(MOMEMNTS_PARAM_K);
                    momentSketch.setCompressed(MOMENTS_PARAM_COMPRESSED);
                    reqSketch =
                        ReqSketch.builder().setK(REQ_PARAM_K).setHighRankAccuracy(REQ_PARAM_HIGH_RANK_ACCURACY)
                            .setLessThanOrEqual(REQ_PARAM_LT_EQ).build();

                    for (int i = 0; i < dataSize; i++) {
                        double sampledValue = sampledFullData[i];
                        ddsketch.accept(sampledValue);
                        kllsketch.update((float) sampledValue);
                        momentSketch.add(sampledValue);
                        reqSketch.update((float) sampledValue);
                        uddsketch.accept(sampledValue);
                    }

                    // Query time test
                    for (int iter = 0; iter < 11; iter++) {
                        System.out.println(
                            "======== Starting query time tests for data size " + dataSize + " and iteration " +
                                iter + "========");
                        long[] queryResults = new long[5];

                        // Moments
                        long start = System.nanoTime();

                        double[] resultsMomentsSketch = momentSketch.getQuantiles(percentiles);

                        long end = System.nanoTime();
                        long elapsedTimeNanos = end - start;
                        long elapsedTimeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeNanos);

                        System.out.println(
                            "MomentSketch - Query time [" + dataSize + "] (micros): " + elapsedTimeMicros);
                        System.out.println("MomentSketch - Query time [" + dataSize + "] (nanos): " + elapsedTimeNanos);
                        queryResults[0] = elapsedTimeMicros;

                        // DDS
                        start = System.nanoTime();

                        double[] resultsDDSketch = ddsketch.getValuesAtQuantiles(percentiles);

                        end = System.nanoTime();
                        elapsedTimeNanos = end - start;

                        elapsedTimeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeNanos);
                        System.out.println(
                            "DDSketch - Query time [" + dataSize + "] (micros): " + elapsedTimeMicros);
                        System.out.println("DDSketch - Query time [" + dataSize + "] (nanos): " + elapsedTimeNanos);
                        System.out.println("Max: " + ddsketch.getMaxValue() + ", Min: " + ddsketch.getMinValue());
                        queryResults[1] = elapsedTimeMicros;

                        // KLL
                        start = System.nanoTime();

                        float[] resultsKLL = kllsketch.getQuantiles(percentiles);

                        end = System.nanoTime();
                        elapsedTimeNanos = end - start;

                        elapsedTimeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeNanos);
                        System.out.println(
                            "KLLSketch - Query time [" + dataSize + "] (micros): " + elapsedTimeMicros);
                        System.out.println("KLLSketch - Query time [" + dataSize + "] (nanos): " + elapsedTimeNanos);
                        queryResults[2] = elapsedTimeMicros;

                        // REQ
                        start = System.nanoTime();

                        float[] resultsReqSketch = reqSketch.getQuantiles(percentiles);

                        end = System.nanoTime();
                        elapsedTimeNanos = end - start;

                        elapsedTimeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeNanos);
                        System.out.println(
                            "REQSketch - Query time [" + dataSize + "] (micros): " + elapsedTimeMicros);
                        System.out.println("REQSketch - Query time [" + dataSize + "] (nanos): " + elapsedTimeNanos);
                        queryResults[3] = elapsedTimeMicros;

                        // UDDS
                        start = System.nanoTime();

                        double[] resultsUDDSketch = uddsketch.getValuesAtQuantiles(percentiles);

                        end = System.nanoTime();
                        elapsedTimeNanos = end - start;

                        elapsedTimeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeNanos);
                        System.out.println(
                            "UDDSketch - Query time [" + dataSize + "] (micros): " + elapsedTimeMicros);
                        System.out.println("UDDSketch - Query time [" + dataSize + "] (nanos): " + elapsedTimeNanos);
                        System.out.println(uddsketch);
                        queryResults[4] = elapsedTimeMicros;

                        if (PRINT_QUERY_RESULTS) {
                            for (int i = 0; i < percentiles.length; i++) {
                                System.out.println(
                                    "MomentsSketch [" + percentiles[i] + "]: " + resultsMomentsSketch[i]);
                                System.out.println("DDSketch [" + percentiles[i] + "]: " + resultsDDSketch[i]);
                                System.out.println("KLLSketch [" + percentiles[i] + "]: " + resultsKLL[i]);
                                System.out.println("REQSketch [" + percentiles[i] + "]: " + resultsReqSketch[i]);
                                System.out.println("UDDSketch [" + percentiles[i] + "]: " + resultsUDDSketch[i]);
                            }
                        }
                        if (iter == 0) {
                            queryWriter.write(dataSize + "," + Arrays.toString(queryResults) + "\n");
                        }
                    }
                }
                queryWriter.close();
            }
        } catch (FileNotFoundException e) {
            System.out.println("File not found exception occurred.");
            e.printStackTrace();
        } catch (IOException e) {
            System.out.println("IO exception occurred.");
            e.printStackTrace();
        }

    }

    private static void runUDDSTests() {
        System.out.println("======= Running initial testing of sketches ===========");
        int paramK = 12;
        double alphaZero =
            Math.tanh(FastMath.atanh(0.001) / Math.pow(2.0, paramK - 1));
        System.out.println("Alpha zero: " + alphaZero);
        DDSketch ex_dds = new DDSketch(0.01);
        UniformDDSketch ex_udds = new UniformDDSketch(1024, alphaZero);

        double[] example_data = {1, 2, 3, 4, 5, 6, 7, 18, 30, 51, 10002412, 123.01123333};
        for (int i = 0; i < example_data.length; i++) {
            ex_dds.accept(example_data[i]);
            ex_udds.accept(example_data[i]);
        }

        double[] ex_quantiles = {0.8, 0.9, 0.95, 0.98, 0.99};

        System.out.println("Example data output");
        System.out.println("DDSketch: " + ex_dds.getValuesAtQuantiles(ex_quantiles)[0]);
        System.out.println("DDSketch: " + ex_dds.getValuesAtQuantiles(ex_quantiles)[1]);
        System.out.println("UDDSketch: " + ex_udds.getValuesAtQuantiles(ex_quantiles)[0]);
        System.out.println("UDDSketch: " + ex_udds.getValuesAtQuantiles(ex_quantiles)[1]);

        ex_dds = new DDSketch(0.01);
        ex_udds = new UniformDDSketch(1024, alphaZero);

        ParetoDistribution ptoD = new ParetoDistribution(1, 1);
        System.out.println("Pareto mean: " + ptoD.getNumericalMean());

        ArrayList<Double> actualValues = new ArrayList<>();
        for (int i = 0; i < 1_000_000; i++) {
            double sampled_value = ptoD.sample();
            actualValues.add(sampled_value);
            ex_dds.accept(sampled_value);
            ex_udds.accept(sampled_value);
        }

        System.out.println("Pareto data output");
        ArrayList<Double> actualQuantiles = getPercentiles(actualValues, ex_quantiles);
        System.out.println("Actual:" + actualQuantiles);
        System.out.println("DDSketch: " + Arrays.toString(ex_dds.getValuesAtQuantiles(ex_quantiles)));
        System.out.println("UDDSketch: " + Arrays.toString(ex_udds.getValuesAtQuantiles(ex_quantiles)));

        System.out.println("======= End UDDS testing ===========");
        System.out.println(ex_udds);
        System.exit(0);
    }

    private static void runKTests() {
        System.out.println("Get K");
        System.out.println(KllFloatsSketch.getKFromEpsilon(0.01, true));
        System.out.println(KllFloatsSketch.getKFromEpsilon(0.01, false));
        System.out.println("End K");
        System.out.println("RSE: " + ReqSketch.builder().build().getRSE(30, 0.98, false, 1_000_000));
    }

    private static void calcPowerDatasetStats() throws IOException {
        String line;
        /* POWER ACTUAL_POWER GENERATOR */
        String delimiterPowerDataset = ";";
        BufferedReader br =
            new BufferedReader(new FileReader("/home/m34ferna/flink-benchmarks/household_power_consumption.txt"));
        ArrayList<Double> valuesRead = new ArrayList<>();
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
        String line;
        String splitBy = ",";
        BufferedReader br = new BufferedReader(new FileReader("/home/m34ferna/flink-benchmarks/nyt-data.csv"));
        ArrayList<Double> valuesRead = new ArrayList<>();
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

    private static void runMergeTests(int dataSize, ArrayList<Integer> numSketchesList) throws IOException {
        /*
        // MERGE TESTS
        ****************************************************
         */
        FileWriter mergeWriter = new FileWriter("merge_times.txt");

        for (Integer numSketches : numSketchesList) {
            System.out.println(
                "======== Starting merge tests for data size " + dataSize + " with " +
                    numSketches + " sketches ========");
            UniformRealDistribution uD2 = new UniformRealDistribution(1, 5000);
            BinomialDistribution binD2 = new BinomialDistribution(100, 0.2);
            ZipfDistribution zipD2 = new ZipfDistribution(20, 0.6);

            ArrayList<DDSketch> ddSketches = new ArrayList<>();
            ArrayList<KllFloatsSketch> kllFloatsSketches = new ArrayList<>();
            ArrayList<SimpleMomentSketch> momentSketches = new ArrayList<>();
            ArrayList<ReqSketch> reqSketches = new ArrayList<>();
            ArrayList<UniformDDSketch> uddSketches = new ArrayList<>();
            long[] mergeResults = new long[5];

            for (int i = 0; i < numSketches; i++) {
                DDSketch ddSketch = new DDSketch(DDS_PARAM_RELATIVE_ACCURACY);
                ddSketches.add(ddSketch);
                KllFloatsSketch kllFloatsSketch = new KllFloatsSketch(KLL_PARAM_K);
                kllFloatsSketches.add(kllFloatsSketch);
                SimpleMomentSketch msSketch = new SimpleMomentSketch(MOMEMNTS_PARAM_K);
                msSketch.setCompressed(MOMENTS_PARAM_COMPRESSED);
                momentSketches.add(msSketch);
                ReqSketch reqSketch =
                    ReqSketch.builder().setK(REQ_PARAM_K).setHighRankAccuracy(REQ_PARAM_HIGH_RANK_ACCURACY)
                        .setLessThanOrEqual(REQ_PARAM_LT_EQ).build();
                reqSketches.add(reqSketch);
                double alphaZero =
                    Math.tanh(FastMath.atanh(UDDS_PARAM_RELATIVE_ACCURACY) / Math.pow(2.0, UDDS_PARAM_K - 1));
                UniformDDSketch uniformDDSketch = new UniformDDSketch(UDDS_PARAM_MAX_NUM_BUCKETS, alphaZero);
                uddSketches.add(uniformDDSketch);
            }

            System.out.println(
                "Starting to populate " + numSketches + " sketches with " + dataSize + " data points each ...");
            int k = 0;
            for (int i = 0; i < numSketches; i++) {
                DDSketch ds = ddSketches.get(i);
                KllFloatsSketch kll = kllFloatsSketches.get(i);
                SimpleMomentSketch msketch = momentSketches.get(i);
                ReqSketch reqSketch = reqSketches.get(i);
                UniformDDSketch uddsketch = uddSketches.get(i);

                if ((k % 3) == 0) {
                    for (int j = 0; j < dataSize; j++) {
                        double sampled_value = uD2.sample();
                        ds.accept(sampled_value);
                        kll.update((float) sampled_value);
                        msketch.add(sampled_value);
                        reqSketch.update((float) sampled_value);
                        uddsketch.accept(sampled_value);
                    }
                }
                if ((k % 3) == 1) {
                    for (int j = 0; j < dataSize; j++) {
                        double sampled_value = binD2.sample();
                        ds.accept(sampled_value);
                        kll.update((float) sampled_value);
                        msketch.add(sampled_value);
                        reqSketch.update((float) sampled_value);
                        uddsketch.accept(sampled_value);
                    }
                }
                if ((k % 3) == 2) {
                    for (int j = 0; j < dataSize; j++) {
                        double sampled_value = zipD2.sample();
                        ds.accept(sampled_value);
                        kll.update((float) sampled_value);
                        msketch.add(sampled_value);
                        reqSketch.update((float) sampled_value);
                        uddsketch.accept(sampled_value);
                    }
                }
                k++;
            }

            DDSketch orig_dds = ddSketches.get(0);
            KllFloatsSketch orig_kll = kllFloatsSketches.get(0);
            SimpleMomentSketch orig_moments = momentSketches.get(0);
            ReqSketch orig_req = reqSketches.get(0);
            UniformDDSketch orig_udds = uddSketches.get(0);

            long startMerge = System.nanoTime();

            for (int i = 1; i < numSketches; i++) {
                orig_moments.merge(momentSketches.get(i));
            }

            long endMerge = System.nanoTime();
            long elapsedTimeMergeNanos = endMerge - startMerge;
            long elapsedTimeMergeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeMergeNanos);
            System.out.println(
                "MomentsSketch - Merge time [" + numSketches + "] (micros): " + elapsedTimeMergeMicros);
            System.out.println("MomentsSketch - Merge time [" + numSketches + "] (nanos): " + elapsedTimeMergeNanos);
            mergeResults[0] = elapsedTimeMergeMicros;

            startMerge = System.nanoTime();

            for (int i = 1; i < numSketches; i++) {
                orig_dds.mergeWith(ddSketches.get(i));
            }

            endMerge = System.nanoTime();

            elapsedTimeMergeNanos = endMerge - startMerge;
            elapsedTimeMergeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeMergeNanos);
            System.out.println(
                "DDSketch - Merge time [" + numSketches + "] (micros): " + elapsedTimeMergeMicros);
            System.out.println("DDSketch - Merge time [" + numSketches + "] (nanos): " + elapsedTimeMergeNanos);
            mergeResults[1] = elapsedTimeMergeMicros;

            startMerge = System.nanoTime();

            for (int i = 1; i < numSketches; i++) {
                orig_kll.merge(kllFloatsSketches.get(i));
            }

            endMerge = System.nanoTime();
            elapsedTimeMergeNanos = endMerge - startMerge;
            elapsedTimeMergeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeMergeNanos);
            System.out.println(
                "KLLSketch - Merge time [" + numSketches + "] (micros): " + elapsedTimeMergeMicros);
            System.out.println("KLLSketch - Merge time [" + numSketches + "] (nanos): " + elapsedTimeMergeNanos);
            mergeResults[2] = elapsedTimeMergeMicros;

            startMerge = System.nanoTime();

            for (int i = 1; i < numSketches; i++) {
                orig_req.merge(reqSketches.get(i));
            }

            endMerge = System.nanoTime();
            elapsedTimeMergeNanos = endMerge - startMerge;
            elapsedTimeMergeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeMergeNanos);
            System.out.println(
                "REQSketch - Merge time [" + numSketches + "] (micros): " + elapsedTimeMergeMicros);
            System.out.println("REQSketch - Merge time [" + numSketches + "] (nanos): " + elapsedTimeMergeNanos);
            mergeResults[3] = elapsedTimeMergeMicros;

            startMerge = System.nanoTime();

            for (int i = 1; i < numSketches; i++) {
                orig_udds.mergeWith(uddSketches.get(i));
            }

            endMerge = System.nanoTime();
            elapsedTimeMergeNanos = endMerge - startMerge;
            elapsedTimeMergeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeMergeNanos);
            System.out.println(
                "UDDSketch - Merge time [" + numSketches + "] (micros): " + elapsedTimeMergeMicros);
            System.out.println("UDDSketch - Merge time [" + numSketches + "] (nanos): " + elapsedTimeMergeNanos);
            mergeResults[4] = elapsedTimeMergeMicros;

            mergeWriter.write(numSketches + "," + Arrays.toString(mergeResults) + "\n");
        }
        mergeWriter.close();
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
        ArrayList<Double> ret_list = new ArrayList<>();
        Collections.sort(all_events);
        for (double p : percentiles) {
            ret_list.add(round(percentile(all_events, p), 4));
        }
        return ret_list;
    }


}
