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
    public static final boolean RUN_K_TESTS = true;
    public static final boolean RUN_INIT_TESTS = false;
    public static final boolean PRINT_QUERY_RESULTS = false;
    public static final boolean RUN_KURT_POWER = true;
    public static final boolean RUN_KURT_NYT = true;
    public static final boolean RUN_WARM_UP_FOR_MERGE = false;
    public static final boolean SAVE_PARETO_SAMPLE = false;
    public static final boolean SAVE_UNIFORM_SAMPLE = false;
    public static final boolean MEASURE_SKETCH_SIZES = false;
    public static final boolean PREPROCESS_INSERT_MERGE_RESULTS = true;
    public static final boolean PREPROCESS_INSERT_TIMES = false;
    public static final boolean PREPROCESS_MERGE_TIMES = true;
    private static final boolean RUN_UDDS_TESTS = false;
    public static final boolean MOMENTS_PARAM_COMPRESSED = true;

    public static void main(String[] args) {
        int runMode = Integer.parseInt(args[0]);
        try {
            double[] percentiles = {0.01, 0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.98, 0.99};
            double alphaZero =
                Math.tanh(FastMath.atanh(UDDS_PARAM_RELATIVE_ACCURACY) / Math.pow(2.0, UDDS_PARAM_K - 1));

            // Sketch Algorithms
            DDSketch ddsketch;
            DDSketch ddSketchCollapsing;
            UniformDDSketch uddsketch;
            KllFloatsSketch kllsketch;
            SimpleMomentSketch momentSketch;
            ReqSketch reqSketch;

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
                numSketchesList.add(10);
                numSketchesList.add(10);

                numSketchesList.add(100);
                numSketchesList.add(100);

                numSketchesList.add(100);
                numSketchesList.add(100);
                numSketchesList.add(100);

                numSketchesList.add(1000);
                numSketchesList.add(1000);
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
                FileWriter adaptabilityDistWriter = new FileWriter("adaptability_dist.csv");
                for (int iter = 0; iter < 12; iter++) {
                    all_data.clear();
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

                    if (iter == 0) {
                        adaptabilityDistWriter.write("index, adapt_val\n");
                        for (int i = 0; i < all_data.size(); i++) {
                            adaptabilityDistWriter.write(i + "," + all_data.get(i) + "\n");
                        }
                        System.out.println(
                            "Wrote adaptability sample of " + all_data.size() + " to adaptability_dist.csv");
                    }
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
                adaptabilityDistWriter.close();
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
                int numIters = 12;
                double realQ, momentsQ, ddsQ, ddscQ, kllQ, reqQ, uddsQ;
                String resultString;
                FileWriter kurtosisWriter = new FileWriter("kurtosis_data.txt");

                // Uniform
                NormalDistribution uniformNormal = new NormalDistribution(100, 25);
                NormalDistribution uniformNormal2 = new NormalDistribution(1000, 100);
                UniformRealDistribution uDKurt =
                    new UniformRealDistribution(uniformNormal.sample(), uniformNormal2.sample());
                for (int i = 0; i < dataSizeKurtosis; i++) {
                    if (i % 50000 == 0) {
                        uDKurt = new UniformRealDistribution(uniformNormal.sample(), uniformNormal2.sample());
                    }
                    double sampled_value = uDKurt.sample();
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
                NormalDistribution paretoNormal = new NormalDistribution(1, 0.05);
                double shapeParam = paretoNormal.sample();
                while (shapeParam < 0.01) {
                    shapeParam = paretoNormal.sample();
                }
                ParetoDistribution ptoDKurt = new ParetoDistribution(shapeParam, shapeParam);
                for (int i = 0; i < dataSizeKurtosis; i++) {
                    if (i % 50000 == 0) {
                        shapeParam = paretoNormal.sample();
                        while (shapeParam < 0.01) {
                            shapeParam = paretoNormal.sample();
                        }
                        ptoDKurt = new ParetoDistribution(shapeParam, shapeParam);
                    }
                    double sampled_value = ptoDKurt.sample();
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

                    if (MEASURE_SKETCH_SIZES) {
                        long momentBytes = ObjectSizeFetcher.getObjectSize(momentSketch) +
                            (long) (momentSketch.getK() + 2) * Double.BYTES;
                        long ddsBytes =
                            ObjectSizeFetcher.getObjectSize(ddsketch) +
                                ddsketch.getPositiveValueStore().serializedSize();
                        long ddscBytes = ObjectSizeFetcher.getObjectSize(ddSketchCollapsing) +
                            ddSketchCollapsing.getPositiveValueStore().serializedSize();
                        long uddsBytes =
                            ObjectSizeFetcher.getObjectSize(uddsketch) + uddsketch.getSerializedStoreSize();
                        System.out.println("Moments:" + momentBytes +
                            ", DDS:" + ddsBytes
                            + ", DDSC:" + ddscBytes
                            + ", KLL:" + kllsketch.getSerializedSizeBytes()
                            + ", REQ:" + reqSketch.getSerializationBytes()
                            + ", UDDS:" + uddsBytes);
                    }
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
                    while ((line = brPower.readLine()) != null && actualKurtosisData.size() <= dataSizeKurtosis) {
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
                    while ((line = brNYT.readLine()) != null && actualKurtosisData.size() <= dataSizeKurtosis) {
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
                dataSizes.add(10_000_000);
                dataSizes.add(10_000_000);
                dataSizes.add(10_000_000);

                dataSizes.add(100_000_000);
                dataSizes.add(100_000_000);
                dataSizes.add(100_000_000);
                dataSizes.add(100_000_000);
                dataSizes.add(100_000_000);

                FileWriter insertWriter = new FileWriter("insert_times.txt");

                int insertIter = 0;
                for (Integer dataSize : dataSizes) {
                    insertIter++;
                    System.out.println("======================================================");
                    long startInsertOp, elapsedTimeOp;
                    long[] insertResults = new long[5];
                    ddsketch = new DDSketch(DDS_PARAM_RELATIVE_ACCURACY);
                    uddsketch = new UniformDDSketch(UDDS_PARAM_MAX_NUM_BUCKETS, alphaZero);
                    kllsketch = new KllFloatsSketch(KLL_PARAM_K);
                    momentSketch = new SimpleMomentSketch(MOMEMNTS_PARAM_K);
                    momentSketch.setCompressed(MOMENTS_PARAM_COMPRESSED);
                    reqSketch =
                        ReqSketch.builder().setK(REQ_PARAM_K).setHighRankAccuracy(REQ_PARAM_HIGH_RANK_ACCURACY)
                            .setLessThanOrEqual(REQ_PARAM_LT_EQ).build();

                    FileWriter momentsInsertWriter =
                        new FileWriter("moments_insert_times_" + dataSize + "_" + insertIter + ".txt");
                    long startInsert = System.nanoTime();

                    for (int i = 0; i < dataSize; i++) {
                        double sampled_value = ptoD.sample();
                        startInsertOp = System.nanoTime();
                        momentSketch.add(sampled_value);
                        elapsedTimeOp = System.nanoTime() - startInsertOp;
                        momentsInsertWriter.write(i + "," + elapsedTimeOp + "\n");
                    }

                    long endInsert = System.nanoTime();

                    long elapsedTimeInsertNanos = endInsert - startInsert;
                    long elapsedTimeInsertMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeInsertNanos);
                    System.out.println(
                        "MomentSketch - Insert time [" + dataSize + "] (micros): " + elapsedTimeInsertMicros);
                    System.out.println(
                        "MomentSketch - Insert time [" + dataSize + "] (nanos): " + elapsedTimeInsertNanos);
                    insertResults[0] = elapsedTimeInsertMicros;
                    momentsInsertWriter.close();

                    // DDSketch
                    FileWriter ddsInsertWriter =
                        new FileWriter("dds_insert_times_" + dataSize + "_" + insertIter + ".txt");
                    startInsert = System.nanoTime();

                    for (int i = 0; i < dataSize; i++) {
                        double sampled_value = ptoD.sample();
                        startInsertOp = System.nanoTime();
                        ddsketch.accept(sampled_value);
                        elapsedTimeOp = System.nanoTime() - startInsertOp;
                        ddsInsertWriter.write(i + "," + elapsedTimeOp + "\n");
                    }

                    endInsert = System.nanoTime();

                    elapsedTimeInsertNanos = endInsert - startInsert;
                    elapsedTimeInsertMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeInsertNanos);
                    System.out.println(
                        "DDSketch - Insert time [" + dataSize + "] (micros): " + elapsedTimeInsertMicros);
                    System.out.println("DDSketch - Insert time [" + dataSize + "] (nanos): " + elapsedTimeInsertNanos);
                    insertResults[1] = elapsedTimeInsertMicros;
                    ddsInsertWriter.close();

                    //KLL Sketch
                    FileWriter kllInsertWriter =
                        new FileWriter("kll_insert_times_" + dataSize + "_" + insertIter + ".txt");
                    startInsert = System.nanoTime();

                    for (int i = 0; i < dataSize; i++) {
                        double sampled_value = ptoD.sample();
                        startInsertOp = System.nanoTime();
                        kllsketch.update((float) sampled_value);
                        elapsedTimeOp = System.nanoTime() - startInsertOp;
                        kllInsertWriter.write(i + "," + elapsedTimeOp + "\n");
                    }

                    endInsert = System.nanoTime();

                    elapsedTimeInsertNanos = endInsert - startInsert;
                    elapsedTimeInsertMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeInsertNanos);
                    System.out.println(
                        "KLLSketch - Insert time [" + dataSize + "] (micros): " + elapsedTimeInsertMicros);
                    System.out.println("KLLSketch - Insert time [" + dataSize + "] (nanos): " + elapsedTimeInsertNanos);
                    insertResults[2] = elapsedTimeInsertMicros;
                    kllInsertWriter.close();

                    // REQ
                    FileWriter reqInsertWriter =
                        new FileWriter("req_insert_times_" + dataSize + "_" + insertIter + ".txt");
                    startInsert = System.nanoTime();

                    for (int i = 0; i < dataSize; i++) {
                        double sampled_value = ptoD.sample();
                        startInsertOp = System.nanoTime();
                        reqSketch.update((float) sampled_value);
                        elapsedTimeOp = System.nanoTime() - startInsertOp;
                        reqInsertWriter.write(i + "," + elapsedTimeOp + "\n");
                    }

                    endInsert = System.nanoTime();

                    elapsedTimeInsertNanos = endInsert - startInsert;
                    elapsedTimeInsertMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeInsertNanos);
                    System.out.println(
                        "REQSketch - Insert time [" + dataSize + "] (micros): " + elapsedTimeInsertMicros);
                    System.out.println("REQSketch - Insert time [" + dataSize + "] (nanos): " + elapsedTimeInsertNanos);
                    insertResults[3] = elapsedTimeInsertMicros;
                    reqInsertWriter.close();

                    // UDDS
                    FileWriter uddsInsertWriter =
                        new FileWriter("udds_insert_times_" + dataSize + "_" + insertIter + ".txt");
                    startInsert = System.nanoTime();

                    for (int i = 0; i < dataSize; i++) {
                        double sampled_value = ptoD.sample();
                        startInsertOp = System.nanoTime();
                        uddsketch.accept(sampled_value);
                        elapsedTimeOp = System.nanoTime() - startInsertOp;
                        uddsInsertWriter.write(i + "," + elapsedTimeOp + "\n");
                    }

                    endInsert = System.nanoTime();

                    elapsedTimeInsertNanos = endInsert - startInsert;
                    elapsedTimeInsertMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeInsertNanos);
                    System.out.println(
                        "UDDSketch - Insert time [" + dataSize + "] (micros): " + elapsedTimeInsertMicros);
                    System.out.println("UDDSketch - Insert time [" + dataSize + "] (nanos): " + elapsedTimeInsertNanos);
                    insertResults[4] = elapsedTimeInsertMicros;
                    uddsInsertWriter.close();

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
            if (SAVE_PARETO_SAMPLE) {
                sampleToFile(1_000_000, "pareto");
            }
            if (SAVE_UNIFORM_SAMPLE) {
                sampleToFile(1_000_000, "uniform");
            }
            if (PREPROCESS_INSERT_MERGE_RESULTS) {
                preprocessInsertMergeResults();
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

                int numThousandMilIters = 12;
                for (int i = 0; i < numThousandMilIters; i++) {
                    dataSizes.add(1_000_000_000);
                }

                if (runMode == 6) {
                    Collections.reverse(dataSizes);
                }

                int itersForWarmUp = 20;
                for (int i = 0; i < itersForWarmUp; i++) {
                    dataSizes.add(0, 10_000_000);
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

    private static void sampleToFile(int sampleSize, String distType) throws IOException {
        String filename = distType + "_sample_" + sampleSize + ".csv";
        FileWriter distSampleWriter = new FileWriter(filename);
        distSampleWriter.write("index, sampled_val\n");
        if (distType.equals("pareto")) {
            double maxPareto = 0;
            NormalDistribution paretoNormal = new NormalDistribution(1, 0.05);
            double shapeParam = paretoNormal.sample();
            while (shapeParam < 0.01) {
                shapeParam = paretoNormal.sample();
            }
            ParetoDistribution ptoD = new ParetoDistribution(shapeParam, shapeParam);
            for (int i = 0; i < sampleSize; i++) {
                if (i % 5 == 0) {
                    shapeParam = paretoNormal.sample();
                    while (shapeParam < 0.01) {
                        shapeParam = paretoNormal.sample();
                    }
                    ptoD = new ParetoDistribution(shapeParam, shapeParam);
                }
                double sampledValue = ptoD.sample();
                if (sampledValue > maxPareto) {
                    maxPareto = sampledValue;
                }
                distSampleWriter.write(i + "," + sampledValue + "\n");
            }
            System.out.println("Max value:" + maxPareto);
        } else if (distType.equals("uniform")) {
            UniformRealDistribution uD = new UniformRealDistribution(0, 1000);
            for (int i = 0; i < sampleSize; i++) {
                double sampledValue = uD.sample();
                distSampleWriter.write(i + "," + sampledValue + "\n");
            }
        }
        distSampleWriter.close();
        System.out.println("Wrote " + distType + " sample of " + sampleSize + " to " + filename);
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
        System.out.println(KllFloatsSketch.getNormalizedRankError(200, true));
        System.out.println("End K");
        System.out.println("RSE: " + ReqSketch.builder().build().getRSE(30, 0.5, true, 1_000_000));
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

    private static void preprocessInsertMergeResults() throws IOException {
        String line;
        String splitBy = ",";
        int iter = 3;
        int numIters = 5;
        String[] algorithms = {"moments", "dds", "kll", "req", "udds"};
        if (PREPROCESS_INSERT_TIMES) {
            FileWriter insertAggWriter = new FileWriter("agg_insert_times.csv");
            int[] dataSizes = {10_000_000, 100_000_000};
            for (int dataSize : dataSizes) {
                for (int i = 0; i < numIters; i++) {
                    iter++;
                    StringBuilder resultString = new StringBuilder().append(dataSize);
                    for (String algorithm : algorithms) {
                        BufferedReader br = new BufferedReader(
                            new FileReader("results/" + algorithm + "_insert_times_" + dataSize + "_" + iter + ".txt"));
                        long totalTime = 0;
                        int totCount = 0;
                        while ((line = br.readLine()) != null) {
                            String[] line_array = line.split(splitBy);    // use comma as separator
                            long insertOpTime = Long.parseLong(line_array[1]);
                            totalTime += insertOpTime;
                            totCount += 1;
                        }
                        System.out.println(algorithm + "(" + totCount + "):" + totalTime);
                        resultString.append(",").append(totalTime / 1000);
                        br.close();
                    }
                    insertAggWriter.write(resultString.append("\n").toString());
                }
            }
            insertAggWriter.close();
        }
        if (PREPROCESS_MERGE_TIMES) {
            FileWriter mergeAggWriter = new FileWriter("agg_merge_times.csv");
            int[] sketchSizes = {100, 1000};
            numIters = 3;
            iter = 5;
            for (int numSketches : sketchSizes) {
                for (int i = 0; i < numIters; i++) {
                    iter++;
                    StringBuilder resultString = new StringBuilder().append(numSketches);
                    for (String algorithm : algorithms) {
                        BufferedReader br = new BufferedReader(
                            new FileReader(
                                "results/" + algorithm + "_merge_times_" + numSketches + "_" + iter + ".txt"));
                        long totalTime = 0;
                        int totCount = 0;
                        while ((line = br.readLine()) != null) {
                            String[] line_array = line.split(splitBy);    // use comma as separator
                            long mergeOpTime = Long.parseLong(line_array[1]);
                            totalTime += mergeOpTime;
                            totCount += 1;
                        }
                        System.out.println(algorithm + "(" + totCount + "):" + totalTime);
                        resultString.append(",").append(totalTime / 1000);
                        br.close();
                    }
                    mergeAggWriter.write(resultString.append("\n").toString());
                }
            }
            mergeAggWriter.close();
        }
    }

    private static void runMergeTests(int dataSize, ArrayList<Integer> numSketchesList) throws IOException {
        /*
        // MERGE TESTS
        ****************************************************
         */
        FileWriter mergeWriter = new FileWriter("merge_times.txt");

        int mergeIter = 0;
        for (Integer numSketches : numSketchesList) {
            System.out.println(
                "======== Starting merge tests for data size " + dataSize + " with " +
                    numSketches + " sketches ========");
            mergeIter++;
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

            if (RUN_WARM_UP_FOR_MERGE) {
                DDSketch ddSketchWarmUp = new DDSketch(DDS_PARAM_RELATIVE_ACCURACY);
                KllFloatsSketch kllFloatsSketchWarmUp = new KllFloatsSketch(KLL_PARAM_K);
                SimpleMomentSketch momentsSketchWarmUp = new SimpleMomentSketch(MOMEMNTS_PARAM_K);
                momentsSketchWarmUp.setCompressed(MOMENTS_PARAM_COMPRESSED);
                ReqSketch reqSketchWarmUp =
                    ReqSketch.builder().setK(REQ_PARAM_K).setHighRankAccuracy(REQ_PARAM_HIGH_RANK_ACCURACY)
                        .setLessThanOrEqual(REQ_PARAM_LT_EQ).build();
                double alphaZero =
                    Math.tanh(FastMath.atanh(UDDS_PARAM_RELATIVE_ACCURACY) / Math.pow(2.0, UDDS_PARAM_K - 1));
                UniformDDSketch uniformDDSketchWarmUp = new UniformDDSketch(UDDS_PARAM_MAX_NUM_BUCKETS, alphaZero);

                int warmUpRuns = Math.min(numSketches, 100);
                for (int i = 1; i < warmUpRuns; i++) {
                    momentsSketchWarmUp.merge(momentSketches.get(i));
                }
                for (int i = 1; i < warmUpRuns; i++) {
                    ddSketchWarmUp.mergeWith(ddSketches.get(i));
                }
                for (int i = 1; i < warmUpRuns; i++) {
                    kllFloatsSketchWarmUp.merge(kllFloatsSketches.get(i));
                }
                for (int i = 1; i < warmUpRuns; i++) {
                    reqSketchWarmUp.merge(reqSketches.get(i));
                }
                for (int i = 1; i < warmUpRuns; i++) {
                    uniformDDSketchWarmUp.mergeWith(uddSketches.get(i));
                }
            }

            DDSketch orig_dds = ddSketches.get(0);
            KllFloatsSketch orig_kll = kllFloatsSketches.get(0);
            SimpleMomentSketch orig_moments = momentSketches.get(0);
            ReqSketch orig_req = reqSketches.get(0);
            UniformDDSketch orig_udds = uddSketches.get(0);

            long elapsedTimeOp, startMergeOp;

            //Moments
            FileWriter momentsMergeWriter =
                new FileWriter("moments_merge_times_" + numSketches + "_" + mergeIter + ".txt");
            long startMerge = System.nanoTime();

            for (int i = 1; i < numSketches; i++) {
                startMergeOp = System.nanoTime();
                orig_moments.merge(momentSketches.get(i));
                elapsedTimeOp = System.nanoTime() - startMergeOp;
                momentsMergeWriter.write(i + "," + elapsedTimeOp + "\n");
            }

            long endMerge = System.nanoTime();
            long elapsedTimeMergeNanos = endMerge - startMerge;
            long elapsedTimeMergeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeMergeNanos);
            System.out.println(
                "MomentsSketch - Merge time [" + numSketches + "] (micros): " + elapsedTimeMergeMicros);
            System.out.println("MomentsSketch - Merge time [" + numSketches + "] (nanos): " + elapsedTimeMergeNanos);
            mergeResults[0] = elapsedTimeMergeMicros;
            momentsMergeWriter.close();

            // DDS
            FileWriter ddsMergeWriter =
                new FileWriter("dds_merge_times_" + numSketches + "_" + mergeIter + ".txt");
            startMerge = System.nanoTime();

            for (int i = 1; i < numSketches; i++) {
                startMergeOp = System.nanoTime();
                orig_dds.mergeWith(ddSketches.get(i));
                elapsedTimeOp = System.nanoTime() - startMergeOp;
                ddsMergeWriter.write(i + "," + elapsedTimeOp + "\n");
            }

            endMerge = System.nanoTime();

            elapsedTimeMergeNanos = endMerge - startMerge;
            elapsedTimeMergeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeMergeNanos);
            System.out.println(
                "DDSketch - Merge time [" + numSketches + "] (micros): " + elapsedTimeMergeMicros);
            System.out.println("DDSketch - Merge time [" + numSketches + "] (nanos): " + elapsedTimeMergeNanos);
            mergeResults[1] = elapsedTimeMergeMicros;
            ddsMergeWriter.close();

            // KLL
            FileWriter kllMergeWriter =
                new FileWriter("kll_merge_times_" + numSketches + "_" + mergeIter + ".txt");
            startMerge = System.nanoTime();

            for (int i = 1; i < numSketches; i++) {
                startMergeOp = System.nanoTime();
                orig_kll.merge(kllFloatsSketches.get(i));
                elapsedTimeOp = System.nanoTime() - startMergeOp;
                kllMergeWriter.write(i + "," + elapsedTimeOp + "\n");
            }

            endMerge = System.nanoTime();
            elapsedTimeMergeNanos = endMerge - startMerge;
            elapsedTimeMergeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeMergeNanos);
            System.out.println(
                "KLLSketch - Merge time [" + numSketches + "] (micros): " + elapsedTimeMergeMicros);
            System.out.println("KLLSketch - Merge time [" + numSketches + "] (nanos): " + elapsedTimeMergeNanos);
            mergeResults[2] = elapsedTimeMergeMicros;
            kllMergeWriter.close();

            // REQ
            FileWriter reqMergeWriter =
                new FileWriter("req_merge_times_" + numSketches + "_" + mergeIter + ".txt");
            startMerge = System.nanoTime();

            for (int i = 1; i < numSketches; i++) {
                startMergeOp = System.nanoTime();
                orig_req.merge(reqSketches.get(i));
                elapsedTimeOp = System.nanoTime() - startMergeOp;
                reqMergeWriter.write(i + "," + elapsedTimeOp + "\n");
            }

            endMerge = System.nanoTime();
            elapsedTimeMergeNanos = endMerge - startMerge;
            elapsedTimeMergeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeMergeNanos);
            System.out.println(
                "REQSketch - Merge time [" + numSketches + "] (micros): " + elapsedTimeMergeMicros);
            System.out.println("REQSketch - Merge time [" + numSketches + "] (nanos): " + elapsedTimeMergeNanos);
            mergeResults[3] = elapsedTimeMergeMicros;
            reqMergeWriter.close();

            // UDDS
            FileWriter uddsMergeWriter =
                new FileWriter("udds_merge_times_" + numSketches + "_" + mergeIter + ".txt");
            startMerge = System.nanoTime();

            for (int i = 1; i < numSketches; i++) {
                startMergeOp = System.nanoTime();
                orig_udds.mergeWith(uddSketches.get(i));
                elapsedTimeOp = System.nanoTime() - startMergeOp;
                uddsMergeWriter.write(i + "," + elapsedTimeOp + "\n");
            }

            endMerge = System.nanoTime();
            elapsedTimeMergeNanos = endMerge - startMerge;
            elapsedTimeMergeMicros = TimeUnit.NANOSECONDS.toMicros(elapsedTimeMergeNanos);
            System.out.println(
                "UDDSketch - Merge time [" + numSketches + "] (micros): " + elapsedTimeMergeMicros);
            System.out.println("UDDSketch - Merge time [" + numSketches + "] (nanos): " + elapsedTimeMergeNanos);
            mergeResults[4] = elapsedTimeMergeMicros;
            uddsMergeWriter.close();

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
