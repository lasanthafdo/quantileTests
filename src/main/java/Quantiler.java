import com.datadoghq.sketch.ddsketch.DDSketch;
import com.github.stanfordfuturedata.momentsketch.MomentStruct;
import com.github.stanfordfuturedata.momentsketch.SimpleMomentSketch;
import org.apache.commons.math3.distribution.*;
import org.apache.datasketches.kll.KllFloatsSketch;
import java.io.IOException;  // Import the IOException class to handle errors
import org.apache.commons.math3.stat.descriptive.moment.Kurtosis;

import java.io.*;
import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.ArrayList;
import java.util.Collections;

import static java.lang.Double.parseDouble;

public class Quantiler {
    public static void main(String[] args) {
        try{

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

            double relativeAccuracy = 0.01;
            DDSketch ddsketch = new DDSketch(relativeAccuracy);

            KllFloatsSketch kllsketch = new KllFloatsSketch();

            // Synthetic Workloads

            System.out.println("Get K");
            System.out.println(KllFloatsSketch.getKFromEpsilon(0.01, true));
            System.out.println(KllFloatsSketch.getKFromEpsilon(0.01, false));
            System.out.println("End K");

            /*
            // MERGE TESTS
            ****************************************************
             */
//            int dataSize = 100;
//            int numSketches = 10;
//
//            UniformRealDistribution uD2 = new UniformRealDistribution(1, 5000);
//            NormalDistribution valnD2 = new NormalDistribution(100, 15);
//            BinomialDistribution ptoD2 = new BinomialDistribution(100, 0.2);
//
//            ArrayList<DDSketch> ddSketches = new ArrayList<DDSketch>();
//            ArrayList<KllFloatsSketch> kllFloatsSketches = new ArrayList<KllFloatsSketch>();
//            ArrayList<MomentStruct> momentSketches = new ArrayList<MomentStruct>();
//
//            for (int i = 0; i < numSketches; i++){
//                DDSketch ddSketch = new DDSketch(0.01);
//                ddSketches.add(ddSketch);
//                KllFloatsSketch kllFloatsSketch = new KllFloatsSketch();
//                kllFloatsSketches.add(kllFloatsSketch);
//                MomentStruct momentStruct = new MomentStruct(18);
//                momentSketches.add(momentStruct);
//                kllFloatsSketches.add(kllFloatsSketch);
//
//            }
//
//            int k = 0;
//            for (int i = 0; i < 100; i++){
//                DDSketch ds =  ddSketches.get(i);
//                KllFloatsSketch kll = kllFloatsSketches.get(i);
//                MomentStruct momsketch = momentSketches.get(i);
//
//                if ((k % 3) == 0){
//                 for (int j = 0; j < dataSize; j++){
//                     double sampled_value = uD2.sample();
//                     ds.accept(sampled_value);
//                     kll.update((float) sampled_value);
//                     momsketch.add(sampled_value);
//                 }
//               }
//                if ((k % 3) == 1){
//                    for (int j = 0; j < dataSize; j++){
//                        double sampled_value = valnD2.sample();
//                        ds.accept(sampled_value);
//                        kll.update((float) sampled_value);
//                        momsketch.add(sampled_value);
//                    }
//                }
//                if ((k % 3) == 2){
//                    for (int j = 0; j < dataSize; j++){
//                        double sampled_value = ptoD2.sample();
//                        ds.accept(sampled_value);
//                        kll.update((float) sampled_value);
//                        momsketch.add(sampled_value);
//                    }
//                }
//               k++;
//            }
//
//            DDSketch orig_dds = ddSketches.get(0);
//            KllFloatsSketch orig_kll = kllFloatsSketches.get(0);
//            MomentStruct orig_moments = momentSketches.get(0);
//            long startMerge = System.nanoTime();
//
//            for (int i = 1; i < 10; i++){
//                orig_dds.mergeWith(ddSketches.get(i));
//            }
//
//            long endMerge = System.nanoTime();
//
//            long elapsedTimeMerge = endMerge - startMerge;
//            System.out.println("DDSketch");
//            System.out.println(TimeUnit.NANOSECONDS.toMicros(elapsedTimeMerge));
//            System.out.println(elapsedTimeMerge);
//
//            startMerge = System.nanoTime();
//
//            for (int i = 1; i < 10; i++){
//                orig_kll.merge(kllFloatsSketches.get(i));
//            }
//
//            endMerge = System.nanoTime();
//            elapsedTimeMerge = endMerge - startMerge;
//            System.out.println("KLLSketch");
//            System.out.println(TimeUnit.NANOSECONDS.toMicros(elapsedTimeMerge));
//            System.out.println(elapsedTimeMerge);
//
//            startMerge = System.nanoTime();
//
//            for (int i = 1; i < 10; i++){
//                orig_moments.merge(momentSketches.get(i));
//            }
//
//            endMerge = System.nanoTime();
//            elapsedTimeMerge = endMerge - startMerge;
//            System.out.println("Moments Sketch");
//            System.out.println(TimeUnit.NANOSECONDS.toMicros(elapsedTimeMerge));
//            System.out.println(elapsedTimeMerge);


            // ADAPTABILITY TEST

            int dataSizeAdaptability = 500000;
            ArrayList<Double> all_data = new ArrayList<Double>();
            BinomialDistribution binD = new BinomialDistribution(30, 0.4);
            System.out.println("Binomial Distribution mean: " + binD.getNumericalMean());

            ZipfDistribution zipD = new ZipfDistribution(20, 0.6);
            System.out.println("Zipf mean: " + zipD.getNumericalMean());

            GammaDistribution gd = new GammaDistribution(2,0.4);
            System.out.println("Gamma mean: " + gd.getNumericalMean());

            UniformRealDistribution unif_adapt_test = new UniformRealDistribution(40, 100);
            System.out.println("Uniform mean: " + unif_adapt_test.getNumericalMean());

            ddsketch = new DDSketch(0.01);
            kllsketch = new KllFloatsSketch(600);
            SimpleMomentSketch simplesketch = new SimpleMomentSketch(18);

            long startInsert = System.nanoTime();

            for (int i = 0; i < dataSizeAdaptability; i++){
                double sampled_value = binD.sample();
                double sampled_value_2 = gd.sample();
                //all_data.add(sampled_value);
                ddsketch.accept(sampled_value);
                ddsketch.accept(sampled_value_2);
                //kllsketch.update((float) sampled_value);
                //msketch.add(sampled_value);
            }

            long endInsert = System.nanoTime();

            long elapsedTimeInsert = endInsert - startInsert;
            System.out.println("DDS Sketch");
            System.out.println(elapsedTimeInsert);
            System.out.println(TimeUnit.NANOSECONDS.toMicros(elapsedTimeInsert));
            System.out.println(TimeUnit.NANOSECONDS.toMillis(elapsedTimeInsert));


            startInsert = System.nanoTime();
            ArrayList<Double> dal = new ArrayList<Double>();

            for (int i = 0; i < dataSizeAdaptability; i++){
                double sampled_value = binD.sample();
                double sampled_value_2 = gd.sample();
                dal.add(sampled_value);
                //all_data.add(sampled_value);
                //ddsketch.accept(sampled_value);
                //kllsketch.update((float) sampled_value);
                //kllsketch.update((float) sampled_value_2);
                simplesketch.add(sampled_value);
                simplesketch.add(sampled_value_2);
            }
            Kurtosis k = new Kurtosis();


            double[] target = new double[dal.size()];
            for (int i = 0; i < target.length; i++) {
                target[i] = dal.get(i);                // java 1.5+ style (outboxing)
            }

            System.out.println("Kurtosis");
            System.out.println(k.evaluate(target));

            endInsert = System.nanoTime();

            elapsedTimeInsert = endInsert - startInsert;

            System.out.println("Moments Sketch");
            System.out.println(elapsedTimeInsert);
            System.out.println(TimeUnit.NANOSECONDS.toMicros(elapsedTimeInsert));
            System.out.println(TimeUnit.NANOSECONDS.toMillis(elapsedTimeInsert));

            startInsert = System.nanoTime();

            for (int i = 0; i < dataSizeAdaptability; i++){
                double sampled_value = binD.sample();
                double sampled_value_2 = gd.sample();
                //all_data.add(sampled_value);
                //ddsketch.accept(sampled_value);
                kllsketch.update((float) sampled_value);
                kllsketch.update((float) sampled_value_2);
            }

            endInsert = System.nanoTime();

            elapsedTimeInsert = endInsert - startInsert;

            System.out.println("KLL Sketch");
            System.out.println(elapsedTimeInsert);
            System.out.println(TimeUnit.NANOSECONDS.toMicros(elapsedTimeInsert));
            System.out.println(TimeUnit.NANOSECONDS.toMillis(elapsedTimeInsert));

            System.out.println(kllsketch.toString(true, false));

            double[] percentilesAdaptability = {.05, .25, .50, .75, .90, .95, .98};

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

            /* PARETO GENERATOR
             */
            ParetoDistribution ptoD = new ParetoDistribution(1, 1);
            ArrayList<Double> ad3 = new ArrayList<Double>();
            for (int i = 0; i < 100; i++){
                double sampled_value = ptoD.sample();
                //ddsketch.accept(sampled_value);
                //kllsketch.update((float) sampled_value);
                //msketch.add(sampled_value);
                ad3.add(sampled_value);
            }
            System.out.println("Pareto min max");
            System.out.println(Collections.min(ad3));
            System.out.println(Collections.max(ad3));

            /* UNIFORM GENERATOR
             */

            /*
            ArrayList<DDSketch> ddSketches = new ArrayList<DDSketch>();
            UniformRealDistribution uD = new UniformRealDistribution(1, 5000);
            for (int i = 0; i < dataSize; i++){
                ddsketch.accept(uD.sample());
                kllsketch.update((float) uD.sample());
                msketch.add(uD.sample());
            }
            */

            /* NORMAL GENERATOR
            NormalDistribution valnD = new NormalDistribution(100, 15);
            for (int i = 0; i < dataSize; i++){
                ddsketch.accept(valnD.sample());
                kllsketch.update(valnD.sample());
                msketch.add(valnD.sample());
            }
             */

            SimpleMomentSketch ex_ms = new SimpleMomentSketch(18);
            DDSketch ex_dds = new DDSketch(0.01);
            double[] example_data = {1, 2, 3, 4, 5, 6, 7, 18, 30, 51};
            for (int i = 0; i < example_data.length; i++){
                ex_ms.add(example_data[i]);
                ex_dds.accept(example_data[i]);
            }

            double[] ex_quantiles = {0.8, 0.9};

            System.out.println("example data output");
            System.out.println("Moments Sketch 0.8: " + ex_ms.getQuantiles(ex_quantiles)[0]);
            System.out.println("Moments Sketch 0.9: " + ex_ms.getQuantiles(ex_quantiles)[1]);
            System.out.println("DDSketch: " + ex_dds.getValuesAtQuantiles(ex_quantiles)[0]);
            System.out.println("DDSketch: " + ex_dds.getValuesAtQuantiles(ex_quantiles)[1]);

            // Real World Workloads
            String line = "";
            String splitBy = ",";

            /* NYT FARES GENERATOR
             */
            BufferedReader br = new BufferedReader(new FileReader("/Users/harshbindra/flink-benchmarks/nyt-data.csv"));
            ArrayList<Double> ad = new ArrayList<Double>();
            double min = 0;
            double max = 0;
            while ((line = br.readLine()) != null)  {
                String[] line_array = line.split(splitBy);    // use comma as separator
                double d = Double.parseDouble(line_array[16]);
                //ddsketch.accept(d);
                //kllsketch.update(d;
                //msketch.add(d);
                ad.add(d);
            }
            System.out.println("NYT min max");
            System.out.println(Collections.min(ad));
            System.out.println(Collections.max(ad));


            String splitBy2 = ";";
            BufferedReader br2 = new BufferedReader(new FileReader("/Users/harshbindra/flink-benchmarks/household_power_consumption.txt"));
            //BufferedReader br2 = new BufferedReader(new FileReader("/Users/harshbindra/flink-benchmarks/empty_file.txt"));
            ArrayList<Double> ad2 = new ArrayList<Double>();
            while ((line = br2.readLine()) != null) {
                String[] line_array = line.split(splitBy2);    // use comma as separator
                double d = Double.parseDouble(line_array[2]);
                ad2.add(d);

                //ddsketch.accept(d);

                //kllsketch.update(d;

                //msketch.add(d);
            }

            System.out.println("power min max");
            System.out.println(Collections.min(ad2));
            System.out.println(Collections.max(ad2));

            /*

            long start = System.nanoTime();

            MomentSolver ms = new MomentSolver(msketch);
            ms.setGridSize(1024);
            ms.solve();

            double[] results_moments = ms.getQuantiles(percentiles);

            long end = System.nanoTime();

            long elapsed_time = end - start;

            System.out.println("Data Size: " + dataSize);
            System.out.println(TimeUnit.NANOSECONDS.toMicros(elapsed_time));
            System.out.println(elapsed_time);

            start = System.nanoTime();

            double[] results_ddsketch = ddsketch.getValuesAtQuantiles(percentiles);

            end = System.nanoTime();
            elapsed_time = end - start;

            System.out.println(TimeUnit.NANOSECONDS.toMicros(elapsed_time));
            System.out.println(elapsed_time);

            start = System.nanoTime();

            float[] results_kll = kllsketch.getQuantiles(percentiles);

            end = System.nanoTime();
            elapsed_time = end - start;

            System.out.println(TimeUnit.NANOSECONDS.toMicros(elapsed_time));
            System.out.println(elapsed_time);

            // Requires a sorted set of values
            //int[] percentiles = {1, 5, 25, 50, 75, 90, 95, 98, 99};
            for (int i = 0 ; i < results_ddsketch.length ; i++){
                System.out.println(results_moments[i]);
                System.out.println(results_ddsketch[i]);
                System.out.println(results_kll[i]);
            }
             */

        } catch (FileNotFoundException e) {
            System.out.println("File not found exception occurred.");
            e.printStackTrace();
        } catch (IOException e){
            System.out.println("IO exception occurred.");
            e.printStackTrace();
        }

    }

    static private double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        BigDecimal bd = new BigDecimal(Double.toString(value));
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }

    public static Double percentile(ArrayList<Double> sort_values, double percentile) {
        assert(percentile > 0 && percentile < 1);
        int index = (int) Math.ceil(percentile * sort_values.size());
        return sort_values.get(index-1);
    }

    public static ArrayList<Double> getPercentiles(ArrayList<Double> all_events, double[] percentiles){
        ArrayList<Double> ret_list = new ArrayList<Double>();
        Collections.sort(all_events);
        for (double p : percentiles){
            ret_list.add(round(percentile(all_events, p), 4));
        }
        return ret_list;
    }


}
