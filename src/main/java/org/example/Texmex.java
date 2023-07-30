package org.example;

import io.jhdf.HdfFile;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.*;
import org.example.util.ListRandomAccessVectorValues;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.stream.IntStream;

/**
 * Tests HNSW against vectors from the Texmex dataset
 */
public class Texmex {
    private record Results(int topK, double recall, long buildNanos, long fingerNanos, long queryNanos) {
    }

    public static Results testRecall(List<float[]> baseVectors, List<float[]> queryVectors, List<Set<Integer>> groundTruth) throws IOException {
        var ravv = new ListRandomAccessVectorValues(baseVectors, baseVectors.get(0).length);
        var topK = groundTruth.get(0).size();

        var start = System.nanoTime();
        var builder = HnswGraphBuilder.create(ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, 16, 100, 42);
        var hnsw = builder.build(ravv.copy());
        long buildNanos = System.nanoTime() - start;

        start = System.nanoTime();
        FingerMetadata<float[]> fm = null; // new FingerMetadata<>(hnsw.getView(), ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.DOT_PRODUCT, 64);
        long fingerNanos = System.nanoTime() - start;

        start = System.nanoTime();

        int queryRuns = 10;
        int topKFound = performQueries(queryVectors, groundTruth, ravv, hnsw, fm, topK, queryRuns);
        long queryNanos = System.nanoTime() - start;
        var recall = ((double) topKFound) / (queryRuns * queryVectors.size() * topK);
        return new Results(topK, recall, buildNanos, fingerNanos, queryNanos);
    }

    private static float normOf(float[] baseVector) {
        float norm = 0;
        for (float v : baseVector) {
            norm += v * v;
        }
        return (float) Math.sqrt(norm);
    }

    private static int performQueries(List<float[]> queryVectors, List<Set<Integer>> groundTruth, ListRandomAccessVectorValues ravv, HnswGraph hnsw, FingerMetadata<float[]> fm, int topK, int queryRuns) throws IOException {
        int topKfound = 0;
        for (int k = 0; k < queryRuns; k++) {
            HnswSearcher<float[]> searcher = new HnswSearcher.Builder<>(hnsw, ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE)
                        .withFinger(fm)
                        .build();
            for (int i = 0; i < queryVectors.size(); i++) {
                var queryVector = queryVectors.get(i);
                NeighborQueue nn;
                nn = searcher.search(queryVector, topK, null, Integer.MAX_VALUE);

                var gt = groundTruth.get(i);
                int[] resultNodes = nn.nodes();
                var n = IntStream.range(0, Math.min(nn.size(), topK)).filter(j -> gt.contains(resultNodes[j])).count();
                topKfound += n;
            }
        }
        return topKfound;
    }

    private static void computeRecallFor(String pathStr) throws IOException {
        HdfFile hdf = new HdfFile(Paths.get(pathStr));
        float[][] baseVectors = (float[][]) hdf.getDatasetByPath("train").getData();
        float[][] queryVectors = (float[][]) hdf.getDatasetByPath("test").getData();
        int[][] groundTruth = (int[][]) hdf.getDatasetByPath("neighbors").getData();

        // verify that vectors are normalized and sane
        List<float[]> scrubbedBaseVectors = new ArrayList<>(baseVectors.length);
        List<float[]> scrubbedQueryVectors = new ArrayList<>(queryVectors.length);
        List<Set<Integer>> gtSet = new ArrayList<>(groundTruth.length);
        if (true) {
            for (float[] v : baseVectors) {
                if (Math.abs(normOf(v)) > 1e-5) {
                    scrubbedBaseVectors.add(v);
                }
            }
            for (int i = 0; i < queryVectors.length; i++) {
                float[] v = queryVectors[i];
                if (Math.abs(normOf(v)) > 1e-5) {
                    scrubbedQueryVectors.add(v);
                    var gt = new HashSet<Integer>();
                    for (int j = 0; j < groundTruth[i].length; j++) {
                        gt.add(groundTruth[i][j]);
                    }
                    gtSet.add(gt);
                }
            }
        }
        assert scrubbedQueryVectors.size() == gtSet.size();
        baseVectors = null;
        queryVectors = null;
        groundTruth = null;

        System.out.format("%s: %d base and %d query vectors loaded, dimensions %d%n",
                pathStr, scrubbedBaseVectors.size(), scrubbedQueryVectors.size(), scrubbedBaseVectors.get(0).length);

        var results = testRecall(scrubbedBaseVectors, scrubbedQueryVectors, gtSet);
        System.out.format("%s: top %d recall %.4f, build %.2fs, finger %.2fs, query %.2fs%n",
                pathStr, results.topK, results.recall, results.buildNanos / 1_000_000_000.0, results.fingerNanos / 1_000_000_000.0, results.queryNanos / 1_000_000_000.0);
    }

    public static void main(String[] args) {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        new Thread(() -> {
            try {
                computeRecallFor("hdf5/glove-100-angular.hdf5");
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }).start();
        new Thread(() -> {
            try {
                computeRecallFor("hdf5/glove-200-angular.hdf5");
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }).start();
        new Thread(() -> {
            try {
                computeRecallFor("hdf5/deep-image-96-angular.hdf5");
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }).start();
        new Thread(() -> {
            try {
                computeRecallFor("hdf5/nytimes-256-angular.hdf5");
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }).start();
    }
}
