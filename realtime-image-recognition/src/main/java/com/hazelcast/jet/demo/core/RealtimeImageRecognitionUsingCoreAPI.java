package com.hazelcast.jet.demo.core;

import com.hazelcast.jet.Jet;
import com.hazelcast.jet.JetInstance;
import com.hazelcast.jet.core.DAG;
import com.hazelcast.jet.core.Processor;
import com.hazelcast.jet.core.TimestampKind;
import com.hazelcast.jet.core.Vertex;
import com.hazelcast.jet.core.processor.DiagnosticProcessors;
import com.hazelcast.jet.datamodel.TimestampedEntry;
import com.hazelcast.jet.demo.SerializableBufferedImage;
import com.hazelcast.jet.demo.WebcamSource;
import com.hazelcast.jet.function.DistributedFunction;
import com.hazelcast.jet.function.DistributedSupplier;
import com.hazelcast.jet.function.DistributedToLongFunction;
import com.hazelcast.jet.pipeline.TumblingWindowDef;
import com.hazelcast.jet.pipeline.WindowDefinition;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map.Entry;

import static com.hazelcast.jet.aggregate.AggregateOperations.maxBy;
import static com.hazelcast.jet.core.Edge.between;
import static com.hazelcast.jet.core.ProcessorSupplier.of;
import static com.hazelcast.jet.core.WatermarkEmissionPolicy.emitByFrame;
import static com.hazelcast.jet.core.WatermarkGenerationParams.wmGenParams;
import static com.hazelcast.jet.core.WatermarkPolicies.limitingTimestampAndWallClockLag;
import static com.hazelcast.jet.core.processor.Processors.accumulateByFrameP;
import static com.hazelcast.jet.core.processor.Processors.combineToSlidingWindowP;
import static com.hazelcast.jet.core.processor.Processors.insertWatermarksP;
import static com.hazelcast.jet.function.DistributedComparator.comparingDouble;
import static java.util.Collections.singletonList;

/**
 * An application which uses webcam frame stream as input and classifies those frames
 * with a model pre-trained with CIFAR-10 dataset.
 * Frames constituting a second of stream will be aggregated together to find
 * maximum scored classification and that will be sent a GUI sink to be shown on the screen.
 */
public class RealtimeImageRecognitionUsingCoreAPI {

    static {
        System.setProperty("hazelcast.logging.type", "slf4j");
    }

    public static void main(String[] args) {
        Path modelPath = Paths.get(args[0]).toAbsolutePath();

        JetInstance jet = Jet.newJetInstance();

        DAG dag = buildDAG(modelPath.toString());
        try {
            jet.newJob(dag).join();
        } finally {
            Jet.shutdownAll();
        }
    }


    private static DAG buildDAG(String modelPath) {
        DAG dag = new DAG();

        Vertex webcamSource = dag.newVertex("webcam source", WebcamSource.metaSupplier());
        Vertex classifierVertex = dag.newVertex("classifier", of(() -> new ClassifierProcessor(modelPath)));

        TumblingWindowDef tumbling = WindowDefinition.tumbling(1000);
        DistributedSupplier<Processor> insertWMP = insertWatermarksP(wmGenParams(
                (DistributedToLongFunction<TimestampedEntry>) TimestampedEntry::getTimestamp,
                limitingTimestampAndWallClockLag(500, 500),
                emitByFrame(tumbling.toSlidingWindowPolicy()),
                60000L
        ));
        Vertex insertWm = dag.newVertex("insertWm", insertWMP);

        Vertex localMaxScore = dag.newVertex("localMaxScore", accumulateByFrameP(
                singletonList((DistributedFunction<TimestampedEntry, String>) input -> "MAX_SCORE"),
                singletonList((DistributedToLongFunction<TimestampedEntry>) TimestampedEntry::getTimestamp),
                TimestampKind.EVENT,
                tumbling.toSlidingWindowPolicy(),
                maxBy(comparingDouble((Entry<SerializableBufferedImage, Entry<String, Double>> input) -> {
                            Entry<String, Double> maxScoredCategory = input.getValue();
                            return maxScoredCategory.getValue();
                        })
                ))
        ).localParallelism(1);
        Vertex globalMaxScore = dag.newVertex("globalMaxScore", combineToSlidingWindowP(
                tumbling.toSlidingWindowPolicy(),
                maxBy(comparingDouble((Entry<SerializableBufferedImage, Entry<String, Double>> input) -> {
                            Entry<String, Double> maxScoredCategory = input.getValue();
                            return maxScoredCategory.getValue();
                        })
                ),
                TimestampedEntry::new)).localParallelism(1);

        Vertex guiSink = dag.newVertex("gui", DiagnosticProcessors.peekInputP(GUISink.metaSupplier()));

        dag.edge(between(webcamSource, classifierVertex))
           .edge(between(classifierVertex, insertWm))
           .edge(between(insertWm, localMaxScore).partitioned(o -> "MAX_SCORE"))
           .edge(between(localMaxScore, globalMaxScore).partitioned(input -> "gui").distributed())
           .edge(between(globalMaxScore, guiSink).partitioned(input -> "gui"));
        return dag;
    }

}
