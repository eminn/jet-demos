package com.hazelcast.jet.demo;

import boofcv.abst.scene.ImageClassifier;
import boofcv.abst.scene.ImageClassifier.Score;
import boofcv.deepboof.ImageClassifierVggCifar10;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.Planar;
import com.hazelcast.jet.core.AbstractProcessor;
import com.hazelcast.jet.impl.pipeline.JetEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Comparator;
import java.util.List;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import static com.hazelcast.jet.Traverser.over;
import static com.hazelcast.jet.Util.entry;
import static com.hazelcast.jet.impl.pipeline.JetEventImpl.jetEvent;

/**
 * date: 1/23/18
 * author: emindemirci
 */
public class ClassifierProcessor extends AbstractProcessor {


    private ImageClassifier<Planar<GrayF32>> classifier;
    private List<String> categories;
    private String modelPath;

    public ClassifierProcessor(String modelPath) {
        this.modelPath = modelPath;
    }

    @Override
    protected void init(Context context) throws Exception {
        classifier = new ImageClassifierVggCifar10();
        classifier.loadModel(new File(modelPath));
        categories = classifier.getCategories();
    }

    @Override
    protected boolean tryProcess(int ordinal, Object item) {
        JetEvent<SerializableBufferedImage> event = (JetEvent<SerializableBufferedImage>) item;
        SerializableBufferedImage serializableBufferedImage = event.payload();
        BufferedImage image = serializableBufferedImage.getImage();

        Planar<GrayF32> planar = new Planar<>(GrayF32.class, image.getWidth(), image.getHeight(), 3);
        ConvertBufferedImage.convertFromPlanar(image, planar, true, GrayF32.class);

        classifier.classify(planar);
        List<Score> results = classifier.getAllResults();
        List<Entry<String, Double>> categoryWithScores = results.stream().map(score -> entry(categories.get(score.category), score.score)).collect(Collectors.toList());
        Entry<String, Double> maxScoredCategory = categoryWithScores
                .stream()
                .max(Comparator.comparing(Entry::getValue))
                .get();
        return emitFromTraverser(over(jetEvent(entry(serializableBufferedImage, maxScoredCategory), event.timestamp())));
    }

}
