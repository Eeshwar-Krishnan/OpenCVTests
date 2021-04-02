package main.java.Pipelines.Advanced;

import main.java.Utils.BetterTowerGoalUtils;
import main.java.Utils.CvUtils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class BetterTowergoalPipeline {
    private Mat cropCopy;
    private MatOfPoint refContour;
    private Rect boundingRect;

    public BetterTowergoalPipeline(){
        boundingRect = new Rect(0, 0, 0, 0);
        cropCopy = new Mat();
        Mat refMat = new Mat();
        try {
            refMat = CvUtils.bufferedImageToMat(ImageIO.read(new File("src/assets/ref.jpg")));
        } catch (IOException e) {
            e.printStackTrace();
        }

        refContour = BetterTowerGoalUtils.getReference(refMat);
        refMat.release();
    }

    public Mat processFrame(Mat input) {
        Mat redChannel = BetterTowerGoalUtils.cropInRange(input);

        Mat redThresh = new Mat();
        Imgproc.threshold(redChannel, redThresh, 155, 200, Imgproc.THRESH_BINARY);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(redThresh, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        Imgproc.cvtColor(redThresh, redThresh, Imgproc.COLOR_GRAY2BGR);

        contours = contours.stream().filter(i -> {
            double distance = Imgproc.matchShapes(i, refContour, Imgproc.CONTOURS_MATCH_I3, 0);
            Rect rect = Imgproc.boundingRect(i);
            double aspect = (double)rect.width/rect.height;
            return distance < 1 && (aspect > 1 && aspect < 2) && Imgproc.contourArea(i) > 500;
        }).collect(Collectors.toList());

        MatOfPoint bestContour = new MatOfPoint();
        if(!contours.isEmpty()){
            bestContour = Collections.max(contours, Comparator.comparingDouble(c0 -> Imgproc.boundingRect(c0).width));

            boundingRect = Imgproc.boundingRect(bestContour);
            cropCopy = input.submat(boundingRect);

            Imgproc.rectangle(redThresh, boundingRect, new Scalar(0, 0, 255), 3);
        }

        return redThresh;
    }

    public Rect getBoundingRect() {
        return boundingRect;
    }
}
