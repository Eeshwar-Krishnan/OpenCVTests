package main.java.Pipelines.Basic;

import main.java.Utils.BetterTowerGoalUtils;
import main.java.Utils.CvUtils;
import main.java.Utils.PnPUtils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;

public class PipelineTester {
    private Mat cropCopy, test;
    private MatOfPoint refContour;
    private Rect boundingRect;

    public PipelineTester(){
        boundingRect = new Rect(0, 0, 0, 0);
        cropCopy = new Mat();
        test = new Mat();
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
        Mat resized = new Mat();
        Imgproc.resize(input, resized, new Size(640, 480));
        //resized = input.clone();
        MatOfPoint m = BetterTowerGoalUtils.getMatchRect(refContour, resized);

        input.copyTo(cropCopy);
        boundingRect = Imgproc.boundingRect(m);
        m = BetterTowerGoalUtils.refineEstimation(resized, boundingRect, test);

        MatOfPoint scl = new MatOfPoint(), scl2 = new MatOfPoint();
        for(Point p : m.toArray()){
            scl.push_back(new MatOfPoint(new Point((p.x/640) * input.width(), (p.y/480) * input.height())));
            scl2.push_back(new MatOfPoint(new Point((p.x/640) * 1280, (p.y/480) * 720)));
        }

        boundingRect = Imgproc.boundingRect(scl);

        Imgproc.rectangle(cropCopy, boundingRect, new Scalar(0, 255, 0), 5);

        if(boundingRect.area() > 5) {
            //CvUtils.drawRRect(cropCopy, Imgproc.minAreaRect(new MatOfPoint2f(scl.toArray())), new Scalar(0, 255, 0), 5);

            Imgproc.rectangle(cropCopy, new Point(0, input.height() - 200), new Point(350, input.height()), new Scalar(255, 255, 255), -1);

            //817.063304531327
            double horDist = BetterTowerGoalUtils.approximateDistanceToGoal(23.87500, Imgproc.boundingRect(scl2).width, 817.063304531327);
            double verDist = BetterTowerGoalUtils.approximateDistanceToGoal(15.75, Imgproc.boundingRect(scl2).height, 819.4690054531818);

            double[] tmp = PnPUtils.getPitchAndYaw(input, boundingRect, cropCopy);

            double goalWallDist = (horDist + verDist)/2;

            DecimalFormat format = new DecimalFormat("#.##");
            Imgproc.putText(cropCopy, "Firing Solution: ", new Point(2, input.height() - 170), 1, 2, new Scalar(0, 0, 0));
            //Imgproc.putText(cropCopy, "Pitch " + format.format(tmp[0]), new Point(2, input.height() - 100), 1, 3, new Scalar(0, 0, 0));
            Imgproc.putText(cropCopy, "Range " + format.format((goalWallDist)), new Point(2, input.height() - 100), 1, 3, new Scalar(0, 0, 0));
            Imgproc.putText(cropCopy, "Heading " + format.format(tmp[1]), new Point(2, input.height() - 10), 1, 2.75, new Scalar(0, 0, 0));
            //Imgproc.putText(cropCopy, "Err " + format.format(horDist-verDist), new Point(2, input.height() - 10), 1, 3, new Scalar(0, 0, 0));
        }
        m.release();

        return cropCopy;
    }

    public Rect getBoundingRect() {
        return boundingRect;
    }
}