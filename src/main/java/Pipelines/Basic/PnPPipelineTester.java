package main.java.Pipelines.Basic;

import main.java.Utils.BetterTowerGoalUtils;
import main.java.Utils.CvUtils;
import main.java.Utils.PnPUtils;
import main.java.Utils.TowerGoalUtils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.awt.image.Kernel;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;

public class PnPPipelineTester {
    private Mat cropCopy;
    private MatOfPoint refContour;
    private Rect boundingRect;
    private Mat refMat, pos;

    public PnPPipelineTester(){
        boundingRect = new Rect(0, 0, 0, 0);
        cropCopy = new Mat();
        refMat = new Mat();
        try {
            refMat = CvUtils.bufferedImageToMat(ImageIO.read(new File("src/assets/ref.jpg")));
        } catch (IOException e) {
            e.printStackTrace();
        }
        pos = new Mat();
        refContour = BetterTowerGoalUtils.getReference(refMat);
        //refMat.release();
    }

    public Mat processFrame(Mat input) {
        Imgproc.resize(input, cropCopy, new Size(640, 480));

        MatOfPoint m = BetterTowerGoalUtils.getMatchRect(refContour, cropCopy);

        MatOfPoint scl = new MatOfPoint();
        for(Point p : m.toArray()){
            scl.push_back(new MatOfPoint(new Point((p.x/640) * input.width(), (p.y/480) * input.height())));
        }

        boundingRect = Imgproc.boundingRect(scl);

        Mat toReturn = input.clone();
        if(boundingRect.area() > 0) {
            try {
                double[] tmp = PnPUtils.getPitchAndYaw(input, boundingRect, toReturn);
                PnPUtils.solvePnP4(toReturn, Imgproc.minAreaRect(new MatOfPoint2f(scl.toArray())));

                Imgproc.rectangle(toReturn, new Point(0, input.height()-200), new Point(350, input.height()), new Scalar(255, 255, 255), -1);

                DecimalFormat format = new DecimalFormat("#.##");
                Imgproc.putText(toReturn, "Firing Solution: ", new Point(2, input.height()-170), 1, 2, new Scalar(0, 0, 0));
                Imgproc.putText(toReturn, "Pitch " + format.format(tmp[0]), new Point(2, input.height()-100), 1, 3, new Scalar(0, 0, 0));
                Imgproc.putText(toReturn, "Yaw " + format.format(tmp[1]), new Point(2, input.height()-10), 1, 3, new Scalar(0, 0, 0));

            }catch (Exception ignored){
                ignored.printStackTrace();
            }

        }

        m.release();
        scl.release();

        return toReturn;
    }

    public Rect getBoundingRect() {
        return boundingRect;
    }

    public Mat getPos() {
        return pos;
    }
}