package main.java;

import main.java.Utils.CvUtils;
import main.java.Utils.TowerGoalUtils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;
import java.util.regex.Matcher;

public class TowerGoalRunner {

    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) throws IOException {
        //MatOfPoint referenceCountour = TowerGoalUtils.getReference();

        Mat refMat = CvUtils.bufferedImageToMat(ImageIO.read(new File("src/assets/ref.jpg")));

        MatOfPoint refContour = TowerGoalUtils.getReference(refMat);

        Mat greyRef = new Mat();
        Imgproc.cvtColor(refMat, greyRef, Imgproc.COLOR_BGR2GRAY);

        String annoFolder = "src/assets/annotations";

        String outFile = "src/assets/outBounding";//out247

        runAll(refContour, annoFolder, outFile);

        //runOne(refContour, refMat);
    }

    public static void runOne(MatOfPoint refContour, Mat refMat) throws IOException {
        long start = System.currentTimeMillis();
        Mat matchMat = CvUtils.bufferedImageToMat(ImageIO.read(new File("src/assets/annotations/out631.png")));//432, 631

        long pt1 = System.currentTimeMillis() - start;
        if(matchMat.width() > 1280)
            Imgproc.resize(matchMat, matchMat, new Size(matchMat.width()/3, matchMat.height()/3));

        //MatOfPoint refContour = TowerGoalUtils.getReference(refMat);

        Mat croppedMatchMat = TowerGoalUtils.cropInRange(matchMat);
        //ImageIO.write(CvUtils.toBufferedImage(HighGui.toBufferedImage(croppedMatchMat)), "PNG", new File("src/assets/outCropped/out" + i + ".png"));
        long pt2 = System.currentTimeMillis() - start;
        Mat writeMat = new Mat();
        MatOfPoint m = TowerGoalUtils.getMatchRect(refContour, matchMat, writeMat);
        Rect r = Imgproc.boundingRect(m);
        long pt3 = System.currentTimeMillis() - start;
        Mat cropCopy = new Mat();
        matchMat.copyTo(cropCopy);
        Mat testCnt = new Mat();
        matchMat.copyTo(testCnt);
        Imgproc.cvtColor(croppedMatchMat, croppedMatchMat, Imgproc.COLOR_GRAY2BGR);
        Imgproc.rectangle(cropCopy, r, new Scalar(0, 255, 0), 5);

        Imgproc.resize(cropCopy, cropCopy, matchMat.size());

        //Imgproc.cvtColor(cropCopy, cropCopy, Imgproc.COLOR_BGR2GRAY);
        //Imgproc.cvtColor(cropCopy, cropCopy, Imgproc.COLOR_GRAY2BGR);

        //if(m.total() > 0 && (m.depth() == CvType.CV_32F || m.depth() == CvType.CV_32S))
        //    TowerGoalUtils.getPos(new MatOfPoint2f(refContour.toArray()), new MatOfPoint2f(m.toArray()), Imgproc.boundingRect(refContour), Imgproc.minAreaRect(new MatOfPoint2f(m.toArray())), cropCopy);

        Mat outMat = new Mat();
        //TowerGoalUtils.drawKeypoints(matchMat, r);

        BufferedImage bi = CvUtils.toBufferedImage(HighGui.toBufferedImage(writeMat));

        //ImageIO.write(bi, "PNG", new File(outFile + "/out" + i + ".png"));
        long pt4 = System.currentTimeMillis() - start;
        ArrayList<Mat> topMats = new ArrayList<>();
        ArrayList<Mat> botMats = new ArrayList<>();
        topMats.add(matchMat);
        topMats.add(croppedMatchMat);
        botMats.add(writeMat);
        botMats.add(cropCopy);

        Mat top = new Mat();
        Mat bottom = new Mat();
        Core.hconcat(topMats, top);
        Core.hconcat(botMats, bottom);
        Mat combined = new Mat();
        Core.vconcat(Arrays.asList(top, bottom), combined);

        long pt5 = System.currentTimeMillis() - start;
        System.out.println("Timings: " + pt1 + " | " + pt2 + " | " + pt3 + " | " + pt4 + " | " + pt5);

        HighGui.imshow("Test", combined);
        HighGui.waitKey();
    }

    public static void runAll(MatOfPoint refContour, String annoFolder, String outFile) throws IOException {
        ArrayList<Mat> arrWrite = new ArrayList<>();
        DecimalFormat dec = new DecimalFormat("#.##");
        for(int i = 1; i < 900; i ++){
            long start = System.currentTimeMillis();
            Mat matchMat = CvUtils.bufferedImageToMat(ImageIO.read(new File(annoFolder + "/out" + i + ".png")));
            long pt1 = System.currentTimeMillis() - start;
            //if(matchMat.width() > 1280)
                //Imgproc.resize(matchMat, matchMat, new Size(matchMat.width()/3, matchMat.height()/3));

            //MatOfPoint refContour = TowerGoalUtils.getReference(refMat);

            Mat croppedMatchMat = TowerGoalUtils.cropInRange(matchMat);
            //ImageIO.write(CvUtils.toBufferedImage(HighGui.toBufferedImage(croppedMatchMat)), "PNG", new File("src/assets/outCropped/out" + i + ".png"));
            long pt2 = System.currentTimeMillis() - start;
            Mat writeMat = new Mat();
            MatOfPoint m = TowerGoalUtils.getMatchRect(refContour, matchMat, writeMat);
            Rect r = Imgproc.boundingRect(m);
            long pt3 = System.currentTimeMillis() - start;
            Mat cropCopy = new Mat();
            matchMat.copyTo(cropCopy);
            Mat testCnt = new Mat();
            matchMat.copyTo(testCnt);
            Imgproc.cvtColor(croppedMatchMat, croppedMatchMat, Imgproc.COLOR_GRAY2BGR);
            Imgproc.rectangle(cropCopy, r, new Scalar(0, 255, 0), 5);

            Imgproc.resize(cropCopy, cropCopy, matchMat.size());

            //Imgproc.cvtColor(cropCopy, cropCopy, Imgproc.COLOR_BGR2GRAY);
            //Imgproc.cvtColor(cropCopy, cropCopy, Imgproc.COLOR_GRAY2BGR);

            //if(m.total() > 0 && (m.depth() == CvType.CV_32F || m.depth() == CvType.CV_32S))
            //    TowerGoalUtils.getPos(new MatOfPoint2f(refContour.toArray()), new MatOfPoint2f(m.toArray()), Imgproc.boundingRect(refContour), Imgproc.minAreaRect(new MatOfPoint2f(m.toArray())), cropCopy);

            Mat outMat = new Mat();
            //TowerGoalUtils.drawKeypoints(matchMat, r);

            BufferedImage bi = CvUtils.toBufferedImage(HighGui.toBufferedImage(writeMat));

            //ImageIO.write(bi, "PNG", new File(outFile + "/out" + i + ".png"));
            long pt4 = System.currentTimeMillis() - start;
            ArrayList<Mat> topMats = new ArrayList<>();
            ArrayList<Mat> botMats = new ArrayList<>();
            topMats.add(matchMat);
            topMats.add(croppedMatchMat);
            botMats.add(writeMat);
            botMats.add(cropCopy);

            Mat top = new Mat();
            Mat bottom = new Mat();
            Core.hconcat(topMats, top);
            Core.hconcat(botMats, bottom);
            Mat combined = new Mat();
            Core.vconcat(Arrays.asList(top, bottom), combined);

            arrWrite.add(combined);
            long pt5 = System.currentTimeMillis() - start;
            System.out.println(" Finished " + i + " of " + 900 + " | " + pt1 + " | " + pt2 + " | " + pt3 + " | " + pt4 + " | " + pt5);
        }
        for(int i = 0; i < arrWrite.size(); i ++){
            Mat combined = arrWrite.get(i);
            ImageIO.write(CvUtils.toBufferedImage(HighGui.toBufferedImage(combined)), "JPG", new File("src/assets/out/out" + i + ".jpg"));
            System.out.println("Wrote " + i + " of " + arrWrite.size());
        }
    }
}
