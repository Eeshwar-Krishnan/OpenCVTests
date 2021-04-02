package main.java.Utils;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import java.util.*;

public class BetterTowerGoalUtils {

    public static Mat cropInRange(Mat inMat){
        Mat hsv = new Mat();
        Imgproc.cvtColor(inMat, hsv, Imgproc.COLOR_RGB2HSV);
        //Imgproc.cvtColor(inMat, inMat, Imgproc.COLOR_RGB2GRAY);
        Imgproc.resize(hsv, hsv, new Size(inMat.width(), inMat.height())); //[112.0, 113.0, 132.0, 0.0] [122.0, 255.0, 255.0, 0.0]
        //[114.0, 134.0, 144.0, 0.0] [119.0, 255.0, 255.0, 0.0]
        Scalar min = new Scalar(112, 113, 132);
        Scalar max = new Scalar(122, 255, 255);
        Mat outMat = new Mat();
        Core.inRange(hsv, min, max, outMat);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(10, 10));
        Imgproc.morphologyEx(outMat, outMat, Imgproc.MORPH_OPEN, kernel);

        hsv.release();

        return outMat;
    }

    public static Mat normCropInRange(Mat inMat){
        Mat hsv = new Mat();
        Imgproc.cvtColor(inMat, hsv, Imgproc.COLOR_RGB2HSV);
        //Imgproc.cvtColor(inMat, inMat, Imgproc.COLOR_RGB2GRAY);
        //Imgproc.resize(hsv, hsv, new Size(inMat.width(), inMat.height())); //[112.0, 113.0, 132.0, 0.0] [122.0, 255.0, 255.0, 0.0]
        //[114.0, 134.0, 144.0, 0.0] [119.0, 255.0, 255.0, 0.0]
        Scalar min = new Scalar(112, 113, 132);
        Scalar max = new Scalar(122, 255, 255);
        Mat outMat = new Mat();
        Core.inRange(hsv, min, max, outMat);

        hsv.release();

        return outMat;
    }

    public static MatOfPoint getReference(Mat referenceMat)  {
        Mat grayReferenceMat = new Mat();
        Imgproc.cvtColor(referenceMat, grayReferenceMat, Imgproc.COLOR_BGR2GRAY);

        Imgproc.resize(grayReferenceMat, grayReferenceMat, new Size(500, 500));

        Mat cannyOut = new Mat();
        Imgproc.Canny(grayReferenceMat, cannyOut, 100, 200);

        List<MatOfPoint> referenceContours = new ArrayList<>();
        Imgproc.findContours(cannyOut, referenceContours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        Imgproc.cvtColor(cannyOut, cannyOut, Imgproc.COLOR_GRAY2BGR);

        double maxVal = 0;
        int maxValIdx = 0;
        for (int contourIdx = 0; contourIdx < referenceContours.size(); contourIdx++)
        {
            double contourArea = Imgproc.contourArea(referenceContours.get(contourIdx));
            if (maxVal < contourArea)
            {
                maxVal = contourArea;
                maxValIdx = contourIdx;
            }
        }

        MatOfPoint match = referenceContours.get(maxValIdx);

        MatOfPoint2f contourMat = new MatOfPoint2f(match.toArray());
        MatOfPoint2f poly = new MatOfPoint2f();
        Imgproc.approxPolyDP(contourMat, poly, 0.01 * Imgproc.arcLength(contourMat, true), true);

        MatOfPoint toReturn = new MatOfPoint(poly.toArray());

        grayReferenceMat.release();
        cannyOut.release();
        referenceContours.clear();
        match.release();
        contourMat.release();
        poly.release();

        return toReturn;
    }

    public static MatOfPoint getNormReference(Mat referenceMat)  {
        Mat grayReferenceMat = new Mat();
        Imgproc.cvtColor(referenceMat, grayReferenceMat, Imgproc.COLOR_BGR2GRAY);

        Imgproc.resize(grayReferenceMat, grayReferenceMat, new Size(500, 500));

        Mat cannyOut = new Mat();
        Imgproc.Canny(grayReferenceMat, cannyOut, 100, 200);

        List<MatOfPoint> referenceContours = new ArrayList<>();
        Imgproc.findContours(cannyOut, referenceContours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

        Imgproc.cvtColor(cannyOut, cannyOut, Imgproc.COLOR_GRAY2BGR);

        double maxVal = 0;
        int maxValIdx = 0;
        for (int contourIdx = 0; contourIdx < referenceContours.size(); contourIdx++)
        {
            double contourArea = Imgproc.contourArea(referenceContours.get(contourIdx));
            if (maxVal < contourArea)
            {
                maxVal = contourArea;
                maxValIdx = contourIdx;
            }
        }

        MatOfPoint match = referenceContours.get(maxValIdx);

        grayReferenceMat.release();
        cannyOut.release();
        referenceContours.clear();

        return match;
    }

    public static MatOfPoint getContour(Mat inMat)  {
        List<MatOfPoint> referenceContours = new ArrayList<>();
        Imgproc.findContours(inMat, referenceContours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        double maxVal = 0;
        int maxValIdx = 0;
        for (int contourIdx = 0; contourIdx < referenceContours.size(); contourIdx++)
        {
            double contourArea = Imgproc.contourArea(referenceContours.get(contourIdx));
            if (maxVal < contourArea)
            {
                maxVal = contourArea;
                maxValIdx = contourIdx;
            }
        }

        if(referenceContours.size() > 0) {
            MatOfPoint toReturn = new MatOfPoint(referenceContours.get(maxValIdx).toArray());
            for(MatOfPoint m : referenceContours){
                m.release();
            }
            return toReturn;
        }else{
            return new MatOfPoint();
        }
    }

    public static MatOfPoint getNormContour(Mat inMat)  {
        List<MatOfPoint> referenceContours = new ArrayList<>();
        Imgproc.findContours(inMat, referenceContours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);

        double maxVal = 0;
        int maxValIdx = 0;
        for (int contourIdx = 0; contourIdx < referenceContours.size(); contourIdx++)
        {
            double contourArea = Imgproc.contourArea(referenceContours.get(contourIdx));
            if (maxVal < contourArea)
            {
                maxVal = contourArea;
                maxValIdx = contourIdx;
            }
        }

        if(referenceContours.size() > 0) {
            MatOfPoint toReturn = new MatOfPoint(referenceContours.get(maxValIdx).toArray());
            for(MatOfPoint m : referenceContours){
                m.release();
            }
            return toReturn;
        }else{
            return new MatOfPoint();
        }
    }

    public static MatOfPoint getChainedContour(Mat inMat)  {
        List<MatOfPoint> referenceContours = new ArrayList<>();
        Imgproc.findContours(inMat, referenceContours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        double maxVal = 0;
        int maxValIdx = 0;
        for (int contourIdx = 0; contourIdx < referenceContours.size(); contourIdx++)
        {
            double contourArea = Imgproc.contourArea(referenceContours.get(contourIdx));
            if (maxVal < contourArea)
            {
                maxVal = contourArea;
                maxValIdx = contourIdx;
            }
        }

        if(referenceContours.size() > 0) {
            MatOfPoint toReturn = new MatOfPoint(referenceContours.get(maxValIdx).toArray());
            for(MatOfPoint m : referenceContours){
                m.release();
            }
            return toReturn;
        }else{
            return new MatOfPoint();
        }
    }

    public static MatOfPoint getMatchRect(MatOfPoint refContour, Mat inMat){
        //Mat resized = new Mat();
        //Imgproc.resize(inMat, resized, new Size(640, 480));
        Mat hsv = new Mat();
        Imgproc.cvtColor(inMat, hsv, Imgproc.COLOR_RGB2HSV);
        Scalar min = new Scalar(112, 113, 132);
        Scalar max = new Scalar(122, 255, 255);

        Mat outMat = new Mat();
        Core.inRange(hsv, min, max, outMat);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(10, 10));
        Imgproc.morphologyEx(outMat, outMat, Imgproc.MORPH_OPEN, kernel);

        kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(10, 10));
        Imgproc.morphologyEx(outMat, outMat, Imgproc.MORPH_CLOSE, kernel);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(outMat, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);

        int idx = -1;
        double minDist = Double.MAX_VALUE;

        ArrayList<MatOfPoint2f> dpPoints = new ArrayList<>();

        Rect matchRect = Imgproc.boundingRect(refContour);

        for(int i = 0; i < contours.size(); i ++){
            MatOfPoint contour = contours.get(i);
            Rect rect = Imgproc.boundingRect(contour);
            rect = CvUtils.enlargeROI(outMat, rect, 10);
            Mat cropped = outMat.submat(rect);

            Imgproc.resize(cropped, cropped, new Size(500, 500));

            MatOfPoint croppedContour = getContour(cropped);

            MatOfPoint2f points = new MatOfPoint2f();

            MatOfPoint2f croppedContour2f = new MatOfPoint2f(croppedContour.toArray());
            MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
            MatOfPoint2f refContour2f = new MatOfPoint2f(refContour.toArray());

            Imgproc.approxPolyDP(croppedContour2f, points, 0.02 * (Imgproc.arcLength(croppedContour2f, true)), true);
            MatOfPoint pointsNorm = new MatOfPoint(points.toArray());

            double distance = Imgproc.matchShapes(pointsNorm, refContour, Imgproc.CONTOURS_MATCH_I3, 0);

            Imgproc.approxPolyDP(contour2f, points, 0.02 * (Imgproc.arcLength(contour2f, true)), true);

            rect = Imgproc.boundingRect((pointsNorm));
            double aspect = ((double)(rect.height)) / ((double)rect.width);
            dpPoints.add(points);

            double numMatch = matchPoints(refContour2f, points, matchRect, Imgproc.minAreaRect(points));

            //System.out.println(distance + " | " + rect.area() + " | " + points.toArray().length + " | " + aspect + " | " + numMatch);

            if(distance < minDist && rect.area() > 100000 && points.toArray().length < 10 && points.toArray().length > 4){
                if(aspect > 0.56 && aspect < 2) {
                    if(numMatch > 4) {
                        //System.out.println("up");
                        minDist = distance;
                        idx = i;
                    }
                }
            }

            contour.release();
            cropped.release();
            croppedContour.release();
            pointsNorm.release();
            croppedContour2f.release();
            contour2f.release();
            refContour2f.release();
        }

        hsv.release();
        outMat.release();
        kernel.release();
        contours.clear();

        MatOfPoint toReturn = new MatOfPoint();
        if(idx != -1){
            if(minDist < 0.2) {
                toReturn.release();
                toReturn = new MatOfPoint(dpPoints.get(idx).toArray());
                for(MatOfPoint2f mp2f : dpPoints){
                    mp2f.release();
                }
                dpPoints.clear();
            }
        }

        return toReturn;
    }

    public static double matchPoints(MatOfPoint2f ref, MatOfPoint2f match, Rect refRect, RotatedRect matchRect){
        double epsilon = 0.175;

        int numMatch = 0;
        for(Point p : ref.toArray()) {
            for(Point pt : match.toArray()) {
                Point scaledRef = new Point((p.x - refRect.x)/refRect.width, (p.y - refRect.y)/refRect.height);
                Point scaledMatch = new Point((pt.x - matchRect.boundingRect().x)/matchRect.boundingRect().width, (pt.y - matchRect.boundingRect().y)/matchRect.boundingRect().height);

                double xDelta = scaledMatch.x - scaledRef.x;
                double yDelta = scaledMatch.y - scaledRef.y;
                double delta = Math.sqrt((xDelta * xDelta) + (yDelta * yDelta));
                if(delta < epsilon){
                    numMatch ++;
                    break;
                }
            }
        }
        return numMatch;
    }

    public static MatOfPoint refineEstimation(Mat inMat, Rect boundingRect, Mat outMat){
        Mat inClone = inMat.clone();
        Rect enlarged = CvUtils.enlargeROI(inClone, boundingRect, 10);

        Mat cropped = inClone.submat(enlarged);

        Mat extracted = cropInRange(cropped);
        //Core.extractChannel(cropped, extracted, 2);
        //Imgproc.adaptiveThreshold(extracted, extracted, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);

        Imgproc.cvtColor(extracted, outMat, Imgproc.COLOR_GRAY2BGR);

        MatOfPoint contour = getContour(extracted);
        MatOfPoint2f cnt2f = new MatOfPoint2f(contour.toArray());
        if(cnt2f.toArray().length > 0 && false) {
            MatOfPoint2f dp = new MatOfPoint2f();
            Imgproc.approxPolyDP(cnt2f, dp, 0.02 * Imgproc.arcLength(cnt2f, true), true);
            contour = new MatOfPoint(dp.toArray());
            dp.release();
        }

        MatOfPoint scl = new MatOfPoint();
        for(Point p : contour.toArray()){
            Point tmp = new Point(p.x + enlarged.x, p.y + enlarged.y);
            scl.push_back(new MatOfPoint(tmp));
        }

        inClone.release();
        cropped.release();
        extracted.release();
        contour.release();

        return scl;
    }

    public static void drawPOI(Mat in, Rect bounds){
        double inset = 0.1;
        //inset = 0;
        double maxInset = 0.2;
        //maxInset = 0;
        double yInst = 0.5;
        double maxYInset = 0.4;

        Rect newBound = new Rect((int)(bounds.x + (bounds.width * inset)), (int) (bounds.y + (bounds.height * yInst)), (int)(bounds.width * (1-maxInset)), (int) (bounds.height * (1-maxYInset)));

        //System.out.println("Bounds " + newBound + " | " + bounds);

        Mat inCopy = in.submat(newBound);
        Mat inNorm = BetterTowerGoalUtils.cropInRange(inCopy);

        //Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        //Imgproc.morphologyEx(inNorm, inNorm, Imgproc.MORPH_OPEN, kernel);

        Mat fullMask = new Mat(inNorm.rows(), inNorm.cols(), inNorm.type(), new Scalar(255,255,255));
        Mat subMask = new Mat();
        Core.subtract(fullMask, inNorm, subMask);

        Imgproc.cvtColor(in, in, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(in, in, Imgproc.COLOR_GRAY2BGR);

        MatOfPoint corners = new MatOfPoint();
        double qualityLevel = 0.01;
        double minDistance = 10;
        int blockSize = 3, gradientSize = 3;
        boolean useHarrisDetector = true;
        double k = 0.04;

        Imgproc.goodFeaturesToTrack(subMask, corners, 4, qualityLevel, minDistance, new Mat(), blockSize, gradientSize, useHarrisDetector, k);

        int[] cornersData = new int[(int) (corners.total() * corners.channels())];
        corners.get(0, 0, cornersData);
        Mat matCorners = new Mat(corners.rows(), 2, CvType.CV_32F);
        float[] matCornersData = new float[(int) (matCorners.total() * matCorners.channels())];
        matCorners.get(0, 0, matCornersData);
        for (int i = 0; i < corners.rows(); i++) {
            matCornersData[i * 2] = cornersData[i * 2];
            matCornersData[i * 2 + 1] = cornersData[i * 2 + 1];
        }
        matCorners.put(0, 0, matCornersData);

        Size winSize = new Size(5, 5);
        Size zeroZone = new Size(-1, -1);
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.COUNT, 40, 0.001);

        Imgproc.cornerSubPix(subMask, matCorners, winSize, zeroZone, criteria);

        matCorners.get(0, 0, matCornersData);

        Scalar[] colors = new Scalar[]{ new Scalar(255, 75, 75), new Scalar(0, 255, 0), new Scalar(0, 0, 255), new Scalar(255, 0, 255) };

        Point[] points  = new Point[4];
        for(int i = 0; i < corners.rows(); i ++){
            Point tmp = new Point((newBound.width/2.0), (newBound.height/2.0));
            Point p = new Point(matCornersData[i * 2], matCornersData[i * 2 + 1]);

            //System.out.println("P " + p + " Tmp " + tmp);

            if(p.x < tmp.x && p.y < tmp.y){
                points[0] = p;
            }
            if(p.x > tmp.x && p.y < tmp.y){
                points[1] = p;
            }
            if(p.x < tmp.x && p.y > tmp.y){
                points[2] = p;
            }
            if(p.x > tmp.x && p.y > tmp.y){
                points[3] = p;
            }
        }

        MatOfPoint imgPoints = new MatOfPoint();
        for (int i = 0; i < points.length; i ++) {
            Point p = points[i];
            if(!(p == null)) {
                Point newPoint = new Point(p.x + newBound.x, p.y + newBound.y);
                //Point newPoint = new Point(p.x, p.y);
                Imgproc.circle(in, newPoint, 3, colors[i], Imgproc.FILLED);
                //System.out.println("(" + newPoint.x + ", " + newPoint.y + ")");
                imgPoints.push_back(new MatOfPoint(newPoint));
            }
        }
    }
}
