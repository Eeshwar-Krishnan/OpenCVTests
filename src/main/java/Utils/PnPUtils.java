package main.java.Utils;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class PnPUtils {
    public static Mat solvePnP(Mat in, Rect bounds){
        Rect rect = CvUtils.enlargeROI(in, bounds, 15);
        Mat toTest = in.submat(rect);

        Mat redChannel = BetterTowerGoalUtils.cropInRange(toTest);
        //Core.extractChannel(toTest, redChannel, 2);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.morphologyEx(redChannel, redChannel, Imgproc.MORPH_CLOSE, kernel);

        Mat outMat = new Mat();
        Imgproc.cvtColor(in, outMat, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(outMat, outMat, Imgproc.COLOR_GRAY2BGR);

        MatOfPoint contour = BetterTowerGoalUtils.getContour(redChannel);

        MatOfPoint2f dp = new MatOfPoint2f();
        Imgproc.approxPolyDP(new MatOfPoint2f(contour.toArray()), dp, 0.01 * Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true), true);

        Point[] data = CvUtils.refineCorners(redChannel, new MatOfPoint(dp.toArray())).toArray();

        MatOfPoint imgPoints = new MatOfPoint();

        for(Point p : data){
            if(p.y > (redChannel.height()*0.55)) {
                Point scl = new Point(p.x + rect.x, p.y + rect.y);

                Imgproc.circle(outMat, scl, 3, CvUtils.getRandomColor(), 2);
                imgPoints.push_back(new MatOfPoint(scl));
            }
        }

        double xPoint = 23.875/2;
        double yPoint = 15.75/2;
        double xInset = 3.9375;
        double yInset = 5.5;

        MatOfPoint3f objPoints = new MatOfPoint3f();
        objPoints.push_back(new MatOfPoint3f(new Point3(-xPoint, -yPoint, 0)));
        objPoints.push_back(new MatOfPoint3f(new Point3(-xPoint + xInset, -yPoint, 0)));
        objPoints.push_back(new MatOfPoint3f(new Point3(-xPoint + xInset, -yPoint + yInset, 0)));
        objPoints.push_back(new MatOfPoint3f(new Point3(xPoint - xInset, -yPoint + yInset, 0)));
        objPoints.push_back(new MatOfPoint3f(new Point3(xPoint - xInset, -yPoint, 0)));
        objPoints.push_back(new MatOfPoint3f(new Point3(xPoint, -yPoint, 0)));

        MatOfDouble distCoeffs = new MatOfDouble();
        Mat intrinsic = new Mat(3, 3, CvType.CV_32FC1);
        Mat rvec = new Mat();
        Mat tvec = new Mat();

        double[] inrinsicFloat = new double[] {817.063304531327,0.0,325.9485286458284,0.0,819.4690054531818,236.2597899599986,0.0,0.0,1.0,0.0};
        intrinsic.put(0, 0, inrinsicFloat);

        double[] distFloat = new double[] {-0.014680796227423968,1.3720322590501144,-0.0028429009326778093,0.0010064951672061734,-5.347658630748131};
        distCoeffs.fromArray(distFloat);

        if(imgPoints.toArray().length == objPoints.toArray().length) {
            Calib3d.solvePnP(objPoints, new MatOfPoint2f(imgPoints.toArray()), intrinsic, distCoeffs, rvec, tvec);
            Calib3d.drawFrameAxes(outMat, intrinsic, distCoeffs, rvec, tvec, 5, 3);
        }

        for(Point p : imgPoints.toArray()){
            //System.out.println("(" + p.x + ", " + p.y + ")");
        }
        //System.out.println();
        for(Point3 p : objPoints.toArray()){
            //System.out.println("(" + p.x + ", " + p.y + ")");
        }

        //Mat pos = CvUtils.toPosition(tvec, rvec);
        //System.out.println(pos.dump());

        //HighGui.imshow("Test", outMat);
        //HighGui.waitKey();
        return outMat;
    }

    public static Mat solvePnP2(Mat in, Rect bounds){
        double inset = 0.1;
        double maxInset = 0.2;
        double yInst = 0.5;
        double maxYInset = 0.4;

        Rect newBound = new Rect((int)(bounds.x + (bounds.width * inset)), (int) (bounds.y + (bounds.height * yInst)), (int)(bounds.width * (1-maxInset)), (int) (bounds.height * (1-maxYInset)));

        //System.out.println("Bounds " + newBound + " | " + bounds);

        Mat inCopy = in.submat(newBound);
        Mat inNorm = BetterTowerGoalUtils.normCropInRange(inCopy);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.morphologyEx(inNorm, inNorm, Imgproc.MORPH_OPEN, kernel);

        Mat fullMask = new Mat(inNorm.rows(), inNorm.cols(), inNorm.type(), new Scalar(255,255,255));
        Mat subMask = new Mat();
        Core.subtract(fullMask, inNorm, subMask);

        Mat outMat = subMask.clone();
        //Imgproc.cvtColor(outMat, outMat, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(outMat, outMat, Imgproc.COLOR_GRAY2BGR);

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
                //Point newPoint = new Point(p.x + newBound.x, p.y + newBound.y);
                Point newPoint = new Point(p.x, p.y);
                Imgproc.circle(outMat, newPoint, 3, colors[i], Imgproc.FILLED);
                //System.out.println("(" + newPoint.x + ", " + newPoint.y + ")");
                imgPoints.push_back(new MatOfPoint(newPoint));
            }
        }

        //Imgproc.rectangle(outMat, newBound, new Scalar(255, 0, 0));

        double insetWidth = 16;
        double tgoalHeight = 15.75;
        double togalWidth = 23.87500;
        double xInset = 3.9375;
        double yInset = 5.5;

        MatOfPoint3f objPoints = new MatOfPoint3f();
        objPoints.push_back(new MatOfPoint3f(new Point3(xInset, 0, 0)));
        objPoints.push_back(new MatOfPoint3f(new Point3(xInset + insetWidth, 0, 0)));
        objPoints.push_back(new MatOfPoint3f(new Point3(xInset, yInset, 0)));
        objPoints.push_back(new MatOfPoint3f(new Point3(xInset + insetWidth, yInset, 0)));

        MatOfDouble distCoeffs = new MatOfDouble();
        Mat intrinsic = new Mat(3, 3, CvType.CV_32FC1);
        Mat rvec = new Mat();
        Mat tvec = new Mat();

        double[] inrinsicFloat = new double[] {817.063304531327,0.0,325.9485286458284,0.0,819.4690054531818,236.2597899599986,0.0,0.0,1.0,0.0};
        intrinsic.put(0, 0, inrinsicFloat);

        double[] distFloat = new double[] {-0.014680796227423968,1.3720322590501144,-0.0028429009326778093,0.0010064951672061734,-5.347658630748131};
        distCoeffs.fromArray(distFloat);

        if(imgPoints.toArray().length == objPoints.toArray().length) {
            //Calib3d.solvePnPRansac(objPoints, new MatOfPoint2f(imgPoints.toArray()), intrinsic, distCoeffs, rvec, tvec);
            //Calib3d.drawFrameAxes(outMat, intrinsic, distCoeffs, rvec, tvec, 5, 3);
        }

        return outMat;
    }

    public static double[] solvePnP3(Mat in, RotatedRect bounds, MatOfPoint contour, Mat outMat){
        MatOfPoint2f imagePoints = new MatOfPoint2f(contour.toArray());
        MatOfPoint3f objectPoints = new MatOfPoint3f();

        double tgoalHeight = 15.75;
        double tgoalWidth = 23.87500;

        for(Point p : imagePoints.toArray()){
            Point scl = new Point(p.x / bounds.size.width, p.y / bounds.size.height);
            objectPoints.push_back(new MatOfPoint3f(new Point3(scl.x * tgoalWidth, scl.y * tgoalHeight, 0)));
        }

        MatOfDouble distCoeffs = new MatOfDouble();
        Mat intrinsic = new Mat(3, 3, CvType.CV_32FC1);
        Mat rvec = new Mat();
        Mat tvec = new Mat();

        //double[] inrinsicFloat = new double[] {817.063304531327,0.0,325.9485286458284,0.0,819.4690054531818,236.2597899599986,0.0,0.0,1.0,0.0};
        double[] intrinsicFloat = new double[]{212.4928484986939, 0, 1767.403745144145, 0, 8775.039060970414, 1226.292868696765, 0, 0, 1};
        intrinsic.put(0, 0, intrinsicFloat);

        //double[] distFloat = new double[] {-0.014680796227423968,1.3720322590501144,-0.0028429009326778093,0.0010064951672061734,-5.347658630748131};
        double[] distFloat = new double[] {0.07622041092946209, 1.676748319435331, 0.1624917317491128, 0.7905723382101558, -0.1488714574814097};
        distCoeffs.fromArray(distFloat);

        if(imagePoints.toArray().length >= 6) {
            Calib3d.solvePnP(objectPoints, imagePoints, intrinsic, distCoeffs, rvec, tvec);
            Calib3d.drawFrameAxes(outMat, intrinsic, distCoeffs, rvec, tvec, 5, 3);
            Mat pos = CvUtils.toPosition(tvec, rvec);
            return new double[]{pos.get(0, 0)[0], pos.get(1, 0)[0], pos.get(2, 0)[0]};
        }
        return new double[]{0, 0, 0};
    }

    public static double[] solvePnP4(Mat in, RotatedRect bounds, MatOfPoint contour, Mat outMat){
        List<Point> cntArr = contour.toList();

        if(cntArr.size() == 8){

            MatOfPoint imgPoints = new MatOfPoint();

            cntArr.sort(Comparator.comparingDouble(o -> o.y));

            List<Point> topPoints = cntArr.subList(0, 2);
            List<Point> inset = cntArr.subList(2, 4);
            List<Point> bottomPoints = cntArr.subList(4, 8);
            topPoints.sort(Comparator.comparingDouble(o -> o.x));
            bottomPoints.sort(Comparator.comparingDouble(o -> o.x));
            inset.sort(Comparator.comparingDouble(o -> o.x));

            Point tl = topPoints.get(0);
            Point tr = topPoints.get(1);

            Point bl = bottomPoints.get(0);
            Point br = bottomPoints.get(3);

            Point itl = inset.get(0);
            Point itr = inset.get(1);
            Point ibl = bottomPoints.get(1);
            Point ibr = bottomPoints.get(2);

            double insetWidth = 16;
            double tgoalHeight = 15.75;
            double tgoalWidth = 23.87500;
            double xInset = 3.9375;
            double yInset = 5.5;

            imgPoints.push_back(new MatOfPoint(tl));
            imgPoints.push_back(new MatOfPoint(tr));
            imgPoints.push_back(new MatOfPoint(bl));
            imgPoints.push_back(new MatOfPoint(ibl));
            imgPoints.push_back(new MatOfPoint(ibr));
            imgPoints.push_back(new MatOfPoint(itl));
            imgPoints.push_back(new MatOfPoint(itr));
            imgPoints.push_back(new MatOfPoint(br));

            MatOfPoint3f objPoints = new MatOfPoint3f();
            objPoints.push_back(new MatOfPoint3f(new Point3(0, tgoalHeight, 0)));
            objPoints.push_back(new MatOfPoint3f(new Point3(tgoalWidth, tgoalHeight, 0)));
            objPoints.push_back(new MatOfPoint3f(new Point3(0, 0, 0)));
            objPoints.push_back(new MatOfPoint3f(new Point3(xInset, 0, 0)));
            objPoints.push_back(new MatOfPoint3f(new Point3(xInset + insetWidth, 0, 0)));
            objPoints.push_back(new MatOfPoint3f(new Point3(xInset, yInset, 0)));
            objPoints.push_back(new MatOfPoint3f(new Point3(xInset + insetWidth, yInset, 0)));
            objPoints.push_back(new MatOfPoint3f(new Point3(tgoalWidth, 0, 0)));

            for(Point p : imgPoints.toArray()){
                System.out.println("(" + p.x + ", " + p.y + ")");
            }
            System.out.println();
            for(Point3 p : objPoints.toArray()){
                System.out.println("(" + p.x + ", " + p.y + ")");
            }

            MatOfDouble distCoeffs = new MatOfDouble();
            Mat intrinsic = new Mat(3, 3, CvType.CV_32FC1);
            Mat rvec = new Mat();
            Mat tvec = new Mat();

            //double[] inrinsicFloat = new double[] {817.063304531327,0.0,325.9485286458284,0.0,819.4690054531818,236.2597899599986,0.0,0.0,1.0,0.0};
            double[] intrinsicFloat = new double[]{212.4928484986939, 0, 1767.403745144145, 0, 8775.039060970414, 1226.292868696765, 0, 0, 1};
            intrinsic.put(0, 0, intrinsicFloat);

            //double[] distFloat = new double[] {-0.014680796227423968,1.3720322590501144,-0.0028429009326778093,0.0010064951672061734,-5.347658630748131};
            double[] distFloat = new double[] {0.07622041092946209, 1.676748319435331, 0.1624917317491128, 0.7905723382101558, -0.1488714574814097};
            distCoeffs.fromArray(distFloat);

            if(imgPoints.toArray().length >= 6) {
                Calib3d.solvePnPRansac(objPoints, new MatOfPoint2f(imgPoints.toArray()), intrinsic, distCoeffs, rvec, tvec);
                Calib3d.drawFrameAxes(outMat, intrinsic, distCoeffs, rvec, tvec, 5, 3);
                Mat pos = CvUtils.toPosition(tvec, rvec);
                return new double[]{pos.get(0, 0)[0], pos.get(1, 0)[0], pos.get(2, 0)[0]};
            }
        }
        return new double[]{0, 0, 0};
    }

    public static double[] getPitchAndYaw(Mat in, Rect bounds, Mat outMat){

        Mat inCopy = in.submat(bounds);
        Mat inNorm = BetterTowerGoalUtils.normCropInRange(inCopy);

        double width = (in.width()/2.0) - 0.5;
        double height = (in.height()/2.0) - 0.5;
        double fov = 123;

        Point center = CvUtils.getCenter(bounds);
        double pitch = -Math.toDegrees(Math.atan2(center.y - height, CvUtils.calcPinholeVert(fov, in.width(), in.height())));
        double yaw = Math.toDegrees(Math.atan2(center.x - width, CvUtils.calcPinholeHor(fov, in.width(), in.height())));
        //System.out.println("Center " + center + " | vert " + CvUtils.calcPinholeVert(fov, in.width(), in.height()) + " | hor " + CvUtils.calcPinholeHor(fov, in.width(), in.height()));
        //System.out.println("Pitch " + pitch + " | Yaw " + yaw);

        //Imgproc.rectangle(outMat, bounds, new Scalar(0, 0, 255));

        return new double[]{ pitch, yaw };
    }

    public static Mat getPitchAndYaw(Mat in, Point center){

        Mat outMat = in.clone();
        Imgproc.cvtColor(outMat, outMat, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(outMat, outMat, Imgproc.COLOR_GRAY2BGR);

        double width = (in.width()/2.0) - 0.5;
        double height = (in.height()/2.0) - 0.5;
        double fov = 123;
        //fov = 180;

        double pitch = -Math.toDegrees(Math.atan2(center.y - height, CvUtils.calcPinholeVert(fov, in.width(), in.height())));
        double yaw = Math.toDegrees(Math.atan2(center.x - width, CvUtils.calcPinholeHor(fov, in.width(), in.height())));
        System.out.println("Center " + center + " | vert " + CvUtils.calcPinholeVert(fov, in.width(), in.height()) + " | hor " + CvUtils.calcPinholeHor(fov, in.width(), in.height()));
        System.out.println("Pitch " + pitch + " | Yaw " + yaw);

        return outMat;
    }
}
