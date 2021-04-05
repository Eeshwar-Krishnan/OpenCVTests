package main.java;

import main.java.Utils.CvUtils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CameraCalibration {

    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) throws IOException {
        //CALIBRATION
        File calibFolder = new File("src/assets/calibration");
        boolean showChessboards = false;
        //END CALIBRATION

        File[] files = calibFolder.listFiles();

        List<Mat> imagePoints = new ArrayList<>();
        List<Mat> objectPoints = new ArrayList<>();
        MatOfPoint3f obj = new MatOfPoint3f();

        int numHor = 9;
        int numVer = 7;
        int numBlocks = 9 * 7;
        for (int j = 0; j < numBlocks; j++)
            obj.push_back(new MatOfPoint3f(new Point3(j / numHor, j % numVer, 0.0f)));

        Size size = new Size();

        assert files != null;
        System.out.println("Calibrating");
        for(int i = 0; i < files.length; i ++){
            File f = files[i];

            Mat inMat = CvUtils.bufferedImageToMat(ImageIO.read(f));

            Mat inGrey = new Mat();
            Imgproc.cvtColor(inMat, inGrey, Imgproc.COLOR_BGR2GRAY);

            Size chessboardSize = new Size(numHor, numVer);

            MatOfPoint2f corners = new MatOfPoint2f();

            boolean foundCorners = Calib3d.findChessboardCorners(inMat, chessboardSize, corners);

            if(foundCorners){
                TermCriteria criteria = new TermCriteria(TermCriteria.EPS | TermCriteria.MAX_ITER, 30, 0.1);
                Imgproc.cornerSubPix(inGrey, corners, new Size(11, 11), new Size(-1, -1), criteria);
                Imgproc.cvtColor(inGrey, inGrey, Imgproc.COLOR_GRAY2BGR);
                Calib3d.drawChessboardCorners(inGrey, chessboardSize, corners, true);

                imagePoints.add(corners);
                objectPoints.add(obj);

                size = inMat.size();
            }
            if(showChessboards) {
                Mat resized = new Mat();
                Imgproc.resize(inGrey, resized, new Size(1280, 720));
                HighGui.imshow(f.getName(), resized);
            }

            CvUtils.printLoadingBar("Processing Images", i+1, files.length);
        }

        if(showChessboards) {
            HighGui.waitKey();
        }

        System.out.println("Solving For Parameters");

        List<Mat> rvecs = new ArrayList<>();
        List<Mat> tvecs = new ArrayList<>();
        Mat intrinsic = new Mat(), distCoeffs = new Mat();
        intrinsic.put(0, 0, 1);
        intrinsic.put(1, 1, 1);

        Calib3d.calibrateCamera(objectPoints, imagePoints, size, intrinsic, distCoeffs, rvecs, tvecs);

        System.out.println(intrinsic.dump());
        System.out.println(distCoeffs.dump());
    }
}
