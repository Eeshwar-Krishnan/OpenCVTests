package main.java;

import main.java.Utils.BetterTowerGoalUtils;
import main.java.Utils.CvUtils;
import main.java.Utils.PnPUtils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;

public class Testing {

    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) throws InterruptedException, IOException {
        Mat matchMat = CvUtils.bufferedImageToMat(ImageIO.read(new File("src/assets/input2/out10.png")));
        PnPUtils.getPitchAndYaw(matchMat, new Point(0, 0));
    }
}
