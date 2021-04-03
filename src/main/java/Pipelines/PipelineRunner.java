package main.java.Pipelines;

import main.java.Pipelines.Basic.PipelineTester;
import main.java.Pipelines.Basic.PnPPipelineTester;
import main.java.Utils.BetterTowerGoalUtils;
import main.java.Utils.CvUtils;
import main.java.Utils.PnPUtils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;

public class PipelineRunner {

    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) throws IOException {
        PipelineTester pipeline = new PipelineTester();
        Mat matchMat = CvUtils.bufferedImageToMat(ImageIO.read(new File("src/assets/input2/out23.png")));//238

        Mat blurMat = CvUtils.bufferedImageToMat(ImageIO.read(new File("src/assets/outBounding/output631.png")));//631
        Imgproc.resize(blurMat, blurMat, matchMat.size());
        //HighGui.imshow("Input", blurMat);

        Mat out = pipeline.processFrame(matchMat);
        //System.out.println(pipeline.getPos().dump());
        Imgproc.resize(out, out, new Size(1280, 720));
        HighGui.imshow("Output", out);
        HighGui.waitKey();
    }
}
