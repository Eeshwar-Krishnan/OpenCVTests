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
        PnPPipelineTester pipeline = new PnPPipelineTester();
        Mat matchMat = CvUtils.bufferedImageToMat(ImageIO.read(new File("src/assets/input2/out10.png")));//238


        Mat out = pipeline.processFrame(matchMat);
        //System.out.println(pipeline.getPos().dump());
        Imgproc.resize(out, out, new Size(1280, 720));
        HighGui.imshow("Output", out);
        HighGui.waitKey();
    }
}
