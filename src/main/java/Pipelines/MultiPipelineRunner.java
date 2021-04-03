package main.java.Pipelines;

import main.java.Pipelines.Advanced.BetterTowergoalPipeline;
import main.java.Pipelines.Basic.PipelineTester;
import main.java.Pipelines.Basic.PnPPipelineTester;
import main.java.Utils.BetterTowerGoalUtils;
import main.java.Utils.CvUtils;
import main.java.Utils.VideoPlayback;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Objects;

public class MultiPipelineRunner {

    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) throws IOException, InterruptedException {
        PipelineTester pipeline = new PipelineTester();
        Mat matchMat = CvUtils.bufferedImageToMat(ImageIO.read(new File("src/assets/annotations/out631.png")));//432, 631

        String annoFolder = "src/assets/annotations";
        System.out.print("Serializing Images... ");
        long tmS = System.currentTimeMillis();
        ArrayList<Mat> inMats = new ArrayList<>();

        int fileLength = Objects.requireNonNull(new File(annoFolder).listFiles()).length-2;
        //fileLength = 400;

        for(int i = 1; i < fileLength; i ++) {
            Mat tmp = CvUtils.bufferedImageToMat(ImageIO.read(new File(annoFolder + "/out" + i + ".png")));
            inMats.add(tmp);
            CvUtils.printLoadingBar("Loading Images", i, fileLength-1.0);
            //System.out.println(i + "/" + 681);
        }
        ArrayList<Long> times = new ArrayList<>();
        ArrayList<Rect> rects = new ArrayList();
        System.out.println("Loading Images Complete! Elapsed Time " + (System.currentTimeMillis() - tmS) + " ms");
        System.out.print("Processing Images... ");
        tmS = System.currentTimeMillis();
        ArrayList<Mat> mts = new ArrayList<>();
        ArrayList<Mat> pos = new ArrayList<>();
        for(Mat m : inMats) {
            long start = System.currentTimeMillis();
            Mat out = pipeline.processFrame(m);
            rects.add(pipeline.getBoundingRect());
            //pos.add(pipeline.getPos().clone());
            long end = System.currentTimeMillis();

            Imgproc.resize(out, out, new Size(1280, 720));
            ImageIO.write(CvUtils.toBufferedImage(HighGui.toBufferedImage(out)), "JPG", new File("src/assets/out/out" + inMats.indexOf(m) + ".jpg"));

            //mts.add(out.clone());
            out.release();
            m.release();
            times.add(end-start);
            CvUtils.printLoadingBar("Processing Images", inMats.indexOf(m), inMats.size()-1);
        }
        System.out.println("Processing Images Complete! Elapsed Time " + (System.currentTimeMillis() - tmS) + " ms");
        double avgTime = 0;
        double avgx = 0, avgy = 0, avgw = 0, avgh = 0;
        for(long l : times){
            avgTime += ((double)l/times.size());
        }
        for(Rect r : rects){
            avgx += (double)r.x/rects.size();
            avgy += (double)r.y/rects.size();
            avgw += (double)r.width/rects.size();
            avgh += (double)r.height/rects.size();
        }
        double lastX = 0;
        double lastY = 0;
        double lastZ = 0;
        StringBuilder str = new StringBuilder();
        for(Mat m : pos){
            double[] arr = new double[(int) m.total()];
            m.get(0, 0, arr);
            if(arr.length == 3){
                double x = arr[0];
                if(x > 0){
                    x = -x;
                }
                double y = arr[1];
                double z = arr[2];

                str.append("[").append(x).append(',').append(y).append(",").append(z).append("]\n");

                lastX = x;
                lastY = y;
                lastZ = z;
            }else{
                str.append("[").append(lastX).append(',').append(lastY).append(",").append(lastZ).append("]\n");
            }
        }

        File f = new File("src/assets/out.txt");
        f.createNewFile();
        FileWriter writer = new FileWriter(f);
        writer.write(str.toString());
        writer.flush();
        writer.close();

        System.out.println("Average Time Taken: " + avgTime + " ms. Full Capacity: " + (1/(avgTime/1000.0)) + " Frames Per Second");
        System.out.println("Average Position | X: " + avgx + " Y: " + avgy + " | W: " + avgw + " H: " + avgh);
        System.out.println();

        /**
        System.out.print("Starting Visualizer... " );
        for(int i = 0; i < mts.size(); i ++){
            Mat combined = mts.get(i);
            Imgproc.resize(combined, combined, new Size(1280, 720));
            ImageIO.write(CvUtils.toBufferedImage(HighGui.toBufferedImage(combined)), "JPG", new File("src/assets/out/out" + i + ".jpg"));
            //System.out.println("Wrote " + i + " of " + mts.size());
            CvUtils.printLoadingBar("Writing Images", i, mts.size()-1);
        }
        System.out.println("Complete");

        //VideoPlayback playback = new VideoPlayback(mts, 24);
        //playback.run();
         ffmpeg -i out%d.jpg -c:v libx264 -vf fps=24 -pix_fmt yuv420p out.mp4
         */
    }
}
