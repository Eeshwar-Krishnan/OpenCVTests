package main.java.Utils;

import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.util.ArrayList;

public class VideoPlayback {
    private ArrayList<Mat> mats;
    private int idx = 0;
    private double fps = 30;
    public VideoPlayback(ArrayList<Mat> mats, double fps){
        this.fps = fps;
        this.mats = mats;
    }

    public void run(){
        JFrame frame = new JFrame();
        frame.setSize(1280/2, 720/2);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);

        frame.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                idx = 0;
            }
        });

        BufferedImage toShow;
        long timer = System.currentTimeMillis();
        while(frame.isVisible()){
            toShow = CvUtils.toBufferedImage(HighGui.toBufferedImage(mats.get(idx)));
            frame.getGraphics().drawImage(toShow, 0, 0, 1280/2,720/2, null);
            if(timer < System.currentTimeMillis()){
                timer = (long) (System.currentTimeMillis() + ((1/fps)*1000.0));
                idx ++;
                //System.out.println(idx);
                if(idx >= mats.size()){
                    idx = mats.size()-1;
                }
            }
        }
    }
}