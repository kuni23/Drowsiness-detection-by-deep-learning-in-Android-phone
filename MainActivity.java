package com.example.androidseries;
import static org.opencv.imgproc.Imgproc.INTER_AREA;
import static org.opencv.imgproc.Imgproc.resize;
import android.Manifest;
import android.content.pm.PackageManager;
import android.media.MediaPlayer;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    Button Btn;
    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    public static final int CAMERA_PERM_CODE = 101;
    int score=0;
    int previous_frame = 0;
    int counter = 0;
    Boolean alarm_signal = false;
    Boolean alarm_signal_in_detection = false;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        Btn = (Button)findViewById(R.id.button);
        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);


        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);

                switch(status){

                    case BaseLoaderCallback.SUCCESS:
                        askCameraPermissions();
                        cameraBridgeViewBase.setCameraIndex(1);
                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }


            }

        };


        //When we press the button
        Btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Toast.makeText(MainActivity.this, "Camera Btn is clicked", Toast.LENGTH_SHORT).show();

                //Set alarm_signal to 1
                alarm_signal = true;
                alarm_signal_in_detection = true;

            }
        });


    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        //Set up the alarm
        MediaPlayer mp = MediaPlayer.create(getApplicationContext(), R.raw.alarm);


        //Get the Mat frame in the right orientation
        Mat frame = inputFrame.rgba();
        Core.transpose(frame, frame);
        Core.flip(frame, frame, 0);
        Core.flip(frame, frame, +1);



        //We do not need to process all the frames
        if(counter % 3 ==0) {

            new DetectionTask().execute(frame);

        }
        counter = counter + 1;

        //Putting info to the frame (fps, sleepiness score, etc.)
        Imgproc.putText(frame, "Drowsiness score: " + String.valueOf(score), new Point(frame.cols()/10 , frame.rows()/10 ) , Core.FONT_HERSHEY_COMPLEX_SMALL , 3 , new Scalar(255, 0, 0));
        Imgproc.putText(frame, "Frame processing time: " + String.valueOf(1000/(SystemClock.elapsedRealtime()-previous_frame)
        ), new Point(frame.cols()/10 , 9*frame.rows()/10 ) , Core.FONT_HERSHEY_COMPLEX_SMALL , 2 , new Scalar(0, 0, 255));
        Imgproc.rectangle(frame, new Point(0, 0), new Point(frame.cols(), frame.rows()),new Scalar(255, 0, 0), score / 3);

        previous_frame= (int) SystemClock.elapsedRealtime();


        //If the button is pressed, we need to make the alarm stop (thanks to asynchron. we need to check two booleans)
        if(alarm_signal || alarm_signal_in_detection) {
            mp.stop();
            alarm_signal = false;
        }



        //Start or stop the alarm
        if(score>10) {
            mp.start();
        } else if(score==10){
            mp.stop();
        }


        return frame;
    }


    @Override
    public void onCameraViewStarted(int width, int height) {

    }


    @Override
    public void onCameraViewStopped() {

    }


    @Override
    protected void onResume() {
        super.onResume();

        if (!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(),"There's a problem, yo!", Toast.LENGTH_SHORT).show();
        }

        else
        {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }


    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase!=null){

            cameraBridgeViewBase.disableView();
        }

    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }

    private void askCameraPermissions() {

        if(ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, CAMERA_PERM_CODE);
        }
    }

    public class DetectionTask extends AsyncTask<Mat, Void, Integer> {

        // run detection method in background thread
        // takes in parameter in the .execute(Mat frame) call on the class that is created
        @Override
        protected Integer doInBackground(Mat... params) {

            Mat frame = params[0];

            //Applying the Python module
            if(!Python.isStarted()) {
                Python.start(new AndroidPlatform(MainActivity.this));
            }
            final Python py = Python.getInstance();

            //Resizing for enhancing the process time

            Mat resizeimage = new Mat();

            //Using the whole image
            //frame.copyTo(resizeimage);

            Size scaleSize = new Size(frame.width()/3,frame.height()/3);
            resize(frame, resizeimage, scaleSize , 0, 0, INTER_AREA);


            //Mat->MatOfByte->byteArray
            MatOfByte matOfByte = new MatOfByte();
            Imgcodecs.imencode(".png", resizeimage, matOfByte);
            byte[] byteArray = matOfByte.toArray();


            //Call the Python module
            PyObject pyo = py.getModule("drowsiness detection_mobile");  //name of the python file
            PyObject obj = pyo.callAttr("main", byteArray, score);


            return obj.toInt();


        }

        // result Integer is passed here after
        // this method is run on maing UI thread
        @Override
        protected void onPostExecute(Integer result) {

            if(alarm_signal || alarm_signal_in_detection) {
                score = 0;
                alarm_signal_in_detection = false;
            } else {
                score = result;
            }


        }

    }


}