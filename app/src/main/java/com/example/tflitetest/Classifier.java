package com.example.tflitetest;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Pair;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

public class Classifier {
    Context context;
    Interpreter interpreter = null;
    int modelInputWidth, modelInputHeight, modelInputChannel;
    int modelOutputClasses;

    private static final String MODEL_NAME = "keras_model.tflite";

    public Classifier(Context context) {
        this.context = context;
    }

    public void init() throws IOException {
        ByteBuffer model = loadModelFile(MODEL_NAME);
        model.order(ByteOrder.nativeOrder());

        // interpreter는 모델에 데이터를 입력하고 추론 결과를 전달받을 수 있는 클래스
        interpreter = new Interpreter(model);

        initModelShape();
    }

    private ByteBuffer loadModelFile(String modelName) throws IOException {
        // AssetManager은 assets 폴더에 저장된 리소스에 접근하기 위함.
        AssetManager am = context.getAssets();
        AssetFileDescriptor afd = am.openFd(modelName);
        FileInputStream fis = new FileInputStream(afd.getFileDescriptor());

        // 성능을 위함?
        FileChannel fc = fis.getChannel();
        long startOffset = afd.getStartOffset();
        long declaredLength = afd.getDeclaredLength();

        return fc.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void initModelShape() {
        // interpreter의 입력 텐서를 구함
        Tensor inputTensor = interpreter.getInputTensor(0);

        int[] inputShape = inputTensor.shape();
        modelInputChannel = inputShape[0];
        modelInputWidth = inputShape[1];
        modelInputHeight = inputShape[2];

        Tensor outputTensor = interpreter.getOutputTensor(0);
        int[] outputShape = outputTensor.shape();
        modelOutputClasses = outputShape[1];
    }

    private Bitmap resizeBitmap(Bitmap bitmap) {
        // 변환할 이미지, 가로 크기, 세로크기를 각각 전달달
        return Bitmap.createScaledBitmap(bitmap, modelInputWidth, modelInputHeight, false);
    }

    private ByteBuffer convertBitmapToGrayByteBuffer(Bitmap bitmap) {
        // allocate 함수는 JVM에 heap에 메모리를 할당하는 방식이고,
        // allocateDirect 함수는 시스템의 메모리에 직접 할당하는 방식, 할당 및 해제에 많은 시간이 소요
        // 그러나 IO나 Copy 등 작업의 성능을 높일 수 있음.
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bitmap.getByteCount());
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[bitmap.getWidth() * bitmap.getHeight()];
        bitmap.getPixels(pixels, 0,bitmap.getWidth(), 0,0,bitmap.getWidth(), bitmap.getHeight());

        for(int pixel : pixels) {
            int r = pixel >> 16 & 0xFF;
            int g = pixel >> 8 & 0xFF;
            int b = pixel & 0xFF;

            float avgPixelValue = (r + g + b) / 3.0f;
            float normalizedPixelValue = avgPixelValue / 255.0f;

            byteBuffer.putFloat(normalizedPixelValue);
        }

        return byteBuffer;
    }

    private Pair<Integer, Float> argmax(float[] array) {
        int argmax = 0;
        float max = array[0];
        for(int i=1;i<array.length;i++) {
            float f = array[i];
            if(f>max) {
                argmax = i;
                max = f;
            }
        }
        return new Pair<>(argmax, max);
    }

    public Pair<Integer, Float> classify(Bitmap image) {
        ByteBuffer buffer = convertBitmapToGrayByteBuffer(resizeBitmap(image));
        float[][]result = new float[1][modelOutputClasses];
        interpreter.run(buffer, result);
        return argmax(result[0]);
    }

    public void finish() {
        if(interpreter != null)
            interpreter.close();
    }
}
