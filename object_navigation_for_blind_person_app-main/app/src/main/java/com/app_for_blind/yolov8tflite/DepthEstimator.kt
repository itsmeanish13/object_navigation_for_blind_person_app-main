package com.app_for_blind.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

// DepthEstimator class handles MiDaS TFLite model for depth estimation
class DepthEstimator(context: Context) {
    private lateinit var interpreter: Interpreter
    private var inputImageWidth: Int = 0
    private var inputImageHeight: Int = 0

    init {
        val compatList = CompatibilityList()
        val options = Interpreter.Options()
        
        // Try initializing with GPU
        var gpuDelegate: GpuDelegate? = null
        var initialized = false
        
        if (compatList.isDelegateSupportedOnThisDevice) {
            try {
                val delegateOptions = compatList.bestOptionsForThisDevice
                gpuDelegate = GpuDelegate(delegateOptions)
                options.addDelegate(gpuDelegate)
                
                val modelFile = FileUtil.loadMappedFile(context, Constants.DEPTH_MODEL_PATH)
                interpreter = Interpreter(modelFile, options)
                initialized = true
            } catch (e: Exception) {
                Log.e("DepthEstimator", "GPU initialization failed, falling back to CPU", e)
                // Clean up failed delegate
                if (gpuDelegate != null) {
                    gpuDelegate?.close()
                    gpuDelegate = null
                }
                options.delegates.clear()
            }
        }
        
        if (!initialized) {
            // Fallback to CPU
            val cpuOptions = Interpreter.Options().apply { setNumThreads(4) }
            val modelFile = FileUtil.loadMappedFile(context, Constants.DEPTH_MODEL_PATH)
            interpreter = Interpreter(modelFile, cpuOptions)
        }

        val inputShape = interpreter.getInputTensor(0).shape() 
        // Handle shapes like [1, 256, 256, 3] (NHWC) or [1, 3, 256, 256] (NCHW)
        if (inputShape[1] == 3) {
            inputImageHeight = inputShape[2]
            inputImageWidth = inputShape[3]
        } else {
            inputImageHeight = inputShape[1]
            inputImageWidth = inputShape[2]
        }
    }

    // Process the frame and return a depth map (float array)
    fun computeDepthMap(bitmap: Bitmap): FloatArray {
        // 1. Resize and Preprocess Input
        val imageProcessor = ImageProcessor.Builder()
            .add(NormalizeOp(0f, 255f)) // Normalize [0, 255] -> [0, 1]
            .build()
            
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputImageWidth, inputImageHeight, true)
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)

        // 2. Run Inference
        val outputTensor = interpreter.getOutputTensor(0)
        val outputShape = outputTensor.shape()
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
        
        interpreter.run(processedImage.buffer, outputBuffer.buffer)
        
        return outputBuffer.floatArray
    }
    
    // Estimate distance for a given bounding box
    // Returns estimated distance in meters
    fun getDistance(depthMap: FloatArray, box: BoundingBox): Float {
        // Ensure box coordinates are relative 0..1
        val x1 = (box.x1 * inputImageWidth).toInt().coerceIn(0, inputImageWidth - 1)
        val y1 = (box.y1 * inputImageHeight).toInt().coerceIn(0, inputImageHeight - 1)
        val x2 = (box.x2 * inputImageWidth).toInt().coerceIn(0, inputImageWidth - 1)
        val y2 = (box.y2 * inputImageHeight).toInt().coerceIn(0, inputImageHeight - 1)

        var sumDepth = 0f
        var count = 0
        
        val centerX = (x1 + x2) / 2
        val centerY = (y1 + y2) / 2
        
        // Sample a small region around center
        val sampleRadius = 5 
        
        for (y in (centerY - sampleRadius).coerceAtLeast(0) .. (centerY + sampleRadius).coerceAtMost(inputImageHeight - 1)) {
            for (x in (centerX - sampleRadius).coerceAtLeast(0) .. (centerX + sampleRadius).coerceAtMost(inputImageWidth - 1)) {
                val index = y * inputImageWidth + x
                if (index >= 0 && index < depthMap.size) {
                    sumDepth += depthMap[index]
                    count++
                }
            }
        }
        
        if (count == 0) return 0f
        
        val avgInverseDepth = sumDepth / count
        Log.d("DepthEstimator", "Avg Inverse Depth: $avgInverseDepth")
        
        // Simple heuristic conversion for MiDaS
        // If avgInverseDepth is very small, distance is large.
        if (avgInverseDepth < 0.0001f) return 10f 
        
        // Adjusted scale factor to avoid 0.0 result for high inverse depth values.
        // Assuming model output range might be large (e.g., 0-255 or 0-1000).
        // Tune this value based on real-world calibration. 
        // For standard MiDaS v2.1 small, a factor of ~100-200 often maps to reasonable meters if output is raw.
        val scaleFactor = 100.0f 
        return scaleFactor / avgInverseDepth
    }

    fun close() {
        interpreter.close()
    }
}