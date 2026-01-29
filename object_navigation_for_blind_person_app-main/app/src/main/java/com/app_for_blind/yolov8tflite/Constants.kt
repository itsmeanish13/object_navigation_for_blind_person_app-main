package com.app_for_blind.yolov8tflite

object Constants {
    const val MODEL_PATH = "best_int8.tflite"
    const val LABELS_PATH = "labels.txt" // provide your labels.txt file if the metadata not present in the model
    const val DEPTH_MODEL_PATH = "depth_model.tflite"
}
