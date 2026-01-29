package com.app_for_blind.yolov8tflite

data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val cx: Float,
    val cy: Float,
    val w: Float,
    val h: Float,
    val cnf: Float,
    val cls: Int,
    val clsName: String,
    var distance: String = "", // Added field to store distance
    var isAnnounced: Boolean = false // Added field to track announcement status
)