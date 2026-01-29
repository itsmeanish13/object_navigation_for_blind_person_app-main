package com.app_for_blind.yolov8tflite

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<BoundingBox>()
    private var boxPaint = Paint()
    private var announcedBoxPaint = Paint() // Paint for announced objects
    private var lockedBoxPaint = Paint() // Paint for locked objects
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()

    private var bounds = Rect()
    
    // ID of the currently locked object (if any)
    private var lockedObject: String? = null

    init {
        initPaints()
    }

    fun clear() {
        results = listOf()
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        announcedBoxPaint.reset()
        lockedBoxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        // Default Green for unannounced
        boxPaint.color = Color.GREEN
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE
        
        // Blue for announced
        announcedBoxPaint.color = Color.BLUE
        announcedBoxPaint.strokeWidth = 8F
        announcedBoxPaint.style = Paint.Style.STROKE
        
        // Red for locked/target objects
        lockedBoxPaint.color = Color.RED
        lockedBoxPaint.strokeWidth = 10F // Slightly thicker
        lockedBoxPaint.style = Paint.Style.STROKE
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        results.forEach {
            val left = it.x1 * width
            val top = it.y1 * height
            val right = it.x2 * width
            val bottom = it.y2 * height

            // Choose paint based on status
            val currentPaint = when {
                lockedObject != null && it.clsName.equals(lockedObject, ignoreCase = true) -> lockedBoxPaint
                it.isAnnounced -> announcedBoxPaint
                else -> boxPaint
            }
            
            canvas.drawRect(left, top, right, bottom, currentPaint)
            
            // Append distance to the label if available
            val drawableText = if (it.distance.isNotEmpty()) {
                "${it.clsName} ${it.distance}"
            } else {
                it.clsName
            }

            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top,
                left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)

        }
    }

    fun setResults(boundingBoxes: List<BoundingBox>) {
        results = boundingBoxes
        invalidate()
    }
    
    fun setLockedObject(objectName: String?) {
        lockedObject = objectName
        invalidate()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}