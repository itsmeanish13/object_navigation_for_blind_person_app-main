package com.app_for_blind.yolov8tflite

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.view.GestureDetector
import android.view.MotionEvent
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.app_for_blind.yolov8tflite.Constants.LABELS_PATH
import com.app_for_blind.yolov8tflite.Constants.MODEL_PATH
import yolov8tflite.R
import yolov8tflite.databinding.ActivityMainBinding
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs

class MainActivity : AppCompatActivity(), Detector.DetectorListener, TextToSpeech.OnInitListener {
    private lateinit var binding: ActivityMainBinding
    private val isFrontCamera = false

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var detector: Detector? = null
    private var depthEstimator: DepthEstimator? = null

    private lateinit var cameraExecutor: ExecutorService
    private var tts: TextToSpeech? = null
    private var isSpeaking = false
    
    // Speech Recognition
    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var recognitionIntent: Intent
    
    // Gesture Detector
    private lateinit var gestureDetector: GestureDetector
    
    // Application States
    enum class AppState {
        DETECTING,
        PROMPTING,
        LISTENING,
        NAVIGATING,
        COMPLETED,
        SCENE_DESCRIPTION // New state for Scene Description feature
    }
    
    private var currentState = AppState.DETECTING
    private var targetObjectName: String? = null
    
    // To help with re-orientation if target is lost
    private var lastKnownDirection: String = "center"

    // Track last spoken time for each object class separately
    private val lastSpokenTimes = ConcurrentHashMap<String, Long>()
    private val SPEECH_COOLDOWN_MS = 10000L // 10 seconds cooldown per object
    
    // Memory of announced objects and when they were last seen
    private val announcedObjects = ConcurrentHashMap<String, Long>()
    private val objectLastSeenTime = ConcurrentHashMap<String, Long>()
    private val OBJECT_PERSISTENCE_MS = 3000L // Keep objects in memory for 3 seconds after they leave frame
    
    // Navigation updates throttling
    private var lastNavigationUpdate = 0L
    private val NAVIGATION_UPDATE_INTERVAL = 3000L // Update every 3 seconds
    private val NAVIGATION_REACHED_THRESHOLD = 0.2f // 0.2 meters (reduced from 0.5)

    @Volatile
    private var currentDepthMap: FloatArray? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize Text-To-Speech
        tts = TextToSpeech(this, this)
        
        // Initialize Speech Recognizer
        initializeSpeechRecognizer()

        // Initialize Gesture Detector
        initializeGestureDetector()

        cameraExecutor = Executors.newSingleThreadExecutor()

        cameraExecutor.execute {
            detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this) {
                toast(it)
            }
            try {
                depthEstimator = DepthEstimator(baseContext)
            } catch (e: Exception) {
                Log.e(TAG, "Error initializing DepthEstimator: ${e.message}")
            }
        }

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        bindListeners()
    }

    private fun initializeGestureDetector() {
        gestureDetector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDoubleTap(e: MotionEvent): Boolean {
                // Handle double tap
                runOnUiThread {
                    handleDoubleTap()
                }
                return true
            }

            override fun onFling(e1: MotionEvent?, e2: MotionEvent, velocityX: Float, velocityY: Float): Boolean {
                if (e1 == null) return false
                val diffY = e2.y - e1.y
                val diffX = e2.x - e1.x
                
                // Swipe Thresholds
                val SWIPE_THRESHOLD = 100
                val SWIPE_VELOCITY_THRESHOLD = 100
                
                if (abs(diffX) > abs(diffY)) {
                    // Horizontal Swipe
                    if (abs(diffX) > SWIPE_THRESHOLD && abs(velocityX) > SWIPE_VELOCITY_THRESHOLD) {
                        if (diffX > 0) {
                            // Swipe Right -> Reset
                            runOnUiThread {
                                handleSwipeRight()
                            }
                            return true
                        } else {
                            // Swipe Left -> Scene Description
                            runOnUiThread {
                                handleSwipeLeft()
                            }
                            return true
                        }
                    }
                }
                return false
            }
        })
    }
    
    private fun handleDoubleTap() {
        // Stop current TTS
        if (tts != null) {
            tts!!.stop()
        }
        isSpeaking = false // Reset speaking flag since we forced stop
        
        // Change state to Listening immediately
        currentState = AppState.LISTENING
        
        // Provide audio feedback and start listening
        speak("Listening for target...", "MANUAL_LISTEN")
        // "MANUAL_LISTEN" will trigger startListening in onDone callback
    }

    private fun handleSwipeRight() {
        // Reset to initial state
        runOnUiThread {
            if (tts != null) {
                tts!!.stop()
            }
            isSpeaking = false
            currentState = AppState.DETECTING
            targetObjectName = null
            binding.overlay.setLockedObject(null)
            announcedObjects.clear()
            lastSpokenTimes.clear()
            objectLastSeenTime.clear()
            
            // Switch back to Camera tab if needed
            binding.bottomNavigation.selectedItemId = R.id.navigation_camera
            
            speak("Resetting. I am looking for objects.", "RESET_SWIPE")
        }
    }

    private fun handleSwipeLeft() {
        // Switch to Scene Description
        runOnUiThread {
            if (currentState != AppState.SCENE_DESCRIPTION) {
                // Stop any ongoing speech
                if (tts != null) {
                    tts!!.stop()
                }
                isSpeaking = false
                
                // Set state to avoid interruptions
                currentState = AppState.SCENE_DESCRIPTION
                
                // Update UI tab
                binding.bottomNavigation.selectedItemId = R.id.navigation_scene
                
                // Trigger Scene Description Logic
                triggerSceneDescription()
            }
        }
    }
    
    private fun triggerSceneDescription() {
        // This function will be called once per activation
        // We will use the latest detection results to formulate a description
        // Since we are in the main thread here, we need to access the latest bounding boxes.
        // However, bounding boxes are passed via onDetect callback.
        // We can create a flag to request a scene snapshot on the next frame or use stored data if available.
        // For simplicity, let's set a flag "pendingSceneDescription" and handle it in onDetect for the immediate next frame.
        
        // Actually, we can just look at announcedObjects or better, wait for the next detections.
        // Let's set a flag.
        pendingSceneDescription = true
    }
    
    private var pendingSceneDescription = false

    // Pass touch events to gesture detector
    override fun onTouchEvent(event: MotionEvent?): Boolean {
        return if (event != null) {
            gestureDetector.onTouchEvent(event) || super.onTouchEvent(event)
        } else {
            super.onTouchEvent(event)
        }
    }

    private fun initializeSpeechRecognizer() {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        recognitionIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
        }
        
        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            
            override fun onError(error: Int) {
                Log.e("SpeechRecognizer", "Error: $error")
                if (currentState == AppState.LISTENING) {
                     runOnUiThread {
                         speak("I didn't catch that. Please try again.", "RETRY_PROMPT")
                     }
                }
            }

            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty()) {
                    val spokenText = matches[0]
                    handleUserSpeech(spokenText)
                } else {
                    runOnUiThread {
                        speak("I didn't catch that. Please try again.", "RETRY_PROMPT")
                    }
                }
            }

            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
    }
    
    private fun handleUserSpeech(text: String) {
        val normalized = normalizeText(text)
        Log.d("Speech", "Original: $text, Normalized: $normalized")
        
        // Check if normalized text matches any previously announced object
        val match = announcedObjects.keys.find { knownObject ->
            normalized.contains(knownObject.lowercase(Locale.ROOT)) || 
            knownObject.lowercase(Locale.ROOT).contains(normalized) 
        }
        
        if (match != null) {
            targetObjectName = match
            currentState = AppState.NAVIGATING
            // Update overlay with locked object
            binding.overlay.setLockedObject(match)
            speak("$match selected. Navigation will begin.", "NAV_START")
        } else {
            speak("I couldn't find $text. Please say the name of one of the detected objects.", "RETRY_PROMPT")
        }
    }
    
    private fun normalizeText(text: String): String {
        var normalized = text.lowercase(Locale.ROOT)
        val fillers = listOf("go to", "guide me to", "please", "navigate to", "find", "where is", "i want", "detect", "see")
        fillers.forEach { normalized = normalized.replace(it, "") }
        return normalized.trim()
    }

    private fun bindListeners() {
        // Prevent manual tab switching to ensure gesture-only access logic works as intended,
        // or allow it but sync state.
        binding.bottomNavigation.setOnItemSelectedListener { item ->
            // If user clicks tabs manually, update state accordingly
            when (item.itemId) {
                R.id.navigation_camera -> {
                    if (currentState == AppState.SCENE_DESCRIPTION) {
                        handleSwipeRight() // Reuse reset/camera logic
                    }
                    true
                }
                R.id.navigation_scene -> {
                    if (currentState != AppState.SCENE_DESCRIPTION) {
                        handleSwipeLeft() // Reuse scene logic
                    }
                    true
                }
                else -> false
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider  = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        val rotation = binding.viewFinder.display.rotation

        val cameraSelector = CameraSelector
            .Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview =  Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            val bitmapBuffer =
                Bitmap.createBitmap(
                    imageProxy.width,
                    imageProxy.height,
                    Bitmap.Config.ARGB_8888
                )
            imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
            imageProxy.close()

            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

                if (isFrontCamera) {
                    postScale(
                        -1f,
                        1f,
                        imageProxy.width.toFloat(),
                        imageProxy.height.toFloat()
                    )
                }
            }

            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
                matrix, true
            )

            currentDepthMap = depthEstimator?.computeDepthMap(rotatedBitmap)
            detector?.detect(rotatedBitmap)
        }

        cameraProvider.unbindAll()

        try {
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            preview?.surfaceProvider = binding.viewFinder.surfaceProvider
        } catch(exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()) {
        if (it[Manifest.permission.CAMERA] == true && it[Manifest.permission.RECORD_AUDIO] == true) { 
            startCamera() 
        }
    }

    private fun toast(message: String) {
        runOnUiThread {
            Toast.makeText(baseContext, message, Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector?.close()
        depthEstimator?.close()
        cameraExecutor.shutdown()
        try {
            speechRecognizer.destroy()
        } catch (e: Exception) {
            Log.e("Speech", "Error destroying speech recognizer", e)
        }
        if (tts != null) {
            tts?.stop()
            tts?.shutdown()
        }
    }

    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted()){
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    companion object {
        private const val TAG = "Camera"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = mutableListOf (
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        ).toTypedArray()
    }

    override fun onEmptyDetect() {
        runOnUiThread {
            binding.overlay.clear()
        }
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        val depthMap = currentDepthMap
        val estimator = depthEstimator
        val currentTime = System.currentTimeMillis()

        // 1. Update last seen times for current objects
        boundingBoxes.forEach { box ->
            objectLastSeenTime[box.clsName] = currentTime
        }

        // 2. Memory Cleanup
        val toRemove = ArrayList<String>()
        announcedObjects.keys.forEach { clsName ->
            val lastSeen = objectLastSeenTime[clsName] ?: 0L
            if (currentTime - lastSeen > OBJECT_PERSISTENCE_MS) {
                toRemove.add(clsName)
            }
        }
        toRemove.forEach { 
            announcedObjects.remove(it) 
            objectLastSeenTime.remove(it)
        }

        // 3. Process Depth & Status
        if (depthMap != null && estimator != null) {
            boundingBoxes.forEach { box ->
                val distance = estimator.getDistance(depthMap, box)
                box.distance = String.format("%.1f m", distance)
                if (announcedObjects.containsKey(box.clsName)) {
                    box.isAnnounced = true
                }
            }
        }

        runOnUiThread {
            binding.overlay.apply {
                setResults(boundingBoxes)
                invalidate()
            }
            
            // Check for Scene Description Request
            if (pendingSceneDescription && currentState == AppState.SCENE_DESCRIPTION) {
                pendingSceneDescription = false
                generateSceneDescription(boundingBoxes)
            } else {
                // Normal Logic if NOT in Scene Description mode
                if (currentState != AppState.SCENE_DESCRIPTION) {
                    when (currentState) {
                        AppState.DETECTING -> handleDetectingState(boundingBoxes, currentTime)
                        AppState.NAVIGATING -> handleNavigatingState(boundingBoxes, currentTime)
                        else -> { 
                            // In PROMPTING and LISTENING states, we pause visual descriptions
                        }
                    }
                }
            }
        }
    }
    
    private fun generateSceneDescription(boundingBoxes: List<BoundingBox>) {
        if (boundingBoxes.isEmpty()) {
            speak("The scene appears to be empty. No objects detected.", "SCENE_DESC")
            // Return to Camera mode after speech is done handled by onUtteranceDone or manual swipe
            // Requirement says: "After completing the description: Stop speaking. Return control to the main camera view."
            // We can schedule a return to DETECTING in onUtteranceDone
            return
        }

        // Analyze objects
        val objectCounts = boundingBoxes.groupingBy { it.clsName }.eachCount()
        val objectsString = objectCounts.entries.joinToString(separator = ", ") { "${it.key}" }
        
        // Analyze Spatial Context
        // Check if there are objects in the center (path blocked)
        val centerObjects = boundingBoxes.filter { it.cx in 0.3..0.7 }
        val pathStatus = if (centerObjects.isEmpty()) "The path ahead appears mostly clear." else "There are objects directly in front of you."
        
        // Environment Type Guess (Simple heuristic based on objects)
        // E.g., Bed/Couch -> Room/Bedroom, Chair/Table -> Room/Office
        val indoorObjects = setOf("bed", "couch", "chair", "dining table", "tv", "laptop", "microwave", "refrigerator")
        val isIndoor = boundingBoxes.any { indoorObjects.contains(it.clsName.lowercase()) }
        val envType = if (isIndoor) "You are in a room" else "You are in an environment"

        val description = "$envType with $objectsString nearby. $pathStatus"
        
        speak(description, "SCENE_DESC")
    }
    
    private fun handleDetectingState(boundingBoxes: List<BoundingBox>, currentTime: Long) {
        if (isSpeaking) return
        
        val unannouncedCandidates = boundingBoxes.filter { !it.isAnnounced }
        
        if (unannouncedCandidates.isNotEmpty()) {
            val bestCandidate = unannouncedCandidates.maxByOrNull { it.cnf }
            if (bestCandidate != null) {
                val objectName = bestCandidate.clsName
                val lastTime = lastSpokenTimes[objectName] ?: 0L
                
                if (currentTime - lastTime > SPEECH_COOLDOWN_MS) {
                    val direction = getDirection(bestCandidate.cx)
                    val speechText = if (bestCandidate.distance.isNotEmpty()) {
                        "I see a $objectName at ${bestCandidate.distance.replace("m", "meters")} on your $direction"
                    } else {
                        "I see a $objectName on your $direction"
                    }
                    
                    announcedObjects[objectName] = currentTime
                    objectLastSeenTime[objectName] = currentTime // Ensure it's marked as seen
                    lastSpokenTimes[objectName] = currentTime
                    bestCandidate.isAnnounced = true
                    speak(speechText, "DESCRIBE")
                }
            }
        } else {
            // Trigger PROMPT if we have announced objects and they are still somewhat recent
            // Check if there are ANY known objects
            if (announcedObjects.isNotEmpty()) {
                 currentState = AppState.PROMPTING
                 speak("Please say the name of the object you want to navigate to.", "PROMPT")
            }
        }
    }
    
    private fun handleNavigatingState(boundingBoxes: List<BoundingBox>, currentTime: Long) {
        if (currentTime - lastNavigationUpdate < NAVIGATION_UPDATE_INTERVAL) return
        if (isSpeaking) return
        
        val target = targetObjectName ?: return
        
        val targetBox = boundingBoxes.find { it.clsName.equals(target, ignoreCase = true) }
        
        if (targetBox != null) {
            // Update last known direction when target is visible
            lastKnownDirection = getDirection(targetBox.cx)
            
            val rawDist = targetBox.distance.split(" ")[0].toFloatOrNull() ?: 100f
            
            // CHECK IF REACHED
            if (rawDist < NAVIGATION_REACHED_THRESHOLD && rawDist > 0) {
                 speak("Object found. You have reached the $target.", "NAV_COMPLETE")
                 currentState = AppState.COMPLETED
                 return
            }
            
            // GENERATE GUIDANCE
            val directionInstruction = getNavigationInstruction(targetBox.cx)
            val distText = targetBox.distance.replace("m", "meters")
            
            // Check for closer obstacles
            val obstacle = boundingBoxes.find { 
                !it.clsName.equals(target, ignoreCase = true) && 
                isCloser(it.distance, targetBox.distance) &&
                isInFront(it.cx) 
            }
            
            if (obstacle != null) {
                 val obstacleDir = getDirection(obstacle.cx)
                 val avoidInstruction = if (obstacleDir == "center") "Obstacle ahead. Move side." else "Obstacle on your $obstacleDir."
                 speak("$avoidInstruction $target is $distText", "NAV_OBSTACLE")
            } else {
                 speak("$directionInstruction. $target is $distText ahead.", "NAV_UPDATE")
            }
            lastNavigationUpdate = currentTime
        } else {
            // TARGET LOST - Provide corrective guidance based on last known position
            val correction = when (lastKnownDirection) {
                "left" -> "Turn left"
                "right" -> "Turn right"
                else -> "Turn around slowly"
            }
            speak("Target not visible. $correction.", "NAV_LOST")
            lastNavigationUpdate = currentTime
        }
    }
    
    private fun getNavigationInstruction(cx: Float): String {
        return when {
            cx < 0.4 -> "Turn slightly left"
            cx > 0.6 -> "Turn right"
            else -> "Move forward"
        }
    }
    
    private fun getDirection(cx: Float): String {
        return when {
            cx < 0.4 -> "left"
            cx > 0.6 -> "right"
            else -> "center"
        }
    }
    
    private fun isCloser(dist1: String, dist2: String): Boolean {
        // Parse "1.2 m"
        val d1 = dist1.split(" ")[0].toFloatOrNull() ?: 100f
        val d2 = dist2.split(" ")[0].toFloatOrNull() ?: 100f
        return d1 < d2
    }
    
    private fun isInFront(cx: Float): Boolean {
        return cx in 0.3..0.7
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts!!.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("TTS", "The Language not supported!")
            } else {
                tts!!.setSpeechRate(0.85f)
                
                tts!!.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {
                        isSpeaking = true
                    }

                    override fun onDone(utteranceId: String?) {
                        isSpeaking = false
                        handleUtteranceDone(utteranceId)
                    }

                    override fun onError(utteranceId: String?) {
                        isSpeaking = false
                    }
                })
                
                speak("Camera started. I am looking for objects.", "INIT")
            }
        }
    }
    
    private fun handleUtteranceDone(utteranceId: String?) {
        runOnUiThread {
            when (utteranceId) {
                "PROMPT", "RETRY_PROMPT", "MANUAL_LISTEN" -> {
                    currentState = AppState.LISTENING
                    startListening()
                }
                "NAV_COMPLETE", "RESET_SWIPE" -> {
                     // Reset to detecting state after completion
                     currentState = AppState.DETECTING
                     targetObjectName = null
                     binding.overlay.setLockedObject(null)
                     announcedObjects.clear()
                     lastSpokenTimes.clear()
                     objectLastSeenTime.clear()
                     binding.bottomNavigation.selectedItemId = R.id.navigation_camera
                }
                "SCENE_DESC" -> {
                    // After scene description finishes, return to Main Camera View (DETECTING state)
                    // Reset UI tab to Camera
                    binding.bottomNavigation.selectedItemId = R.id.navigation_camera
                    currentState = AppState.DETECTING
                }
            }
        }
    }
    
    private fun startListening() {
        try {
            speechRecognizer.startListening(recognitionIntent)
            Toast.makeText(this, "Listening...", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e("Speech", "Start listening failed", e)
        }
    }

    private fun speak(text: String, utteranceId: String) {
        val params = Bundle()
        params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, utteranceId)
        
        if (utteranceId == "PROMPT" || utteranceId == "RETRY_PROMPT" || utteranceId == "NAV_START" || utteranceId == "NAV_COMPLETE" || utteranceId == "MANUAL_LISTEN" || utteranceId == "RESET_SWIPE" || utteranceId == "SCENE_DESC") {
             tts!!.speak(text, TextToSpeech.QUEUE_FLUSH, params, utteranceId)
        } else {
             tts!!.speak(text, TextToSpeech.QUEUE_ADD, params, utteranceId)
        }
    }
}
