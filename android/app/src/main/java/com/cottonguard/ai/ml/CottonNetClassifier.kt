package com.cottonguard.ai.ml

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.*
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.exp

/**
 * CottonNet Mobile Classifier using ONNX Runtime Android.
 * Runs 4.87M parameter CottonNet with 224x224 input and ImageNet normalization.
 */
class CottonNetClassifier(private val context: Context) {

    private var ortEnvironment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null

    companion object {
        const val MODEL_FILE = "cottonnet.onnx"
        const val INPUT_SIZE = 224
        
        // Exact class names mapping from SAR-CLD-2024 training
        val CLASS_NAMES = listOf(
            "Bacterial Blight",
            "Curl Virus",
            "Healthy Leaf",
            "Herbicide Growth Damage",
            "Leaf Hopper Jassids",
            "Leaf Redding",
            "Leaf Variegation"
        )
        
        // ImageNet normalization constants
        val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    data class Recognition(
        val className: String,
        val classIndex: Int,
        val confidence: Float,
        val latencyMs: Long,
        val allProbabilities: FloatArray
    )

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            val modelBytes = context.assets.open(MODEL_FILE).readBytes()
            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            ortSession = ortEnvironment.createSession(modelBytes, sessionOptions)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * Preprocesses Bitmap -> [1, 3, 224, 224] Float Tensor and runs inference.
     */
    fun classify(bitmap: Bitmap): Recognition {
        val session = ortSession ?: throw IllegalStateException("ONNX Session not initialized")
        val startTime = System.currentTimeMillis()

        // 1. Center crop to square (preserves leaf aspect ratio without distortion)
        val squareBitmap = if (bitmap.width == bitmap.height) {
            bitmap
        } else {
            val size = minOf(bitmap.width, bitmap.height)
            val x = (bitmap.width - size) / 2
            val y = (bitmap.height - size) / 2
            Bitmap.createBitmap(bitmap, x, y, size, size)
        }

        // 2. High-quality Bilinear Resize to 224x224
        val resized = Bitmap.createScaledBitmap(squareBitmap, INPUT_SIZE, INPUT_SIZE, true)
        
        // 3. Normalize and construct CHW FloatBuffer
        val inputBuffer = FloatBuffer.allocate(1 * 3 * INPUT_SIZE * INPUT_SIZE)
        inputBuffer.rewind()

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // CHW format: Channel R -> G -> B
        for (c in 0..2) {
            for (i in 0 until INPUT_SIZE * INPUT_SIZE) {
                val pixel = pixels[i]
                val channelVal = when (c) {
                    0 -> ((pixel shr 16 and 0xFF) / 255.0f - MEAN[0]) / STD[0]
                    1 -> ((pixel shr 8 and 0xFF) / 255.0f - MEAN[1]) / STD[1]
                    else -> ((pixel and 0xFF) / 255.0f - MEAN[2]) / STD[2]
                }
                inputBuffer.put(channelVal)
            }
        }
        inputBuffer.rewind()

        // 3. Create Tensor & Execute Session
        val inputShape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, inputBuffer, inputShape)

        val results = session.run(Collections.singletonMap("input", inputTensor))
        val outputTensor = results[0].value as Array<FloatArray>
        val logits = outputTensor[0]

        // 4. Softmax computation
        val probs = softmax(logits)
        var maxIdx = 0
        var maxProb = probs[0]
        for (i in 1 until probs.size) {
            if (probs[i] > maxProb) {
                maxProb = probs[i]
                maxIdx = i
            }
        }

        val latency = System.currentTimeMillis() - startTime

        return Recognition(
            className = CLASS_NAMES[maxIdx],
            classIndex = maxIdx,
            confidence = maxProb,
            latencyMs = latency,
            allProbabilities = probs
        )
    }

    private fun softmax(logits: FloatArray): FloatArray {
        var maxLogit = Float.NEGATIVE_INFINITY
        for (v in logits) if (v > maxLogit) maxLogit = v
        var sum = 0.0f
        val expArray = FloatArray(logits.size)
        for (i in logits.indices) {
            expArray[i] = exp((logits[i] - maxLogit).toDouble()).toFloat()
            sum += expArray[i]
        }
        for (i in expArray.indices) {
            expArray[i] /= sum
        }
        return expArray
    }

    fun close() {
        ortSession?.close()
        ortEnvironment.close()
    }
}
