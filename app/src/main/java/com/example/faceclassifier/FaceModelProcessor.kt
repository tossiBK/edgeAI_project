package com.example.faceclassifier

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import androidx.core.graphics.scale

/**
 * Helper class for TFLite model operations on face bitmaps with automatic shape detection
 */
class FaceModelProcessor(
    private val context: Context,
    private val model1Name: String,
    private val model2Name: String,
    private val model3Name: String,
    private val outputSize1: Int,
    private val outputSize2: Int,
    private val outputSize3: Int,
    private val model1ClassLabels: Array<String>,
    private val model2ClassLabels: Array<String>,
    private val model3ClassLabels: Array<String>,
    private val useGpu: Boolean = true
) {
    private var interpreter1: Interpreter? = null
    private var interpreter2: Interpreter? = null
    private var interpreter3: Interpreter? = null

    private var gpuDelegate: GpuDelegate? = null

    // Dynamic model configuration detected from loaded models
    private var model1Config: ModelConfig? = null
    private var model2Config: ModelConfig? = null
    private var model3Config: ModelConfig? = null

    /**
     * Configuration class for each model
     */
    data class ModelConfig(
        val inputWidth: Int,
        val inputHeight: Int,
        val channels: Int,
        val dataType: org.tensorflow.lite.DataType,
        val bytesPerChannel: Int,
        val outputSize: Int
    )

    companion object {
        private const val TAG = "FaceModelProcessor"
    }

    init {
        // Load all three models on initialization
        loadModels()
    }

    /**
     * Loads all three TFLite models and detects their configurations
     */
    private fun loadModels() {
        val options = Interpreter.Options()

        // try and set GPU if available
        if (useGpu) {
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                val delegateOptions = compatList.bestOptionsForThisDevice
                gpuDelegate = GpuDelegate(delegateOptions)
                options.addDelegate(gpuDelegate)
            } else {
                // Fall back to CPU if GPU not supported
                options.setNumThreads(4)
            }
        } else {
            // Use CPU with multiple threads
            options.setNumThreads(4)
        }

        try {
            // Load all three models
            interpreter1 = Interpreter(loadModelFile(model1Name), options)
            interpreter2 = Interpreter(loadModelFile(model2Name), options)
            interpreter3 = Interpreter(loadModelFile(model3Name), options)

            // Detect and configure model parameters
            model1Config = detectModelConfig(interpreter1!!, outputSize1)
            model2Config = detectModelConfig(interpreter2!!, outputSize2)
            model3Config = detectModelConfig(interpreter3!!, outputSize3)

            // Log detected configurations for debugging
            logModelConfig("Model 1", model1Config)
            logModelConfig("Model 2", model2Config)
            logModelConfig("Model 3", model3Config)

            // Validate class labels match model output sizes
            validateClassLabels()

        } catch (e: Exception) {
            Log.e(TAG, "Error loading models: ${e.message}", e)
        }
    }

    /**
     * Validate that class labels match model output sizes. Should fit, but better to be sure
     */
    private fun validateClassLabels() {
        model1Config?.let {
            if (model1ClassLabels.size != it.outputSize) {
                Log.w(TAG, "Model 1 output size (${it.outputSize}) doesn't match provided class labels (${model1ClassLabels.size})")
            }
        }

        model2Config?.let {
            if (model2ClassLabels.size != it.outputSize) {
                Log.w(TAG, "Model 2 output size (${it.outputSize}) doesn't match provided class labels (${model2ClassLabels.size})")
            }
        }

        model3Config?.let {
            if (model3ClassLabels.size != it.outputSize) {
                Log.w(TAG, "Model 3 output size (${it.outputSize}) doesn't match provided class labels (${model3ClassLabels.size})")
            }
        }
    }

    /**
     * Log detected model configuration for debugging
     */
    private fun logModelConfig(name: String, config: ModelConfig?) {
        config?.let {
            Log.d(TAG, "$name config: ${it.inputWidth}x${it.inputHeight}x${it.channels}, " +
                    "dataType: ${it.dataType}, bytesPerChannel: ${it.bytesPerChannel}, " +
                    "outputSize: ${it.outputSize}")
        } ?: Log.e(TAG, "$name configuration detection failed")
    }

    /**
     * Detect model configuration from interpreter
     */
    private fun detectModelConfig(interpreter: Interpreter, outputSize: Int): ModelConfig {
        val inputTensor = interpreter.getInputTensor(0)
        val shape = inputTensor.shape()

        // Analyze tensor properties
        // shapes for images in tflite: [1, height, width, channels]
        val batchSize = shape[0]
        val height = shape[1]
        val width = shape[2]
        val channels = if (shape.size >= 4) shape[3] else 1

        val dataType = inputTensor.dataType()
        val bytesPerChannel = when (dataType) {
            org.tensorflow.lite.DataType.FLOAT32 -> 4
            org.tensorflow.lite.DataType.UINT8 -> 1
            else -> 4 // Default to float32
        }

        return ModelConfig(
            inputWidth = width,
            inputHeight = height,
            channels = channels,
            dataType = dataType,
            bytesPerChannel = bytesPerChannel,
            outputSize = outputSize
        )
    }

    /**
     * Load a TFLite model file from assets
     */
    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Preprocess a bitmap for model input according to detected configuration
     */
    private fun preprocessBitmap(bitmap: Bitmap, config: ModelConfig): ByteBuffer {
        // Resize the bitmap to match the model's input dimensions
        val resizedBitmap = if (bitmap.width != config.inputWidth || bitmap.height != config.inputHeight) {
            bitmap.scale(config.inputWidth, config.inputHeight)
        } else {
            bitmap
        }

        // Calculate buffer size based on detected properties
        val bufferSize = 1 * config.inputWidth * config.inputHeight * config.channels * config.bytesPerChannel

        // Allocate a ByteBuffer for the model input
        val inputBuffer = ByteBuffer.allocateDirect(bufferSize)
        inputBuffer.order(ByteOrder.nativeOrder())

        // Fill the ByteBuffer with pixel values
        val pixels = IntArray(config.inputWidth * config.inputHeight)
        resizedBitmap.getPixels(pixels, 0, config.inputWidth, 0, 0, config.inputWidth, config.inputHeight)

        for (pixel in pixels) {
            val r = pixel shr 16 and 0xFF
            val g = pixel shr 8 and 0xFF
            val b = pixel and 0xFF

            if (config.channels == 1) {
                // Grayscale processing
                val gray = (0.299f * r + 0.587f * g + 0.114f * b).toInt()

                if (config.dataType == org.tensorflow.lite.DataType.FLOAT32) {
                    // Normalize to [0, 1]
                    inputBuffer.putFloat(gray / 255.0f)
                } else {
                    // Use as-is for UINT8
                    inputBuffer.put(gray.toByte())
                }
            } else {
                // RGB processing
                if (config.dataType == org.tensorflow.lite.DataType.FLOAT32) {
                    // Normalize to [0, 1]
                    inputBuffer.putFloat(r / 255.0f)
                    inputBuffer.putFloat(g / 255.0f)
                    inputBuffer.putFloat(b / 255.0f)
                } else {
                    // Use as-is for UINT8
                    inputBuffer.put(r.toByte())
                    inputBuffer.put(g.toByte())
                    inputBuffer.put(b.toByte())
                }
            }
        }

        // Reset the buffer position to beginning
        inputBuffer.rewind()

        if (resizedBitmap != bitmap) {
            resizedBitmap.recycle()
        }

        return inputBuffer
    }

    /**
     * Run inference on model 1
     */
    fun runModel1(faceBitmap: Bitmap): FloatArray? {
        val config = model1Config ?: return null

        try {
            val inputBuffer = preprocessBitmap(faceBitmap, config)

            // Create output buffer based on model's output size
            val outputBuffer = Array(1) { FloatArray(config.outputSize) }

            interpreter1?.run(inputBuffer, outputBuffer)

            return outputBuffer[0]
        } catch (e: Exception) {
            Log.e(TAG, "Error running model 1: ${e.message}", e)
            return null
        }
    }

    /**
     * Run inference on model 2
     */
    fun runModel2(faceBitmap: Bitmap): FloatArray? {
        val config = model2Config ?: return null

        try {
            val inputBuffer = preprocessBitmap(faceBitmap, config)

            // Create output buffer based on model's output size
            val outputBuffer = Array(1) { FloatArray(config.outputSize) }

            interpreter2?.run(inputBuffer, outputBuffer)

            return outputBuffer[0]
        } catch (e: Exception) {
            Log.e(TAG, "Error running model 2: ${e.message}", e)
            return null
        }
    }

    /**
     * Run inference on model 3
     */
    fun runModel3(faceBitmap: Bitmap): FloatArray? {
        val config = model3Config ?: return null

        try {
            val inputBuffer = preprocessBitmap(faceBitmap, config)

            // Create output buffer based on model's output size
            val outputBuffer = Array(1) { FloatArray(config.outputSize) }

            interpreter3?.run(inputBuffer, outputBuffer)

            return outputBuffer[0]
        } catch (e: Exception) {
            Log.e(TAG, "Error running model 3: ${e.message}", e)
            return null
        }
    }

    /**
     * Run all three models on a face bitmap and return combined results
     */
    fun analyzeFace(faceBitmap: Bitmap): FaceAnalysisResult? {
        val startTime = SystemClock.elapsedRealtime()

        val results1 = runModel1(faceBitmap)
        val results2 = runModel2(faceBitmap)
        val results3 = runModel3(faceBitmap)

        // Check if any model failed
        if (results1 == null || results2 == null || results3 == null) {
            return null
        }

        val inferenceTime = SystemClock.elapsedRealtime() - startTime

        // Get top class predictions
        val topClass1 = getTopClass(results1, model1ClassLabels)
        val topClass2 = getTopClass(results2, model2ClassLabels)
        val topClass3 = getTopClass(results3, model3ClassLabels)

        return FaceAnalysisResult(
            model1Results = results1,
            model2Results = results2,
            model3Results = results3,
            topClass1 = topClass1,
            topClass2 = topClass2,
            topClass3 = topClass3,
            inferenceTimeMs = inferenceTime
        )
    }

    /**
     * Find the class with the highest probability
     */
    private fun getTopClass(probabilities: FloatArray, classLabels: Array<String>): TopClassPrediction {
        var maxIndex = 0
        var maxProbability = probabilities[0]

        for (i in 1 until probabilities.size) {
            if (probabilities[i] > maxProbability) {
                maxProbability = probabilities[i]
                maxIndex = i
            }
        }

        // Get the class label if available (index is in bounds)
        val className = if (maxIndex < classLabels.size) classLabels[maxIndex] else "unknown"

        return TopClassPrediction(
            classIndex = maxIndex,
            className = className,
            confidence = maxProbability
        )
    }

    /**
     * Clean up resources
     */
    fun close() {
        interpreter1?.close()
        interpreter2?.close()
        interpreter3?.close()
        gpuDelegate?.close()
    }

    /**
     * Data class representing a single top class prediction
     */
    data class TopClassPrediction(
        val classIndex: Int,       // Index of the top class
        val className: String,     // Name of the top class
        val confidence: Float      // Confidence score (0-1)
    )

    /**
     * Data class to hold all results from the three models, so we can use for the view
     */
    data class FaceAnalysisResult(
        val model1Results: FloatArray,
        val model2Results: FloatArray,
        val model3Results: FloatArray,
        val topClass1: TopClassPrediction,
        val topClass2: TopClassPrediction,
        val topClass3: TopClassPrediction,
        val inferenceTimeMs: Long
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as FaceAnalysisResult

            if (!model1Results.contentEquals(other.model1Results)) return false
            if (!model2Results.contentEquals(other.model2Results)) return false
            if (!model3Results.contentEquals(other.model3Results)) return false
            if (topClass1 != other.topClass1) return false
            if (topClass2 != other.topClass2) return false
            if (topClass3 != other.topClass3) return false
            if (inferenceTimeMs != other.inferenceTimeMs) return false

            return true
        }

        override fun hashCode(): Int {
            var result = model1Results.contentHashCode()
            result = 31 * result + model2Results.contentHashCode()
            result = 31 * result + model3Results.contentHashCode()
            result = 31 * result + topClass1.hashCode()
            result = 31 * result + topClass2.hashCode()
            result = 31 * result + topClass3.hashCode()
            result = 31 * result + inferenceTimeMs.hashCode()
            return result
        }
    }
}