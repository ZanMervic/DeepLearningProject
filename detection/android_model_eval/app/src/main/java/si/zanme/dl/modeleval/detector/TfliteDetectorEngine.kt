package si.zanme.dl.modeleval.detector

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import si.zanme.dl.modeleval.model.AppModelSpec
import si.zanme.dl.modeleval.model.DetectionBox
import si.zanme.dl.modeleval.model.DetectionRunResult

class TfliteDetectorEngine(
    private val context: Context,
) {
    private var interpreter: Interpreter? = null
    private var loadedModel: AppModelSpec? = null
    private var inputWidth: Int = 0
    private var inputHeight: Int = 0
    private var inputDataType: DataType = DataType.FLOAT32

    fun loadModel(model: AppModelSpec): Result<Unit> = runCatching {
        close()
        val options = Interpreter.Options().apply {
            numThreads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
        }
        val mappedModel = loadMappedModel(model.assetFile)
        val createdInterpreter = Interpreter(mappedModel, options)
        val inputTensor = createdInterpreter.getInputTensor(0)
        val outputTensor = createdInterpreter.getOutputTensor(0)
        
        Log.d("TfliteDetector", "Interpreter created. Input shape: ${inputTensor.shape().contentToString()}, dtype: ${inputTensor.dataType()}")
        Log.d("TfliteDetector", "Interpreter created. Output shape: ${outputTensor.shape().contentToString()}, dtype: ${outputTensor.dataType()}")
        
        val shape = inputTensor.shape()
        require(shape.size == 4) { "Expected 4D input tensor, got ${shape.contentToString()}" }
        inputHeight = shape[1]
        inputWidth = shape[2]
        inputDataType = inputTensor.dataType()
        interpreter = createdInterpreter
        loadedModel = model

        val warmBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        detect(
            bitmap = warmBitmap,
            threshold = model.defaultThreshold,
            iouThreshold = 0.45f,
            maxDet = 300,
        ).getOrThrow()
    }

    fun detect(
        bitmap: Bitmap,
        threshold: Float,
        iouThreshold: Float,
        maxDet: Int,
    ): Result<DetectionRunResult> = runCatching {
        val activeInterpreter = interpreter ?: error("Model is not loaded")
        val activeModel = loadedModel ?: error("Model metadata missing")

        val preprocessed = ImagePreprocessor.prepare(
            bitmap = bitmap,
            inputWidth = inputWidth,
            inputHeight = inputHeight,
            dataType = inputDataType,
        )
        val outputTensor = activeInterpreter.getOutputTensor(0)
        val outputBuffer = allocateOutputBuffer(outputTensor.shape(), outputTensor.dataType())

        val start = SystemClock.elapsedRealtimeNanos()
        activeInterpreter.run(preprocessed.buffer, outputBuffer)
        val end = SystemClock.elapsedRealtimeNanos()

        val rawBoxes = YoloOutputParser.parse(
            output = outputBuffer,
            imageWidth = inputWidth,
            imageHeight = inputHeight,
            threshold = threshold,
            iouThreshold = iouThreshold,
            maxDet = maxDet,
        )
        val mappedBoxes = rawBoxes.map { box ->
            mapToOriginal(box, preprocessed.metadata, bitmap.width, bitmap.height)
        }

        DetectionRunResult(
            modelKey = activeModel.key,
            modelLabel = activeModel.label,
            threshold = threshold,
            boxes = mappedBoxes,
            inferenceLatencyMs = (end - start) / 1_000_000f,
        )
    }

    fun close() {
        interpreter?.close()
        interpreter = null
        loadedModel = null
    }

    private fun loadMappedModel(assetPath: String): MappedByteBuffer {
        val descriptor = context.assets.openFd(assetPath)
        FileInputStream(descriptor.fileDescriptor).use { input ->
            return input.channel.map(
                FileChannel.MapMode.READ_ONLY,
                descriptor.startOffset,
                descriptor.declaredLength,
            )
        }
    }

    private fun allocateOutputBuffer(shape: IntArray, dataType: DataType): Any {
        require(dataType == DataType.FLOAT32) { "Only FLOAT32 outputs are supported in v1" }
        return when (shape.size) {
            2 -> Array(shape[0]) { FloatArray(shape[1]) }
            3 -> Array(shape[0]) { Array(shape[1]) { FloatArray(shape[2]) } }
            4 -> Array(shape[0]) { Array(shape[1]) { Array(shape[2]) { FloatArray(shape[3]) } } }
            else -> {
                val size = shape.fold(1) { acc, value -> acc * value }
                ByteBuffer.allocateDirect(size * 4).order(ByteOrder.nativeOrder())
            }
        }
    }

    private fun mapToOriginal(
        box: DetectionBox,
        metadata: LetterboxMetadata,
        originalWidth: Int,
        originalHeight: Int,
    ): DetectionBox {
        val left = ((box.left - metadata.padX) / metadata.scale).coerceIn(0f, originalWidth.toFloat())
        val top = ((box.top - metadata.padY) / metadata.scale).coerceIn(0f, originalHeight.toFloat())
        val right = ((box.right - metadata.padX) / metadata.scale).coerceIn(0f, originalWidth.toFloat())
        val bottom = ((box.bottom - metadata.padY) / metadata.scale).coerceIn(0f, originalHeight.toFloat())
        return DetectionBox(
            left = left,
            top = top,
            right = right,
            bottom = bottom,
            confidence = box.confidence,
        )
    }
}
