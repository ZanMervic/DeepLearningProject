package si.zanme.dl.modeleval.detector

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min
import org.tensorflow.lite.DataType

data class LetterboxMetadata(
    val scale: Float,
    val padX: Float,
    val padY: Float,
    val inputWidth: Int,
    val inputHeight: Int,
)

data class PreprocessedImage(
    val buffer: ByteBuffer,
    val metadata: LetterboxMetadata,
)

object ImagePreprocessor {
    fun prepare(
        bitmap: Bitmap,
        inputWidth: Int,
        inputHeight: Int,
        dataType: DataType,
    ): PreprocessedImage {
        val scale = min(
            inputWidth.toFloat() / bitmap.width.toFloat(),
            inputHeight.toFloat() / bitmap.height.toFloat(),
        )
        val scaledWidth = (bitmap.width * scale).toInt().coerceAtLeast(1)
        val scaledHeight = (bitmap.height * scale).toInt().coerceAtLeast(1)
        val padX = (inputWidth - scaledWidth) / 2f
        val padY = (inputHeight - scaledHeight) / 2f

        val scaled = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)
        val letterboxed = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(letterboxed)
        canvas.drawColor(Color.rgb(114, 114, 114))
        canvas.drawBitmap(scaled, padX, padY, Paint(Paint.ANTI_ALIAS_FLAG))

        val buffer = when (dataType) {
            DataType.FLOAT32 -> floatInputBuffer(letterboxed)
            DataType.UINT8 -> byteInputBuffer(letterboxed)
            else -> throw IllegalArgumentException("Unsupported input tensor type: $dataType")
        }

        return PreprocessedImage(
            buffer = buffer,
            metadata = LetterboxMetadata(
                scale = scale,
                padX = padX,
                padY = padY,
                inputWidth = inputWidth,
                inputHeight = inputHeight,
            ),
        )
    }

    private fun floatInputBuffer(bitmap: Bitmap): ByteBuffer {
        val pixelCount = bitmap.width * bitmap.height
        val buffer = ByteBuffer.allocateDirect(pixelCount * 3 * 4).order(ByteOrder.nativeOrder())
        val pixels = IntArray(pixelCount)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        pixels.forEach { pixel ->
            buffer.putFloat(Color.red(pixel) / 255f)
            buffer.putFloat(Color.green(pixel) / 255f)
            buffer.putFloat(Color.blue(pixel) / 255f)
        }
        buffer.rewind()
        return buffer
    }

    private fun byteInputBuffer(bitmap: Bitmap): ByteBuffer {
        val pixelCount = bitmap.width * bitmap.height
        val buffer = ByteBuffer.allocateDirect(pixelCount * 3).order(ByteOrder.nativeOrder())
        val pixels = IntArray(pixelCount)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        pixels.forEach { pixel ->
            buffer.put(Color.red(pixel).toByte())
            buffer.put(Color.green(pixel).toByte())
            buffer.put(Color.blue(pixel).toByte())
        }
        buffer.rewind()
        return buffer
    }
}
