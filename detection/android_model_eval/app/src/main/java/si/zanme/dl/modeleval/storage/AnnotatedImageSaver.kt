package si.zanme.dl.modeleval.storage

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import si.zanme.dl.modeleval.model.DetectionBox

class AnnotatedImageSaver {
    fun save(
        context: Context,
        sourceBitmap: Bitmap,
        boxes: List<DetectionBox>,
        sourceDisplayName: String,
        modelKey: String,
        threshold: Float,
    ): Uri {
        val annotated = drawAnnotatedBitmap(sourceBitmap, boxes)
        val displayName = buildFileName(sourceDisplayName, modelKey, threshold)
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/ModelEval")
            }
        }

        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: error("Unable to create MediaStore entry")

        resolver.openOutputStream(uri)?.use { output ->
            check(annotated.compress(Bitmap.CompressFormat.JPEG, 95, output)) {
                "Failed to write annotated image"
            }
        } ?: error("Unable to open output stream for saved image")

        return uri
    }

    private fun drawAnnotatedBitmap(source: Bitmap, boxes: List<DetectionBox>): Bitmap {
        val copy = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(copy)
        val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(255, 87, 34)
            style = Paint.Style.STROKE
            strokeWidth = 5f
        }
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 28f
        }
        val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(180, 0, 0, 0)
            style = Paint.Style.FILL
        }

        boxes.forEach { box ->
            canvas.drawRect(box.left, box.top, box.right, box.bottom, strokePaint)
            val label = String.format(Locale.US, "%.2f", box.confidence)
            val labelWidth = textPaint.measureText(label) + 18f
            val labelHeight = textPaint.textSize + 12f
            val labelRect = RectF(
                box.left,
                (box.top - labelHeight).coerceAtLeast(0f),
                box.left + labelWidth,
                box.top,
            )
            canvas.drawRect(labelRect, labelBgPaint)
            canvas.drawText(label, labelRect.left + 9f, labelRect.bottom - 6f, textPaint)
        }

        return copy
    }

    private fun buildFileName(sourceDisplayName: String, modelKey: String, threshold: Float): String {
        val stem = sourceDisplayName.substringBeforeLast('.').replace(Regex("[^A-Za-z0-9_.-]+"), "_")
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        return "${stem}_${modelKey}_thr_${String.format(Locale.US, "%.2f", threshold)}_$timestamp.jpg"
    }
}
