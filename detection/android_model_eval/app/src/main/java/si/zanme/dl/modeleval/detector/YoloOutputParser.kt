package si.zanme.dl.modeleval.detector

import android.util.Log
import si.zanme.dl.modeleval.model.DetectionBox
import kotlin.math.max
import kotlin.math.min

object YoloOutputParser {
    fun parse(
        output: Any,
        imageWidth: Int,
        imageHeight: Int,
        threshold: Float,
        iouThreshold: Float,
        maxDet: Int,
    ): List<DetectionBox> {
        val candidates = when (output) {
            is Array<*> -> parseNested(output, imageWidth, imageHeight, threshold)
            else -> emptyList()
        }

        // The model provides end-to-end detections already thresholded and NMS-ed if it outputs [1, 300, 6]
        // We will just sort them descending by confidence and take up to maxDet.
        return candidates.sortedByDescending { it.confidence }.take(maxDet)
    }

    private fun parseNested(
        output: Array<*>,
        imageWidth: Int,
        imageHeight: Int,
        threshold: Float,
    ): List<DetectionBox> {
        if (output.isEmpty()) {
            return emptyList()
        }
        val batch0 = output.firstOrNull()
        if (batch0 !is Array<*>) {
            return emptyList()
        }

        val rows = when {
            batch0.all { it is FloatArray } && batch0.size in setOf(5, 6) ->
                transpose(batch0.filterIsInstance<FloatArray>())
            batch0.all { it is FloatArray } ->
                batch0.filterIsInstance<FloatArray>().toList()
            else -> emptyList()
        }

        // --- ADDED LOGS ---
        val debugThreshold = 0.01f
        val aboveThreshold = rows.filter { it.size >= 5 && it[4] >= debugThreshold }
        Log.d("YoloOutputParser", "Parsed ${rows.size} total rows. Rows with conf >= $debugThreshold: ${aboveThreshold.size}")
        
        if (aboveThreshold.isNotEmpty()) {
            val topRow = aboveThreshold.maxByOrNull { it[4] }
            if (topRow != null) {
                Log.d("YoloOutputParser", "Top row by confidence: conf=${topRow[4]}, coords=[${topRow[0]}, ${topRow[1]}, ${topRow[2]}, ${topRow[3]}]")
                if (topRow.size > 5) {
                    val classIds = aboveThreshold.map { it[5] }.distinct()
                    Log.d("YoloOutputParser", "Distinct class IDs above threshold: $classIds")
                }
            }
        }
        // ------------------

        return rows.mapNotNull { row ->
            toBox(
                attributes = row,
                imageWidth = imageWidth.toFloat(),
                imageHeight = imageHeight.toFloat(),
                threshold = threshold,
            )
        }
    }

    private fun transpose(columns: List<FloatArray>): List<FloatArray> {
        if (columns.isEmpty()) {
            return emptyList()
        }
        val rowCount = columns.first().size
        return List(rowCount) { rowIndex ->
            FloatArray(columns.size) { columnIndex -> columns[columnIndex][rowIndex] }
        }
    }

    private fun toBox(
        attributes: FloatArray,
        imageWidth: Float,
        imageHeight: Float,
        threshold: Float,
    ): DetectionBox? {
        if (attributes.size < 5) {
            return null
        }

        // YOLO26 End-to-End detection format: [x1, y1, x2, y2, confidence, class_id]
        // Coordinates are normalized [0..1], scale them to imageWidth / imageHeight
        val left = (attributes[0] * imageWidth).coerceIn(0f, imageWidth)
        val top = (attributes[1] * imageHeight).coerceIn(0f, imageHeight)
        val right = (attributes[2] * imageWidth).coerceIn(0f, imageWidth)
        val bottom = (attributes[3] * imageHeight).coerceIn(0f, imageHeight)

        val confidence = attributes[4]

        // Restore correct thresholding behavior
        if (confidence < threshold) {
            return null
        }

        // Check if there is a class constraint if needed. The model may just output class 0.

        if (right <= left || bottom <= top) {
            return null
        }

        return DetectionBox(
            left = left,
            top = top,
            right = right,
            bottom = bottom,
            confidence = confidence,
        )
    }

    private fun nonMaxSuppression(
        boxes: List<DetectionBox>,
        iouThreshold: Float,
        maxDet: Int,
    ): List<DetectionBox> {
        val remaining = boxes.sortedByDescending { it.confidence }.toMutableList()
        val selected = mutableListOf<DetectionBox>()

        while (remaining.isNotEmpty() && selected.size < maxDet) {
            val current = remaining.removeFirst()
            selected += current
            remaining.removeAll { candidate -> intersectionOverUnion(current, candidate) > iouThreshold }
        }

        return selected
    }

    private fun intersectionOverUnion(a: DetectionBox, b: DetectionBox): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)

        val width = (right - left).coerceAtLeast(0f)
        val height = (bottom - top).coerceAtLeast(0f)
        val intersection = width * height
        if (intersection <= 0f) {
            return 0f
        }

        val union = a.area + b.area - intersection
        if (union <= 0f) {
            return 0f
        }
        return intersection / union
    }
}
