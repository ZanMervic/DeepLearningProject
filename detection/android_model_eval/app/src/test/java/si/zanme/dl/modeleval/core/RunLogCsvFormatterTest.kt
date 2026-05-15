package si.zanme.dl.modeleval.core

import org.junit.Assert.assertEquals
import org.junit.Test
import si.zanme.dl.modeleval.logging.RunLogCsvFormatter
import si.zanme.dl.modeleval.logging.RunRecord

class RunLogCsvFormatterTest {

    @Test
    fun writes_header_and_escapes_commas_and_quotes() {
        val record = RunRecord(
            timestampIso = "2026-05-14T10:15:30",
            sourceType = "gallery",
            sourceDisplayName = "test, \"image\".jpg",
            modelKey = "yolo26n_chv_wp_coco_och_640",
            inputSize = 640,
            threshold = 0.20f,
            imageWidth = 1080,
            imageHeight = 1920,
            detectionCount = 3,
            inferenceLatencyMs = 12.34f,
        )

        val csv = RunLogCsvFormatter.toCsv(listOf(record))

        val expected = buildString {
            appendLine("timestamp_iso,source_type,source_display_name,model_key,input_size,threshold,image_width,image_height,detection_count,inference_latency_ms")
            appendLine("2026-05-14T10:15:30,gallery,\"test, \"\"image\"\".jpg\",yolo26n_chv_wp_coco_och_640,640,0.20,1080,1920,3,12.34")
        }

        assertEquals(expected, csv)
    }
}
