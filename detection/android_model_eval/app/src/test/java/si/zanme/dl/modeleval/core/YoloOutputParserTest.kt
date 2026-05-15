package si.zanme.dl.modeleval.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import si.zanme.dl.modeleval.detector.YoloOutputParser

class YoloOutputParserTest {

    @Test
    fun parses_1x5xN_tensor_and_filters_overlapping_boxes() {
        val tensor = arrayOf(
            arrayOf(
                floatArrayOf(320f, 330f), // cx
                floatArrayOf(320f, 330f), // cy
                floatArrayOf(100f, 100f), // w
                floatArrayOf(120f, 120f), // h
                floatArrayOf(0.92f, 0.88f), // conf
            ),
        )

        val boxes = YoloOutputParser.parse(
            output = tensor,
            imageWidth = 640,
            imageHeight = 640,
            threshold = 0.20f,
            iouThreshold = 0.45f,
            maxDet = 300,
        )

        assertEquals(1, boxes.size)
        assertTrue(boxes.first().confidence > 0.9f)
        assertEquals(270f, boxes.first().left, 0.01f)
        assertEquals(260f, boxes.first().top, 0.01f)
        assertEquals(370f, boxes.first().right, 0.01f)
        assertEquals(380f, boxes.first().bottom, 0.01f)
    }

    @Test
    fun parses_1xNx6_tensor_with_objectness_and_class_score() {
        val tensor = arrayOf(
            arrayOf(
                floatArrayOf(320f, 320f, 100f, 100f, 0.9f, 0.8f),
                floatArrayOf(100f, 100f, 40f, 80f, 0.3f, 0.5f),
            ),
        )

        val boxes = YoloOutputParser.parse(
            output = tensor,
            imageWidth = 640,
            imageHeight = 640,
            threshold = 0.50f,
            iouThreshold = 0.45f,
            maxDet = 300,
        )

        assertEquals(1, boxes.size)
        assertEquals(0.72f, boxes.first().confidence, 0.001f)
    }
}
