package si.zanme.dl.modeleval.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import si.zanme.dl.modeleval.model.ThresholdOptions

class ThresholdOptionsTest {

    @Test
    fun exposes_expected_presets_and_default_value() {
        val values = ThresholdOptions.presets

        assertEquals(listOf(0.10f, 0.15f, 0.20f, 0.25f, 0.30f, 0.40f), values)
        assertEquals(0.20f, ThresholdOptions.default)
        assertTrue(values.contains(ThresholdOptions.default))
    }
}
