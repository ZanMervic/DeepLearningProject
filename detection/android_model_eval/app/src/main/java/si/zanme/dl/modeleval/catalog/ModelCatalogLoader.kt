package si.zanme.dl.modeleval.catalog

import android.content.res.AssetManager
import org.json.JSONArray
import si.zanme.dl.modeleval.model.AppModelSpec

class ModelCatalogLoader {
    fun load(assetManager: AssetManager): List<AppModelSpec> {
        return runCatching {
            assetManager.open("models/manifest.json").use { input ->
                val json = input.bufferedReader().readText()
                parse(JSONArray(json))
            }
        }.getOrElse { fallbackModels() }
    }

    private fun parse(array: JSONArray): List<AppModelSpec> {
        return buildList {
            for (index in 0 until array.length()) {
                val item = array.getJSONObject(index)
                add(
                    AppModelSpec(
                        key = item.getString("key"),
                        label = item.getString("label"),
                        assetFile = item.getString("assetFile"),
                        inputSize = item.getInt("inputSize"),
                        defaultThreshold = item.optDouble("defaultThreshold", 0.20).toFloat(),
                    ),
                )
            }
        }
    }

    private fun fallbackModels(): List<AppModelSpec> = listOf(
        AppModelSpec(
            key = "yolo26n_chv_wp_coco_och_640",
            label = "YOLO26n CHV+WP+COCO+OCH 640",
            assetFile = "models/yolo26n_chv_wp_coco_och_640.tflite",
            inputSize = 640,
            defaultThreshold = 0.20f,
        ),
        AppModelSpec(
            key = "yolo26n_chv_wp_coco_och_960",
            label = "YOLO26n CHV+WP+COCO+OCH 960",
            assetFile = "models/yolo26n_chv_wp_coco_och_960.tflite",
            inputSize = 960,
            defaultThreshold = 0.20f,
        ),
    )
}
