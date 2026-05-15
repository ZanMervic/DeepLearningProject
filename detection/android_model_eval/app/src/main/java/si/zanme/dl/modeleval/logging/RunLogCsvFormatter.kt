package si.zanme.dl.modeleval.logging

object RunLogCsvFormatter {
    private val header = listOf(
        "timestamp_iso",
        "source_type",
        "source_display_name",
        "model_key",
        "input_size",
        "threshold",
        "image_width",
        "image_height",
        "detection_count",
        "inference_latency_ms",
    )

    fun toCsv(records: List<RunRecord>): String {
        val rows = buildList {
            add(header.joinToString(","))
            records.forEach { record ->
                add(
                    listOf(
                        record.timestampIso,
                        record.sourceType,
                        record.sourceDisplayName,
                        record.modelKey,
                        record.inputSize.toString(),
                        "%.2f".format(record.threshold),
                        record.imageWidth.toString(),
                        record.imageHeight.toString(),
                        record.detectionCount.toString(),
                        "%.2f".format(record.inferenceLatencyMs),
                    ).joinToString(",") { escape(it) },
                )
            }
        }
        return rows.joinToString(separator = "\n", postfix = "\n")
    }

    private fun escape(value: String): String {
        val mustQuote = value.contains(',') || value.contains('"') || value.contains('\n')
        if (!mustQuote) {
            return value
        }
        return "\"" + value.replace("\"", "\"\"") + "\""
    }
}
