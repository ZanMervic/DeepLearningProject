package si.zanme.dl.modeleval.logging

import android.content.Context
import android.net.Uri
import androidx.core.content.FileProvider
import java.io.File

class RunLogStore(
    private val context: Context,
) {
    private val exportsDir: File = File(context.filesDir, "exports").apply { mkdirs() }
    private val csvFile: File = File(exportsDir, "session_log.csv")

    fun append(record: RunRecord) {
        if (!csvFile.exists()) {
            csvFile.writeText(RunLogCsvFormatter.toCsv(listOf(record)))
            return
        }

        val line = RunLogCsvFormatter.toCsv(listOf(record)).lineSequence().drop(1).joinToString("\n") + "\n"
        csvFile.appendText(line)
    }

    fun readAll(): List<RunRecord> {
        if (!csvFile.exists()) {
            return emptyList()
        }
        val lines = csvFile.readLines().drop(1).filter { it.isNotBlank() }
        return lines.mapNotNull(::parseRecord)
    }

    fun clear() {
        if (csvFile.exists()) {
            csvFile.delete()
        }
    }

    fun shareUri(): Uri {
        if (!csvFile.exists()) {
            csvFile.writeText(RunLogCsvFormatter.toCsv(emptyList()))
        }
        return FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            csvFile,
        )
    }

    private fun parseRecord(line: String): RunRecord? {
        val parts = parseCsvLine(line)
        if (parts.size != 10) {
            return null
        }
        return runCatching {
            RunRecord(
                timestampIso = parts[0],
                sourceType = parts[1],
                sourceDisplayName = parts[2],
                modelKey = parts[3],
                inputSize = parts[4].toInt(),
                threshold = parts[5].toFloat(),
                imageWidth = parts[6].toInt(),
                imageHeight = parts[7].toInt(),
                detectionCount = parts[8].toInt(),
                inferenceLatencyMs = parts[9].toFloat(),
            )
        }.getOrNull()
    }

    private fun parseCsvLine(line: String): List<String> {
        val out = mutableListOf<String>()
        val current = StringBuilder()
        var insideQuotes = false
        var index = 0

        while (index < line.length) {
            val char = line[index]
            when {
                char == '"' && insideQuotes && index + 1 < line.length && line[index + 1] == '"' -> {
                    current.append('"')
                    index += 1
                }
                char == '"' -> insideQuotes = !insideQuotes
                char == ',' && !insideQuotes -> {
                    out += current.toString()
                    current.clear()
                }
                else -> current.append(char)
            }
            index += 1
        }
        out += current.toString()
        return out
    }
}
