package si.zanme.dl.modeleval.storage

import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.provider.OpenableColumns

data class LoadedImage(
    val bitmap: Bitmap,
    val displayName: String,
)

class UriImageLoader {
    fun load(context: Context, uri: Uri): LoadedImage {
        val source = ImageDecoder.createSource(context.contentResolver, uri)
        val bitmap = ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
            decoder.isMutableRequired = false
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
        }
        return LoadedImage(
            bitmap = bitmap,
            displayName = resolveDisplayName(context.contentResolver, uri),
        )
    }

    private fun resolveDisplayName(contentResolver: ContentResolver, uri: Uri): String {
        val projection = arrayOf(OpenableColumns.DISPLAY_NAME)
        contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
            val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (index >= 0 && cursor.moveToFirst()) {
                return cursor.getString(index)
            }
        }
        return uri.lastPathSegment ?: "image_${System.currentTimeMillis()}.jpg"
    }
}
