package com.algorigo.smartchair.decision_model_app

import android.os.Bundle
import android.os.Environment
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.core.LinearOutSlowInEasing
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.algorigo.smartchair.decision_model_app.ui.theme.DecisionModelAppTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            DecisionModelAppTheme {
                Scaffold(modifier = Modifier.fillMaxSize().padding(16.dp)) { innerPadding ->
                    ModelDownloadScreen(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(innerPadding)
                    )
                }
            }
        }
    }
}

@Composable
fun ModelDownloadScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    val modelFile = remember(context) {
        val documentsDir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: context.filesDir
        File(documentsDir, "model/decision_nox_4b.pte")
    }

    var buttonTitle by remember { mutableStateOf("Download") }
    var isButtonEnabled by remember { mutableStateOf(true) }
    var isDownloading by remember { mutableStateOf(false) }
    var downloadProgress by remember { mutableFloatStateOf(0f) }

    var fileExists by remember { mutableStateOf(modelFile.exists()) }
    var noxModelRunner by remember { mutableStateOf<NoxModelRunner?>(null) }
    var isModelLoaded by remember { mutableStateOf(false) }
    var isModelLoading by remember { mutableStateOf(false) }

    LaunchedEffect(modelFile) {
        fileExists = modelFile.exists()
        if (fileExists) {
            buttonTitle = "Download Completed"
            isButtonEnabled = false
            isDownloading = false
            downloadProgress = 1f
        } else {
            buttonTitle = "Download"
            isButtonEnabled = true
            isDownloading = false
            downloadProgress = 0f
        }
    }

    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        ProgressButton(
            text = buttonTitle,
            progress = downloadProgress,
            isDownloading = isDownloading,
            enabled = isButtonEnabled,
            onClick = {
                isDownloading = true
                isButtonEnabled = false
                downloadProgress = 0f
                buttonTitle = "Downloading 0%"

                coroutineScope.launch {
                    val success = downloadModelFile(
                        downloadUrl = "https://woon.s3.ap-northeast-2.amazonaws.com/rouddy/decision_nox_4b.pte",
                        targetFile = modelFile,
                        onProgress = { progress ->
                            downloadProgress = progress
                            buttonTitle = "Downloading ${(progress * 100).toInt()}%"
                        }
                    )

                    if (success) {
                        fileExists = true
                        buttonTitle = "Download Completed"
                        isButtonEnabled = false
                        isDownloading = false
                        downloadProgress = 1f
                    } else {
                        Toast.makeText(context, "Download failed", Toast.LENGTH_SHORT).show()
                        buttonTitle = "Download"
                        isButtonEnabled = true
                        isDownloading = false
                        downloadProgress = 0f
                    }
                }
            }
        )

        Spacer(modifier = Modifier.height(16.dp))

        val isLoadButtonEnabled = fileExists && !isModelLoading
        val loadButtonText = when {
            isModelLoading -> "Loading Model..."
            isModelLoaded -> "Model Unload"
            else -> "Model Load"
        }

        Button(
            onClick = {
                if (isModelLoaded) {
                    noxModelRunner?.destroy()
                    noxModelRunner = null
                    isModelLoaded = false
                    Toast.makeText(context, "Model Unloaded", Toast.LENGTH_SHORT).show()
                } else {
                    isModelLoading = true
                    coroutineScope.launch {
                        val runner = withContext(Dispatchers.IO) {
                            try {
                                val r = NoxModelRunner(context, modelFile.absolutePath)
                                r.loadModel()
                                r
                            } catch (e: Exception) {
                                e.printStackTrace()
                                null
                            }
                        }

                        isModelLoading = false
                        if (runner != null) {
                            noxModelRunner = runner
                            isModelLoaded = true
                            Toast.makeText(context, "Model Loaded", Toast.LENGTH_SHORT).show()
                        } else {
                            Toast.makeText(context, "Model Load Failed", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            },
            enabled = isLoadButtonEnabled,
            modifier = Modifier
                .height(50.dp)
                .widthIn(min = 220.dp)
        ) {
            Text(text = loadButtonText)
        }

        Button(
            onClick = {
                coroutineScope.launch {
                    val result = noxModelRunner?.choice(
                        state = "Customer requests a refund.",
                        instructions = "Which team should handle this?",
                        criteria = linkedMapOf(
                            "billing" to "Payments and refunds",
                            "technical" to "Product faults"
                        )
                    )

                    println("choice = ${result?.choice}")
                    println("confidence = ${result?.confidence}")
                    println("probabilities = ${result?.probabilities}")
                }
            },
            enabled = isModelLoaded,
            modifier = Modifier
                .height(50.dp)
                .widthIn(min = 220.dp)
        ) {
            Text(text = "Test Model")
        }
    }
}

@Composable
fun ProgressButton(
    text: String,
    progress: Float,
    isDownloading: Boolean,
    enabled: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val shape = RoundedCornerShape(12.dp)
    val animatedProgress by animateFloatAsState(
        targetValue = progress,
        animationSpec = tween(durationMillis = 150, easing = LinearOutSlowInEasing),
        label = "ProgressAnimation"
    )

    val containerColor = when {
        isDownloading -> MaterialTheme.colorScheme.surfaceVariant
        enabled -> MaterialTheme.colorScheme.primary
        else -> MaterialTheme.colorScheme.surfaceContainerHigh
    }

    val contentColor = when {
        isDownloading -> MaterialTheme.colorScheme.onSurface
        enabled -> MaterialTheme.colorScheme.onPrimary
        else -> MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
    }

    Surface(
        onClick = onClick,
        enabled = enabled,
        shape = shape,
        color = containerColor,
        contentColor = contentColor,
        modifier = modifier
            .height(50.dp)
            .widthIn(min = 220.dp)
            .clip(shape)
    ) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            if (isDownloading && animatedProgress > 0f) {
                Box(
                    modifier = Modifier
                        .fillMaxHeight()
                        .fillMaxWidth(animatedProgress)
                        .align(Alignment.CenterStart)
                        .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.4f))
                )
            }

            Text(
                text = text,
                style = MaterialTheme.typography.labelLarge,
                color = contentColor,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(horizontal = 24.dp)
            )
        }
    }
}

private suspend fun downloadModelFile(
    downloadUrl: String,
    targetFile: File,
    onProgress: suspend (Float) -> Unit
): Boolean {
    return withContext(Dispatchers.IO) {
        try {
            val parentDir = targetFile.parentFile
            if (parentDir != null && !parentDir.exists()) {
                parentDir.mkdirs()
            }

            val tempFile = File(parentDir, "${targetFile.name}.tmp")
            val url = URL(downloadUrl)
            val connection = url.openConnection() as HttpURLConnection
            connection.connectTimeout = 15000
            connection.readTimeout = 15000
            connection.connect()

            if (connection.responseCode == HttpURLConnection.HTTP_OK) {
                val totalBytes = connection.contentLengthLong
                var bytesDownloaded = 0L
                var lastReportedPercent = -1

                connection.inputStream.use { input ->
                    tempFile.outputStream().use { output ->
                        val buffer = ByteArray(8192)
                        var read: Int
                        while (input.read(buffer).also { read = it } != -1) {
                            output.write(buffer, 0, read)
                            bytesDownloaded += read

                            if (totalBytes > 0) {
                                val currentPercent = ((bytesDownloaded * 100) / totalBytes).toInt()
                                if (currentPercent != lastReportedPercent) {
                                    lastReportedPercent = currentPercent
                                    val progress = (bytesDownloaded.toFloat() / totalBytes).coerceIn(0f, 1f)
                                    withContext(Dispatchers.Main) {
                                        onProgress(progress)
                                    }
                                }
                            }
                        }
                    }
                }

                if (tempFile.exists() && tempFile.length() > 0) {
                    if (targetFile.exists()) {
                        targetFile.delete()
                    }
                    val renamed = tempFile.renameTo(targetFile)
                    if (!renamed) {
                        tempFile.copyTo(targetFile, overwrite = true)
                        tempFile.delete()
                    }
                    true
                } else {
                    false
                }
            } else {
                false
            }
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }
}

@Preview(showBackground = true)
@Composable
fun ModelDownloadScreenPreview() {
    DecisionModelAppTheme {
        ModelDownloadScreen()
    }
}
