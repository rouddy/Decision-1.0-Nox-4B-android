package com.algorigo.smartchair.decision_model_app

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import android.content.Context
import android.util.Log
import com.facebook.soloader.SoLoader
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream
import kotlin.math.exp
import kotlin.math.max

class NoxModelRunner(
    private val context: Context,
    private val modelPath: String,
    private val maxSeqLen: Int = 1024,
    private val maxCandidates: Int = 16
) {

    companion object {
        /**
         * Decision-1.0-Nox-4B main release calibration.
         *
         * temperature.json:
         * 1.3231350559653137
         */
        private const val DEFAULT_TEMPERATURE = 1.3231350559653137f

        private const val TOKENIZER_ASSET = "nox/tokenizer.json"
    }

    private var modelModule: Module? = null
    private var tokenizer: HuggingFaceTokenizer? = null
    private var promptBuilder: NoxPromptBuilder? = null

    @Synchronized
    fun loadModel() {
        val file = File(modelPath)

        require(file.exists()) {
            "Model file does not exist: $modelPath"
        }

        require(file.length() > 0L) {
            "Model file is empty: $modelPath"
        }

        SoLoader.init(context, false)

        modelModule?.destroy()
        modelModule = null

        tokenizer?.close()
        tokenizer = null

        // ----------------------------------------------------------
        // Copy tokenizer.json from assets to internal storage.
        // ----------------------------------------------------------

        val tokenizerFile = File(
            context.filesDir, "nox/tokenizer.json"
        )

        if (!tokenizerFile.exists()) {
            tokenizerFile.parentFile?.mkdirs()
        }

        context.assets.open(TOKENIZER_ASSET).use { input ->
            if ((if (tokenizerFile.exists()) tokenizerFile.length() else 0L) != input.available()
                    .toLong()
            ) {
                tokenizerFile.delete()
            }
            Log.e("!!!", "input : ${input.available()} bytes")
            FileOutputStream(tokenizerFile).use { output ->
                input.copyTo(output)
            }
        }
        Log.e("!!!", "output : ${tokenizerFile.exists()}")
        Log.e("!!!", "output : ${tokenizerFile.length()} bytes")

        // ----------------------------------------------------------
        // HuggingFace tokenizer
        // ----------------------------------------------------------
        Log.e("!!!", "tokenizerFile.toPath() = ${tokenizerFile.toPath()}")
        tokenizer = HuggingFaceTokenizer.newInstance(
            tokenizerFile.toPath()
        )

        promptBuilder = NoxPromptBuilder(
            tokenizer = tokenizer!!, maxSeqLen = maxSeqLen, maxCandidates = maxCandidates
        )

        // ----------------------------------------------------------
        // ExecuTorch
        // ----------------------------------------------------------

        modelModule = Module.load(
            modelPath, Module.LOAD_MODE_MMAP
        )
    }

    fun destroy() {

        tokenizer?.close()
        tokenizer = null

        modelModule?.destroy()
        modelModule = null

        promptBuilder = null
    }

    fun choice(
        state: String, instructions: String, criteria: LinkedHashMap<String, String?>
    ): NoxAnswer.Choice {

        val module = requireModule()
        val builder = requireBuilder()

        val question = NoxChoiceQuestion(
            instructions = instructions, criteria = criteria
        )

        val encoded = builder.encodeChoice(
            state = state, question = question
        )

        val scores = runRaw(
            module = module, input = encoded
        )

        val probabilities = softmax(
            scores = scores, temperature = DEFAULT_TEMPERATURE
        )

        val winnerIndex = probabilities.indices.maxByOrNull {
            probabilities[it]
        } ?: error("Model returned no candidates")

        val sorted = probabilities.sortedDescending()

        val confidence = if (sorted.size == 1) {
            1.0f
        } else {
            (sorted[0] - sorted[1]).coerceIn(0.0f, 1.0f)
        }

        val probabilityMap = LinkedHashMap<String, Float>()

        encoded.candidateKeys.forEachIndexed { index, key ->
            probabilityMap[key] = probabilities[index]
        }

        return NoxAnswer.Choice(
            choice = encoded.candidateKeys[winnerIndex],
            confidence = confidence,
            probabilities = probabilityMap
        )
    }

    fun score(
        state: String, instructions: String, criteria: List<String>
    ): NoxAnswer.Score {

        require(criteria.size >= 2) {
            "Score requires at least two levels"
        }

        val builder = requireBuilder()

        val encoded = builder.encodeScore(
            state = state, question = NoxScoreQuestion(
                instructions = instructions, criteria = criteria
            )
        )

        val scores = runRaw(
            module = requireModule(), input = encoded
        )

        val probabilities = softmax(
            scores = scores, temperature = DEFAULT_TEMPERATURE
        )

        val expectedScore = probabilities.indices.sumOf { index ->
            index.toDouble() * probabilities[index].toDouble()
        }.toFloat()

        val mean = expectedScore.toDouble()

        val variance = probabilities.indices.sumOf { index ->
            probabilities[index].toDouble() * (index - mean) * (index - mean)
        }

        val uniformVariance =
            (probabilities.size.toDouble() * probabilities.size.toDouble() - 1.0) / 12.0

        val confidence = if (uniformVariance <= 0.0) {
            1.0f
        } else {
            (1.0 - variance / uniformVariance).coerceIn(0.0, 1.0).toFloat()
        }

        val legend = LinkedHashMap<String, String>()

        criteria.forEachIndexed { index, description ->
            legend[index.toString()] = description
        }

        return NoxAnswer.Score(
            score = expectedScore,
            confidence = confidence,
            probabilities = probabilities.toList(),
            legend = legend
        )
    }

    fun noul(
        state: String, instructions: String
    ): NoxAnswer.Noul {

        val builder = requireBuilder()

        val encoded = builder.encodeNoul(
            state = state, question = NoxNoulQuestion(
                instructions = instructions
            )
        )

        val scores = runRaw(
            module = requireModule(), input = encoded
        )

        val probabilities = softmax(
            scores = scores, temperature = DEFAULT_TEMPERATURE
        )

        // Nox runtime returns candidate #1 ("true"/yes)
        // as the Noul probability.
        return NoxAnswer.Noul(
            probability = probabilities[1]
        )
    }

    /**
     * Direct raw inference.
     *
     * The exported graph expects:
     *
     * 1. input_ids
     * 2. attention_mask
     * 3. candidate_positions
     * 4. candidate_mask
     * 5. query_positions
     */
    fun runRaw(
        input: NoxEncodedInput
    ): FloatArray {

        return runRaw(
            module = requireModule(), input = input
        )
    }

    private fun runRaw(
        module: Module, input: NoxEncodedInput
    ): FloatArray {

        val inputIdsTensor = Tensor.fromBlob(
            input.inputIds, longArrayOf(
                1L, input.inputIds.size.toLong()
            )
        )

        val attentionMaskTensor = Tensor.fromBlob(
            input.attentionMask, longArrayOf(
                1L, input.attentionMask.size.toLong()
            )
        )

        val candidatePositionsTensor = Tensor.fromBlob(
            input.candidatePositions, longArrayOf(
                1L, input.candidatePositions.size.toLong()
            )
        )

        /*
         * IMPORTANT:
         *
         * This assumes the exported graph uses an int64
         * candidate_mask and converts it internally:
         *
         * candidate_mask = candidate_mask.to(torch.bool)
         *
         * I recommend exporting the graph this way because
         * Android Tensor creation is then simple and stable.
         */
        val candidateMaskTensor = Tensor.fromBlob(
            input.candidateMask, longArrayOf(
                1L, input.candidateMask.size.toLong()
            )
        )

        val queryPositionsTensor = Tensor.fromBlob(
            longArrayOf(input.queryPosition), longArrayOf(1L)
        )

        val outputs = module.forward(
            EValue.from(inputIdsTensor),
            EValue.from(attentionMaskTensor),
            EValue.from(candidatePositionsTensor),
            EValue.from(candidateMaskTensor),
            EValue.from(queryPositionsTensor)
        )

        require(outputs.isNotEmpty()) {
            "ExecuTorch returned no outputs"
        }

        val resultTensor = outputs[0].toTensor()

        val data = resultTensor.dataAsFloatArray

        return data.copyOf(
            input.candidateKeys.size
        )
    }

    private fun softmax(
        scores: FloatArray, temperature: Float
    ): FloatArray {

        require(scores.isNotEmpty())

        val scaled = FloatArray(scores.size)

        var maxValue = Float.NEGATIVE_INFINITY

        for (i in scores.indices) {

            scaled[i] = scores[i] / temperature

            maxValue = max(maxValue, scaled[i])
        }

        var sum = 0.0

        for (i in scaled.indices) {
            scaled[i] = exp(
                (scaled[i] - maxValue).toDouble()
            ).toFloat()

            sum += scaled[i].toDouble()
        }

        require(sum > 0.0)

        for (i in scaled.indices) {
            scaled[i] = (scaled[i] / sum).toFloat()
        }

        return scaled
    }

    private fun requireModule(): Module = modelModule ?: error("Nox model is not loaded")

    private fun requireBuilder(): NoxPromptBuilder =
        promptBuilder ?: error("Nox tokenizer is not loaded")
}
