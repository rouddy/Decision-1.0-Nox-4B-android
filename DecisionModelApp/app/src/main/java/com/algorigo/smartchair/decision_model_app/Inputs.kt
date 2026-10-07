package com.algorigo.smartchair.decision_model_app

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer

data class NoxEncodedInput(
    val inputIds: LongArray,
    val attentionMask: LongArray,
    val candidatePositions: LongArray,
    val candidateMask: LongArray,
    val queryPosition: Long,
    val candidateKeys: List<String>
)

class NoxPromptBuilder(
    private val tokenizer: HuggingFaceTokenizer,
    private val maxSeqLen: Int,
    private val maxCandidates: Int
) {

    companion object {
        private const val SUFFIX =
            "\n\nSelect the single option best supported by the context and instructions.\nDecision:"
    }

    private fun tokenize(text: String): LongArray {
        val encoding = tokenizer.encode(
            text,
            false, // addSpecialTokens = false
            false  // withOverflowingTokens = false
        )

        return encoding.ids
    }

    /**
     * Nox's content_text():
     *
     * String -> 그대로 사용
     *
     * 여기서는 Android API의 state/instructions를 String으로 받기 때문에
     * 그대로 반환한다.
     */
    private fun contentText(value: String): String {
        return value
    }

    /**
     * Nox의 canonical_json({"key": ..., "description": ...})과
     * 동일한 key ordering을 유지한다.
     *
     * Python implementation:
     * json.dumps(..., sort_keys=True, separators=(",", ":"))
     */
    private fun canonicalCandidateJson(
        key: String,
        description: String?
    ): String {

        val resolvedDescription = description ?: key

        return buildString {
            append("{")
            append("\"description\":")
            append(quoteJson(resolvedDescription))
            append(",")
            append("\"key\":")
            append(quoteJson(key))
            append("}")
        }
    }

    private fun quoteJson(value: String): String {
        return org.json.JSONObject.quote(value)
    }

    fun encodeChoice(
        state: String,
        question: NoxChoiceQuestion
    ): NoxEncodedInput {

        require(question.criteria.isNotEmpty()) {
            "Choice question needs at least one candidate"
        }

        require(question.criteria.size <= maxCandidates) {
            "candidate count ${question.criteria.size} exceeds maxCandidates=$maxCandidates"
        }

        val prefix =
            "Context:\n" +
                    contentText(state) +
                    "\n\n" +
                    "Task type: choice\n" +
                    "Question:\n" +
                    contentText(question.instructions) +
                    "\n" +
                    "Options:"

        val ids = ArrayList<Long>()
        val candidatePositions = ArrayList<Long>()

        ids.addAll(tokenize(prefix).toList())

        for ((key, description) in question.criteria) {

            val option =
                "\n<option>\n" +
                        canonicalCandidateJson(
                            key = key,
                            description = description
                        ) +
                        "\n</option>"

            val part = tokenize(option)

            require(part.isNotEmpty()) {
                "Candidate '$key' produced no tokens"
            }

            ids.addAll(part.toList())

            // Nox candidate position = final token of option.
            candidatePositions.add(
                (ids.size - 1).toLong()
            )
        }

        // Final global query.
        ids.addAll(
            tokenize(SUFFIX).toList()
        )

        val queryPosition = ids.lastIndex.toLong()

        require(ids.size <= maxSeqLen) {
            "Input has ${ids.size} tokens, maxSeqLen=$maxSeqLen"
        }

        return pad(
            ids = ids,
            candidatePositions = candidatePositions,
            queryPosition = queryPosition,
            candidateKeys = question.criteria.keys.toList()
        )
    }

    fun encodeScore(
        state: String,
        question: NoxScoreQuestion
    ): NoxEncodedInput {

        require(question.criteria.isNotEmpty()) {
            "Score question needs at least one candidate"
        }

        require(question.criteria.size <= maxCandidates) {
            "candidate count ${question.criteria.size} exceeds maxCandidates=$maxCandidates"
        }

        val prefix =
            "Context:\n" +
                    contentText(state) +
                    "\n\n" +
                    "Task type: score\n" +
                    "Question:\n" +
                    contentText(question.instructions) +
                    "\n" +
                    "Options:"

        val ids = ArrayList<Long>()
        val candidatePositions = ArrayList<Long>()

        ids.addAll(tokenize(prefix).toList())

        question.criteria.forEachIndexed { index, description ->

            val key = index.toString()

            val option =
                "\n<option>\n" +
                        canonicalCandidateJson(
                            key = key,
                            description = description
                        ) +
                        "\n</option>"

            val part = tokenize(option)

            require(part.isNotEmpty()) {
                "Score candidate '$index' produced no tokens"
            }

            ids.addAll(part.toList())

            candidatePositions.add(
                (ids.size - 1).toLong()
            )
        }

        ids.addAll(tokenize(SUFFIX).toList())

        val queryPosition = ids.lastIndex.toLong()

        require(ids.size <= maxSeqLen) {
            "Input has ${ids.size} tokens, maxSeqLen=$maxSeqLen"
        }

        return pad(
            ids = ids,
            candidatePositions = candidatePositions,
            queryPosition = queryPosition,
            candidateKeys = question.criteria.mapIndexed { i, _ ->
                i.toString()
            }
        )
    }

    /**
     * SystemOne의 Noul은 yes/no 두 후보를 사용한다.
     */
    fun encodeNoul(
        state: String,
        question: NoxNoulQuestion
    ): NoxEncodedInput {

        val prefix =
            "Context:\n" +
                    contentText(state) +
                    "\n\n" +
                    "Task type: noul\n" +
                    "Question:\n" +
                    contentText(question.instructions) +
                    "\n" +
                    "Options:"

        val candidates = listOf(
            "false" to "The answer to the question is no.",
            "true" to "The answer to the question is yes."
        )

        val ids = ArrayList<Long>()
        val candidatePositions = ArrayList<Long>()

        ids.addAll(tokenize(prefix).toList())

        for ((key, description) in candidates) {

            val option =
                "\n<option>\n" +
                        canonicalCandidateJson(
                            key = key,
                            description = description
                        ) +
                        "\n</option>"

            val part = tokenize(option)

            require(part.isNotEmpty())

            ids.addAll(part.toList())

            candidatePositions.add(
                (ids.size - 1).toLong()
            )
        }

        ids.addAll(tokenize(SUFFIX).toList())

        val queryPosition = ids.lastIndex.toLong()

        require(ids.size <= maxSeqLen) {
            "Input has ${ids.size} tokens, maxSeqLen=$maxSeqLen"
        }

        return pad(
            ids = ids,
            candidatePositions = candidatePositions,
            queryPosition = queryPosition,
            candidateKeys = candidates.map { it.first }
        )
    }

    private fun pad(
        ids: List<Long>,
        candidatePositions: List<Long>,
        queryPosition: Long,
        candidateKeys: List<String>
    ): NoxEncodedInput {

        val paddedIds = LongArray(maxSeqLen)
        val attentionMask = LongArray(maxSeqLen)

        ids.forEachIndexed { index, value ->
            paddedIds[index] = value
            attentionMask[index] = 1L
        }

        val positions = LongArray(maxCandidates)
        val mask = LongArray(maxCandidates)

        candidatePositions.forEachIndexed { index, value ->
            positions[index] = value
            mask[index] = 1L
        }

        return NoxEncodedInput(
            inputIds = paddedIds,
            attentionMask = attentionMask,
            candidatePositions = positions,
            candidateMask = mask,
            queryPosition = queryPosition,
            candidateKeys = candidateKeys
        )
    }
}