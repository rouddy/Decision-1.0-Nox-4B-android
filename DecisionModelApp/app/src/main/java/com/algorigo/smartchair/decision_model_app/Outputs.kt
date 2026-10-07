package com.algorigo.smartchair.decision_model_app

data class NoxChoiceQuestion(
    val instructions: String,
    val criteria: LinkedHashMap<String, String?>
)

data class NoxScoreQuestion(
    val instructions: String,
    val criteria: List<String>
)

data class NoxNoulQuestion(
    val instructions: String
)

sealed interface NoxAnswer {

    data class Choice(
        val choice: String,
        val confidence: Float,
        val probabilities: LinkedHashMap<String, Float>
    ) : NoxAnswer

    data class Noul(
        val probability: Float
    ) : NoxAnswer

    data class Score(
        val score: Float,
        val confidence: Float,
        val probabilities: List<Float>,
        val legend: LinkedHashMap<String, String>
    ) : NoxAnswer
}