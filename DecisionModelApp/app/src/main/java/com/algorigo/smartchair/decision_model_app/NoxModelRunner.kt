package com.algorigo.smartchair.decision_model_app

import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import org.pytorch.executorch.EValue
import android.content.Context
import java.io.File

class NoxModelRunner(private val context: Context, private val modelPath: String) {
    private var modelModule: Module? = null

    fun loadModel() {
        // ExecuTorch 모듈 로드
        modelModule = Module.load(modelPath)
    }

    fun runInference(inputIds: LongArray): LongArray {
        if (modelModule == null) return longArrayOf()

        // 1. 입력 데이터를 Tensor 형태로 변환
        val shape = longArrayOf(1, inputIds.size.toLong())
        val inputTensor = Tensor.fromBlob(inputIds, shape) //

        // 2. 입력값을 EValue(ExecuTorch Value) 래퍼로 감싸기
        val inputs = arrayOf(EValue.from(inputTensor))

        // 3. 모델 실행 (Forward Pass)
        val outputs = modelModule!!.forward(inputs.first())

        // 4. 출력 텐서 분석 및 후처리
        val outputTensor = outputs[0].toTensor()
        val outputData = outputTensor.dataAsLongArray

        return outputData
    }
}
