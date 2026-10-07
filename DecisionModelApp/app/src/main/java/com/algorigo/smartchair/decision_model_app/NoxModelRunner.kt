package com.algorigo.smartchair.decision_model_app

import android.content.Context
import com.facebook.soloader.SoLoader
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File

class NoxModelRunner(private val context: Context, private val modelPath: String) {
    private var modelModule: Module? = null

    fun loadModel() {
        val file = File(modelPath)
        if (!file.exists() || file.length() == 0L) {
            throw IllegalArgumentException("Model file does not exist or is empty: $modelPath")
        }

        // Native SoLoader 초기화
        try {
            SoLoader.init(context, false)
        } catch (e: Exception) {
            e.printStackTrace()
        }

        // ExecuTorch 모듈 로드 (대용량 모델 메모리 절약을 위해 LOAD_MODE_MMAP 사용)
        modelModule = Module.load(modelPath, Module.LOAD_MODE_MMAP)
    }

    fun destroy() {
        modelModule?.destroy()
        modelModule = null
    }

    fun runInference(inputIds: LongArray): LongArray {
        val module = modelModule ?: return longArrayOf()

        // 1. 입력 데이터를 Tensor 형태로 변환
        val shape = longArrayOf(1, inputIds.size.toLong())
        val inputTensor = Tensor.fromBlob(inputIds, shape)

        // 2. 입력값을 EValue(ExecuTorch Value) 래퍼로 감싸기
        val inputs = arrayOf(EValue.from(inputTensor))

        // 3. 모델 실행 (Forward Pass)
        val outputs = module.forward(inputs.first())

        // 4. 출력 텐서 분석 및 후처리
        val outputTensor = outputs[0].toTensor()
        val outputData = outputTensor.dataAsLongArray

        return outputData
    }
}
